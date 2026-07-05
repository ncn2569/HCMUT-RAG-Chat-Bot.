"""
rag/agent.py
============
RAGAgent — Class thực thi luồng Plan-and-Execute Agent.

LUỒNG BÊN TRONG run():
  1. PLAN: Gọi LLM (với structured output JSON) để lập kế hoạch:
     - Chọn tool chính (search_db hoặc search_web).
     - Chuẩn bị query cho tool.
     - Dự phòng fallback action.
  2. ACT: Thực thi tool đã được chọn.
  3. GENERATE: Gọi LLM để sinh câu trả lời từ context thu được.
  4. REFLECT: Nếu LLM báo không đủ thông tin → kích hoạt fallback (web search).
  5. RE-GENERATE: Tổng hợp lại câu trả lời với context mới.

"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from rag.chat.history import add_turn, get_history
from rag.generation.build_prompt import build_prompt

if TYPE_CHECKING:
    from rag.core.container import ResourceContainer
    from rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Pydantic models cho Structured Output (Plan của LLM)
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """Một hành động cụ thể mà Agent sẽ thực thi."""

    tools: str = Field(
        description="Công cụ cụ thể để hỗ trợ tìm kiếm thông tin (search_db, search_web, hoặc predict_admission)"
    )
    query: str = Field(
        description="Câu truy vấn cụ thể cho công cụ (nếu dùng search_db/search_web)"
    )
    major: Optional[str] = Field(
        default=None,
        description="Tên ngành cần tra cứu điểm chuẩn (nếu hỏi nhiều ngành, ghi cách nhau bằng dấu phẩy)",
    )
    dgnl_score: Optional[float] = Field(default=None, description="Điểm ĐGNL")
    thpt_score: Optional[float] = Field(
        default=None, description="Tổng điểm 3 môn thi THPT (0-30)"
    )
    hocba_score: Optional[float] = Field(
        default=None, description="Tổng điểm trung bình học bạ 3 môn (0-30)"
    )


class Plan(BaseModel):
    """Kế hoạch đầy đủ của Agent cho 1 lượt hỏi."""

    thought: str = Field(description="Suy luận cụ thể để lập kế hoạch")
    rewritten_query: str = Field(
        description="Câu hỏi gốc được viết lại cho rõ nghĩa dựa trên ngữ cảnh lịch sử. Nếu không cần, giữ nguyên câu gốc."
    )
    primary_action: Action
    fallback_action: Optional[Action] = Field(
        default=None,
        description="Hành động dự phòng nếu hành động chính không có kết quả",
    )


# ---------------------------------------------------------------------------
# RAGAgent class
# ---------------------------------------------------------------------------


class RAGAgent:
    """
    Plan-and-Execute Agent cho hệ thống chatbot HCMUT.

    Agent có 2 tools:
      - search_db  : Tìm trong vector database nội bộ (dùng HybridRetriever).
      - search_web : Tìm trên Web qua Tavily API.

    Usage:
        from rag.core.container import container
        from rag.pipeline import RAGPipeline
        from rag.agent import RAGAgent

        pipeline = RAGPipeline(container)
        agent = RAGAgent(pipeline, container)
        answer = agent.run("Học phí ngành CNTT năm 2025 là bao nhiêu?")
    """

    # Sentinel string để detect khi LLM không tìm được thông tin
    _NO_INFO_MARKER = "tôi không tìm thấy thông tin này trong cơ sở dữ liệu."

    def __init__(self, pipeline: "RAGPipeline", container: "ResourceContainer"):
        """
        Args:
            pipeline  : RAGPipeline —cung cấp retriever và client thông qua container.
            container : ResourceContainer — dùng trực tiếp để gọi LLM.
        """
        self._pipeline = pipeline
        self._container = container
        self._tavily_key = os.getenv("tavily_key")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, history: list = None) -> str:
        """
        Chạy toàn bộ luồng Plan → Act → Reflect → (Fallback) → Generate.

        Args:
            query   : Câu hỏi của người dùng.
            history : (Không dùng, tự lấy từ rag.chat.history).

        Returns:
            Câu trả lời cuối cùng dạng string.
        """
        # --- Bước 0: Kiểm tra Semantic Cache ---
        if cached := self._pipeline.cache.check(query):
            print(" [DEBUG - CACHE] Hit cache! Trả về ngay.")
            add_turn(query, cached, query)
            return cached

        if history_list := get_history():
            history_text = "\n".join(
                [
                    f"User: {t.get('rewritten', t['user'])}\nBot: {t['assistant']}"
                    for t in history_list[-5:]
                ]
            )
        else:
            history_text = ""
        # --- Bước 1: PLAN (Gộp Rewrite + Plan) ---
        try:
            plan = self._plan(query, history_text)
            print("\n" + "*" * 15 + " KẾ HOẠCH CỦA LLM " + "*" * 15)
            print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))
            print("*" * 48)
        except Exception as e:
            print(f"[RAGAgent] Lỗi khi lập kế hoạch: {e}")
            return "Ối chết model của gemini đang nghẽn server rồi bạn iu ơi, thử lại sau nha, khổ lắm hàng free nó thế."

        # --- Bước 2: ACT (Primary action) ---
        primary = plan.primary_action
        print(
            f" [DEBUG - TOOL] Dùng tool: '{primary.tools}' với query: '{primary.query}'"
        )

        contexts = self._dispatch_tool(primary)

        self._extracted_from_run(" NGUỒN TÀI LIỆU ", contexts)
        # --- Bước 3: GENERATE ---
        try:
            answer = self._generate(plan.rewritten_query, contexts, history_text)
        except Exception as e:
            print(f"[RAGAgent] Lỗi khi sinh câu trả lời: {e}")
            return "Ối chết model của gemini đang nghẽn server rồi bạn iu ơi, thử lại sau nha, khổ lắm hàng free nó thế."

        # --- Bước 4: REFLECT — nếu LLM báo thiếu thông tin → fallback ---
        if self._NO_INFO_MARKER in answer.lower():
            print(
                " [DEBUG - REFLECT] Không đủ thông tin, kích hoạt Fallback Web Search..."
            )
            try:
                answer, new_contexts = self._fallback_web_search(
                    plan.rewritten_query, contexts, plan.fallback_action, history_text
                )
                self._extracted_from_run(" NGUỒN TÀI LIỆU BỔ SUNG (WEB) ", new_contexts)
            except Exception as e:
                print(f"[RAGAgent] Lỗi khi tìm kiếm web dự phòng: {e}")
                # Giữ nguyên câu trả lời "không tìm thấy" nếu fallback lỗi

        # --- Bước 5: Lưu History và Cache ---
        add_turn(query, answer, plan.rewritten_query)
        if "Xin lỗi" not in answer:
            self._pipeline.cache.add(plan.rewritten_query, answer)

        return answer

    def _extracted_from_run(self, arg0, arg1):
        print("\n" + "*" * 15 + arg0 + "*" * 15)
        print(json.dumps(arg1, ensure_ascii=False, indent=2))
        print("*" * 48)

    # ------------------------------------------------------------------
    # Private: Tools
    # ------------------------------------------------------------------

    def _tool_search_db(self, query: str) -> list[str]:
        """
        Tool 1: Tìm trong vector database nội bộ.
        Gọi HybridRetriever qua pipeline.retriever — dùng full pipeline: Dense + BM25 + Rerank.

        Args:
            query : Câu truy vấn đã được Agent tối ưu hóa cho DB search.

        Returns:
            Danh sách đoạn context liên quan.
        """
        contexts, _ = self._pipeline.retriever.search(query, use_hyde=False)
        return contexts

    def _tool_search_web(self, query: str) -> list[str]:
        """
        Tool 2: Tìm trên Internet qua Tavily API.

        Args:
            query : Câu truy vấn tìm kiếm web.

        Returns:
            Danh sách context từ web (answer + các kết quả tìm kiếm).
        """
        from tavily import TavilyClient

        client = TavilyClient(api_key=self._tavily_key)
        try:
            response = client.search(
                query, max_results=3, search_depth="basic", include_answer="basic"
            )
            contexts = [response.get("answer", "")]
            contexts.extend(
                f"Nguồn: {res.get('title', '')} - Nội dung: {res.get('content', '')}"
                for res in response.get("results", [])
            )
            return [c for c in contexts if c]
        except Exception as e:
            print(f"[RAGAgent] Lỗi khi gọi Tavily API: {e}")
            return []

    def _dispatch_tool(self, action: Action) -> list[str]:
        """Thực thi tool và trả về list contexts."""
        tool_name = action.tools.lower()
        if tool_name == "search_db":
            return self._tool_search_db(action.query)
        elif tool_name == "search_web":
            return self._tool_search_web(action.query)
        elif tool_name == "predict_admission":
            return self._tool_predict_admission(action)
        else:
            print(f"[RAGAgent] Tool không hợp lệ: {tool_name}")
            return []

    # ------------------------------------------------------------------
    # Private: Plan & Generate
    # ------------------------------------------------------------------

    def _plan(self, query: str, history_text: str) -> Plan:
        """
        Gọi LLM để lập kế hoạch (tool nào, query gì, fallback gì).

        Args:
            query        : Câu hỏi của người dùng.
            history_text : Chuỗi lịch sử hội thoại (nếu có).

        Returns:
            Đối tượng Plan (Pydantic model).
        """
        history_prompt = (
            f"\nLịch sử trò chuyện gần đây:\n{history_text}\n" if history_text else ""
        )

        planner_prompt = f"""
    Bạn là bộ não Lập kế hoạch (Planner) của Trợ lý tư vấn tuyển sinh Đại học Bách Khoa TP.HCM (HCMUT).
    Nhiệm vụ của bạn là lập kế hoạch thu thập thông tin để trả lời câu hỏi: "{query}"
    {history_prompt}
    
    YÊU CẦU:
    1. Đọc lịch sử để hiểu ngữ cảnh. Nếu người dùng dùng đại từ (như "nó", "ngành đó", "cái này"), hãy phân tích lịch sử để viết lại thành câu hỏi độc lập, đầy đủ chủ ngữ vào `rewritten_query`. Nếu câu đã đủ rõ ràng, giữ nguyên.
    2. Viết suy luận logic của bạn vào `thought` (tại sao chọn tool này, bạn đang cần tìm thông tin gì).
    3. Chọn tool phù hợp nhất cho `primary_action`:
       - predict_admission: Dùng khi người dùng cung cấp điểm số (ĐGNL, THPT, Học bạ) và hỏi về cơ hội đậu/điểm chuẩn một ngành cụ thể. Trích xuất các điểm số vào schema.
       - search_db(query): Ưu tiên hàng đầu cho các thông tin tuyển sinh, điểm chuẩn chung, quy chế, học phí, ngành học của Bách Khoa.
       - search_web(query): Chỉ dùng cho thông tin rất mới (2025, 2026, dự đoán xu hướng), chính trị, xã hội, hoặc khi biết chắc chắn DB trường không có (công thức nấu ăn, review ngoài lề).
    4. Xác định `fallback_action` nếu tool chính thất bại.
    """
        response = self._container.client.models.generate_content(
            model=os.getenv("model_name"),
            contents=planner_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": Plan,
            },
        )
        return Plan.model_validate_json(response.text)

    def _generate(self, query: str, contexts: list[str], history_text: str) -> str:
        """
        Gọi LLM để sinh câu trả lời từ query và contexts.

        Args:
            query        : Câu hỏi gốc.
            contexts     : Danh sách context thu được từ tool.
            history_text : Lịch sử trò chuyện.

        Returns:
            Câu trả lời string.
        """
        context_str = "\n\n".join(contexts) if contexts else "Không tìm thấy thông tin."
        final_prompt = build_prompt(query, [context_str], history_text)
        response = self._container.client.models.generate_content(
            model=os.getenv("model_name"), contents=final_prompt
        )
        return response.text

    def _tool_predict_admission(self, action: Action) -> list[str]:
        # sourcery skip: low-code-quality
        """Tính toán điểm và dự đoán cơ hội trúng tuyển dựa trên công thức 2026 (Demo)"""
        major = action.major or ""
        dgnl = action.dgnl_score
        thpt = action.thpt_score
        hocba = action.hocba_score

        # Nếu không có điểm nào được cung cấp
        if dgnl is None and thpt is None and hocba is None:
            return [
                "Để tôi có thể tư vấn cơ hội đậu, bạn vui lòng cung cấp điểm ĐGNL, điểm THPT hoặc điểm Học bạ nhé."
            ]

        # Giả lập tính điểm theo thang 100 (Simplified 2026 formula)
        # Điểm học lực = ĐGNL*0.7 + THPT*0.2 + Học bạ*0.1
        diem_nang_luc = (
            (dgnl / 12) if dgnl else 0
        )  # Gỉa sử ĐGNL thang 1200 cho phổ biến hiện tại
        diem_thpt_quy_doi = (thpt / 30 * 100) if thpt else 0
        diem_hocba_quy_doi = (hocba / 30 * 100) if hocba else 0

        # Nếu thí sinh thiếu ĐGNL, dùng THPT bù vào theo công thức 2.2 của 2026
        if dgnl is None and thpt is not None:
            diem_nang_luc = diem_thpt_quy_doi * 0.75

        diem_hoc_luc = round(
            diem_nang_luc * 0.7 + diem_thpt_quy_doi * 0.2 + diem_hocba_quy_doi * 0.1, 2
        )
        diem_xet_tuyen = diem_hoc_luc  # Chưa cộng điểm thưởng/ưu tiên cho bản demo

        # Đọc file điểm chuẩn
        try:
            with open("data/admission_scores.json", "r", encoding="utf-8") as f:
                scores = json.load(f)
        except Exception:
            scores = {}

        # Lấy năm mới nhất
        years = [y for y in scores if y.isdigit()]
        latest_year = max(years, default=None)
        year_data = scores.get(latest_year, {}) if latest_year else {}

        majors = [m.strip() for m in major.split(",")] if major else [""]
        results = []

        # Dictionary dịch viết tắt phổ biến
        abbrev_map = {
            "khmt": "khoa học máy tính",
            "ktmt": "kỹ thuật máy tính",
            "ck": "cơ khí",
            "ô tô": "kỹ thuật ô tô",
            "logistics": "hệ thống công nghiệp",
            "vi mạch": "thiết kế vi mạch",
        }

        available_majors = [k for k in year_data.keys() if k != "phuong_thuc_xet_tuyen"]

        for m in majors:
            if not m:
                continue

            search_term = m.lower()
            # Áp dụng map viết tắt nếu có, hoặc giữ nguyên nhưng cố gắng map từ khóa
            for abbr, full_name in abbrev_map.items():
                if abbr in search_term.split():
                    search_term = search_term.replace(abbr, full_name)
                elif search_term == abbr:
                    search_term = full_name

            matched_major = None
            matched_score = None

            # 1. Thử tìm chuỗi con trước
            for k in available_majors:
                if search_term in k.lower():
                    matched_major = k
                    matched_score = float(year_data[k])
                    break

            # 2. Dùng thư viện difflib nếu chưa tìm thấy
            if not matched_major:
                import difflib

                if matches := difflib.get_close_matches(
                    search_term, available_majors, n=1, cutoff=0.4
                ):
                    matched_major = matches[0]
                    matched_score = float(year_data[matched_major])

            if not matched_major:
                results.append(
                    f"- Ngành '{m}': Không tìm thấy điểm chuẩn trong dữ liệu để so sánh."
                )
                continue

            diff = diem_xet_tuyen - matched_score
            if diff >= 2:
                status = "An toàn"
            elif diff >= -1:
                status = "Cạnh tranh"
            else:
                status = "Nguy hiểm"

            results.append(
                f"- Ngành {matched_major} (Điểm chuẩn: {matched_score}/100): Cơ hội **{status}** (Chênh lệch {diff:+.2f} điểm)."
            )

        if not results:
            return [
                "Vui lòng cung cấp tên ngành cụ thể để tôi có thể so sánh cơ hội trúng tuyển."
            ]

        result = (
            f"Tính toán dựa trên điểm bạn cung cấp (ĐGNL: {dgnl or 'Không'}, THPT: {thpt or 'Không'}, Học bạ: {hocba or 'Không'}):\n"
            f"- Điểm xét tuyển tổng hợp (ước tính): **{diem_xet_tuyen}/100**\n\n"
            f"**Đánh giá cơ hội trúng tuyển:**\n" + "\n".join(results)
        )
        return [result]

    def _fallback_web_search(
        self,
        original_query: str,
        existing_contexts: list[str],
        fallback_action: Optional[Action],
        history_text: str,
    ) -> tuple[str, list[str]]:
        """
        Khi LLM báo không đủ thông tin, tự động fallback sang web search
        rồi tổng hợp lại câu trả lời với context mới (DB + Web).

        Args:
            original_query    : Câu hỏi gốc của người dùng.
            existing_contexts : Context đã thu được từ primary tool.
            fallback_action   : Fallback action từ Plan (có thể None).
            history_text      : Lịch sử trò chuyện.

        Returns:
            Câu trả lời tổng hợp mới.
        """
        web_query = fallback_action.query if fallback_action else original_query

        # Đảm bảo query web luôn có context về trường BK
        if "Bách Khoa" not in web_query and "HCMUT" not in web_query:
            web_query += " Đại học Bách Khoa TP.HCM"

        print(f" [DEBUG - TOOL] Fallback search_web với query: '{web_query}'")

        if web_contexts := self._tool_search_web(web_query):
            combined = existing_contexts + web_contexts
            print(" [DEBUG - RE-GENERATE] Tổng hợp lại câu trả lời với context mới...")
            return self._generate(original_query, combined, history_text), web_contexts

        return self._generate(original_query, existing_contexts, history_text), []

    def __repr__(self):
        return f"RAGAgent(pipeline={self._pipeline})"
