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
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from rag.generation.build_prompt import build_prompt
from rag.chat.history import add_turn, get_history

if TYPE_CHECKING:
    from rag.core.container import ResourceContainer
    from rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Pydantic models cho Structured Output (Plan của LLM)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Một hành động cụ thể mà Agent sẽ thực thi."""
    tools: str = Field(
        description="Công cụ cụ thể để hỗ trợ tìm kiếm thông tin (search_db hoặc search_web)"
    )
    query: str = Field(description="Câu truy vấn cụ thể cho công cụ")


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
        answer = agent.run("Học phí ngành CNTT năm 2025 là bao nhiêu?", history=[])
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
            return "Xin lỗi, hiện tại hệ thống đang bận. Vui lòng thử lại sau."

        # --- Bước 2: ACT (Primary action) ---
        primary = plan.primary_action
        print(f" [DEBUG - TOOL] Dùng tool: '{primary.tools}' với query: '{primary.query}'")

        contexts = self._dispatch_tool(primary.tools, primary.query)

        self._extracted_from_run_59(" NGUỒN TÀI LIỆU ", contexts)
        # --- Bước 3: GENERATE ---
        try:
            answer = self._generate(plan.rewritten_query, contexts, history_text)
        except Exception as e:
            print(f"[RAGAgent] Lỗi khi sinh câu trả lời: {e}")
            return "Xin lỗi, đã xảy ra lỗi trong quá trình tạo câu trả lời. Vui lòng thử lại sau."

        # --- Bước 4: REFLECT — nếu LLM báo thiếu thông tin → fallback ---
        if self._NO_INFO_MARKER in answer.lower():
            print(" [DEBUG - REFLECT] Không đủ thông tin, kích hoạt Fallback Web Search...")
            try:
                answer, new_contexts = self._fallback_web_search(plan.rewritten_query, contexts, plan.fallback_action, history_text)
                self._extracted_from_run(" NGUỒN TÀI LIỆU BỔ SUNG (WEB) ", new_contexts)
            except Exception as e:
                print(f"[RAGAgent] Lỗi khi tìm kiếm web dự phòng: {e}")
                # Giữ nguyên câu trả lời "không tìm thấy" nếu fallback lỗi

        # --- Bước 5: Lưu History và Cache ---
        add_turn(query, answer, plan.rewritten_query)
        if "Xin lỗi" not in answer:
            self._pipeline.cache.add(plan.rewritten_query, answer)

        return answer

    # TODO Rename this here and in `run`
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

    def _dispatch_tool(self, tool_name: str, query: str) -> list[str]:
        """
        Routing: gọi đúng tool dựa trên tên.

        Args:
            tool_name : "search_db" hoặc "search_web".
            query     : Câu truy vấn cho tool.

        Returns:
            Kết quả contexts từ tool, hoặc list rỗng nếu tool không hợp lệ.
        """
        if tool_name == "search_db":
            return self._tool_search_db(query)
        if tool_name == "search_web":
            return self._tool_search_web(query)

        print(f" [DEBUG - TOOL] Tool không hợp lệ: '{tool_name}', bỏ qua.")
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
        history_prompt = f"\nLịch sử trò chuyện gần đây:\n{history_text}\n" if history_text else ""
        
        planner_prompt = f"""
    Bạn là bộ não Lập kế hoạch (Planner) của Trợ lý tư vấn tuyển sinh Đại học Bách Khoa TP.HCM (HCMUT).
    Nhiệm vụ của bạn là lập kế hoạch thu thập thông tin để trả lời câu hỏi: "{query}"
    {history_prompt}
    
    YÊU CẦU:
    1. Đọc lịch sử để hiểu ngữ cảnh. Nếu người dùng dùng đại từ (như "nó", "ngành đó", "cái này"), hãy phân tích lịch sử để viết lại thành câu hỏi độc lập, đầy đủ chủ ngữ vào `rewritten_query`. Nếu câu đã đủ rõ ràng, giữ nguyên.
    2. Viết suy luận logic của bạn vào `thought` (tại sao chọn tool này, bạn đang cần tìm thông tin gì).
    3. Chọn tool phù hợp nhất cho `primary_action`:
       - search_db(query): Ưu tiên hàng đầu cho các thông tin tuyển sinh, điểm chuẩn, quy chế, học phí, ngành học của Bách Khoa.
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

    def _fallback_web_search(
        self,
        original_query: str,
        existing_contexts: list[str],
        fallback_action: Optional[Action],
        history_text: str
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
