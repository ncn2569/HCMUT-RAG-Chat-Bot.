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
            pipeline  : RAGPipeline — cung cấp retriever và client thông qua container.
            container : ResourceContainer — dùng trực tiếp để gọi LLM.
        """
        self._pipeline = pipeline
        self._container = container
        self._tavily_key = os.getenv("tavily_key")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, history: list) -> str:
        """
        Chạy toàn bộ luồng Plan → Act → Reflect → (Fallback) → Generate.

        Args:
            query   : Câu hỏi của người dùng.
            history : Lịch sử hội thoại (hiện tại chưa dùng trong plan, có thể mở rộng).

        Returns:
            Câu trả lời cuối cùng dạng string.
        """
        # --- Bước 1: PLAN ---
        plan = self._plan(query)
        print("\n" + "*" * 15 + " KẾ HOẠCH CỦA LLM " + "*" * 15)
        print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))
        print("*" * 48)

        # --- Bước 2: ACT (Primary action) ---
        primary = plan.primary_action
        print(f" [DEBUG - TOOL] Dùng tool: '{primary.tools}' với query: '{primary.query}'")

        contexts = self._dispatch_tool(primary.tools, primary.query)

        # --- Bước 3: GENERATE ---
        answer = self._generate(query, contexts)

        # --- Bước 4: REFLECT — nếu LLM báo thiếu thông tin → fallback ---
        if self._NO_INFO_MARKER in answer.lower():
            print(" [DEBUG - REFLECT] Không đủ thông tin, kích hoạt Fallback Web Search...")
            answer = self._fallback_web_search(query, contexts, plan.fallback_action)

        return answer

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
        response = client.search(
            query, max_results=3, search_depth="basic", include_answer="basic"
        )
        contexts = [response["answer"]]
        contexts.extend(
            f"Nguồn: {res['title']} - Nội dung: {res['content']}"
            for res in response.get("results", [])
        )
        return contexts

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

    def _plan(self, query: str) -> Plan:
        """
        Gọi LLM để lập kế hoạch (tool nào, query gì, fallback gì).

        Args:
            query : Câu hỏi của người dùng.

        Returns:
            Đối tượng Plan (Pydantic model).
        """
        planner_prompt = f"""
    Bạn là trợ lý tuyển sinh Đại học Bách Khoa TP.HCM. 
    Hãy lập kế hoạch tìm thông tin cho câu hỏi: "{query}"

    Công cụ hiện có:
    - search_db(query): Tìm trong vector database nội bộ về thông tin của trường BK để trả lời.
    - search_web(query): Tìm trên Web/Wikipedia về các thông tin và bối cảnh (thông tin mới năm 2024/2025/2026) để trả lời.
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

    def _generate(self, query: str, contexts: list[str]) -> str:
        """
        Gọi LLM để sinh câu trả lời từ query và contexts.

        Args:
            query    : Câu hỏi gốc.
            contexts : Danh sách context thu được từ tool.

        Returns:
            Câu trả lời string.
        """
        context_str = "\n\n".join(contexts) if contexts else "Không tìm thấy thông tin."
        final_prompt = build_prompt(query, [context_str])
        response = self._container.client.models.generate_content(
            model=os.getenv("model_name"), contents=final_prompt
        )
        return response.text

    def _fallback_web_search(
        self,
        original_query: str,
        existing_contexts: list[str],
        fallback_action: Optional[Action],
    ) -> str:
        """
        Khi LLM báo không đủ thông tin, tự động fallback sang web search
        rồi tổng hợp lại câu trả lời với context mới (DB + Web).

        Args:
            original_query    : Câu hỏi gốc của người dùng.
            existing_contexts : Context đã thu được từ primary tool.
            fallback_action   : Fallback action từ Plan (có thể None).

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
            return self._generate(original_query, combined)

        return self._generate(original_query, existing_contexts)

    def __repr__(self):
        return f"RAGAgent(pipeline={self._pipeline})"
