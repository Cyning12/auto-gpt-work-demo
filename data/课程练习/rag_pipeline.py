"""
RAG 问答编排：FAISS 检索 → 可选 DashScope 精排 → 拼 system 提示 → Generation。

依赖 ``constants``、``langchain_faiss_store``、``dashscope_rerank``、``dashscope_generation``、``utils``。
将 ``data/课程练习`` 加入 ``sys.path`` 后：

    from rag_pipeline import generate_rag_answer
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from constants import DEFAULT_RAG_SIMILARITY_THRESHOLD, DEFAULT_RAG_TOP_K
from dashscope_generation import call_dashscope_chat, chat_answer_text
from dashscope_rerank import rerank_retrieval_hits
from langchain_faiss_store import FAISS_INDEX_NAME, faiss_similarity_search
from utils import build_rag_prompt_text, format_rag_context


def generate_rag_answer(
    query: str,
    *,
    embeddings: Embeddings,
    api_key: str,
    chat_model: str,
    top_k: int = DEFAULT_RAG_TOP_K,
    score_threshold: float | None = DEFAULT_RAG_SIMILARITY_THRESHOLD,
    vector_db: FAISS | None = None,
    save_dir: str | Path | None = None,
    use_rerank: bool = True,
    index_name: str = FAISS_INDEX_NAME,
    api_key_for_rerank: str | None = None,
) -> dict[str, Any]:
    """
    检索制度片段 → 可选精排 → 调用通用对话模型生成回答。

    :param api_key: 用于 Generation；精排默认使用同一 key，可用 ``api_key_for_rerank`` 覆盖。
    :returns: ``answer``、``model``、``retrieval``、``context``
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")

    rk = api_key if api_key_for_rerank is None else api_key_for_rerank

    hits = faiss_similarity_search(
        q,
        vector_db=vector_db,
        embeddings=embeddings,
        save_dir=save_dir,
        index_name=index_name,
        k=top_k,
        score_threshold=score_threshold,
        rerank=False,
    )
    if use_rerank and hits:
        hits = rerank_retrieval_hits(q, hits, top_n=len(hits), api_key=rk)

    system_text = build_rag_prompt_text(q, hits)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": q},
    ]
    resp = call_dashscope_chat(messages, model=chat_model, api_key=api_key)
    if getattr(resp, "status_code", None) != HTTPStatus.OK:
        code = getattr(resp, "code", None)
        message = getattr(resp, "message", None)
        raise RuntimeError(
            f"DashScope 对话失败：status_code={getattr(resp, 'status_code', None)!r}, "
            f"code={code!r}, message={message!r}"
        )
    answer = chat_answer_text(resp)
    return {
        "answer": answer,
        "model": chat_model,
        "retrieval": hits,
        "context": format_rag_context(hits),
    }
