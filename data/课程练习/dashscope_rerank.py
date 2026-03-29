"""
DashScope 文本精排（TextReRank）统一入口，与 ``dashscope_embedding.py``、``dashscope_generation.py`` 并列。

封装 ``dashscope.TextReRank.call`` 与网络退避重试；可重试异常类型与嵌入模块一致。

使用前将 ``data/课程练习`` 加入 ``sys.path`` 后：

    from dashscope_rerank import call_text_rerank, DEFAULT_RERANK_MODEL, rerank_retrieval_hits
"""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from typing import Any

import dashscope

from dashscope_embedding import RETRYABLE_EMBEDDING_ERRORS
from dashscope_generation import get_dashscope_api_key_from_env

_RERANK_NETWORK_RETRY = 5
_RERANK_NETWORK_RETRY_BASE_SEC = 2.0

_logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "gte-rerank-v2"


def call_text_rerank(
    query: str,
    documents: list[str],
    *,
    top_n: int,
    model: str = DEFAULT_RERANK_MODEL,
    api_key: str | None = None,
    return_documents: bool = False,
) -> Any:
    """
    调用 ``dashscope.TextReRank.call``，对 SSL/断连等网络类异常做有限次指数退避重试。

    :param query: 查询语句
    :param documents: 待重排的纯文本列表（与向量召回片段的 ``page_content`` 对应）
    :param top_n: 返回前 n 条
    :param model: 精排模型名，默认 ``gte-rerank-v2``
    :param api_key: 可选；不传则依赖 SDK 全局或环境（与 DashScope 其它接口一致）
    :param return_documents: 是否让接口返回文档全文（一般与向量侧重复，默认 False）
    """
    delay = _RERANK_NETWORK_RETRY_BASE_SEC
    last: BaseException | None = None
    for attempt in range(1, _RERANK_NETWORK_RETRY + 1):
        try:
            return dashscope.TextReRank.call(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=return_documents,
                api_key=api_key or None,
            )
        except RETRYABLE_EMBEDDING_ERRORS as e:
            last = e
            _logger.warning(
                "DashScope TextReRank 网络异常（%d/%d）: %s，%.0fs 后重试",
                attempt,
                _RERANK_NETWORK_RETRY,
                e,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    assert last is not None
    raise last


def rerank_retrieval_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_n: int | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    对向量检索得到的 ``hits``（每项含 ``content``）调用精排，为每条增加 ``rerank_score``。

    失败时记录警告并**原样返回** ``hits``（不中断上层 RAG）。
    ``api_key`` 为空时从环境变量读取（与 Generation 一致）。
    """
    if not hits:
        return []
    key = (api_key or "").strip() or get_dashscope_api_key_from_env()
    n = top_n if top_n is not None else len(hits)
    n = max(1, min(n, len(hits)))
    docs = [str(h.get("content", "") or "") for h in hits]
    try:
        resp = call_text_rerank(query, docs, top_n=n, api_key=key)
    except Exception as e:
        _logger.warning("Rerank 请求异常，回退为向量检索顺序：%s", e)
        return hits
    if getattr(resp, "status_code", None) != HTTPStatus.OK:
        _logger.warning(
            "Rerank 非成功响应，回退为向量检索顺序：status_code=%r message=%r",
            getattr(resp, "status_code", None),
            getattr(resp, "message", None),
        )
        return hits
    output = getattr(resp, "output", None)
    if output is None or not getattr(output, "results", None):
        _logger.warning("Rerank 返回无 output.results，回退为向量检索顺序")
        return hits
    out: list[dict[str, Any]] = []
    for res in output.results:
        idx = int(res.index if hasattr(res, "index") else res["index"])
        if idx < 0 or idx >= len(hits):
            continue
        row = dict(hits[idx])
        rs = float(
            res.relevance_score
            if hasattr(res, "relevance_score")
            else res["relevance_score"]
        )
        row["rerank_score"] = rs
        out.append(row)
    return out if out else hits
