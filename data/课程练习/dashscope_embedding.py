"""
DashScope 文本嵌入（TextEmbedding）统一入口，供各课程练习脚本复用。

与 ``dashscope_generation.py`` 一致：封装 ``dashscope.TextEmbedding.call``、网络退避重试，
并提供 LangChain ``Embeddings`` 适配实现 ``DashScopeEmbeddingsSafe``。
同目录另有 ``dashscope_rerank.py``（``TextReRank`` 精排与重试）。

使用前将 ``data/课程练习`` 加入 ``sys.path`` 后：

    from dashscope_embedding import (
        call_text_embedding,
        DashScopeEmbeddingsSafe,
        get_dashscope_embeddings,
        DEFAULT_EMBEDDING_MODEL,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any, List

import dashscope
import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

from dashscope_generation import set_dashscope_api_key

# 与 langchain_community.embeddings.dashscope 中 BATCH_SIZE 一致
DASHSCOPE_EMBED_BATCH_SIZE: dict[str, int] = {
    "text-embedding-v1": 25,
}

_EMBED_NETWORK_RETRY = 5
_EMBED_NETWORK_RETRY_BASE_SEC = 2.0

RETRYABLE_EMBEDDING_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

_logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-v1"


def call_text_embedding(
    model: str,
    input_data: str | List[str],
    *,
    text_type: str,
    api_key: str | None = None,
) -> Any:
    """
    调用 ``dashscope.TextEmbedding.call``，对网络类异常做有限次指数退避重试。

    :param model: 如 ``text-embedding-v2``
    :param input_data: 单条字符串或字符串列表（列表长度受模型单批上限约束，上层宜自行分批）
    :param text_type: ``document`` 或 ``query``
    :param api_key: 可选；不传则使用 ``set_dashscope_api_key`` 从环境读取后的全局 key
    """
    set_dashscope_api_key(api_key)

    delay = _EMBED_NETWORK_RETRY_BASE_SEC
    last: BaseException | None = None
    for attempt in range(1, _EMBED_NETWORK_RETRY + 1):
        try:
            return dashscope.TextEmbedding.call(
                model=model,
                input=input_data,
                text_type=text_type,
                api_key=api_key or None,
            )
        except RETRYABLE_EMBEDDING_ERRORS as e:
            last = e
            _logger.warning(
                "DashScope TextEmbedding 网络异常（%d/%d）: %s，%.0fs 后重试",
                attempt,
                _EMBED_NETWORK_RETRY,
                e,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        except OSError as e:
            if "SSL" in type(e).__name__:
                last = e
                _logger.warning(
                    "DashScope TextEmbedding SSL 类 OSError（%d/%d）: %s，%.0fs 后重试",
                    attempt,
                    _EMBED_NETWORK_RETRY,
                    e,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            else:
                raise
    assert last is not None
    _logger.error(
        "提示：SSL/连接错误多见于公司代理、VPN、抓包或证书链问题；"
        "可检查 HTTPS_PROXY/NO_PROXY，或换网络后用 curl -v https://dashscope.aliyuncs.com 对比。"
    )
    raise last


class DashScopeEmbeddingsSafe(BaseModel, Embeddings):
    """
    直接调用 DashScope SDK，避免 langchain 用 requests.HTTPError 包装响应导致 KeyError('request')；
    对 SSL/断连等网络异常按批重试，减轻偶发抖动。
    """

    model: str = DEFAULT_EMBEDDING_MODEL
    dashscope_api_key: str = Field(default="")

    model_config = ConfigDict(extra="forbid")

    def _raise_if_bad(self, resp: Any, *, context: str) -> None:
        if getattr(resp, "status_code", None) == 200:
            return
        code = getattr(resp, "code", None)
        message = getattr(resp, "message", None)
        raise RuntimeError(
            f"{context}：status_code={getattr(resp, 'status_code', None)!r}, "
            f"code={code!r}, message={message!r}"
        )

    def _call_embedding(
        self,
        input_data: str | List[str],
        *,
        text_type: str,
    ) -> Any:
        return call_text_embedding(
            self.model,
            input_data,
            text_type=text_type,
            api_key=self.dashscope_api_key or None,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        batch_size = DASHSCOPE_EMBED_BATCH_SIZE.get(self.model, 25)
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._call_embedding(batch, text_type="document")
            self._raise_if_bad(resp, context="DashScope embed_documents 失败")
            embeddings = resp.output["embeddings"]
            out.extend(item["embedding"] for item in embeddings)
        return out

    def embed_query(self, text: str) -> List[float]:
        resp = self._call_embedding(text, text_type="query")
        self._raise_if_bad(resp, context="DashScope embed_query 失败")
        return resp.output["embeddings"][0]["embedding"]


def log_embedding_failure_hint(
    exc: BaseException,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """向量化失败时补充典型原因说明（KeyError request / SSL 等），便于排查。"""
    log = logger or _logger
    if isinstance(exc, KeyError) and exc.args == ("request",):
        log.error(
            "提示：上述 KeyError('request') 多见于 DashScope 返回非 200 时，"
            "langchain_community 用 requests.HTTPError 包装了 DashScope 的响应对象，"
            "初始化 HTTPError 时访问 .request 失败，真正的 status_code/message 未展示。"
            "请检查：代理/网络、API Key、embedding 额度与限流；"
            "text-embedding-v3 单批上限 10 条（本库内已分批，若仍失败可看控制台原始返回）。"
        )
    elif isinstance(exc, RETRYABLE_EMBEDDING_ERRORS) or (
        isinstance(exc, OSError) and "SSL" in type(exc).__name__
    ):
        log.error(
            "提示：SSL/连接类错误常见于公司代理、VPN、防火墙截断 HTTPS，或本机证书链异常。"
            "可尝试：换网络、检查 HTTPS_PROXY/NO_PROXY、暂时关闭抓包代理；"
            "仍失败时在终端用 curl -v https://dashscope.aliyuncs.com 做连通性对比。"
        )


def get_dashscope_embeddings(
    api_key: str,
    *,
    model: str | None = None,
) -> DashScopeEmbeddingsSafe:
    """与 LangChain FAISS 等配合：设置全局 key 并返回嵌入实例。"""
    key = (api_key or "").strip()
    set_dashscope_api_key(key)
    return DashScopeEmbeddingsSafe(
        model=model or DEFAULT_EMBEDDING_MODEL,
        dashscope_api_key=key,
    )
