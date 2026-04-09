"""
Web Search 统一入口（当前仅接入 Tavily）。

目的：
- 给课程练习脚本提供可复用的联网搜索能力
- 统一：API Key 读取、超时、重试、返回结构

使用前：
- 在 .env 或环境变量中配置 TAVILY_API_KEY
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

from tavily import TavilyClient

_logger = logging.getLogger(__name__)


def get_tavily_api_key_from_env() -> str:
    key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("未配置 TAVILY_API_KEY（请在环境变量或 .env 中设置）")
    return key


SearchDepth = Literal["basic", "advanced"]
TimeRange = Literal["day", "week", "month", "year"]


def tavily_search(
    query: str,
    *,
    api_key: str | None = None,
    max_results: int = 3,
    search_depth: SearchDepth = "basic",
    time_range: TimeRange | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    include_answer: bool = False,
    include_raw_content: bool = False,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict[str, Any]:
    """
    调用 Tavily Python SDK（官方推荐写法）。

    返回（示例字段）：
    - query
    - results: [{title, url, content, score, raw_content?}, ...]
    - answer?（可选）
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")

    key = (api_key or "").strip() or get_tavily_api_key_from_env()
    client = TavilyClient(key)

    params: dict[str, Any] = {
        "query": q,
        "search_depth": search_depth,
        "max_results": int(max_results),
        "include_answer": bool(include_answer),
        "include_raw_content": bool(include_raw_content),
        "include_domains": include_domains,
        "exclude_domains": exclude_domains,
    }
    if time_range is not None:
        params["time_range"] = time_range
    if start_date is not None:
        params["start_date"] = start_date
    if end_date is not None:
        params["end_date"] = end_date

    # 网络抖动做有限次重试（SDK 内部也可能抛异常）
    retry = 3
    base_delay = 1.5
    last: BaseException | None = None
    for attempt in range(1, retry + 1):
        try:
            data = client.search(**params)
            if not isinstance(data, dict):
                raise RuntimeError("Tavily 返回不是 JSON 对象")
            return data
        except Exception as e:
            last = e
            if attempt >= retry:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), 8.0)
            _logger.warning(
                "Tavily 搜索异常（%d/%d）: %s，%.1fs 后重试",
                attempt,
                retry,
                e,
                delay,
            )
            time.sleep(delay)
    assert last is not None
    raise last


def search_web(
    query: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    统一入口：后续可扩展多 provider（目前仅 tavily）。
    """
    p = (provider or os.getenv("WEB_SEARCH_PROVIDER") or "tavily").strip().lower()
    if p != "tavily":
        raise ValueError(f"不支持的 WEB_SEARCH_PROVIDER={p!r}（当前仅支持 'tavily'）")
    return tavily_search(query, **kwargs)
