"""
DashScope 文本生成（Generation）统一入口，供各课程练习脚本复用。

使用前请保证已 ``load_dotenv``，且将本文件所在目录（``data/课程练习``）加入 ``sys.path``，
与其它脚本 ``from utils import ...`` 的写法一致：

    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))
    from dashscope_generation import call_generation, get_model
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import dashscope
import requests

# 与嵌入调用类似：代理/VPN/抖动下 Generation 易出现 SSLEOFError，做有限次退避重试
_GEN_NETWORK_RETRY = 5
_GEN_NETWORK_RETRY_BASE_SEC = 2.0
_RETRYABLE_GENERATION_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

_logger = logging.getLogger(__name__)


def get_dashscope_api_key_from_env() -> str:
    """从环境变量读取 API Key（百炼 / DashScope 二选一）。"""
    key = (
        os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "未配置 API Key：请在环境变量或 .env 中设置 BAILIAN_API_KEY 或 DASHSCOPE_API_KEY"
        )
    return key


def set_dashscope_api_key(api_key: str | None = None) -> str:
    """
    将 ``dashscope.api_key`` 设为给定值；若为 ``None`` 则从环境变量读取。
    返回实际使用的 Key（不脱敏）。
    """
    key = (api_key or "").strip() or get_dashscope_api_key_from_env()
    dashscope.api_key = key
    return key


def call_generation(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    prompt: str | None = None,
    result_format: str = "message",
    **kwargs: Any,
) -> Any:
    """
    调用 ``dashscope.Generation.call``，封装模型名与消息列表。

    :param model: 模型名，如 ``qwen-turbo``、``qwen-plus`` 等
    :param messages: OpenAI 风格 role/content 列表
    :param api_key: 可选，不传则使用环境变量
    :param prompt: 可选；部分旧示例会传，与 ``messages`` 并存时由 DashScope SDK 决定优先级；
        若为空串则不传入 call
    :param result_format: 默认 ``message``
    :param kwargs: 透传给 ``Generation.call``（如 ``temperature``、``functions`` 等）
    """
    set_dashscope_api_key(api_key)
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "result_format": result_format,
        **kwargs,
    }
    if prompt is not None and str(prompt).strip() != "":
        params["prompt"] = prompt

    delay = _GEN_NETWORK_RETRY_BASE_SEC
    last: BaseException | None = None
    for attempt in range(1, _GEN_NETWORK_RETRY + 1):
        try:
            return dashscope.Generation.call(**params)
        except _RETRYABLE_GENERATION_ERRORS as e:
            last = e
            _logger.warning(
                "DashScope Generation 网络异常（%d/%d）: %s，%.0fs 后重试",
                attempt,
                _GEN_NETWORK_RETRY,
                e,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        except OSError as e:
            if "SSL" in type(e).__name__:
                last = e
                _logger.warning(
                    "DashScope Generation SSL 类 OSError（%d/%d）: %s，%.0fs 后重试",
                    attempt,
                    _GEN_NETWORK_RETRY,
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


def get_model(model_name: str, prompt: str, messages: list[dict[str, Any]]) -> Any:
    """
    与早期练习脚本兼容的签名：``get_model(model_name, prompt, messages)``。

    ``prompt`` 非空时会一并传给 SDK；若你仅使用 ``messages``，可将 ``prompt`` 设为 ``""``。
    """
    return call_generation(
        model_name,
        messages,
        prompt=prompt if prompt else None,
    )
