"""
百炼（DashScope）全模态对话：``MultiModalConversation``，默认模型 ``qwen3.5-omni-plus-2026-03-15``。

将 ``data/课程练习`` 加入 ``sys.path`` 后：

    from bailian_omni_multimodal import (
        DEFAULT_OMNI_MODEL,
        call_omni_multimodal,
        omni_answer_text,
        user_turn_image_and_text,
    )

``messages`` 中每条 ``content`` 为 **列表**，元素为 ``{"text": "..."}``、``{"image": "https://..."}``
或本地路径（由 DashScope SDK 预处理上传）。与 ``AI大模型原理与API使用/lessons1_pre.py`` case3 一致。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import dashscope
import requests

from dashscope_generation import set_dashscope_api_key

# 与文本 Generation 一致：网络抖动时有限次退避重试
_OMNI_NETWORK_RETRY = 5
_OMNI_NETWORK_RETRY_BASE_SEC = 2.0
_RETRYABLE_OMNI_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

_logger = logging.getLogger(__name__)

# 百炼全模态模型（可按需覆盖或通过参数传入）
DEFAULT_OMNI_MODEL = "qwen-omni-turbo-realtime"


def user_turn_image_and_text(
    image: str,
    text: str,
    *,
    role: str = "user",
) -> dict[str, Any]:
    """
    构造单轮用户消息：一张图 + 一段文字。

    :param image: 图片 URL、``file://`` 路径或本地绝对/相对路径（由 SDK 处理上传）
    :param text: 用户问题或指令
    """
    return {
        "role": role,
        "content": [
            {"image": image},
            {"text": text},
        ],
    }


def call_omni_multimodal(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_OMNI_MODEL,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    调用 ``dashscope.MultiModalConversation.call``。

    :param messages: 多模态消息列表（``role`` + ``content`` 为 ``text`` / ``image`` 等块列表）
    :param model: 模型名，默认 ``DEFAULT_OMNI_MODEL``
    :param api_key: 可选；不传则使用 ``BAILIAN_API_KEY`` / ``DASHSCOPE_API_KEY``
    :param kwargs: 透传（如 ``temperature``、``max_length``、``stream`` 等）
    """
    key = set_dashscope_api_key(api_key)
    delay = _OMNI_NETWORK_RETRY_BASE_SEC
    last: BaseException | None = None
    for attempt in range(1, _OMNI_NETWORK_RETRY + 1):
        try:
            return dashscope.MultiModalConversation.call(
                model=model,
                messages=messages,
                api_key=key,
                **kwargs,
            )
        except _RETRYABLE_OMNI_ERRORS as e:
            last = e
            _logger.warning(
                "DashScope MultiModalConversation 网络异常（%d/%d）: %s，%.0fs 后重试",
                attempt,
                _OMNI_NETWORK_RETRY,
                e,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        except OSError as e:
            if "SSL" in type(e).__name__:
                last = e
                _logger.warning(
                    "DashScope MultiModalConversation SSL 类 OSError（%d/%d）: %s，%.0fs 后重试",
                    attempt,
                    _OMNI_NETWORK_RETRY,
                    e,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            else:
                raise
    assert last is not None
    _logger.error(
        "提示：SSL/连接错误多见于代理、VPN 或证书链问题；可检查 HTTPS_PROXY/NO_PROXY。"
    )
    raise last


def omni_answer_text(response: Any) -> str:
    """
    从全模态响应中取出助手文本：优先 ``output.choices[0].message.content``，
    若为多块结构则拼接 ``text`` 字段；否则尝试 ``output.text``。
    """
    from http import HTTPStatus

    from utils import generation_first_message

    if response is None:
        raise RuntimeError("全模态响应为空")
    code = getattr(response, "status_code", None)
    if code != HTTPStatus.OK:
        msg_err = getattr(response, "message", None)
        err_code = getattr(response, "code", None)
        raise RuntimeError(
            "DashScope 全模态调用失败："
            f"status_code={code!r}, code={err_code!r}, message={msg_err!r}。"
            "若为 403 Access denied：通常是当前 API Key / Workspace 未开通该模型权限，"
            "请在百炼控制台确认模型可用与授权；或临时切换到你有权限的多模态模型（如 qwen-vl-plus）。"
        )

    msg = generation_first_message(response)
    if msg is not None:
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text")
                    if t:
                        parts.append(str(t))
                elif isinstance(item, str) and item.strip():
                    parts.append(item.strip())
            if parts:
                return "\n".join(parts).strip()

    output = (
        response.get("output")
        if hasattr(response, "get")
        else getattr(response, "output", None)
    )
    if output is not None:
        text = (
            output.get("text")
            if hasattr(output, "get")
            else getattr(output, "text", None)
        )
        if text:
            return str(text).strip()

    raise RuntimeError(
        "全模态响应中未解析到有效文本（choices/message 与 output.text 均为空）"
    )
