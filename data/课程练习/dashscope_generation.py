"""
DashScope 文本生成（Generation）统一入口，供各课程练习脚本复用。

使用前请保证已 ``load_dotenv``，且将本文件所在目录（``data/课程练习``）加入 ``sys.path``，
与其它脚本 ``from utils import ...`` 的写法一致：

    _root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_root))
    from dashscope_generation import call_generation, get_model
"""

from __future__ import annotations

import json
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
    key = (os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
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


def call_dashscope_chat(
    messages: list[dict[str, Any]],
    *,
    model: str,
    api_key: str | None = None,
) -> Any:
    """OpenAI 风格 ``messages`` 对话；与嵌入 / 精排模型相互独立。"""
    return call_generation(model, messages, api_key=api_key)


def chat_answer_text(response: Any) -> str:
    """从 Generation 响应中取出 assistant 文本；无有效内容时抛 ``RuntimeError``。"""
    from utils import generation_first_message

    msg = generation_first_message(response)
    if msg is None:
        code = getattr(response, "status_code", None)
        msg_err = getattr(response, "message", None)
        raise RuntimeError(
            f"DashScope 对话无有效回复：status_code={code!r}, message={msg_err!r}"
        )
    content = (
        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    )
    if content:
        # 兼容 content 为多块（少见）：拼接 text
        if isinstance(content, list):
            parts: list[str] = []
            for it in content:
                if isinstance(it, dict):
                    t = it.get("text")
                    if t:
                        parts.append(str(t))
                elif isinstance(it, str) and it.strip():
                    parts.append(it.strip())
            if parts:
                return "\n".join(parts).strip()
        return str(content).strip()

    # 兜底：部分返回可能把文本放在 output.text（而非 choices.message.content）
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

    code = getattr(response, "status_code", None)
    msg_err = getattr(response, "message", None)
    raise RuntimeError(
        f"DashScope 返回的 assistant message 无 content（status_code={code!r}, message={msg_err!r}）"
    )


def call_generation_can_search(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    max_tool_turns: int = 3,
    search_max_results: int = 5,
    search_depth: str = "basic",
    **kwargs: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    """
    在 ``call_generation`` 的基础上增加“可联网搜索”的工具调用闭环（Function Calling 风格）。

    - 不影响其它 demo：只有显式调用本函数时才启用 tools。
    - 当前仅内置一个工具：``web_search``（使用 Tavily，见 ``web_search.search_web``）。

    返回：(最终 response, 完整 messages 轨迹)。
    """
    from utils import generation_first_message, message_function_call
    from web_search import search_web

    tool_spec = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "联网搜索：在互联网上检索最新/实时信息，返回结构化结果。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询"},
                        "max_results": {
                            "type": "integer",
                            "description": "返回结果条数",
                            "default": search_max_results,
                        },
                        "search_depth": {
                            "type": "string",
                            "description": "搜索深度：basic/advanced",
                            "default": search_depth,
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    msgs: list[dict[str, Any]] = [dict(m) for m in (messages or [])]
    turns = 0
    _logger.info(
        "call_generation_can_search: start model=%r tool=web_search max_tool_turns=%d",
        model,
        int(max_tool_turns),
    )
    while True:
        resp = call_generation(
            model,
            msgs,
            api_key=api_key,
            tools=tool_spec,
            result_format="message",
            **kwargs,
        )
        msg = generation_first_message(resp)
        if msg is None:
            return resp, msgs

        # 记录 assistant 消息（可能带 function_call）
        if isinstance(msg, dict):
            msgs.append(dict(msg))
        else:
            # 兜底：尽量转成 dict
            msgs.append(
                {
                    "role": getattr(msg, "role", "assistant"),
                    "content": getattr(msg, "content", ""),
                }
            )

        # DashScope 可能返回 function_call 或 tool_calls（OpenAI 风格）
        fc = message_function_call(msg)
        if not fc and isinstance(msg, dict):
            tc = msg.get("tool_calls")
            if isinstance(tc, list) and tc:
                first = tc[0] if isinstance(tc[0], dict) else None
                if first and (first.get("type") == "function" or "function" in first):
                    fn = first.get("function") or {}
                    if isinstance(fn, dict):
                        fc = {
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments"),
                        }
        if not fc:
            _logger.info("call_generation_can_search: done (no tool call)")
            return resp, msgs
        if turns >= int(max_tool_turns):
            raise RuntimeError("工具调用轮次超过上限，已中止（防止死循环）")
        turns += 1

        name = fc.get("name") if isinstance(fc, dict) else getattr(fc, "name", "")
        args_raw = (
            fc.get("arguments")
            if isinstance(fc, dict)
            else getattr(fc, "arguments", "")
        )
        if name != "web_search":
            raise RuntimeError(f"不支持的 function_call: {name!r}")
        try:
            args = (
                json.loads(args_raw)
                if isinstance(args_raw, str) and args_raw.strip()
                else {}
            )
        except Exception as e:
            raise RuntimeError(f"web_search arguments 解析失败: {args_raw!r}") from e

        q = str(args.get("query") or "").strip()
        if not q:
            tool_out = {"error": "empty_query"}
            _logger.warning(
                "call_generation_can_search: tool_call web_search (turn=%d) empty query",
                turns,
            )
        else:
            _logger.info(
                "call_generation_can_search: tool_call web_search (turn=%d) query=%r max_results=%s depth=%s",
                turns,
                q,
                args.get("max_results"),
                args.get("search_depth"),
            )
            tool_out = search_web(
                q,
                max_results=int(args.get("max_results") or search_max_results),
                search_depth=str(args.get("search_depth") or search_depth),
            )
            n = tool_out.get("results")
            n2 = len(n) if isinstance(n, list) else None
            _logger.info(
                "call_generation_can_search: web_search ok (turn=%d) results=%s",
                turns,
                n2,
            )

        # 按 function_call 约定回填 function 结果
        msgs.append(
            {
                "role": "function",
                "name": "web_search",
                "content": json.dumps(tool_out, ensure_ascii=False),
            }
        )
