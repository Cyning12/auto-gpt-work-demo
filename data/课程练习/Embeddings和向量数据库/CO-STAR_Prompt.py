# CO-STAR_Prompt.py

# 课程练习：Embeddings和向量数据库

# 1. 导入必要的库
import os
import sys
import uuid
from pathlib import Path

_practice_root = Path(__file__).resolve().parents[1]
if str(_practice_root) not in sys.path:
    sys.path.insert(0, str(_practice_root))
from utils import append_practice_jsonl_line, list_doc_files, pick, read_local_file
from constants import PROFESSIONAL_SYSTEM_TEMPLATE

import dashscope
from dotenv import load_dotenv
from dashscope.api_entities.dashscope_response import Role
import json
import random
from collections.abc import Mapping
from http import HTTPStatus

_here = Path(__file__).resolve().parent
# 与脚本同目录；也可用环境变量 CO_STAR_LOG_DIR / PRACTICE_JSONL_LOG_DIR 指定绝对路径
_log_override = (
    os.environ.get("CO_STAR_LOG_DIR") or os.environ.get("PRACTICE_JSONL_LOG_DIR") or ""
).strip()
LOG_DIR = (
    Path(_log_override).expanduser().resolve() if _log_override else (_here / "log")
)
# 单条 tool 返回写入「本轮汇总行」时的最大字符数
_LOG_TOOL_CONTENT_MAX = 12000
# 快照里每条 message 的 content 最长（含 tool 角色），避免 JSONL 单条过大
_LOG_MESSAGE_SNAPSHOT_MAX = 8000

# read_local_file 返回超过该字符数时，经二级摘要模型压缩后再交给主对话（测试阶段宜小；正式可调大）
# 也可用环境变量覆盖：export TOOL_OUTPUT_SUMMARY_THRESHOLD_CHARS=8000
TOOL_OUTPUT_SUMMARY_THRESHOLD_CHARS = int(
    os.environ.get("TOOL_OUTPUT_SUMMARY_THRESHOLD_CHARS") or "100"
)

load_dotenv(_here / ".env")
load_dotenv()
api_key = (os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
dashscope.api_key = api_key or None

# System 主模板见 constants.PROFESSIONAL_SYSTEM_TEMPLATE；工具分流规则在 _case4_system_content 末尾拼接。


def _case4_system_content() -> str:
    """合并 CO-STAR 与工具分流规则，作为唯一 system 消息。"""
    tool_rules = (
        "你是企业内的运维与制度咨询助手，只能通过工具获取事实，禁止编造文档内容。\n"
        "【工具分流——必须遵守】\n"
        "1）用户问「怎么处理、操作步骤、SOP、手册、制度、规定、策略、标准、联系人、文档编号」"
        "或告警场景下需要「按公司书面流程处置」：必须先依次调用 list_doc_files，再 read_local_file 读取相关文件，"
        "最后基于文件内容作答；不可仅凭 get_current_status 回答流程类问题。\n"
        "2）仅当用户明确索要「当前/此刻/现在」的监控数值（连接数、CPU、内存）时，才调用 get_current_status；"
        "可与文档工具配合，但文档问题不得省略文档工具。\n"
        "3）若用户同时需要流程与实时数据：先完成文档检索与阅读，再决定是否调用 get_current_status。\n"
        "回答时注明信息来自哪份文件名中的条款（若读过文件）。"
    )
    system_template = PROFESSIONAL_SYSTEM_TEMPLATE.format(
        role_definition="你是一名资深运维专家。根据工具返回的企业内部文档片段或监控数据，回答用户关于服务器异常与制度类问题。",
        audience="面向值班运维工程师；回答应可直接用于值班与排障决策，给出清晰、可执行的操作建议。",
        constraints="仅基于 tool 返回的文档正文或 JSON 作答；不得编造工具结果中不存在的条款、数字、联系人。信息不足时明确说明。",
        output_format="输出格式：\n1. 问题定位：...\n2. 参考文档：[文档标题/编号]\n3. 建议操作：...",
        language="中文",
        style="模仿高级 SRE 工程师的排障日志格式。使用 Markdown 代码块展示技术参数，用无序列表呈现步骤；每段文字尽量不超过 3 行。",
        tone="保持专业、客观；避免「我觉得」「可能」等模糊表述；指出系统问题时语气直接、果断。",
    )
    return f"{system_template}\n\n---\n\n{tool_rules}"


# 通过第三方接口获取数据库服务器状态
def get_current_status():
    # 生成连接数数据
    connections = random.randint(10, 100)
    # 生成CPU使用率数据
    cpu_usage = round(random.uniform(1, 100), 1)
    # 生成内存使用率数据
    memory_usage = round(random.uniform(10, 100), 1)
    status_info = {
        "连接数": connections,
        "CPU使用率": f"{cpu_usage}%",
        "内存使用率": f"{memory_usage}%",
    }
    return json.dumps(status_info, ensure_ascii=False)


# 工具顺序会影响模型倾向：文档类工具置前，避免「告警」场景一律打监控接口
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_doc_files",
            "description": (
                "【文档检索第一步】列出知识库 doc 目录下可读取文件的相对路径（JSON）。"
                "在以下任一情况必须优先调用本工具（再调用 read_local_file），不要调用 get_current_status："
                "用户问处理步骤/SOP/手册/制度/政策/规范/备份策略/报销标准/团建安排/前端规范；"
                "或用户描述告警但需要你「按制度给出处置步骤、联系人、编号」；"
                "或问题里出现「依据哪份文档、公司规定、书面流程」等。"
                "默认不递归子目录；需要时再设 recursive=true。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recursive": {
                        "type": "boolean",
                        "description": "是否包含子目录，默认 false，仅在需要时设为 true",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "最多返回多少个文件路径，默认 100，最大 300",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": (
                "【文档检索第二步】读取 doc 内某个文件的完整文本。"
                "relative_path 必须来自 list_doc_files 的 files 列表。"
                "典型文件名：运维助手工具说明.md（各工具何时调用）、服务器异常处置手册 (SOP).txt、"
                "DB_Backup_Policy.md、员工差旅报销制度 (Policy).txt、Frontend_Review_Guide.txt、Q1_Teambuilding.txt 等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "相对于 doc 的文件路径，例如 服务器异常处置手册 (SOP).txt",
                    },
                },
                "required": ["relative_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_status",
            "description": (
                "【仅实时指标】仅当用户**明确**要「当前/此刻/现在」的监控数值时使用："
                "数据库连接数、CPU 使用率、内存使用率等快照。"
                "若用户要的是处置流程、制度条款、备份时间、报销额度等——禁止只用本工具，必须先 list_doc_files + read_local_file。"
                "若既要文档又要实时数据，应先读完相关文档再视需要调用本工具辅助。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


LAST_API_ERROR: str | None = None


# 封装模型相应函数
def get_case4_response(messages: list[dict]):
    global LAST_API_ERROR
    try:
        response = dashscope.Generation.call(
            model="qwen-turbo",
            messages=messages,
            tools=tools,
            result_format="message",
        )
        LAST_API_ERROR = None
        return response
    except Exception as e:
        LAST_API_ERROR = repr(e)
        return None


def _tool_arguments_to_dict(raw) -> dict:
    if isinstance(raw, str):
        return json.loads(raw) if raw.strip() else {}
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


# Map-Reduce 长文预处理（与主对话 tools 无关的纯文本补全）
_MAP_REDUCE_MODEL = os.environ.get("MAP_REDUCE_MODEL") or "qwen-turbo"
MAP_REDUCE_CHUNK_SIZE = int(os.environ.get("MAP_REDUCE_CHUNK_SIZE") or "4000")
MAP_REDUCE_OVERLAP = int(os.environ.get("MAP_REDUCE_OVERLAP") or "400")


def _split_content_chunks(content: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符长度滑动窗口切分；overlap 为相邻块尾部与头部的重叠长度。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 须满足 0 <= overlap < chunk_size")
    n = len(content)
    if n == 0:
        return []
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(content[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def _generation_plain_text(system: str, user: str) -> str | None:
    """无 tools 的单次生成，返回 assistant 文本；失败返回 None。"""
    try:
        response = dashscope.Generation.call(
            model=_MAP_REDUCE_MODEL,
            messages=[
                {"role": Role.SYSTEM, "content": system},
                {"role": Role.USER, "content": user},
            ],
            result_format="message",
        )
        if response is None or getattr(response, "status_code", None) != HTTPStatus.OK:
            return None
        output = pick(response, "output")
        choices = pick(output, "choices") if output else None
        if not choices:
            return None
        msg = pick(choices[0], "message")
        text = pick(msg, "content")
        if text is None:
            return None
        s = str(text).strip()
        return s if s else None
    except Exception:
        return None


def process_long_file(
    content: str,
    chunk_size: int = MAP_REDUCE_CHUNK_SIZE,
    overlap: int = MAP_REDUCE_OVERLAP,
) -> str:
    """
    Map-Reduce 式预处理：长文本切块 → 每块单独摘要（Map）→ 合并为一份总摘要（Reduce）。

    - 按字符切片，不保证在段落/句子边界截断（课程练习用）。
    - 需配置可用的 DashScope API Key；任一步失败则返回已得部分的降级文本说明。
    """
    if not isinstance(content, str):
        content = str(content)
    if not content:
        return ""

    chunks = _split_content_chunks(content, chunk_size, overlap)
    if not chunks:
        return ""

    map_system = (
        "你是文档摘要助手。用户会给出长文档中的一段连续文字（可能从句中切开）。"
        "请用中文输出该段的结构化短摘要：保留出现的标题/编号/条款号、数字、联系人、步骤要点；"
        "不要编造片段中不存在的信息。"
    )
    partial_summaries: list[str] = []
    for i, ch in enumerate(chunks):
        header = f"[分块 {i + 1}/{len(chunks)}，约 {len(ch)} 字符]\n"
        piece = _generation_plain_text(map_system, header + ch)
        if piece is None:
            piece = f"（第 {i + 1} 块摘要失败，保留原文前 500 字）\n{ch[:500]}"
        partial_summaries.append(f"### 分块 {i + 1} 摘要\n{piece}")

    merged_input = "\n\n".join(partial_summaries)
    reduce_system = (
        "下面是同一篇长文档各分块的摘要，块与块之间可能有少量内容重叠。"
        "请合并为一份连贯的中文总摘要：去重、统一术语，保留全部关键事实与编号；"
        "不要添加各分块摘要中均未出现的内容。"
    )
    final_text = _generation_plain_text(
        reduce_system,
        f"共分 {len(chunks)} 块。\n\n{merged_input}",
    )
    if final_text is None:
        return "[Map-Reduce] Reduce 步骤失败，仅返回各分块摘要拼接：\n\n" + merged_input
    return (
        f"[Map-Reduce] 原文字符 {len(content)}，块大小 {chunk_size}，重叠 {overlap}，共 {len(chunks)} 块。\n\n"
        + final_text
    )


def _json_safe(obj):
    """转为可 json.dumps 的结构（用于 JSONL）；失败时降级为可序列化占位，避免整轮日志丢失。"""
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except (TypeError, ValueError, RecursionError) as e:
        return {
            "_json_safe_error": repr(e),
            "_repr": repr(obj)[:4000],
        }


def _truncate_log_text(s: str, max_len: int = _LOG_TOOL_CONTENT_MAX) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...[truncated,total_len={len(s)}]"


def _summarize_messages_for_log(messages: list) -> list:
    """本次请求发送前的完整 messages 快照（截断长 content），用于 JSONL 复盘。"""
    snap: list[dict] = []
    lim = _LOG_MESSAGE_SNAPSHOT_MAX
    for m in messages:
        if not isinstance(m, Mapping):
            m = dict(m) if hasattr(m, "items") else {}
        role = m.get("role")
        item: dict = {"role": str(role) if role is not None else ""}
        raw_c = m.get("content")
        if raw_c is None:
            item["content"] = ""
        elif isinstance(raw_c, str):
            item["content"] = _truncate_log_text(raw_c, lim)
        else:
            item["content"] = _json_safe(raw_c)
        if m.get("tool_calls"):
            item["tool_calls"] = _json_safe(m["tool_calls"])
        if m.get("tool_call_id") is not None:
            item["tool_call_id"] = m["tool_call_id"]
        if m.get("name") is not None:
            item["name"] = m["name"]
        snap.append(item)
    return snap


def _append_jsonl_line(record: dict) -> Path:
    """委托 utils.append_practice_jsonl_line（含 fsync）。"""
    record.setdefault("script", "CO-STAR_Prompt")
    return append_practice_jsonl_line(LOG_DIR, record)


def _log_secondary_summary(extra: dict) -> None:
    """写入 ``LOG_DIR/YYYY-MM-DD_secondary.jsonl``。"""
    row = {
        "script": "CO-STAR_Prompt",
        "event": "secondary_summary",
        **extra,
    }
    append_practice_jsonl_line(LOG_DIR, row, secondary=True)


def _secondary_summary_tool_output(
    raw_text: str, *, source_path: str | None = None
) -> str:
    """
    对过长 ``read_local_file`` 原文走 Map-Reduce（``process_long_file``），再交给主对话。
    不携带 tools；失败时退回截断原文。每次调用写一行 ``*_secondary.jsonl``。
    """
    orig_len = len(raw_text)
    cap = 8000
    base_log = {
        "source_path": source_path,
        "original_chars": orig_len,
        "threshold_chars": TOOL_OUTPUT_SUMMARY_THRESHOLD_CHARS,
        "model": _MAP_REDUCE_MODEL,
        "mode": "map_reduce",
        "chunk_size": MAP_REDUCE_CHUNK_SIZE,
        "overlap": MAP_REDUCE_OVERLAP,
    }
    try:
        merged = process_long_file(
            raw_text,
            chunk_size=MAP_REDUCE_CHUNK_SIZE,
            overlap=MAP_REDUCE_OVERLAP,
        )
        reduce_failed = "Reduce 步骤失败" in merged
        partial_map_failed = "块摘要失败" in merged
        _log_secondary_summary(
            {
                **base_log,
                "ok": not reduce_failed,
                "outcome": (
                    "map_reduce_partial"
                    if partial_map_failed and not reduce_failed
                    else (
                        "map_reduce_reduce_failed" if reduce_failed else "map_reduce_ok"
                    )
                ),
                "result_chars": len(merged),
                "summary_preview": merged[:600],
            }
        )
        return (
            f"[read_local_file → Map-Reduce 预处理] 原文 {orig_len} 字符"
            f"（阈值 {TOOL_OUTPUT_SUMMARY_THRESHOLD_CHARS}，块 {MAP_REDUCE_CHUNK_SIZE}"
            f"/重叠 {MAP_REDUCE_OVERLAP}）。\n\n"
            f"{merged}"
        )
    except Exception as e:
        _log_secondary_summary(
            {
                **base_log,
                "ok": False,
                "outcome": "exception",
                "exception": repr(e),
            }
        )
        tail = f"\n...[truncated, total={orig_len}]" if orig_len > cap else ""
        return (
            f"[Map-Reduce 预处理异常 {e!r}，返回原文前 {cap} 字]\n"
            + raw_text[:cap]
            + tail
        )


def _invoke_tool_function(name: str, arguments_json: dict) -> str:
    if not name or name not in globals():
        return json.dumps({"error": f"未知函数: {name}"}, ensure_ascii=False)
    function = globals()[name]
    try:
        out = function(**arguments_json)
    except TypeError:
        try:
            out = function()
        except Exception as e2:
            return json.dumps({"error": str(e2)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    if isinstance(out, str):
        if (
            name == "read_local_file"
            and TOOL_OUTPUT_SUMMARY_THRESHOLD_CHARS > 0
            and len(out) > TOOL_OUTPUT_SUMMARY_THRESHOLD_CHARS
        ):
            rel = arguments_json.get("relative_path")
            sp = (rel.strip() or None) if isinstance(rel, str) else None
            out = _secondary_summary_tool_output(out, source_path=sp)
        return out
    if isinstance(out, (dict, list)):
        return json.dumps(out, ensure_ascii=False)
    return str(out)


def case4():
    """单次用户提问内多轮 tool 调用，直到模型返回纯文本（无 tool_calls）。"""
    max_tool_rounds = 16

    while True:
        content = input(
            "请输入问题（back 返回菜单 / exit 退出）。\n"
            "  · 要查文档/SOP/制度：例如「连接数暴涨按手册该怎么处理」「一线城市住宿报销上限」\n"
            "  · 要实时指标：例如「现在数据库连接数和 CPU、内存各是多少」\n"
            "> "
        )
        cmd = content.strip().lower()
        if cmd == "exit":
            print("再见")
            sys.exit(0)
        if cmd == "back":
            return
        messages = [
            {"role": Role.SYSTEM, "content": _case4_system_content()},
            {"role": Role.USER, "content": content},
        ]

        session_id = uuid.uuid4().hex
        try:
            snap0 = _summarize_messages_for_log(messages)
        except Exception as e:
            snap0 = [
                {
                    "role": "error",
                    "content": f"messages_snapshot_failed: {e!r}",
                }
            ]
        log_path = _append_jsonl_line(
            {
                "event": "session_start",
                "session_id": session_id,
                "user_input": content,
                "messages_snapshot": snap0,
            }
        )
        print(f"[log] 本回话日志文件: {log_path}", flush=True)

        for _round in range(max_tool_rounds):
            messages_before_request = _summarize_messages_for_log(messages)
            response = get_case4_response(messages)
            log_base = {
                "event": "api_turn",
                "session_id": session_id,
                "user_input": content,
                "api_round": _round + 1,
                "messages_before_request": messages_before_request,
                "request_id": pick(response, "request_id") if response else None,
            }

            if (
                response is None
                or getattr(response, "status_code", None) != HTTPStatus.OK
            ):
                _append_jsonl_line(
                    {
                        **log_base,
                        "error": "request_failed",
                        "status_code": getattr(response, "status_code", None),
                        "exception": LAST_API_ERROR,
                        "response_preview": str(response)[:2000] if response else None,
                    }
                )
                print("请求失败，详情已写入日志。")
                break

            output = pick(response, "output")
            if not output:
                _append_jsonl_line({**log_base, "error": "no_output"})
                print("响应无 output，详情已写入日志。")
                break

            choices = pick(output, "choices")
            if not choices:
                _append_jsonl_line({**log_base, "error": "no_choices"})
                print("响应无 choices，详情已写入日志。")
                break

            choice0 = choices[0]
            response_message = pick(choice0, "message")
            finish_reason = pick(choice0, "finish_reason")
            assistant_content = pick(response_message, "content")
            tool_calls_list = pick(response_message, "tool_calls")

            if not tool_calls_list:
                ac = assistant_content if assistant_content is not None else ""
                _append_jsonl_line(
                    {
                        **log_base,
                        "status_code": getattr(response, "status_code", None),
                        "finish_reason": finish_reason,
                        "called_tool": False,
                        "assistant_content": ac,
                        "assistant_content_chars": len(ac),
                        "assistant_content_note": (
                            "本轮无 tool_calls，assistant_content 即为模型对用户可见的回复全文。"
                        ),
                        "tool_calls_requested": None,
                        "tool_results": None,
                    }
                )
                if ac:
                    print(ac)
                break

            tool_calls_logged = _json_safe(tool_calls_list)
            tool_results_logged: list[dict] = []

            assistant_content = (
                assistant_content if assistant_content is not None else ""
            )
            messages.append(
                {
                    "role": Role.ASSISTANT,
                    "content": assistant_content,
                    "tool_calls": tool_calls_list,
                }
            )

            for tool_call in tool_calls_list:
                call_function = pick(tool_call, "function") or {}
                function_call_name = pick(call_function, "name")
                function_call_args_raw = pick(call_function, "arguments", "{}")
                arguments_json = _tool_arguments_to_dict(function_call_args_raw)
                tool_response = _invoke_tool_function(
                    function_call_name, arguments_json
                )
                tool_info = {
                    "role": "tool",
                    "tool_call_id": pick(tool_call, "id"),
                    "name": function_call_name,
                    "content": tool_response,
                }
                messages.append(tool_info)
                tool_results_logged.append(
                    {
                        "tool_call_id": pick(tool_call, "id"),
                        "name": function_call_name,
                        "arguments": arguments_json,
                        "content": _truncate_log_text(tool_response),
                    }
                )

            _append_jsonl_line(
                {
                    **log_base,
                    "status_code": getattr(response, "status_code", None),
                    "finish_reason": finish_reason,
                    "called_tool": True,
                    "assistant_content": assistant_content,
                    "assistant_content_chars": len(assistant_content),
                    "assistant_content_note": (
                        "本轮模型返回了 tool_calls；按 Chat/Tool 协议，assistant 的 content 经常为空字符串。"
                        "tool_results 是在收到本行上述模型响应之后，由本地执行工具得到的，不是「在 function call 之前多了一轮对话」。"
                    ),
                    "tool_calls_requested": tool_calls_logged,
                    "tool_results": tool_results_logged,
                }
            )
        else:
            _append_jsonl_line(
                {
                    "event": "session_error",
                    "session_id": session_id,
                    "user_input": content,
                    "api_round": max_tool_rounds,
                    "messages_before_request": _summarize_messages_for_log(messages),
                    "error": "max_tool_rounds_exceeded",
                    "called_tool": True,
                }
            )
            print(f"已达到单轮工具调用上限（{max_tool_rounds}），已记入日志。")


def main():
    case4()


if __name__ == "__main__":
    main()
