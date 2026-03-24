# CO-STAR_Prompt.py

# 课程练习：Embeddings和向量数据库

# 1. 导入必要的库
import os
import sys
from datetime import date, datetime
from pathlib import Path

_practice_root = Path(__file__).resolve().parents[1]
if str(_practice_root) not in sys.path:
    sys.path.insert(0, str(_practice_root))
from utils import list_doc_files, pick, read_local_file

import dashscope
from dotenv import load_dotenv
from dashscope.api_entities.dashscope_response import Role
import json
import random
from collections.abc import Mapping
from http import HTTPStatus

_here = Path(__file__).resolve().parent
# 与课程约定一致：日志写在「AI大模型原理与API使用」目录下（勿与脚本所在 Embeddings 子目录混淆）
LOG_DIR = _practice_root / "AI大模型原理与API使用" / "log"
# 单条 tool 返回写入日志时的最大字符数，避免 JSONL 过大
_LOG_TOOL_CONTENT_MAX = 12000

load_dotenv(_here / ".env")
load_dotenv()
api_key = (os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
dashscope.api_key = api_key or None

prompt_test = ""
# Context
"""
你是一名资深运维专家。现在你需要根据提供的【企业内部文档】片段，回答用户关于服务器异常的问题。
"""
# Objective
"""
仅基于提供的文档内容回答。如果文档中没有相关信息，请诚实告知“知识库中暂无此记录”，严禁幻觉。
"""
# Style & Tone
"""
模仿高级 SRE 工程师的排障日志格式。使用 Markdown 的代码块展示技术参数，使用无序列表呈现逻辑步骤。每一段文字不要超过 3 行 。
"""
# Tone
"""
保持极度的专业和客观。避免使用“我觉得”、“可能”等模糊词汇。在指出系统错误时，语气要直接、果断，不带任何感情色彩。
"""
# Audience
"""
面向值班运维工程师，需要直接给处操作建议。
"""
# Response
"""
输出格式：
1.问题定位：...
2.参考文档：[文档标题/编号]
3.建议操作：...
    # Information (RAG 注入位)
    这里是检索到的片段：{{retrieved_context}}
    
"""


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
            prompt=prompt_test,
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


def _json_safe(obj):
    """转为可 json.dumps 的结构（用于 JSONL）。"""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def _truncate_log_text(s: str, max_len: int = _LOG_TOOL_CONTENT_MAX) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...[truncated,total_len={len(s)}]"


def _append_jsonl_line(record: dict) -> None:
    record.setdefault("logged_at", datetime.now().isoformat(timespec="seconds"))
    path = LOG_DIR / f"{date.today().isoformat()}.jsonl"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        print(f"[日志写入失败] {path}: {e}", file=sys.stderr)


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
            {
                "role": Role.SYSTEM,
                "content": (
                    "你是企业内的运维与制度咨询助手，只能通过工具获取事实，禁止编造文档内容。\n"
                    "【工具分流——必须遵守】\n"
                    "1）用户问「怎么处理、操作步骤、SOP、手册、制度、规定、策略、标准、联系人、文档编号」"
                    "或告警场景下需要「按公司书面流程处置」：必须先依次调用 list_doc_files，再 read_local_file 读取相关文件，"
                    "最后基于文件内容作答；不可仅凭 get_current_status 回答流程类问题。\n"
                    "2）仅当用户明确索要「当前/此刻/现在」的监控数值（连接数、CPU、内存）时，才调用 get_current_status；"
                    "可与文档工具配合，但文档问题不得省略文档工具。\n"
                    "3）若用户同时需要流程与实时数据：先完成文档检索与阅读，再决定是否调用 get_current_status。\n"
                    "回答时注明信息来自哪份文件名中的条款（若读过文件）。"
                ),
            },
            {"role": Role.USER, "content": content},
        ]

        for _round in range(max_tool_rounds):
            response = get_case4_response(messages)
            log_base = {
                "user_input": content,
                "api_round": _round + 1,
                "request_id": pick(response, "request_id") if response else None,
            }

            if response is None or getattr(response, "status_code", None) != HTTPStatus.OK:
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
                _append_jsonl_line(
                    {
                        **log_base,
                        "status_code": getattr(response, "status_code", None),
                        "finish_reason": finish_reason,
                        "called_tool": False,
                        "assistant_content": assistant_content,
                        "tool_calls_requested": None,
                        "tool_results": None,
                    }
                )
                if assistant_content:
                    print(assistant_content)
                break

            tool_calls_logged = _json_safe(tool_calls_list)
            tool_results_logged: list[dict] = []

            assistant_content = assistant_content if assistant_content is not None else ""
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
                    "tool_calls_requested": tool_calls_logged,
                    "tool_results": tool_results_logged,
                }
            )
        else:
            _append_jsonl_line(
                {
                    "user_input": content,
                    "api_round": max_tool_rounds,
                    "error": "max_tool_rounds_exceeded",
                    "called_tool": True,
                }
            )
            print(
                f"已达到单轮工具调用上限（{max_tool_rounds}），已记入日志。"
            )


def main():
    case4()


if __name__ == "__main__":
    main()
