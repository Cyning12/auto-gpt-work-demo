"""课程练习通用工具：映射取值、百炼 Generation 响应解析、简单 CLI 提示等。"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from datetime import date, datetime
from http import HTTPStatus
from pathlib import Path

# 供 AI function / tool 读取的本地文档根目录（与 utils.py 同级的 doc/）
DOC_ROOT = Path(__file__).resolve().parent / "doc"


def list_doc_files(**kwargs) -> str:
    """
    列出 ``doc`` 目录下可供读取的文件相对路径（JSON 字符串），供模型先枚举再 ``read_local_file``。

    - 默认**不递归**子目录，减少输出与遍历范围。
    - ``max_entries`` 默认 100、上限 300，防止列表过长。
    """
    recursive = bool(kwargs.get("recursive", False))
    try:
        max_entries = int(kwargs.get("max_entries", 100))
    except (TypeError, ValueError):
        max_entries = 100
    max_entries = max(1, min(max_entries, 300))

    base = DOC_ROOT.resolve()
    if not base.is_dir():
        return json.dumps(
            {"files": [], "message": "doc 目录不存在", "root": str(base)},
            ensure_ascii=False,
        )

    files: list[str] = []
    truncated = False

    if recursive:
        iterator = sorted(base.rglob("*"), key=lambda p: str(p).lower())
    else:
        iterator = sorted(base.iterdir(), key=lambda p: str(p).lower())

    for p in iterator:
        if len(files) >= max_entries:
            truncated = True
            break
        if p.is_file():
            try:
                rel = p.relative_to(base)
            except ValueError:
                continue
            files.append(rel.as_posix())

    return json.dumps(
        {
            "files": files,
            "count": len(files),
            "truncated": truncated,
            "recursive": recursive,
            "hint": "请用 read_local_file(relative_path=...) 读取上列路径之一",
        },
        ensure_ascii=False,
    )


def read_local_file(relative_path: str, *, encoding: str = "utf-8") -> str:
    """
    读取固定目录 ``data/课程练习/doc`` 下的文件内容，供模型 function / tool 调用。

    - ``relative_path`` 为相对 doc 的路径，如 ``服务器异常处置手册 (SOP).txt``、``subdir/a.md``。
    - 禁止绝对路径与 ``..`` 跳出 doc 目录（路径穿越防护）。
    """
    if not relative_path or not relative_path.strip():
        raise ValueError("relative_path 不能为空")
    rel = Path(relative_path.strip())
    if rel.is_absolute():
        raise ValueError("仅允许相对路径（相对于 doc 目录）")
    base = DOC_ROOT.resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ValueError("路径必须位于 doc 目录内") from e
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在或不是普通文件: {relative_path}")
    return target.read_text(encoding=encoding)


def pick(obj, key: str, default=None):
    """优先用 Mapping.get，否则 getattr；兼容 dict、UserDict 及带属性的 SDK 对象。"""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def coerce_for_jsonl(obj):
    """
    递归规范化，使 ``json.dumps(..., allow_nan=False)`` 能写出合法 JSON。
    处理非 str 的 dict 键、NaN/Inf、SDK 返回对象等。
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Mapping):
        return {str(k): coerce_for_jsonl(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [coerce_for_jsonl(v) for v in obj]
    return str(obj)


def append_practice_jsonl_line(
    log_dir: Path | str,
    record: dict,
    *,
    secondary: bool = False,
) -> Path:
    """
    向 ``log_dir/YYYY-MM-DD.jsonl`` 追加一行练习日志。

    - ``secondary=True`` 时写入 ``YYYY-MM-DD_secondary.jsonl``（如二级摘要专用）。
    - 写入后 ``flush`` + ``fsync``，避免「脚本未结束则文件看起来为空」。
    - 主失败时尝试写入一条 ``event: log_write_error`` 占位行。
    """
    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    row.setdefault("logged_at", datetime.now().isoformat(timespec="seconds"))
    d = date.today().isoformat()
    filename = f"{d}_secondary.jsonl" if secondary else f"{d}.jsonl"
    path = (log_dir / filename).resolve()
    try:
        line = json.dumps(
            coerce_for_jsonl(row),
            ensure_ascii=False,
            allow_nan=False,
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    except Exception as e:
        print(f"[append_practice_jsonl_line] 写入失败 {path}: {e}", file=sys.stderr)
        try:
            fb = {
                "event": "log_write_error",
                "logged_at": datetime.now().isoformat(timespec="seconds"),
                "detail": repr(e),
                "original_event": record.get("event"),
            }
            line2 = json.dumps(
                coerce_for_jsonl(fb),
                ensure_ascii=False,
                allow_nan=False,
            )
            with path.open("a", encoding="utf-8") as f:
                f.write(line2 + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        except Exception as e2:
            print(f"[append_practice_jsonl_line] 兜底仍失败: {e2}", file=sys.stderr)
    return path


def generation_first_message(response):
    """
    从 DashScope Generation 响应里取出第一条 assistant message。

    与 OpenAI 不同：百炼返回里 choices 在 output 下，即 response.output.choices，
    而不是顶层的 response.choices（访问后者会 KeyError）。
    """
    if response is None:
        print("获取第一条 assistant message 失败，response 为空")
        return None
    if getattr(response, "status_code", None) != HTTPStatus.OK:
        print("获取第一条 assistant message 失败，状态码：", response.status_code)
        return None
    output = (
        response.get("output")
        if hasattr(response, "get")
        else getattr(response, "output", None)
    )
    if not output:
        print("获取第一条 assistant message 失败，output 为空")
        return None
    choices = output.get("choices") if hasattr(output, "get") else output["choices"]
    if not choices:
        return None
    first = choices[0]
    msg = first.get("message") if hasattr(first, "get") else first["message"]
    return msg


def message_function_call(message):
    """message 可能是 dict 或 DashScope 的类 dict 对象。"""
    if message is None:
        return None
    if isinstance(message, dict):
        return message.get("function_call")
    return getattr(message, "function_call", None)


def prompt_back_or_exit() -> None:
    """case 结束后统一提示：back 回主菜单，exit 结束程序。"""
    while True:
        cmd = input("\n输入 back 返回主菜单，exit 退出程序: ").strip().lower()
        if cmd == "exit":
            print("再见")
            sys.exit(0)
        if cmd == "back":
            return
        print("无效输入，请输入 back 或 exit")

        # INSERT_YOUR_CODE


def list_files_in_directory(folder: str | Path) -> list[str]:
    """
    返回目标文件夹下所有非隐藏、非目录的文件名列表（不含子目录）。
    """
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"目录不存在: {folder}")
    return [
        f.name
        for f in sorted(p.iterdir())
        if f.is_file() and not f.name.startswith(".")
    ]
