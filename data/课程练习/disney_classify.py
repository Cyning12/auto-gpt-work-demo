"""
迪士尼 RAG 练习：按扩展名递归分类知识库目录下的文件（doc/pdf/ppt/images/other）。
"""

from __future__ import annotations

from pathlib import Path

# 与脚本同目录为 data/课程练习；未指定 base_dir 时默认从此根扫描（一般调用方会传入知识库路径）
_PRACTICE_ROOT = Path(__file__).resolve().parent

_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".svg",
    ".heic",
}
_IGNORE_FILE_NAMES = {".DS_Store"}


def classify_disney_knowledge_files(
    base_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    """
    递归获取「迪士尼RAG知识库」下所有文件，并按类型分类。

    返回结构：
    - doc（旧版 Word，含 .doc/.docx）
    - pdf
    - ppt（包含 .ppt 与 .pptx）
    - images（常见图片格式）
    - other（无法识别或不在上述分类中的文件）
    """
    root = Path(base_dir) if base_dir is not None else _PRACTICE_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    result: dict[str, list[str]] = {
        "doc": [],
        "pdf": [],
        "ppt": [],
        "images": [],
        "other": [],
    }

    for fp in sorted(root.rglob("*")):
        if not fp.is_file():
            continue
        if fp.name in _IGNORE_FILE_NAMES:
            continue
        rel_path = str(fp.relative_to(root))
        ext = fp.suffix.lower()
        if ext in {".docx", ".doc"}:
            result["doc"].append(rel_path)
        elif ext == ".pdf":
            result["pdf"].append(rel_path)
        elif ext in {".ppt", ".pptx"}:
            result["ppt"].append(rel_path)
        elif ext in _IMAGE_EXTS:
            result["images"].append(rel_path)
        else:
            result["other"].append(rel_path)
    return result
