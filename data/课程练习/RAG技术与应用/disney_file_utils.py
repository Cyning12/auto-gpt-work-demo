from __future__ import annotations

from pathlib import Path
import subprocess
from PIL import Image
import pytesseract

# 课程练习根目录（.../data/课程练习）
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent

# 图片常见扩展名（统一小写比较）
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


def parse_docx(file_path: str | Path):
    """解析 DOCX 文件，提取文本和表格（转为 Markdown）。"""
    p = Path(file_path)
    if p.suffix.lower() != ".docx":
        raise ValueError(f"仅支持解析 .docx 文件，当前为: {p.name}")
    try:
        from docx import Document as DocxDocument
    except Exception as e:
        raise ImportError(
            "DOCX 解析依赖异常：请卸载旧版 docx 并安装 python-docx（pip uninstall docx && pip install python-docx）"
        ) from e

    doc = DocxDocument(str(p))
    content_chunks = []
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    # 按 body 的一层子节点顺序遍历，避免部分文档上 element.tag 非字符串导致 endswith 报错
    for child in doc.element.body.iterchildren():
        tag = getattr(child, "tag", None)
        if tag == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p":
            # 段落：提取该段落内全部文本节点
            texts = [t.text for t in child.findall(".//w:t", ns) if t.text]
            paragraph_text = "".join(texts).strip()
            if paragraph_text:
                content_chunks.append({"type": "text", "content": paragraph_text})
        elif tag == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl":
            # 表格：直接按 XML 行列提取，避免 doc.tables 与 body 节点映射不稳定
            rows = child.findall("./w:tr", ns)
            if not rows:
                continue

            md_table: list[str] = []

            def _row_cells(tr) -> list[str]:
                cells = tr.findall("./w:tc", ns)
                vals: list[str] = []
                for tc in cells:
                    ts = [t.text for t in tc.findall(".//w:t", ns) if t.text]
                    vals.append("".join(ts).strip())
                return vals

            header = _row_cells(rows[0])
            if not header:
                continue
            md_table.append("| " + " | ".join(header) + " |")
            md_table.append("|" + "---|" * len(header))
            for tr in rows[1:]:
                md_table.append("| " + " | ".join(_row_cells(tr)) + " |")

            table_content = "\n".join(md_table).strip()
            if table_content:
                content_chunks.append({"type": "table", "content": table_content})

    return content_chunks


def parse_doc(file_path: str | Path):
    """
    解析 DOC（旧版 Word）文件，返回统一的 content_chunks 结构。

    优先使用 macOS 自带 textutil；若不可用或转换失败，再尝试 antiword。
    """
    p = Path(file_path)
    if p.suffix.lower() != ".doc":
        raise ValueError(f"仅支持解析 .doc 文件，当前为: {p.name}")

    # 1) macOS 首选：textutil
    try:
        resp = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(p)],
            capture_output=True,
            text=True,
            check=True,
        )
        text_content = (resp.stdout or "").strip()
        if text_content:
            return [{"type": "text", "content": text_content}]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2) 兜底：antiword（若已安装）
    try:
        resp = subprocess.run(
            ["antiword", str(p)],
            capture_output=True,
            text=True,
            check=True,
        )
        text_content = (resp.stdout or "").strip()
        if text_content:
            return [{"type": "text", "content": text_content}]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        "DOC 解析失败：请确认文件未损坏；在 macOS 可用 textutil，或安装 antiword 后重试。"
    )


def parse_word_document(file_path: str | Path):
    """统一入口：按后缀自动解析 .docx / .doc。"""
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext == ".docx":
        return parse_docx(p)
    if ext == ".doc":
        return parse_doc(p)
    raise ValueError(f"仅支持 .doc/.docx，当前文件: {p.name}")


def chunks_to_markdown(chunks: list[dict[str, str]]) -> str:
    """将解析出的文本/表格块拼接为 Markdown。"""
    lines: list[str] = []
    for idx, ch in enumerate(chunks, start=1):
        ctype = ch.get("type", "text")
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        if ctype == "table":
            lines.append(f"### 表格块 {idx}")
            lines.append(content)
        else:
            lines.append(content)
        lines.append("")  # 块间空行
    return "\n".join(lines).strip() + "\n"


def export_doc_and_docx_to_markdown(
    classified: dict[str, list[str]],
    base_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> list[str]:
    """解析 doc/docx 并导出为 Markdown 文件。"""
    ok_count = 0
    fail_count = 0
    failed_paths = []
    for rel in classified.get("doc", []):
        src = base_dir / rel
        try:
            chunks = parse_word_document(src)
            md_text = chunks_to_markdown(chunks)
            out_name = f"{Path(rel).stem}.md"
            out_file = out_dir / out_name
            out_file.write_text(md_text, encoding="utf-8")
            ok_count += 1
        except Exception as e:
            fail_count += 1
            failed_paths.append(rel)
            print(f"[解析失败] {rel} -> {e}")
    print(f"\nWord 解析完成：成功 {ok_count}，失败 {fail_count}")
    print(f"Markdown 输出目录：{out_dir}")
    return failed_paths


def image_to_text(image_path):
    """对图片进行OCR和CLIP描述。"""
    try:
        image = Image.open(image_path)
        # OCR
        ocr_text = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
        return {"ocr": ocr_text}
    except Exception as e:
        print(f"处理图片失败 {image_path}: {e}")
        return {"ocr": ""}
