"""
PDF 解析与 RAG 分块：可单独按页抽取文本，也可配合 ``RecursiveCharacterTextSplitter`` 切成向量库用块。

与 ``langchain_faiss_store`` 解耦，仅依赖 PyPDF2、LangChain TextSplitter、``utils.list_files_in_directory``。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from utils import list_files_in_directory

# 与 langchain_faiss_store 默认元数据保持一致，便于 ``build_chunks_for_pdf`` 直接入库
DEFAULT_PDF_METADATA_DEPARTMENT = "company"
DEFAULT_PDF_METADATA_UPDATE_TIME = "2026-03-27"

# PDF 抽取常见控制字符与 Latin-1 增补（\x7f-\xff）；中文等不在该范围内
_PDF_CTRL_AND_LATIN1_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]")
_PDF_NL_SURROUND_SPACES_RE = re.compile(r"[ \t]*\n[ \t]*")


def get_file_hash(file_path: str | Path) -> str:
    """计算文件 MD5，用于判断源文件是否变更。"""
    p = Path(file_path)
    with p.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def create_rag_text_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> RecursiveCharacterTextSplitter:
    """与课程练习向量库默认一致的递归字符切分器。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def clean_pdf_extracted_text(text: str) -> str:
    """清洗 PyPDF2 抽取文本：剔除控制符、替换私用区列表符、压缩空行。"""
    if not text:
        return text
    s = _PDF_CTRL_AND_LATIN1_RE.sub("", text)
    for sym in ("\uf0b2", "\uf0b7", "\uf0a7", "\uf0a8", "\uf09e", "\uf0fc"):
        s = s.replace(sym, "•")
    s = _PDF_NL_SURROUND_SPACES_RE.sub("\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def read_pdf(pdf_path: str | Path) -> PdfReader:
    """打开 PDF，返回 PyPDF2 ``PdfReader``。"""
    return PdfReader(str(pdf_path))


def extract_text_with_page_numbers(pdf: PdfReader) -> list[tuple[str, int]]:
    """
    从 PDF 按页提取文本，每页一条记录，便于分块后标注准确页码。

    返回列表元素为 (该页全文, 页码)，页码从 1 起；无正文的页不加入列表。
    """
    pages: list[tuple[str, int]] = []
    for page_number, page in enumerate(pdf.pages, start=1):
        extracted = page.extract_text()
        if not extracted:
            continue
        stripped = extracted.strip()
        if not stripped:
            continue
        cleaned = clean_pdf_extracted_text(stripped)
        if cleaned:
            pages.append((cleaned, page_number))
    return pages


def extract_pdf_text_by_page(pdf_path: str | Path) -> list[tuple[str, int]]:
    """
    便捷封装：路径 → 按页文本列表，等价于 ``extract_text_with_page_numbers(read_pdf(pdf_path))``。
    """
    return extract_text_with_page_numbers(read_pdf(pdf_path))


def split_pdf_pages_with_splitter(
    pages: list[tuple[str, int]],
    splitter: RecursiveCharacterTextSplitter,
) -> list[tuple[str, int]]:
    """
    对已抽取的 (页文本, 页码) 列表做字符级切分；每个子块继承原页码。

    返回 ``(chunk 文本, page_number)`` 扁平列表。
    """
    out: list[tuple[str, int]] = []
    for page_text, page_number in pages:
        for chunk in splitter.split_text(page_text):
            out.append((chunk, page_number))
    return out


def build_chunks_for_pdf(
    pdf_full: Path,
    pdf_name: str,
    doc_idx: int,
    file_hash: str,
    splitter: RecursiveCharacterTextSplitter,
    *,
    metadata_department: str = DEFAULT_PDF_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_PDF_METADATA_UPDATE_TIME,
) -> list[dict[str, Any]]:
    """单个 PDF 分块；metadata 含 file_hash、department、update_time、page_number。"""
    pdf = read_pdf(pdf_full)
    per_page = extract_text_with_page_numbers(pdf)

    char_total = sum(len(t) for t, _ in per_page)
    print(
        f"读取文件：{pdf_name}，页数（有文本）：{len(per_page)}，约 {char_total} 字符"
    )

    rows: list[dict[str, Any]] = []
    chunk_in_doc = 0
    for page_text, page_number in per_page:
        for chunk in splitter.split_text(page_text):
            chunk_in_doc += 1
            rows.append(
                {
                    "content": chunk,
                    "metadata": {
                        "source_file": pdf_name,
                        "page_number": page_number,
                        "chunk_id": f"doc_{doc_idx:02d}_ch_{chunk_in_doc}",
                        "file_hash": file_hash,
                        "department": metadata_department,
                        "update_time": metadata_update_time,
                    },
                }
            )
    return rows


def read_pdf_files(
    folder: str | Path,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    processed_files: dict[str, str] | None = None,
    metadata_department: str = DEFAULT_PDF_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_PDF_METADATA_UPDATE_TIME,
) -> list[dict[str, Any]]:
    """
    遍历同一目录下 PDF，按页提取文本后分块。

    若传入 ``processed_files``（文件名 -> 上次入库时的 MD5），则与当前文件哈希比对：
    一致则跳过该文件；变更或新文件则重新分块并带上最新 metadata。
    """
    folder_path = Path(folder)
    splitter = create_rag_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    proc = processed_files if processed_files is not None else {}

    names = [
        n
        for n in list_files_in_directory(folder_path)
        if Path(n).suffix.lower() == ".pdf"
    ]
    all_chunks_data: list[dict[str, Any]] = []
    for doc_idx, pdf_name in enumerate(names, start=1):
        pdf_full = folder_path / pdf_name
        current_hash = get_file_hash(pdf_full)
        if proc.get(pdf_name) == current_hash:
            print(f"跳过已存在的文件: {pdf_name}")
            continue
        all_chunks_data.extend(
            build_chunks_for_pdf(
                pdf_full,
                pdf_name,
                doc_idx,
                current_hash,
                splitter,
                metadata_department=metadata_department,
                metadata_update_time=metadata_update_time,
            )
        )

    return all_chunks_data
