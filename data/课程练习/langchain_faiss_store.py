"""
LangChain + FAISS 向量库工具：PDF 分块、建库、增量同步、按来源文件增删改。

将 ``data/课程练习`` 加入 ``sys.path`` 后与其它练习脚本一致：

    from langchain_faiss_store import sync_vector_store, load_faiss_vector_store
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from constants import DEFAULT_RAG_SIMILARITY_THRESHOLD, DEFAULT_RAG_TOP_K
from dashscope_rerank import rerank_retrieval_hits
from utils import list_files_in_directory

_logger = logging.getLogger(__name__)

# LangChain FAISS 落盘默认文件名（save_local 生成 index.faiss + index.pkl）
FAISS_INDEX_NAME = "index"
# 记录「文件名 -> 内容 MD5」，用于判断 PDF 是否变更；与向量库同目录持久化
PROCESSED_FILES_JSON = "processed_files.json"

DEFAULT_METADATA_DEPARTMENT = "company"
DEFAULT_METADATA_UPDATE_TIME = "2026-03-27"

# PDF 抽取常见控制字符与 Latin-1 增补（\x7f-\xff）；中文等不在该范围内
_PDF_CTRL_AND_LATIN1_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]")
_PDF_NL_SURROUND_SPACES_RE = re.compile(r"[ \t]*\n[ \t]*")


def vector_bundle_exists(save_dir: str | Path, *, index_name: str = FAISS_INDEX_NAME) -> bool:
    """判断 LangChain FAISS 目录是否已完整（向量文件 + 文档库 pkl）。"""
    p = Path(save_dir)
    return p.is_dir() and (
        (p / f"{index_name}.faiss").is_file() and (p / f"{index_name}.pkl").is_file()
    )


def get_file_hash(file_path: str | Path) -> str:
    """计算文件 MD5，用于判断源文件是否变更。"""
    p = Path(file_path)
    with p.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def processed_files_path(save_dir: Path) -> Path:
    return save_dir / PROCESSED_FILES_JSON


def load_processed_files(save_dir: str | Path) -> dict[str, str]:
    """加载「文件名 -> 文件 MD5」映射；文件不存在时返回空字典。"""
    p = processed_files_path(Path(save_dir))
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_processed_files(save_dir: str | Path, mapping: dict[str, str]) -> None:
    """持久化已处理文件的哈希表。"""
    save_p = Path(save_dir)
    save_p.mkdir(parents=True, exist_ok=True)
    out = processed_files_path(save_p)
    with out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def remove_faiss_bundle(
    save_dir: Path,
    *,
    index_name: str = FAISS_INDEX_NAME,
) -> None:
    """删除落盘的 FAISS 向量与文档序列化文件（保留同目录下其它文件如 processed_files.json）。"""
    for suffix in (".faiss", ".pkl"):
        p = save_dir / f"{index_name}{suffix}"
        if p.is_file():
            p.unlink()


def faiss_doc_ids_for_source_file(vs: FAISS, source_file: str) -> list[str]:
    """收集 metadata.source_file 等于给定文件名的文档 id，供增量删除。"""
    seen: set[str] = set()
    out: list[str] = []
    for _idx, doc_id in vs.index_to_docstore_id.items():
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc = vs.docstore.search(doc_id)
        if not isinstance(doc, Document):
            continue
        if doc.metadata.get("source_file") == source_file:
            out.append(doc_id)
    return out


def indexed_source_file_names(vs: FAISS) -> set[str]:
    """向量库 docstore 中曾出现过的 ``metadata.source_file`` 集合（与目录文件名比对用）。"""
    seen_doc_ids: set[str] = set()
    sources: set[str] = set()
    for doc_id in vs.index_to_docstore_id.values():
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        doc = vs.docstore.search(doc_id)
        if not isinstance(doc, Document):
            continue
        sf = doc.metadata.get("source_file")
        if sf:
            sources.add(str(sf))
    return sources


def chunks_dicts_from_faiss(vs: FAISS) -> list[dict[str, Any]]:
    """从 FAISS 文档库导出与 read_pdf_files 一致结构的列表，用于写 chunk JSON。"""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc_id in vs.index_to_docstore_id.values():
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc = vs.docstore.search(doc_id)
        if not isinstance(doc, Document):
            continue
        out.append({"content": doc.page_content, "metadata": dict(doc.metadata)})
    return out


def enrich_page_content_for_embedding(
    content: str,
    metadata: dict[str, Any],
) -> str:
    """
    嵌入前拼接可检索的「身份」短语（资料来源、部门、页码等），提升同义不同问法下的召回；
    向量与 docstore 中存的是增强后全文，metadata 结构不变。
    """
    src = metadata.get("source_file") or "未知文件"
    dept = metadata.get("department")
    page = metadata.get("page_number")
    lines: list[str] = [f"资料来源：{src}"]
    if dept:
        lines.append(f"所属部门：{dept}")
    if page is not None:
        lines.append(f"页码：{page}")
    head = "\n".join(lines)
    body = (content or "").strip()
    return f"{head}\n正文内容：\n{body}"


def delete_documents_by_source_file(vector_store: FAISS, source_file: str) -> int:
    """按 ``metadata.source_file`` 删除向量库中该 PDF 对应的全部文档；返回删除的条数。"""
    ids = faiss_doc_ids_for_source_file(vector_store, source_file)
    if ids:
        vector_store.delete(ids)
    return len(ids)


def add_chunks_to_faiss(vector_store: FAISS, chunks: list[dict[str, Any]]) -> None:
    """将已分块列表写入已有 FAISS（仅 ``add_texts``，不负责 save_local）。"""
    if not chunks:
        return
    vector_store.add_texts(
        [
            enrich_page_content_for_embedding(c["content"], c["metadata"])
            for c in chunks
        ],
        metadatas=[dict(c["metadata"]) for c in chunks],
    )


def rebuild_faiss_from_chunks(
    chunks: list[dict[str, Any]],
    embeddings: Embeddings,
    save_dir: str | Path,
    *,
    index_name: str = FAISS_INDEX_NAME,
    embedding_model_label: str = "",
    on_embedding_failure: Callable[[BaseException], None] | None = None,
) -> FAISS:
    """
    全量用分块重建向量库并落盘（等价于冷启动 ``FAISS.from_texts`` + ``save_local``）。
    若目录已有索引，请先 ``remove_faiss_bundle`` 或改用 ``process_text_with_splitter`` 指定新目录。
    """
    if not chunks:
        raise ValueError("chunks 为空，无法建库")
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [
        enrich_page_content_for_embedding(item["content"], item["metadata"])
        for item in chunks
    ]
    metadatas = [dict(item["metadata"]) for item in chunks]
    try:
        vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vs.save_local(str(out_dir), index_name=index_name)
    except Exception as e:
        _logger.exception(
            "向量化或保存 FAISS 失败：分块数=%d，输出目录=%s，模型标签=%s，异常类型=%s",
            len(texts),
            out_dir,
            embedding_model_label or "?",
            type(e).__name__,
        )
        if on_embedding_failure is not None:
            on_embedding_failure(e)
        raise
    print(
        f"已向量化 {len(texts)} 条分块，已保存至 {out_dir} "
        f"（{index_name}.faiss + {index_name}.pkl）"
    )
    return vs


def process_text_with_splitter(
    all_chunks_data: list[dict[str, Any]],
    embeddings: Embeddings,
    save_path: str | Path | None = None,
    *,
    index_name: str = FAISS_INDEX_NAME,
    default_save_dir: Path | None = None,
    embedding_model_label: str = "",
    on_embedding_failure: Callable[[BaseException], None] | None = None,
) -> FAISS:
    """
    将已分块的文本批量向量化，构建 FAISS 并写入 ``save_path``（若为空则使用 ``default_save_dir``）。

    落盘内容（与「单独的 page_info.pkl」区别见原课程脚本注释）：
    - ``{index_name}.faiss``：仅向量索引。
    - ``{index_name}.pkl``：序列化的 docstore，``metadata`` 含 source_file、页码等。
    """
    if not all_chunks_data:
        raise ValueError("all_chunks_data 为空，无法建库")
    out_dir = Path(save_path) if save_path is not None else default_save_dir
    if out_dir is None:
        raise ValueError("save_path 与 default_save_dir 不能同时为空")
    return rebuild_faiss_from_chunks(
        all_chunks_data,
        embeddings,
        out_dir,
        index_name=index_name,
        embedding_model_label=embedding_model_label,
        on_embedding_failure=on_embedding_failure,
    )


def _clean_pdf_extracted_text(text: str) -> str:
    """清洗 PyPDF2 抽取文本：剔除控制符、替换私用区列表符、压缩空行。"""
    if not text:
        return text
    s = _PDF_CTRL_AND_LATIN1_RE.sub("", text)
    for sym in ("\uf0b2", "\uf0b7", "\uf0a7", "\uf0a8", "\uf09e", "\uf0fc"):
        s = s.replace(sym, "•")
    s = _PDF_NL_SURROUND_SPACES_RE.sub("\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


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
        cleaned = _clean_pdf_extracted_text(stripped)
        if cleaned:
            pages.append((cleaned, page_number))
    return pages


def read_pdf(pdf_path: Path) -> PdfReader:
    return PdfReader(pdf_path)


def build_chunks_for_pdf(
    pdf_full: Path,
    pdf_name: str,
    doc_idx: int,
    file_hash: str,
    splitter: RecursiveCharacterTextSplitter,
    *,
    metadata_department: str = DEFAULT_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_METADATA_UPDATE_TIME,
) -> list[dict[str, Any]]:
    """单个 PDF 分块；metadata 含 file_hash、department、update_time。"""
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
    metadata_department: str = DEFAULT_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_METADATA_UPDATE_TIME,
) -> list[dict[str, Any]]:
    """
    遍历同一目录下 PDF，按页提取文本后分块。

    若传入 ``processed_files``（文件名 -> 上次入库时的 MD5），则与当前文件哈希比对：
    一致则跳过该文件；变更或新文件则重新分块并带上最新 metadata。
    """
    folder_path = Path(folder)
    splitter = RecursiveCharacterTextSplitter(
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


def save_pdf_chunks_json(
    chunks: list[dict[str, Any]],
    *,
    save_dir: str | Path,
    chunk_filename_stem: str | None = None,
) -> Path:
    """
    将分块数据写入 ``save_dir`` 下的 ``{stem}_chunk.json``（默认 stem 为目录名）。
    """
    if not chunks:
        raise ValueError("chunks 为空，跳过写入")
    out_parent = Path(save_dir)
    out_parent.mkdir(parents=True, exist_ok=True)
    stem = chunk_filename_stem if chunk_filename_stem is not None else out_parent.name
    out_path = out_parent / f"{stem}_chunk.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"分块 JSON 已保存：{out_path}（共 {len(chunks)} 条）")
    return out_path


def sync_vector_store(
    *,
    embeddings: Embeddings,
    source_folder: str | Path,
    save_dir: str | Path,
    index_name: str = FAISS_INDEX_NAME,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    embedding_model_label: str = "",
    on_embedding_failure: Callable[[BaseException], None] | None = None,
    metadata_department: str = DEFAULT_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_METADATA_UPDATE_TIME,
) -> FAISS | None:
    """
    全量扫描源目录并比对 ``processed_files.json``（文件名 -> MD5），**Embedding 只对变动文件做**：

    - **冷启动**：无 ``index.faiss`` 时全量分块后 ``FAISS.from_texts``。
    - **增量**：对已存在索引，按文件 ``delete`` 旧向量后 ``add_texts`` 仅嵌入变动 PDF。
    - **清理**：源目录已删除的 PDF，从向量库按 ``source_file`` 删除残留块。
    """
    folder_path = Path(source_folder)
    target = Path(save_dir)
    target.mkdir(parents=True, exist_ok=True)

    names = [
        n
        for n in list_files_in_directory(folder_path)
        if Path(n).suffix.lower() == ".pdf"
    ]
    current_hashes = {n: get_file_hash(folder_path / n) for n in names}
    processed = load_processed_files(target)

    if vector_bundle_exists(target, index_name=index_name) and not processed_files_path(
        target
    ).is_file():
        _logger.warning(
            "检测到向量库存在但未找到 %s，将删除旧索引并全量重建（与 file_hash 等元数据对齐）",
            PROCESSED_FILES_JSON,
        )
        remove_faiss_bundle(target, index_name=index_name)
        processed = {}

    to_refresh = [n for n in names if processed.get(n) != current_hashes[n]]

    if not vector_bundle_exists(target, index_name=index_name):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks: list[dict[str, Any]] = []
        for doc_idx, pdf_name in enumerate(names, start=1):
            pdf_full = folder_path / pdf_name
            h = current_hashes[pdf_name]
            chunks.extend(
                build_chunks_for_pdf(
                    pdf_full,
                    pdf_name,
                    doc_idx,
                    h,
                    splitter,
                    metadata_department=metadata_department,
                    metadata_update_time=metadata_update_time,
                )
            )
        if not chunks:
            print("无 PDF 分块，跳过建库")
            return None
        save_processed_files(target, {n: current_hashes[n] for n in names})
        save_pdf_chunks_json(chunks, save_dir=target)
        return process_text_with_splitter(
            chunks,
            embeddings,
            target,
            index_name=index_name,
            embedding_model_label=embedding_model_label,
            on_embedding_failure=on_embedding_failure,
        )

    names_set = set(names)
    vs = FAISS.load_local(
        str(target),
        embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )

    indexed_sources = indexed_source_file_names(vs)
    orphan_sources = sorted(indexed_sources - names_set)

    for k in list(processed.keys()):
        if k not in names_set:
            processed.pop(k, None)

    if not to_refresh and not orphan_sources:
        save_processed_files(target, processed)
        print("所有 PDF 与已处理记录一致，跳过向量库更新")
        return None

    if orphan_sources:
        for pdf_name in orphan_sources:
            delete_documents_by_source_file(vs, pdf_name)
            print(f"已清理源目录已删除文件对应向量: {pdf_name}")

    if to_refresh:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        incremental_chunks: list[dict[str, Any]] = []
        for pdf_name in to_refresh:
            delete_documents_by_source_file(vs, pdf_name)
            doc_idx = names.index(pdf_name) + 1
            pdf_full = folder_path / pdf_name
            h = current_hashes[pdf_name]
            incremental_chunks.extend(
                build_chunks_for_pdf(
                    pdf_full,
                    pdf_name,
                    doc_idx,
                    h,
                    splitter,
                    metadata_department=metadata_department,
                    metadata_update_time=metadata_update_time,
                )
            )
            processed[pdf_name] = h
        if incremental_chunks:
            add_chunks_to_faiss(vs, incremental_chunks)

    save_processed_files(target, processed)
    vs.save_local(str(target), index_name=index_name)
    save_pdf_chunks_json(chunks_dicts_from_faiss(vs), save_dir=target)
    print("向量库已按文件哈希校验并完成更新")
    return vs


def create_vector_model(
    embeddings: Embeddings,
    source_folder: Path,
    save_dir: Path,
    chunks: list[dict[str, Any]] | None = None,
    *,
    index_name: str = FAISS_INDEX_NAME,
    embedding_model_label: str = "",
    on_embedding_failure: Callable[[BaseException], None] | None = None,
    metadata_department: str = DEFAULT_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_METADATA_UPDATE_TIME,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> FAISS | None:
    """
    默认走 ``sync_vector_store``：按 MD5 检查源 PDF，变更则更新向量库。
    若显式传入 ``chunks``，仅在向量库尚不存在时写入（与历史行为一致）。
    """
    target = save_dir
    if chunks is not None:
        if vector_bundle_exists(target, index_name=index_name):
            print("向量模型已存在")
            return None
        print("向量模型不存在，使用传入分块创建向量模型")
        target.mkdir(parents=True, exist_ok=True)
        save_pdf_chunks_json(chunks, save_dir=target)
        return process_text_with_splitter(
            chunks,
            embeddings,
            target,
            index_name=index_name,
            embedding_model_label=embedding_model_label,
            on_embedding_failure=on_embedding_failure,
        )
    return sync_vector_store(
        embeddings=embeddings,
        source_folder=source_folder,
        save_dir=target,
        index_name=index_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_label=embedding_model_label,
        on_embedding_failure=on_embedding_failure,
        metadata_department=metadata_department,
        metadata_update_time=metadata_update_time,
    )


def load_faiss_vector_store(
    save_dir: str | Path,
    embeddings: Embeddings,
    *,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """从本地目录加载 LangChain FAISS 向量库（与 ``sync_vector_store`` 落盘路径一致）。"""
    target = Path(save_dir)
    if not vector_bundle_exists(target, index_name=index_name):
        raise FileNotFoundError(
            f"向量库不存在或不完整：{target}（请先运行同步建库）"
        )
    return FAISS.load_local(
        str(target),
        embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def faiss_l2_distance_to_similarity(distance: float) -> float:
    """
    将 FAISS ``IndexFlatL2`` 返回的距离转为单调相似度 ``(0, 1]``（``1/(1+d)``，越大越近）。
    适用于未做 L2 单位化的嵌入（如常见 DashScope text-embedding），避免 ``1-d/√2`` 出现负数。
    """
    d = float(distance)
    if d < 0:
        d = 0.0
    return 1.0 / (1.0 + d)


def faiss_similarity_search(
    query: str,
    *,
    vector_db: FAISS | None = None,
    embeddings: Embeddings | None = None,
    save_dir: str | Path | None = None,
    index_name: str = FAISS_INDEX_NAME,
    k: int = DEFAULT_RAG_TOP_K,
    score_threshold: float | None = DEFAULT_RAG_SIMILARITY_THRESHOLD,
    rerank: bool = False,
    api_key_for_rerank: str | None = None,
) -> list[dict[str, Any]]:
    """
    向量检索：``query`` 经 ``embed_query`` 后在 FAISS 中取 Top-K，可选 similarity 阈值过滤与 DashScope 精排。

    - 若未传 ``vector_db``，需同时提供 ``embeddings`` 与 ``save_dir``，内部 ``load_faiss_vector_store``。
    - 返回字典列表，字段含 ``content``、``distance``、``similarity``、``source_file``、``page_number`` 等。
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")

    db = vector_db
    if db is None:
        if embeddings is None or save_dir is None:
            raise ValueError("未提供 vector_db 时，必须同时提供 embeddings 与 save_dir")
        db = load_faiss_vector_store(save_dir, embeddings, index_name=index_name)

    fetch_n = k
    if score_threshold is not None:
        fetch_n = min(300, max(k * 20, 80))

    pairs = db.similarity_search_with_score(q, k=fetch_n)
    rows: list[dict[str, Any]] = []
    for doc, dist in pairs:
        similarity = faiss_l2_distance_to_similarity(dist)
        if score_threshold is not None and similarity < score_threshold:
            continue
        meta = dict(doc.metadata)
        rows.append(
            {
                "content": doc.page_content,
                "distance": float(dist),
                "similarity": similarity,
                "relevance_score": similarity,
                "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"),
                "chunk_id": meta.get("chunk_id"),
                "file_hash": meta.get("file_hash"),
                "department": meta.get("department"),
                "update_time": meta.get("update_time"),
            }
        )
        if len(rows) >= k:
            break
    if rerank and rows:
        rows = rerank_retrieval_hits(
            q, rows, top_n=len(rows), api_key=api_key_for_rerank
        )
    return rows


def faiss_search_knowledge_base(
    query: str,
    vector_db: FAISS,
    top_k: int = DEFAULT_RAG_TOP_K,
    *,
    score_threshold: float | None = DEFAULT_RAG_SIMILARITY_THRESHOLD,
    rerank: bool = False,
    api_key_for_rerank: str | None = None,
) -> list[dict[str, Any]]:
    """在已加载的 ``vector_db`` 上检索，避免重复 ``load_local``；语义同 ``faiss_similarity_search(..., vector_db=...)``。"""
    return faiss_similarity_search(
        query,
        vector_db=vector_db,
        k=top_k,
        score_threshold=score_threshold,
        rerank=rerank,
        api_key_for_rerank=api_key_for_rerank,
    )
