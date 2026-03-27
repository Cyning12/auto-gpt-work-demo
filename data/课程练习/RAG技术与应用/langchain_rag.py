# 2026-3-26 课程练习：RAG技术与应用
# 任务：结合你的业务场景，创建你的RAG问答（LangChain+DeepSeek+Faiss）
# 结合你的业务场景，创建你的RAG问答（LangChain+DeepSeek+Faiss）
# Step1，收集整理知识库 (./source/焦点科技公司制度)
# Step2，从PDF中提取文本并记录每行文本对应的页码
# Step3，处理文本并创建向量存储
# Step4，执行相似度搜索，找到与查询相关的文档
# Step5，使用问到链对用户问题进行回答 （使用你的DASHSCOPE_API_KEY）
# Step6，显示每个文档块的来源页码
from __future__ import annotations

import os
import sys

# macOS：FAISS / NumPy / PyTorch 等若各自链接 libomp，OpenMP 重复初始化会 abort（OMP: Error #15）
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from PyPDF2 import PdfReader

import dashscope
import requests
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from typing import Any, List, Tuple

PROFESSIONAL_SYSTEM_TEMPLATE = """
# ROLE（角色）
你是一个严谨的公司行政助手。
请基于以下提供的【制度片段】回答用户的问题。

【已知制度】：
{context}

【用户问题】：
{query}

【注意事项】：
1. 如果提供的【已知制度】中没有关于“请假”、“假种”、“申请流程”的具体说明，请直接回答：“抱歉，目前的规章制度库中未包含员工请假相关的具体申请流程，建议咨询 HR 部门。”
2. 严禁根据你自身的知识储备来回答公司制度，只能依据上方提供的片段。
3. 如果能够回答，请务必注明出处（如：参考自《XX制度》第X页）。
"""


# 与 langchain_community.embeddings.dashscope 中 BATCH_SIZE 一致
_DASHSCOPE_EMBED_BATCH_SIZE: dict[str, int] = {
    "text-embedding-v1": 25,
    "text-embedding-v2": 25,
}

# 网络抖动 / 代理中断时常见，单批重试，避免整条流水线直接失败
_EMBED_NETWORK_RETRY = 5
_EMBED_NETWORK_RETRY_BASE_SEC = 2.0

_RETRYABLE_REQUEST_ERRORS: tuple[type[BaseException], ...] = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

# LangChain FAISS 落盘默认文件名（save_local 生成 index.faiss + index.pkl）
_FAISS_INDEX_NAME = "index"
# 记录「文件名 -> 内容 MD5」，用于判断 PDF 是否变更；与向量库同目录持久化
_PROCESSED_FILES_JSON = "processed_files.json"
# 部门字段暂写死，后续可从配置或业务接口注入
_METADATA_DEPARTMENT = "company"
_METADATA_UPDATE_TIME = "2026-03-27"
# similarity = 1/(1+distance)，非概率；仅作单调标尺。默认仅保留 similarity >= 该值的片段
_DEFAULT_SIMILARITY_THRESHOLD = 0.5

_logger = logging.getLogger(__name__)


def _setup_cli_logging() -> None:
    """脚本直接运行时启用日志（含异常栈）。"""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _log_embedding_failure_hint(exc: BaseException) -> None:
    """在典型包装异常后补充说明，便于对照阿里云/网络问题排查。"""
    if isinstance(exc, KeyError) and exc.args == ("request",):
        _logger.error(
            "提示：上述 KeyError('request') 多见于 DashScope 返回非 200 时，"
            "langchain_community 用 requests.HTTPError 包装了 DashScope 的响应对象，"
            "初始化 HTTPError 时访问 .request 失败，真正的 status_code/message 未展示。"
            "请检查：代理/网络、API Key、embedding 额度与限流；"
            "text-embedding-v3 单批上限 10 条（本库内已分批，若仍失败可看控制台原始返回）。"
        )
    elif isinstance(exc, _RETRYABLE_REQUEST_ERRORS) or (
        isinstance(exc, OSError) and "SSL" in type(exc).__name__
    ):
        _logger.error(
            "提示：SSL/连接类错误常见于公司代理、VPN、防火墙截断 HTTPS，或本机证书链异常。"
            "可尝试：换网络、检查 HTTPS_PROXY/NO_PROXY、暂时关闭抓包代理；"
            "仍失败时在终端用 curl -v https://dashscope.aliyuncs.com 做连通性对比。"
        )


_practice_root = Path(__file__).resolve().parents[1]
if str(_practice_root) not in sys.path:
    sys.path.insert(0, str(_practice_root))
from utils import list_files_in_directory

# 课程练习根目录（…/data/课程练习）
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent
_FILES_PATH = Path(_PRACTICE_ROOT / "source" / "焦点科技公司制度")
_VECTOR_MODEL_PATH = Path(_PRACTICE_ROOT / "vectorModels" / "焦点科技公司制度")


load_dotenv()
_api_key = (
    os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
).strip()

if not _api_key:
    raise RuntimeError(
        "未配置 API Key：请在环境变量或本目录 .env 中设置 BAILIAN_API_KEY 或 DASHSCOPE_API_KEY"
    )
EMBEDDING_MODEL = "text-embedding-v2"


class DashScopeEmbeddingsSafe(BaseModel, Embeddings):
    """
    直接调用 DashScope SDK，避免 langchain 用 requests.HTTPError 包装响应导致 KeyError('request')；
    对 SSL/断连等网络异常按批重试，减轻偶发抖动。
    """

    model: str = EMBEDDING_MODEL
    dashscope_api_key: str = Field(default="")

    model_config = ConfigDict(extra="forbid")

    def _raise_if_bad(self, resp: Any, *, context: str) -> None:
        if getattr(resp, "status_code", None) == 200:
            return
        code = getattr(resp, "code", None)
        message = getattr(resp, "message", None)
        raise RuntimeError(
            f"{context}：status_code={getattr(resp, 'status_code', None)!r}, "
            f"code={code!r}, message={message!r}"
        )

    def _call_embedding(
        self,
        input_data: str | List[str],
        *,
        text_type: str,
    ) -> Any:
        delay = _EMBED_NETWORK_RETRY_BASE_SEC
        last: BaseException | None = None
        for attempt in range(1, _EMBED_NETWORK_RETRY + 1):
            try:
                return dashscope.TextEmbedding.call(
                    model=self.model,
                    input=input_data,
                    text_type=text_type,
                    api_key=self.dashscope_api_key or None,
                )
            except _RETRYABLE_REQUEST_ERRORS as e:
                last = e
                _logger.warning(
                    "DashScope 网络异常（批次 %d/%d）: %s，%.0fs 后重试",
                    attempt,
                    _EMBED_NETWORK_RETRY,
                    e,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        assert last is not None
        raise last

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        batch_size = _DASHSCOPE_EMBED_BATCH_SIZE.get(self.model, 25)
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._call_embedding(batch, text_type="document")
            self._raise_if_bad(resp, context="DashScope embed_documents 失败")
            embeddings = resp.output["embeddings"]
            out.extend(item["embedding"] for item in embeddings)
        return out

    def embed_query(self, text: str) -> List[float]:
        resp = self._call_embedding(text, text_type="query")
        self._raise_if_bad(resp, context="DashScope embed_query 失败")
        return resp.output["embeddings"][0]["embedding"]


def get_embeddings() -> DashScopeEmbeddingsSafe:
    dashscope.api_key = _api_key
    return DashScopeEmbeddingsSafe(model=EMBEDDING_MODEL, dashscope_api_key=_api_key)


def vector_bundle_exists(save_dir: str | Path) -> bool:
    """判断 LangChain FAISS 目录是否已完整（向量文件 + 文档库 pkl）。"""
    p = Path(save_dir)
    return p.is_dir() and (
        (p / f"{_FAISS_INDEX_NAME}.faiss").is_file()
        and (p / f"{_FAISS_INDEX_NAME}.pkl").is_file()
    )


def get_file_hash(file_path: str | Path) -> str:
    """计算文件 MD5，用于判断源文件是否变更。"""
    p = Path(file_path)
    with p.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _processed_files_path(save_dir: Path) -> Path:
    return save_dir / _PROCESSED_FILES_JSON


def load_processed_files(save_dir: str | Path) -> dict[str, str]:
    """加载「文件名 -> 文件 MD5」映射；文件不存在时返回空字典。"""
    p = _processed_files_path(Path(save_dir))
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
    out = _processed_files_path(save_p)
    with out.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def _remove_faiss_bundle(save_dir: Path) -> None:
    """删除落盘的 FAISS 向量与文档序列化文件（保留同目录下其它文件如 processed_files.json）。"""
    for suffix in (".faiss", ".pkl"):
        p = save_dir / f"{_FAISS_INDEX_NAME}{suffix}"
        if p.is_file():
            p.unlink()


def _faiss_doc_ids_for_source_file(vs: FAISS, source_file: str) -> list[str]:
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


def _indexed_source_file_names(vs: FAISS) -> set[str]:
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


def _chunks_dicts_from_faiss(vs: FAISS) -> list[dict[str, Any]]:
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


def process_text_with_splitter(
    all_chunks_data: list[dict[str, Any]],
    save_path: str | Path | None = None,
    *,
    index_name: str = _FAISS_INDEX_NAME,
) -> FAISS:
    """
    将已分块的文本批量向量化，构建 FAISS 并写入 save_path。

    落盘内容（与「单独的 page_info.pkl」区别见下）：
    - ``{index_name}.faiss``：仅向量索引。
    - ``{index_name}.pkl``：序列化的 (docstore, index_to_docstore_id)，其中每条 Document
      的 ``page_content`` 为正文，``metadata`` 含 source_file、file_hash、department、
      update_time、page_number、chunk_id 等。
      检索时可直接从返回的 Document 取页码，一般**不必再维护**额外的 page_info.pkl。

    「page_info」类文件的作用：在**不用 LangChain 文档对象**、只持有裸 FAISS 索引时，
    用行号/向量 id 去查表得到「对应哪一页、哪一段原文」；本脚本把这类信息已放进
    ``index.pkl`` 的 metadata 里，与向量一一对应。
    """
    if not all_chunks_data:
        raise ValueError("all_chunks_data 为空，无法建库")

    out_dir = Path(save_path) if save_path is not None else _VECTOR_MODEL_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [item["content"] for item in all_chunks_data]
    metadatas = [dict(item["metadata"]) for item in all_chunks_data]

    embeddings = get_embeddings()
    try:
        knowledge_base = FAISS.from_texts(
            texts,
            embeddings,
            metadatas=metadatas,
        )
        knowledge_base.save_local(str(out_dir), index_name=index_name)
    except Exception as e:
        _logger.exception(
            "向量化或保存 FAISS 失败：分块数=%d，输出目录=%s，模型=%s，异常类型=%s",
            len(texts),
            out_dir,
            EMBEDDING_MODEL,
            type(e).__name__,
        )
        _log_embedding_failure_hint(e)
        raise

    print(
        f"已向量化 {len(texts)} 条分块，已保存至 {out_dir} "
        f"（{index_name}.faiss + {index_name}.pkl）"
    )
    return knowledge_base


# PDF 抽取常见控制字符与 Latin-1 增补（\x7f-\xff）；中文等不在该范围内
_PDF_CTRL_AND_LATIN1_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]")
# 换行两侧空白，避免出现「\\n + 空格 + \\n」类冗余
_PDF_NL_SURROUND_SPACES_RE = re.compile(r"[ \t]*\n[ \t]*")


def _clean_pdf_extracted_text(text: str) -> str:
    """
    清洗 PyPDF2 抽取文本：剔除控制符与指定区段字符、替换 PDF 私用区列表符、压缩空行。
    在分块前调用，减轻 Token 浪费与 LLM 上下文杂乱（不影响向量检索语义为主）。
    """
    if not text:
        return text
    s = _PDF_CTRL_AND_LATIN1_RE.sub("", text)
    # Symbol/Wingdings 等常映射到私用区（如 \\uf0b2），统一为项目符号
    for sym in ("\uf0b2", "\uf0b7", "\uf0a7", "\uf0a8", "\uf09e", "\uf0fc"):
        s = s.replace(sym, "•")
    s = _PDF_NL_SURROUND_SPACES_RE.sub("\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_text_with_page_numbers(pdf) -> List[Tuple[str, int]]:
    """
    从 PDF 按页提取文本，每页一条记录，便于分块后标注准确页码。

    参数:
        pdf: PdfReader 实例

    返回:
        列表，元素为 (该页全文, 页码)，页码从 1 起；无正文的页不加入列表。
    """
    pages: List[Tuple[str, int]] = []
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


def _build_chunks_for_pdf(
    pdf_full: Path,
    pdf_name: str,
    doc_idx: int,
    file_hash: str,
    splitter: RecursiveCharacterTextSplitter,
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
                        "department": _METADATA_DEPARTMENT,
                        "update_time": _METADATA_UPDATE_TIME,
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
            _build_chunks_for_pdf(pdf_full, pdf_name, doc_idx, current_hash, splitter)
        )

    return all_chunks_data


def sync_vector_store(
    source_folder: str | Path | None = None,
    save_dir: str | Path | None = None,
) -> FAISS | None:
    """
    每次运行全量扫描源目录并比对 ``processed_files.json``（文件名 -> MD5），但 **Embedding 只对变动文件做**：

    - **冷启动**（目录下尚无 ``index.faiss``）：才会把所有 PDF 分块后一次性 ``FAISS.from_texts``（经 ``process_text_with_splitter``）。
    - **增量**（索引已存在且仅有部分文件新增/变更）：对每个待更新文件先按 ``source_file`` 从索引中 ``delete`` 旧向量，再 **仅将变动文件的分块** 合并后调用 **一次** ``add_texts``；未变更文件既不读 PDF 也不调用嵌入接口。
    - **清理**：源目录中已删除的 PDF，若仍出现在向量库 metadata 的 ``source_file`` 中，会按文件名找到 docstore id 并 ``delete``（不依赖 ``processed_files.json`` 是否仍记着该文件，避免残留）。

    说明：不能仅用 docstore 里 ``file_hash`` 的集合判断「跳过」——文件内容从 A 变到 B 时，旧块仍带着哈希 A，
    若不做按文件删除，会出现同一路径下新旧版本向量并存。因此用 ``processed_files.json`` 按 **文件名** 绑定当前应入库的 MD5，
    与「按 source_file 删旧再写入」配套使用。
    """
    folder_path = Path(source_folder if source_folder is not None else _FILES_PATH)
    target = Path(save_dir if save_dir is not None else _VECTOR_MODEL_PATH)
    target.mkdir(parents=True, exist_ok=True)

    names = [
        n
        for n in list_files_in_directory(folder_path)
        if Path(n).suffix.lower() == ".pdf"
    ]
    current_hashes = {n: get_file_hash(folder_path / n) for n in names}
    processed = load_processed_files(target)

    # 旧向量库无 processed 记录时无法可靠做增量对齐，全量重建
    if vector_bundle_exists(target) and not _processed_files_path(target).is_file():
        _logger.warning(
            "检测到向量库存在但未找到 %s，将删除旧索引并全量重建（与 file_hash 等元数据对齐）",
            _PROCESSED_FILES_JSON,
        )
        _remove_faiss_bundle(target)
        processed = {}

    to_refresh = [n for n in names if processed.get(n) != current_hashes[n]]

    if not vector_bundle_exists(target):
        # 仅冷启动：全量分块 + 一次性建索引（会嵌入全部 PDF）
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
        )
        chunks: list[dict[str, Any]] = []
        for doc_idx, pdf_name in enumerate(names, start=1):
            pdf_full = folder_path / pdf_name
            h = current_hashes[pdf_name]
            chunks.extend(
                _build_chunks_for_pdf(pdf_full, pdf_name, doc_idx, h, splitter)
            )
        if not chunks:
            print("无 PDF 分块，跳过建库")
            return None
        save_processed_files(target, {n: current_hashes[n] for n in names})
        save_pdf_chunks_json(chunks, save_dir=target)
        return process_text_with_splitter(chunks, target)

    names_set = set(names)
    embeddings = get_embeddings()
    vs = FAISS.load_local(
        str(target),
        embeddings,
        index_name=_FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )

    indexed_sources = _indexed_source_file_names(vs)
    # 源目录已不存在，但向量库仍带该 source_file 的分块（含 processed 未记录的旧数据）
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
            ids = _faiss_doc_ids_for_source_file(vs, pdf_name)
            if ids:
                vs.delete(ids)
            print(f"已清理源目录已删除文件对应向量: {pdf_name}")

    if to_refresh:
        # 增量：先按文件删掉旧向量，再只对变动文件解析 PDF；嵌入仅针对本批 new_chunks（一次 add_texts）
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
        )
        incremental_chunks: list[dict[str, Any]] = []
        for pdf_name in to_refresh:
            ids = _faiss_doc_ids_for_source_file(vs, pdf_name)
            if ids:
                vs.delete(ids)
            doc_idx = names.index(pdf_name) + 1
            pdf_full = folder_path / pdf_name
            h = current_hashes[pdf_name]
            incremental_chunks.extend(
                _build_chunks_for_pdf(pdf_full, pdf_name, doc_idx, h, splitter)
            )
            processed[pdf_name] = h
        if incremental_chunks:
            vs.add_texts(
                [c["content"] for c in incremental_chunks],
                metadatas=[c["metadata"] for c in incremental_chunks],
            )

    save_processed_files(target, processed)
    vs.save_local(str(target), index_name=_FAISS_INDEX_NAME)
    save_pdf_chunks_json(_chunks_dicts_from_faiss(vs), save_dir=target)
    print("向量库已按文件哈希校验并完成更新")
    return vs


def save_pdf_chunks_json(
    chunks: list[dict[str, Any]],
    *,
    save_dir: str | Path | None = None,
    chunk_filename_stem: str | None = None,
) -> Path:
    """
    将 ``read_pdf_files`` 返回的分块数据写入与 ``index.faiss`` 同目录的
    ``{stem}_chunk.json``（默认 stem 为向量库目录名，与 FAISS 落盘目录一致）。
    """
    if not chunks:
        raise ValueError("chunks 为空，跳过写入")
    out_parent = Path(save_dir) if save_dir is not None else _VECTOR_MODEL_PATH
    out_parent.mkdir(parents=True, exist_ok=True)
    stem = chunk_filename_stem if chunk_filename_stem is not None else out_parent.name
    out_path = out_parent / f"{stem}_chunk.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"分块 JSON 已保存：{out_path}（共 {len(chunks)} 条）")
    return out_path


def create_vector_model(
    chunks: list[dict[str, Any]] | None = None,
    *,
    save_dir: Path | None = None,
) -> FAISS | None:
    """
    默认走 ``sync_vector_store``：每次按 MD5 检查源 PDF，变更则更新向量库。
    若显式传入 ``chunks``，仅在向量库尚不存在时写入（与历史行为一致）。
    """
    target = save_dir if save_dir is not None else _VECTOR_MODEL_PATH
    if chunks is not None:
        if vector_bundle_exists(target):
            print("向量模型已存在")
            return None
        print("向量模型不存在，使用传入分块创建向量模型")
        target.mkdir(parents=True, exist_ok=True)
        save_pdf_chunks_json(chunks, save_dir=target)
        return process_text_with_splitter(chunks, target)
    return sync_vector_store(source_folder=_FILES_PATH, save_dir=target)


def _faiss_distance_to_similarity(distance: float) -> float:
    """
    将 FAISS 检索返回的距离转为便于阅读的相似度 ``(0, 1]``，越大越相关。

    LangChain 默认的 ``1 - d/√2`` 假定向量已 **L2 单位化**（距离上界约 ``√2``）。
    DashScope text-embedding 等常见 **未单位化**，距离常大于 ``√2``，会得到 **负数** 并触发
    ``Relevance scores must be between 0 and 1`` 警告，与「检索好坏」无关。

    此处用单调映射 ``1/(1+d)``：距离越小相似度越高；``d`` 为 ``IndexFlatL2`` 的原始值（一般为 **平方 L2**，
    仅用于排序与展示，不必与物理距离单位对齐）。
    """
    d = float(distance)
    if d < 0:
        d = 0.0
    return 1.0 / (1.0 + d)


def load_knowledge_base(
    save_dir: str | Path | None = None,
    *,
    index_name: str = _FAISS_INDEX_NAME,
) -> FAISS:
    """
    从本地目录加载 LangChain FAISS 向量库（与 ``sync_vector_store`` 落盘路径一致）。
    需与建库时使用同一套 Embedding 模型（本脚本为 DashScope ``EMBEDDING_MODEL``）。
    """
    target = Path(save_dir if save_dir is not None else _VECTOR_MODEL_PATH)
    if not vector_bundle_exists(target):
        raise FileNotFoundError(
            f"向量库不存在或不完整：{target}（请先运行同步：python langchain_rag.py sync）"
        )
    return FAISS.load_local(
        str(target),
        get_embeddings(),
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def similarity_search(
    query: str,
    *,
    k: int = 5,
    vector_db: FAISS | None = None,
    save_dir: str | Path | None = None,
    score_threshold: float | None = _DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    相似度检索（Step4）：将用户白话 ``query`` 经 ``embed_query`` 向量化，在 FAISS 中检索最相近的制度分块。

    返回字典列表，每项含：

    - ``distance``：FAISS ``IndexFlatL2`` 原始距离（一般为 **平方 L2**），**越小越相似**；
    - ``similarity``：``1/(1+distance)``，落在 ``(0, 1]``，**越大越相似**；
    - ``relevance_score``：与 ``similarity`` 相同；
    - 以及 ``source_file``、``page_number`` 等 metadata。

    **关于 ``similarity``**：由距离单调变换得到，**不是**校准后的「概率」或语义置信度，
    但同一模型、同一索引下 **越大表示向量空间越近**，适合排序与统一阈值。

    :param k: 至多返回 k 条**通过阈值**的结果
    :param score_threshold: 仅保留 ``similarity >= score_threshold``；``None`` 表示不过滤。
        默认 ``_DEFAULT_SIMILARITY_THRESHOLD``（0.5）。设阈值时会向 FAISS 多取邻居再过滤，以尽量凑满 k 条。
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")

    db = vector_db if vector_db is not None else load_knowledge_base(save_dir=save_dir)
    # 有阈值时多取 Top-N 再过滤，否则常出现「前 k 个邻居全不达标」而结果为空
    fetch_n = k
    if score_threshold is not None:
        fetch_n = min(300, max(k * 20, 80))

    pairs = db.similarity_search_with_score(q, k=fetch_n)
    rows: list[dict[str, Any]] = []
    for doc, dist in pairs:
        similarity = _faiss_distance_to_similarity(dist)
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
    return rows


def search_knowledge_base(
    query: str,
    vector_db: FAISS,
    top_k: int = 3,
    *,
    score_threshold: float | None = _DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    在**已加载**的 ``vector_db`` 上检索，避免重复 ``load_local``。

    与 ``similarity_search(..., vector_db=...)`` 等价；默认同样应用最小相似度
    ``_DEFAULT_SIMILARITY_THRESHOLD``。若需看全部 Top-K 不过滤，传入 ``score_threshold=None``。
    """
    return similarity_search(
        query,
        k=top_k,
        vector_db=vector_db,
        score_threshold=score_threshold,
    )


def _print_similarity_results(results: list[dict[str, Any]]) -> None:
    """CLI 下打印检索结果（简体中文）。"""
    if not results:
        print(
            "未找到相关制度片段（可提高 -k、降低 --min-score，或使用 --no-min-score 取消过滤）。"
        )
        return
    for i, row in enumerate(results, start=1):
        sim = row.get("similarity", row.get("relevance_score"))
        dist = row.get("distance")
        src = row.get("source_file") or "?"
        page = row.get("page_number")
        dist_s = f"{dist:.4f}" if dist is not None else "?"
        print(
            f"\n--- 片段 {i} | 相似度={sim:.4f}（越大越好）| 距离={dist_s}（越小越好）| "
            f"来源={src} | 页码={page} ---"
        )
        print(row.get("content", "").strip())


def _cmd_sync(_args: argparse.Namespace) -> None:
    sync_vector_store(source_folder=_FILES_PATH, save_dir=_VECTOR_MODEL_PATH)


def _cmd_search(args: argparse.Namespace) -> None:
    query = " ".join(args.query).strip()
    thr: float | None = None if args.no_min_score else args.min_score
    results = similarity_search(
        query,
        k=args.top_k,
        save_dir=_VECTOR_MODEL_PATH,
        score_threshold=thr,
    )
    _print_similarity_results(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="制度 RAG：同步 PDF 向量库（sync）或白话相似度检索（search）",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_sync = sub.add_parser("sync", help="扫描 PDF、按 MD5 增量更新 FAISS")
    p_sync.set_defaults(handler=_cmd_sync)

    p_search = sub.add_parser(
        "search",
        help="将白话转为向量，在 FAISS 中检索最相关的制度分块",
    )
    p_search.add_argument(
        "query",
        nargs="+",
        help="用户问题或关键词（白话）",
    )
    p_search.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        metavar="N",
        help="最多返回 N 条分块（默认 5）",
    )
    p_search.add_argument(
        "--min-score",
        type=float,
        default=_DEFAULT_SIMILARITY_THRESHOLD,
        metavar="S",
        help=(
            f"仅保留 similarity(=1/(1+距离)) >= S，默认 {_DEFAULT_SIMILARITY_THRESHOLD}；"
            "越大越严"
        ),
    )
    p_search.add_argument(
        "--no-min-score",
        action="store_true",
        help="取消相似度下限，返回距离最近的 top-k 条（调试/对比用）",
    )
    p_search.set_defaults(handler=_cmd_search)

    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is not None:
        args.handler(args)
        return
    _cmd_sync(args)


if __name__ == "__main__":
    _setup_cli_logging()
    main()
