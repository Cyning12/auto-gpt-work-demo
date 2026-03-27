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

import json
import logging
import os
import sys
import time
from pathlib import Path
from PyPDF2 import PdfReader

import dashscope
import requests
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from typing import Any, List, Tuple

# 与 langchain_community.embeddings.dashscope 中 BATCH_SIZE 一致
_DASHSCOPE_EMBED_BATCH_SIZE: dict[str, int] = {
    "text-embedding-v1": 25,
    "text-embedding-v2": 25,
    "text-embedding-v3": 10,
    "text-embedding-v4": 10,
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
      的 ``page_content`` 为正文，``metadata`` 含 source_file、page_number、chunk_id 等。
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
        if stripped:
            pages.append((stripped, page_number))
    return pages


def read_pdf(pdf_path: Path) -> PdfReader:
    return PdfReader(pdf_path)


def read_pdf_files(
    folder: str | Path,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict[str, Any]]:
    """遍历同一目录下 PDF，按页提取文本后分块，整理为待向量化数据结构。"""
    folder_path = Path(folder)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    all_chunks_data: list[dict[str, Any]] = []

    names = [
        n
        for n in list_files_in_directory(folder_path)
        if Path(n).suffix.lower() == ".pdf"
    ]
    for doc_idx, pdf_name in enumerate(names, start=1):
        pdf_full = folder_path / pdf_name
        pdf = read_pdf(pdf_full)
        per_page = extract_text_with_page_numbers(pdf)

        char_total = sum(len(t) for t, _ in per_page)
        print(
            f"读取文件：{pdf_name}，页数（有文本）：{len(per_page)}，约 {char_total} 字符"
        )

        chunk_in_doc = 0
        for page_text, page_number in per_page:
            for chunk in splitter.split_text(page_text):
                chunk_in_doc += 1
                all_chunks_data.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "source_file": pdf_name,
                            "page_number": page_number,
                            "chunk_id": f"doc_{doc_idx:02d}_ch_{chunk_in_doc}",
                        },
                    }
                )

    return all_chunks_data


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
    """若向量目录已存在则跳过；否则从 PDF 读取分块并建库。"""
    target = save_dir if save_dir is not None else _VECTOR_MODEL_PATH
    if vector_bundle_exists(target):
        print("向量模型已存在")
        return None
    data = chunks if chunks is not None else read_pdf_files(_FILES_PATH)
    print("向量模型不存在，创建向量模型")
    save_pdf_chunks_json(data, save_dir=target)
    return process_text_with_splitter(data, target)


# text, page_numbers = extract_text_with_page_numbers(
#     read_pdf(
#         Path(
#             _PRACTICE_ROOT
#             / "source"
#             / "焦点科技公司制度"
#             / "内幕信息知情人登记制度（2025年10月）.pdf"
#         )
#     )
# )
# print(text)
# print(page_numbers)


def main() -> None:
    if vector_bundle_exists(_VECTOR_MODEL_PATH):
        print("向量模型已存在")
        return
    chunks = read_pdf_files(_FILES_PATH)
    print("向量模型不存在，创建向量模型")
    save_pdf_chunks_json(chunks, save_dir=_VECTOR_MODEL_PATH)
    process_text_with_splitter(chunks, _VECTOR_MODEL_PATH)


if __name__ == "__main__":
    _setup_cli_logging()
    main()
