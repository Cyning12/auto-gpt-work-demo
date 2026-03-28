# 2026-3-26 课程练习：RAG技术与应用
# 任务：结合你的业务场景，创建你的RAG问答（LangChain+DeepSeek+Faiss）
# 结合你的业务场景，创建你的RAG问答（LangChain+DeepSeek+Faiss）
# Step1，收集整理知识库 (./source/焦点科技公司制度)
# Step2，从PDF中提取文本并记录每行文本对应的页码
# Step3，处理文本并创建向量存储
# Step4，执行相似度搜索，找到与查询相关的文档（可选 gte-rerank-v2 精排）
# Step5，使用问到链对用户问题进行回答 （使用你的DASHSCOPE_API_KEY；默认先精排再拼上下文）
# Step6，显示每个文档块的来源页码
# 直接运行本脚本：进入终端交互问答；子命令 sync / search / ask 见 --help
from __future__ import annotations

import os
import sys

# macOS：FAISS / NumPy / PyTorch 等若各自链接 libomp，OpenMP 重复初始化会 abort（OMP: Error #15）
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import time
from pathlib import Path

import dashscope
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from http import HTTPStatus
from typing import Any

PROFESSIONAL_SYSTEM_TEMPLATE = """
# ROLE（角色）
你是一个严谨的公司行政助手。
请基于以下提供的【制度片段】回答用户的问题。

【已知制度】：
{context}

【用户问题】：
{query}

【注意事项】：
1. 仅根据【已知信息】回答。例如:如果信息中没有提到该委员会的特定人数要求，请直说“未找到该委员会的具体人数规定”。
2. 严禁使用“参考其他委员会”或“根据常识”进行类比推理。
3. 如果已知信息之间存在冲突，请全部列出并提示差异。
4. 如果已知信息中提到比例（如三分之二），请根据该比例计算并回答用户关于具体人数（如一半）的问题。
5. 请在回答时，必须明确指出信息来源于哪份文件以及对应的页码（例如：根据《XXX》第 X 页所述...）。
"""


# 向量库构建与增量同步见 langchain_faiss_store；以下为练习脚本侧可改的默认元数据
_METADATA_DEPARTMENT = "company"
_METADATA_UPDATE_TIME = "2026-03-27"
# similarity = 1/(1+distance)，非概率；仅作单调标尺。默认仅保留 similarity >= 该值的片段
_DEFAULT_SIMILARITY_THRESHOLD = 0.3
# 检索参与上下文的分块条数默认上限（向量召回条数；RAG 作答后接精排重排同一批）
_DEFAULT_TOP_K = 20
# DashScope 文本精排模型（与向量 embedding 模型独立）
_RERANK_MODEL = "gte-rerank-v2"
_RERANK_RETRY = 5
_RERANK_RETRY_BASE_SEC = 2.0

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
from dashscope_embedding import RETRYABLE_EMBEDDING_ERRORS, get_dashscope_embeddings
from dashscope_generation import call_generation
from langchain_faiss_store import (
    FAISS_INDEX_NAME,
    create_vector_model as _create_vector_model_faiss,
    load_faiss_vector_store,
    process_text_with_splitter as _process_text_with_splitter_faiss,
    save_pdf_chunks_json as _save_pdf_chunks_json_core,
    sync_vector_store as _sync_vector_store_faiss,
)
from utils import generation_first_message

# 精排等网络重试与嵌入模块共用同一组可重试异常类型
_RETRYABLE_REQUEST_ERRORS = RETRYABLE_EMBEDDING_ERRORS

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
# 通用对话模型（RAG 生成阶段）；可通过环境变量覆盖，例如 deepseek-v3、qwen-plus
CHAT_MODEL = "qwen-turbo".strip()


def get_embeddings() -> Embeddings:
    return get_dashscope_embeddings(_api_key, model=EMBEDDING_MODEL)


def process_text_with_splitter(
    all_chunks_data: list[dict[str, Any]],
    save_path: str | Path | None = None,
    *,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """
    分块 → 嵌入 → 构建 FAISS 并落盘。实现细节与元数据字段说明见 ``langchain_faiss_store``。
    """
    return _process_text_with_splitter_faiss(
        all_chunks_data,
        get_embeddings(),
        save_path,
        index_name=index_name,
        default_save_dir=_VECTOR_MODEL_PATH,
        embedding_model_label=EMBEDDING_MODEL,
        on_embedding_failure=_log_embedding_failure_hint,
    )


def sync_vector_store(
    source_folder: str | Path | None = None,
    save_dir: str | Path | None = None,
) -> FAISS | None:
    """
    扫描 PDF 目录，按 ``processed_files.json`` 与 MD5 做冷启动 / 增量更新 / 孤儿清理。
    底层增删（``delete`` / ``add_texts``）见 ``langchain_faiss_store.sync_vector_store``。
    """
    return _sync_vector_store_faiss(
        embeddings=get_embeddings(),
        source_folder=source_folder if source_folder is not None else _FILES_PATH,
        save_dir=save_dir if save_dir is not None else _VECTOR_MODEL_PATH,
        embedding_model_label=EMBEDDING_MODEL,
        on_embedding_failure=_log_embedding_failure_hint,
        metadata_department=_METADATA_DEPARTMENT,
        metadata_update_time=_METADATA_UPDATE_TIME,
    )


def save_pdf_chunks_json(
    chunks: list[dict[str, Any]],
    *,
    save_dir: str | Path | None = None,
    chunk_filename_stem: str | None = None,
) -> Path:
    """
    将分块列表写入 ``{stem}_chunk.json``；默认目录为本练习向量库路径。
    """
    return _save_pdf_chunks_json_core(
        chunks,
        save_dir=save_dir if save_dir is not None else _VECTOR_MODEL_PATH,
        chunk_filename_stem=chunk_filename_stem,
    )


def create_vector_model(
    chunks: list[dict[str, Any]] | None = None,
    *,
    save_dir: Path | None = None,
) -> FAISS | None:
    """
    默认走 ``sync_vector_store``：每次按 MD5 检查源 PDF，变更则更新向量库。
    若显式传入 ``chunks``，仅在向量库尚不存在时写入（与历史行为一致）。
    """
    return _create_vector_model_faiss(
        get_embeddings(),
        _FILES_PATH,
        save_dir if save_dir is not None else _VECTOR_MODEL_PATH,
        chunks=chunks,
        embedding_model_label=EMBEDDING_MODEL,
        on_embedding_failure=_log_embedding_failure_hint,
        metadata_department=_METADATA_DEPARTMENT,
        metadata_update_time=_METADATA_UPDATE_TIME,
    )


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


def _call_dashscope_rerank(
    query: str,
    documents: list[str],
    *,
    top_n: int,
    api_key: str,
) -> Any:
    """调用 TextReRank，网络类错误按指数退避重试。"""
    delay = _RERANK_RETRY_BASE_SEC
    last: BaseException | None = None
    for attempt in range(1, _RERANK_RETRY + 1):
        try:
            return dashscope.TextReRank.call(
                model=_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False,
                api_key=api_key or None,
            )
        except _RETRYABLE_REQUEST_ERRORS as e:
            last = e
            _logger.warning(
                "DashScope Rerank 网络异常（%d/%d）: %s，%.0fs 后重试",
                attempt,
                _RERANK_RETRY,
                e,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    assert last is not None
    raise last


def rerank_retrieval_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_n: int | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    使用 DashScope ``gte-rerank-v2`` 对已有向量检索结果重排序，提升问答上下文相关性。

    - 在每条 hit 上增加 ``rerank_score``；保留原 ``similarity`` / ``distance`` 为向量检索标尺。
    - 若接口失败则记录警告并**原样返回** ``hits``（不中断 RAG）。
    """
    if not hits:
        return []
    key = (api_key or _api_key).strip()
    n = top_n if top_n is not None else len(hits)
    n = max(1, min(n, len(hits)))
    docs = [str(h.get("content", "") or "") for h in hits]
    try:
        resp = _call_dashscope_rerank(query, docs, top_n=n, api_key=key)
    except Exception as e:
        _logger.warning("Rerank 请求异常，回退为向量检索顺序：%s", e)
        return hits
    if getattr(resp, "status_code", None) != HTTPStatus.OK:
        _logger.warning(
            "Rerank 非成功响应，回退为向量检索顺序：status_code=%r message=%r",
            getattr(resp, "status_code", None),
            getattr(resp, "message", None),
        )
        return hits
    output = getattr(resp, "output", None)
    if output is None or not getattr(output, "results", None):
        _logger.warning("Rerank 返回无 output.results，回退为向量检索顺序")
        return hits
    out: list[dict[str, Any]] = []
    for res in output.results:
        idx = int(res.index if hasattr(res, "index") else res["index"])
        if idx < 0 or idx >= len(hits):
            continue
        row = dict(hits[idx])
        rs = float(
            res.relevance_score
            if hasattr(res, "relevance_score")
            else res["relevance_score"]
        )
        row["rerank_score"] = rs
        out.append(row)
    return out if out else hits


def load_knowledge_base(
    save_dir: str | Path | None = None,
    *,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """
    从本地目录加载 LangChain FAISS 向量库（与 ``sync_vector_store`` 落盘路径一致）。
    需与建库时使用同一套 Embedding 模型（本脚本为 DashScope ``EMBEDDING_MODEL``）。
    """
    target = save_dir if save_dir is not None else _VECTOR_MODEL_PATH
    try:
        return load_faiss_vector_store(
            target,
            get_embeddings(),
            index_name=index_name,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"向量库不存在或不完整：{target}（请先运行同步：python {Path(__file__).name} sync）"
        ) from e


def similarity_search(
    query: str,
    *,
    k: int = _DEFAULT_TOP_K,
    vector_db: FAISS | None = None,
    save_dir: str | Path | None = None,
    score_threshold: float | None = _DEFAULT_SIMILARITY_THRESHOLD,
    rerank: bool = False,
) -> list[dict[str, Any]]:
    """
    相似度检索（Step4）：将用户白话 ``query`` 经 ``embed_query`` 向量化，在 FAISS 中检索最相近的制度分块。

    返回字典列表，每项含：

    - ``distance``：FAISS ``IndexFlatL2`` 原始距离（一般为 **平方 L2**），**越小越相似**；
    - ``similarity``：``1/(1+distance)``，落在 ``(0, 1]``，**越大越相似**；
    - ``relevance_score``：与 ``similarity`` 相同；
    - 以及 ``source_file``、``page_number`` 等 metadata。
    - 若 ``rerank=True``，在通过阈值后的前 ``k`` 条上调用 ``gte-rerank-v2`` 重排，并附加 ``rerank_score``。

    **关于 ``similarity``**：由距离单调变换得到，**不是**校准后的「概率」或语义置信度，
    但同一模型、同一索引下 **越大表示向量空间越近**，适合排序与统一阈值。

    :param k: 至多返回 k 条**通过阈值**的结果
    :param score_threshold: 仅保留 ``similarity >= score_threshold``；``None`` 表示不过滤。
        默认 ``_DEFAULT_SIMILARITY_THRESHOLD``。设阈值时会向 FAISS 多取邻居再过滤，以尽量凑满 k 条。
    :param rerank: 是否对召回结果做 DashScope 精排（默认否，``ask``/交互 RAG 在生成阶段单独开启）。
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
    if rerank and rows:
        rows = rerank_retrieval_hits(q, rows, top_n=len(rows), api_key=_api_key)
    return rows


def search_knowledge_base(
    query: str,
    vector_db: FAISS,
    top_k: int = _DEFAULT_TOP_K,
    *,
    score_threshold: float | None = _DEFAULT_SIMILARITY_THRESHOLD,
    rerank: bool = False,
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
        rerank=rerank,
    )


def _format_rag_context(hits: list[dict[str, Any]]) -> str:
    """将检索分块拼成注入大模型的上下文。"""
    if not hits:
        return "（未检索到达到相似度阈值的制度片段。）"
    parts: list[str] = []
    for i, h in enumerate(hits, start=1):
        src = h.get("source_file") or "未知文件"
        page = h.get("page_number")
        sim = h.get("similarity")
        if page is not None:
            head = f"[{i}] 《{src}》第 {page} 页"
        else:
            head = f"[{i}] 《{src}》"
        rr = h.get("rerank_score")
        if rr is not None:
            head += f"（rerank≈{float(rr):.3f}）"
        if sim is not None:
            head += f"（向量 similarity≈{float(sim):.3f}）"
        parts.append(f"{head}\n{h.get('content', '').strip()}")
    return "\n\n---\n\n".join(parts)


def build_rag_prompt_text(query: str, hits: list[dict[str, Any]]) -> str:
    """拼装完整系统提示（含制度上下文与用户问题）。"""
    return PROFESSIONAL_SYSTEM_TEMPLATE.format(
        context=_format_rag_context(hits),
        query=(query or "").strip(),
    )


def call_dashscope_chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> Any:
    """调用 DashScope 文本生成（通用对话模型），与向量嵌入模型相互独立。"""
    return call_generation(
        model or CHAT_MODEL,
        messages,
        api_key=_api_key,
    )


def get_model(model_name: str, prompt: str, messages: list[dict[str, Any]]) -> Any:
    """兼容练习脚本签名；使用本模块已校验的 ``_api_key`` 调用 ``dashscope_generation``。"""
    return call_generation(
        model_name,
        messages,
        api_key=_api_key,
        prompt=prompt or None,
    )


def chat_answer_text(response: Any) -> str:
    """从 Generation 响应中取出 assistant 文本。"""
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
    if not content:
        raise RuntimeError("DashScope 返回的 assistant message 无 content")
    return str(content).strip()


def generate_rag_answer(
    query: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    score_threshold: float | None = _DEFAULT_SIMILARITY_THRESHOLD,
    chat_model: str | None = None,
    vector_db: FAISS | None = None,
    save_dir: str | Path | None = None,
    use_rerank: bool = True,
) -> dict[str, Any]:
    """
    RAG 问答（Step5）：先用本地 FAISS 检索制度片段，再调用通用对话模型（默认 ``CHAT_MODEL``）生成回答。

    - **向量模型**：仅负责 ``similarity_search`` 中的 ``embed_query`` / 索引距离。
    - **精排模型**：默认对召回的 ``top_k`` 条调用 DashScope ``gte-rerank-v2`` 重排后再拼上下文。
    - **通用模型**：负责阅读理解片段并组织自然语言答案（可通过环境变量 ``DASHSCOPE_CHAT_MODEL`` 指定）。

    返回 ``answer``、``model``、``retrieval``（检索分块列表）、``context``（拼好的片段正文）。
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")

    hits = similarity_search(
        q,
        k=top_k,
        vector_db=vector_db,
        save_dir=save_dir,
        score_threshold=score_threshold,
    )
    if use_rerank and hits:
        hits = rerank_retrieval_hits(q, hits, top_n=len(hits), api_key=_api_key)
    system_text = build_rag_prompt_text(q, hits)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": q},
    ]
    resp = call_dashscope_chat(messages, model=chat_model)
    if getattr(resp, "status_code", None) != HTTPStatus.OK:
        code = getattr(resp, "code", None)
        message = getattr(resp, "message", None)
        raise RuntimeError(
            f"DashScope 对话失败：status_code={getattr(resp, 'status_code', None)!r}, "
            f"code={code!r}, message={message!r}"
        )
    answer = chat_answer_text(resp)
    return {
        "answer": answer,
        "model": chat_model or CHAT_MODEL,
        "retrieval": hits,
        "context": _format_rag_context(hits),
    }


def _print_similarity_results(results: list[dict[str, Any]]) -> None:
    """CLI 下打印检索结果（简体中文）。"""
    if not results:
        print(
            "未找到相关制度片段（可提高 -k、降低 --min-score，或使用 --no-min-score 取消过滤）。"
        )
        return
    for i, row in enumerate(results, start=1):
        sim = row.get("similarity", row.get("relevance_score"))
        rr = row.get("rerank_score")
        dist = row.get("distance")
        src = row.get("source_file") or "?"
        page = row.get("page_number")
        dist_s = f"{dist:.4f}" if dist is not None else "?"
        extra = f" | 精排={float(rr):.4f}" if rr is not None else ""
        sim_s = f"{float(sim):.4f}" if sim is not None else "?"
        print(
            f"\n--- 片段 {i} | 向量相似度={sim_s}（越大越好）{extra} | 距离={dist_s}（越小越好）| "
            f"来源={src} | 页码={page} ---"
        )
        print(row.get("content", "").strip())


def _cli_score_threshold(args: argparse.Namespace) -> float | None:
    """命令行：是否启用 similarity 下限。"""
    return None if args.no_min_score else args.min_score


def _add_query_and_retrieval_args(
    p: argparse.ArgumentParser, *, query_help: str
) -> None:
    p.add_argument("query", nargs="+", help=query_help)
    p.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        dest="top_k",
        metavar="N",
        help=f"检索片段数上限（默认 {_DEFAULT_TOP_K}）",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=_DEFAULT_SIMILARITY_THRESHOLD,
        metavar="S",
        help=(
            f"similarity(=1/(1+距离)) 下限，默认 {_DEFAULT_SIMILARITY_THRESHOLD}；"
            "越大越严"
        ),
    )
    p.add_argument(
        "--no-min-score",
        action="store_true",
        help="取消 similarity 下限（调试用）",
    )


def _cmd_sync(_args: argparse.Namespace) -> None:
    sync_vector_store(source_folder=_FILES_PATH, save_dir=_VECTOR_MODEL_PATH)


def _cmd_search(args: argparse.Namespace) -> None:
    query = " ".join(args.query).strip()
    results = similarity_search(
        query,
        k=args.top_k,
        save_dir=_VECTOR_MODEL_PATH,
        score_threshold=_cli_score_threshold(args),
        rerank=args.rerank,
    )
    _print_similarity_results(results)


def _cmd_ask(args: argparse.Namespace) -> None:
    query = " ".join(args.query).strip()
    out = generate_rag_answer(
        query,
        top_k=args.top_k,
        score_threshold=_cli_score_threshold(args),
        save_dir=_VECTOR_MODEL_PATH,
        chat_model=args.chat_model or None,
        use_rerank=not args.no_rerank,
    )
    print(f"[模型: {out['model']}]\n")
    print(out["answer"])
    if args.verbose:
        print("\n---------- 检索到的片段摘要 ----------")
        for i, h in enumerate(out["retrieval"], start=1):
            src = h.get("source_file") or "?"
            page = h.get("page_number")
            sim = h.get("similarity")
            rr = h.get("rerank_score")
            rr_s = f" rerank={rr:.3f}" if rr is not None else ""
            print(f"  {i}. {src} p.{page} sim={sim}{rr_s}")


def _cmd_interactive(_args: argparse.Namespace) -> None:
    """无子命令：终端循环输入问题，RAG 作答。"""
    print(
        "制度 RAG 交互问答（直接回车，或输入 exit / quit / q 结束）\n"
        f"对话模型：{CHAT_MODEL}；更新知识库请另开终端执行：python {Path(__file__).name} sync\n"
    )
    while True:
        try:
            line = input("请输入问题 > ").strip()
        except EOFError:
            print()
            break
        if not line or line.lower() in ("exit", "quit", "q"):
            break
        try:
            out = generate_rag_answer(
                line, top_k=_DEFAULT_TOP_K, save_dir=_VECTOR_MODEL_PATH
            )
        except Exception as e:
            _logger.exception("交互问答失败")
            print(f"错误：{e}\n")
            continue
        print(f"\n[模型: {out['model']}]\n{out['answer']}\n")


def _cmd_help(args: argparse.Namespace) -> None:
    """打印总帮助或某一子命令的帮助（与 ``<子命令> -h`` 等价）。"""
    parser = _build_cli_parser()
    if args.help_topic:
        parser.parse_args([args.help_topic, "-h"])
        return
    parser.print_help()
    script = Path(__file__).name
    print(
        "\n常用示例：\n"
        f"  python {script}                    # 交互 RAG 问答（默认）\n"
        f"  python {script} help               # 本帮助\n"
        f"  python {script} help search        # 仅看 search 的选项\n"
        f"  python {script} sync               # 同步 PDF → FAISS\n"
        f"  python {script} search 考勤 -k {_DEFAULT_TOP_K}\n"
        f"  python {script} ask 年假怎么休 -v\n"
        "\n环境变量：BAILIAN_API_KEY / DASHSCOPE_API_KEY；"
        f"对话模型 DASHSCOPE_CHAT_MODEL（默认 {CHAT_MODEL!r}）。"
        f"\nRAG 默认召回 top_k={_DEFAULT_TOP_K} 后精排模型 {_RERANK_MODEL}；search 加 --rerank、ask 可加 --no-rerank。\n"
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "默认：交互式 RAG 问答。子命令：sync / search / ask / help（或传 -h）。"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("sync", help="扫描 PDF、按 MD5 增量更新 FAISS").set_defaults(
        handler=_cmd_sync
    )

    p_search = sub.add_parser("search", help="仅向量检索，打印制度片段")
    _add_query_and_retrieval_args(p_search, query_help="关键词或问题（白话）")
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help=f"对召回结果使用 {_RERANK_MODEL} 精排后再输出（额外一次 DashScope 调用）",
    )
    p_search.set_defaults(handler=_cmd_search)

    p_ask = sub.add_parser("ask", help="单次 RAG 问答（非交互）")
    _add_query_and_retrieval_args(p_ask, query_help="用户问题（白话）")
    p_ask.add_argument(
        "--chat-model",
        type=str,
        default="",
        metavar="NAME",
        help=f"覆盖对话模型（默认 {CHAT_MODEL!r} 或环境变量）",
    )
    p_ask.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="打印检索片段摘要",
    )
    p_ask.add_argument(
        "--no-rerank",
        action="store_true",
        help=f"跳过 {_RERANK_MODEL}，仅用向量相似度作为上下文顺序",
    )
    p_ask.set_defaults(handler=_cmd_ask)

    p_help = sub.add_parser(
        "help",
        help="显示用法说明；help <子命令> 等同于 <子命令> --help",
    )
    p_help.add_argument(
        "help_topic",
        nargs="?",
        default=None,
        metavar="子命令",
        choices=("sync", "search", "ask"),
        help="可选：sync / search / ask",
    )
    p_help.set_defaults(handler=_cmd_help)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is not None:
        args.handler(args)
        return
    _cmd_interactive(args)


if __name__ == "__main__":
    _setup_cli_logging()
    main()
