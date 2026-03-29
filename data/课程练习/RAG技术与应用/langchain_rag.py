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
#
# 可复用模块（data/课程练习）：constants、langchain_faiss_store、dashscope_*、rag_pipeline、utils
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# macOS：FAISS / NumPy / PyTorch 等若各自链接 libomp，OpenMP 重复初始化会 abort（OMP: Error #15）
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

_practice_root = Path(__file__).resolve().parents[1]
if str(_practice_root) not in sys.path:
    sys.path.insert(0, str(_practice_root))

from constants import DEFAULT_RAG_SIMILARITY_THRESHOLD, DEFAULT_RAG_TOP_K
from dashscope_embedding import (
    RETRYABLE_EMBEDDING_ERRORS,
    get_dashscope_embeddings,
    log_embedding_failure_hint,
)
from dashscope_generation import (
    call_dashscope_chat as _call_dashscope_chat_impl,
    call_generation,
)
from dashscope_rerank import DEFAULT_RERANK_MODEL
from langchain_faiss_store import (
    FAISS_INDEX_NAME,
    create_vector_model as _create_vector_model_faiss,
    faiss_search_knowledge_base,
    faiss_similarity_search,
    load_faiss_vector_store,
    process_text_with_splitter as _process_text_with_splitter_faiss,
    save_pdf_chunks_json as _save_pdf_chunks_json_core,
    sync_vector_store as _sync_vector_store_faiss,
)
from rag_pipeline import generate_rag_answer as _generate_rag_answer_core
from utils import (
    add_rag_query_and_retrieval_args,
    print_rag_similarity_results,
    rag_cli_score_threshold,
)

_RETRYABLE_REQUEST_ERRORS = RETRYABLE_EMBEDDING_ERRORS

# 向量库构建与增量同步见 langchain_faiss_store；以下为练习脚本侧可改的默认元数据
_METADATA_DEPARTMENT = "company"
_METADATA_UPDATE_TIME = "2026-03-27"

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
CHAT_MODEL = "qwen-turbo".strip()


def get_embeddings() -> Embeddings:
    return get_dashscope_embeddings(_api_key, model=EMBEDDING_MODEL)


def process_text_with_splitter(
    all_chunks_data: list[dict[str, Any]],
    save_path: str | Path | None = None,
    *,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """分块 → 嵌入 → 构建 FAISS 并落盘。"""
    return _process_text_with_splitter_faiss(
        all_chunks_data,
        get_embeddings(),
        save_path,
        index_name=index_name,
        default_save_dir=_VECTOR_MODEL_PATH,
        embedding_model_label=EMBEDDING_MODEL,
        on_embedding_failure=log_embedding_failure_hint,
    )


def sync_vector_store(
    source_folder: str | Path | None = None,
    save_dir: str | Path | None = None,
) -> FAISS | None:
    """扫描 PDF 目录，按 MD5 做冷启动 / 增量更新 / 孤儿清理。"""
    return _sync_vector_store_faiss(
        embeddings=get_embeddings(),
        source_folder=source_folder if source_folder is not None else _FILES_PATH,
        save_dir=save_dir if save_dir is not None else _VECTOR_MODEL_PATH,
        embedding_model_label=EMBEDDING_MODEL,
        on_embedding_failure=log_embedding_failure_hint,
        metadata_department=_METADATA_DEPARTMENT,
        metadata_update_time=_METADATA_UPDATE_TIME,
    )


def save_pdf_chunks_json(
    chunks: list[dict[str, Any]],
    *,
    save_dir: str | Path | None = None,
    chunk_filename_stem: str | None = None,
) -> Path:
    """将分块列表写入 ``{stem}_chunk.json``。"""
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
    """默认走 ``sync_vector_store``；若传入 ``chunks`` 则仅在向量库不存在时建库。"""
    return _create_vector_model_faiss(
        get_embeddings(),
        _FILES_PATH,
        save_dir if save_dir is not None else _VECTOR_MODEL_PATH,
        chunks=chunks,
        embedding_model_label=EMBEDDING_MODEL,
        on_embedding_failure=log_embedding_failure_hint,
        metadata_department=_METADATA_DEPARTMENT,
        metadata_update_time=_METADATA_UPDATE_TIME,
    )


def load_knowledge_base(
    save_dir: str | Path | None = None,
    *,
    index_name: str = FAISS_INDEX_NAME,
) -> FAISS:
    """从本地目录加载 FAISS。"""
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
    k: int = DEFAULT_RAG_TOP_K,
    vector_db: FAISS | None = None,
    save_dir: str | Path | None = None,
    score_threshold: float | None = DEFAULT_RAG_SIMILARITY_THRESHOLD,
    rerank: bool = False,
) -> list[dict[str, Any]]:
    """向量检索 + 可选精排；实现见 ``langchain_faiss_store.faiss_similarity_search``。"""
    save = None if vector_db is not None else (
        save_dir if save_dir is not None else _VECTOR_MODEL_PATH
    )
    return faiss_similarity_search(
        query,
        vector_db=vector_db,
        embeddings=get_embeddings() if vector_db is None else None,
        save_dir=save,
        index_name=FAISS_INDEX_NAME,
        k=k,
        score_threshold=score_threshold,
        rerank=rerank,
        api_key_for_rerank=_api_key,
    )


def search_knowledge_base(
    query: str,
    vector_db: FAISS,
    top_k: int = DEFAULT_RAG_TOP_K,
    *,
    score_threshold: float | None = DEFAULT_RAG_SIMILARITY_THRESHOLD,
    rerank: bool = False,
) -> list[dict[str, Any]]:
    """在已加载 ``vector_db`` 上检索。"""
    return faiss_search_knowledge_base(
        query,
        vector_db,
        top_k=top_k,
        score_threshold=score_threshold,
        rerank=rerank,
        api_key_for_rerank=_api_key,
    )


def call_dashscope_chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> Any:
    """本练习默认模型与 Key 的对话封装。"""
    return _call_dashscope_chat_impl(
        messages,
        model=model or CHAT_MODEL,
        api_key=_api_key,
    )


def get_model(model_name: str, prompt: str, messages: list[dict[str, Any]]) -> Any:
    """兼容练习脚本签名。"""
    return call_generation(
        model_name,
        messages,
        api_key=_api_key,
        prompt=prompt or None,
    )


def generate_rag_answer(
    query: str,
    *,
    top_k: int = DEFAULT_RAG_TOP_K,
    score_threshold: float | None = DEFAULT_RAG_SIMILARITY_THRESHOLD,
    chat_model: str | None = None,
    vector_db: FAISS | None = None,
    save_dir: str | Path | None = None,
    use_rerank: bool = True,
) -> dict[str, Any]:
    """RAG 问答；编排逻辑见 ``rag_pipeline.generate_rag_answer``。"""
    return _generate_rag_answer_core(
        query,
        embeddings=get_embeddings(),
        api_key=_api_key,
        chat_model=chat_model or CHAT_MODEL,
        top_k=top_k,
        score_threshold=score_threshold,
        vector_db=vector_db,
        save_dir=save_dir if save_dir is not None else _VECTOR_MODEL_PATH,
        use_rerank=use_rerank,
        index_name=FAISS_INDEX_NAME,
        api_key_for_rerank=_api_key,
    )


def _cmd_sync(_args: argparse.Namespace) -> None:
    sync_vector_store(source_folder=_FILES_PATH, save_dir=_VECTOR_MODEL_PATH)


def _cmd_search(args: argparse.Namespace) -> None:
    query = " ".join(args.query).strip()
    results = similarity_search(
        query,
        k=args.top_k,
        save_dir=_VECTOR_MODEL_PATH,
        score_threshold=rag_cli_score_threshold(args),
        rerank=args.rerank,
    )
    print_rag_similarity_results(results)


def _cmd_ask(args: argparse.Namespace) -> None:
    query = " ".join(args.query).strip()
    out = generate_rag_answer(
        query,
        top_k=args.top_k,
        score_threshold=rag_cli_score_threshold(args),
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
                line, top_k=DEFAULT_RAG_TOP_K, save_dir=_VECTOR_MODEL_PATH
            )
        except Exception as e:
            _logger.exception("交互问答失败")
            print(f"错误：{e}\n")
            continue
        print(f"\n[模型: {out['model']}]\n{out['answer']}\n")


def _cmd_help(args: argparse.Namespace) -> None:
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
        f"  python {script} search 考勤 -k {DEFAULT_RAG_TOP_K}\n"
        f"  python {script} ask 年假怎么休 -v\n"
        "\n环境变量：BAILIAN_API_KEY / DASHSCOPE_API_KEY；"
        f"对话模型 DASHSCOPE_CHAT_MODEL（默认 {CHAT_MODEL!r}）。"
        f"\nRAG 默认召回 top_k={DEFAULT_RAG_TOP_K} 后精排模型 {DEFAULT_RERANK_MODEL}；search 加 --rerank、ask 可加 --no-rerank。\n"
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
    add_rag_query_and_retrieval_args(p_search, query_help="关键词或问题（白话）")
    p_search.add_argument(
        "--rerank",
        action="store_true",
        help=f"对召回结果使用 {DEFAULT_RERANK_MODEL} 精排后再输出（额外一次 DashScope 调用）",
    )
    p_search.set_defaults(handler=_cmd_search)

    p_ask = sub.add_parser("ask", help="单次 RAG 问答（非交互）")
    add_rag_query_and_retrieval_args(p_ask, query_help="用户问题（白话）")
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
        help=f"跳过 {DEFAULT_RERANK_MODEL}，仅用向量相似度作为上下文顺序",
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


# 兼容旧笔记/外部引用
_DEFAULT_TOP_K = DEFAULT_RAG_TOP_K
_DEFAULT_SIMILARITY_THRESHOLD = DEFAULT_RAG_SIMILARITY_THRESHOLD

if __name__ == "__main__":
    _setup_cli_logging()
    main()
