# 课程练习：搭建完整的Disney RAG助手（原生RAG应用）
# Step1，数据层
# • 文档处理：解析Word文档(.docx/.doc)，提取文本段落和表格（转为Markdown格式）
# • 图像处理：支持图片OCR（Tesseract）和CLIP视觉特征提取
# Step2. 向量化层
# • 文本Embedding：使用阿里云百炼的 text-embedding-v4 模型（1024维）
# • 图像Embedding：使用CLIP模型提取图像特征（512维）
# • 双索引系统：FAISS分别构建文本和图像的向量索引
# Step3. 检索层
# • 混合检索：文本查询使用语义相似度检索，图像查询使用CLIP文本编码器
# • 关键词触发：检测特定关键词（如"海报"、"图片"）触发图像检索
# Step4. 生成层：将检索到的上下文组织成结构化提示

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# macOS：FAISS / NumPy / PyTorch 等与 OpenMP 冲突时可 abort，与 langchain_rag 一致
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 课程练习根目录（.../data/课程练习）；与 langchain_rag 等一致，由此导入拆分的工具模块
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent
if str(_PRACTICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_ROOT))

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from dashscope_embedding import get_dashscope_embeddings, log_embedding_failure_hint
from disney_classify import classify_disney_knowledge_files
from doc_file_utils import (
    chunks_json_dir_to_faiss_chunks,
    export_doc_and_docx_to_markdown,
    export_parsed_markdown_chunks_for_doc_paths,
    persist_word_chunks_to_faiss,
)
from hf_clip_utils import load_clip_model_and_processor
from image_file_utils import image_to_text

_DISNEY_KNOWLEDGE_BASE = _PRACTICE_ROOT / "source" / "迪士尼RAG知识库"
_PARSED_MD_DIR = _PRACTICE_ROOT / "doc" / "迪士尼RAG知识库" / "parsed_markdown"
_CHUNKS_JSON_DIR = _PRACTICE_ROOT / "doc" / "迪士尼RAG知识库" / "chunks_json"
# 与 langchain_rag._VECTOR_MODEL_PATH 风格一致：向量库落盘目录
_DISNEY_FAISS_DIR = _PRACTICE_ROOT / "vectorModels" / "迪士尼RAG知识库"
_EMBEDDING_MODEL = "text-embedding-v2"

# CLIP：默认缓存 ~/.cache/huggingface/hub/；可传 revision= 钉死权重，见 hf_clip_utils 模块说明
print("正在加载 CLIP 模型...")
try:
    clip_model, clip_processor = load_clip_model_and_processor()
    print("CLIP 模型加载成功。")
except Exception as e:
    print(f"加载 CLIP 模型失败，请检查网络连接或 Hugging Face Token。错误: {e}")
    raise SystemExit(1) from e


def _resolve_dashscope_api_key() -> str:
    return (
        os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    ).strip()


def _print_classification_summary(data: dict[str, list[str]]) -> None:
    """打印分类摘要与完整 JSON（中文保真）。"""
    print("迪士尼知识库文件分类统计：")
    for k in ("doc", "pdf", "ppt", "images", "other"):
        print(f"- {k}: {len(data[k])} 个")
    print("\n完整分类结果（JSON）：")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_knowledge_base(
    *,
    chunks_json_dir: Path | None = None,
    faiss_save_dir: Path | None = None,
    embeddings: Embeddings | None = None,
    api_key: str | None = None,
    embedding_model: str | None = None,
) -> tuple[list[dict[str, Any]], Any, list[Any]]:
    """
    读取已切片 ``*_chunks.json``，经 ``doc_file_utils.persist_word_chunks_to_faiss`` 调用
    ``langchain_faiss_store.process_text_with_splitter``（与 ``langchain_rag.process_text_with_splitter`` 同源），
    将向量写入 ``{faiss_save_dir}/index.faiss`` 与 ``index.pkl``。

    返回 ``(metadata_store, faiss_vector_store, image_vectors)``：``metadata_store`` 与入库分块顺序一致；
    ``image_vectors`` 仍为空列表占位。向量仅存 FAISS，不再在内存中单独保留 float 列表。
    """
    metadata_store: list[dict[str, Any]] = []
    image_vectors: list[Any] = []

    json_dir = chunks_json_dir if chunks_json_dir is not None else _CHUNKS_JSON_DIR
    target = faiss_save_dir if faiss_save_dir is not None else _DISNEY_FAISS_DIR
    chunks = chunks_json_dir_to_faiss_chunks(json_dir)

    if not chunks:
        print(f"[build_knowledge_base] 未在 {json_dir} 找到有效切片，跳过 FAISS 写入。")
        return metadata_store, None, image_vectors

    embedder = embeddings
    if embedder is None:
        key = (api_key or "").strip() or _resolve_dashscope_api_key()
        if not key:
            raise ValueError(
                "写入 FAISS 需要 API Key：请设置 BAILIAN_API_KEY 或 DASHSCOPE_API_KEY，"
                "或传入 build_knowledge_base(api_key=...)。"
            )
        embedder = get_dashscope_embeddings(
            key, model=embedding_model or _EMBEDDING_MODEL
        )

    model_label = embedding_model or _EMBEDDING_MODEL
    vs = persist_word_chunks_to_faiss(
        chunks,
        embedder,
        target,
        embedding_model_label=model_label,
        on_embedding_failure=log_embedding_failure_hint,
    )
    metadata_store = [dict(c["metadata"]) for c in chunks]
    print(f"[build_knowledge_base] 已写入 FAISS：{target}，分块数 {len(chunks)}。")
    return metadata_store, vs, image_vectors


if __name__ == "__main__":
    load_dotenv()
    classified = classify_disney_knowledge_files(_DISNEY_KNOWLEDGE_BASE)
    _print_classification_summary(classified)
    export_doc_and_docx_to_markdown(classified, _DISNEY_KNOWLEDGE_BASE, _PARSED_MD_DIR)
    # 对已导出的 Markdown（由 chunks_to_markdown 生成）还原块并切片，写入 _CHUNKS_JSON_DIR

    export_parsed_markdown_chunks_for_doc_paths(
        classified["doc"],
        _DISNEY_KNOWLEDGE_BASE,
        _PARSED_MD_DIR,
        _CHUNKS_JSON_DIR,
    )

    metadata_store, _, _image_vecs = build_knowledge_base()
    print(
        f"知识库文本索引：metadata_store={len(metadata_store)} 条；"
        f"本地向量库目录：{_DISNEY_FAISS_DIR}（index.faiss / index.pkl）。"
    )

    # for image_path in classified["images"]:
    #     img_text_info = image_to_text(str(_DISNEY_KNOWLEDGE_BASE / image_path))

    #     print(img_text_info)
