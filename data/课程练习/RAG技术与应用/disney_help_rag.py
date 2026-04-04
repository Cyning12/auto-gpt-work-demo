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

import hashlib
import json
import os
import sys
import pickle
from pathlib import Path
from typing import Any
import argparse

# macOS：FAISS / NumPy / PyTorch 等与 OpenMP 冲突时可 abort，与 langchain_rag 一致
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 课程练习根目录（.../data/课程练习）；与 langchain_rag 等一致，由此导入拆分的工具模块
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent
if str(_PRACTICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_ROOT))

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from dashscope_embedding import get_dashscope_embeddings, log_embedding_failure_hint
from disney_classify import classify_disney_knowledge_files
from doc_file_utils import (
    DEFAULT_WORD_CHUNK_METADATA_DEPARTMENT,
    DEFAULT_WORD_CHUNK_METADATA_UPDATE_TIME,
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
_DISNEY_IMAGE_FAISS_DIR = _DISNEY_FAISS_DIR / "images"
_EMBEDDING_MODEL = "text-embedding-v2"

# CLIP：默认缓存 ~/.cache/huggingface/hub/；可传 revision= 钉死权重，见 hf_clip_utils 模块说明
print("正在加载 CLIP 模型...")
try:
    clip_model, clip_processor = load_clip_model_and_processor()
    print("CLIP 模型加载成功。")
except Exception as e:
    print(f"加载 CLIP 模型失败，请检查网络连接或 Hugging Face Token。错误: {e}")
    raise SystemExit(1) from e

from PIL import Image
import torch


def _image_file_md5_hex(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_image_embedding(image_path):
    """获取图片的 Embedding。"""
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].numpy()


def persist_image_vectors_to_faiss(
    metadata_store: list[dict[str, Any]],
    image_vectors: list[Any],
    save_dir: str | Path,
    *,
    index_name: str = "images.index",
) -> Path:
    """
    将图片向量写入本地 FAISS，并落盘 metadata 映射。

    说明：图片向量（CLIP，通常 512 维）与文本向量（DashScope text-embedding-v2，通常 1024 维）
    **不在同一向量空间**，生产上应维护**单独索引**。本函数保存：

    - ``{save_dir}/{index_name}.faiss``：FAISS 索引
    - ``{save_dir}/{index_name}.pkl``：Python pickle，保存 ``metadata_store``（顺序与向量一致）
    """
    import numpy as np
    import faiss

    if len(metadata_store) != len(image_vectors):
        raise ValueError(
            f"metadata_store 与 image_vectors 长度不一致：{len(metadata_store)} vs {len(image_vectors)}"
        )
    if not image_vectors:
        raise ValueError("image_vectors 为空，无法创建索引")

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vec = np.array(image_vectors, dtype=np.float32)
    if vec.ndim != 2:
        raise ValueError(f"image_vectors 期望二维数组 (n, d)，实际 ndim={vec.ndim}")

    dim = int(vec.shape[1])
    index = faiss.IndexFlatL2(dim)
    index.add(vec)

    index_path = out_dir / f"{index_name}.faiss"
    meta_path = out_dir / f"{index_name}.pkl"
    faiss.write_index(index, str(index_path))
    with meta_path.open("wb") as f:
        pickle.dump(metadata_store, f)
    return out_dir


def load_image_faiss_and_metadata(
    save_dir: str | Path,
    *,
    index_name: str = "images.index",
) -> tuple[Any, list[dict[str, Any]]]:
    """
    读取 ``persist_image_vectors_to_faiss`` 落盘的图片索引与 metadata。

    返回 ``(faiss_index, metadata_store)``；其中 ``metadata_store`` 的顺序与向量写入顺序一致。
    """
    import faiss

    base = Path(save_dir)
    index_path = base / f"{index_name}.faiss"
    meta_path = base / f"{index_name}.pkl"

    if not index_path.is_file():
        raise FileNotFoundError(f"图片索引不存在: {index_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"图片 metadata 不存在: {meta_path}")

    index = faiss.read_index(str(index_path))
    with meta_path.open("rb") as f:
        metadata_store = pickle.load(f)
    if not isinstance(metadata_store, list):
        raise ValueError(
            f"图片 metadata 格式异常，期望 list，实际 {type(metadata_store)}"
        )
    if getattr(index, "ntotal", None) is not None and index.ntotal != len(
        metadata_store
    ):
        raise ValueError(
            f"图片索引与 metadata 数量不一致: index.ntotal={index.ntotal} vs meta={len(metadata_store)}"
        )
    return index, metadata_store


def load_text_faiss_vector_store(
    save_dir: str | Path,
    embeddings: Embeddings,
    *,
    index_name: str = "index",
) -> FAISS:
    """读取 LangChain `FAISS.save_local` 生成的文本向量库（index.faiss + index.pkl）。"""
    return FAISS.load_local(
        str(save_dir),
        embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def create_images_vector_store(
    image_paths: list[str] | list[Path],
    *,
    metadata_department: str = DEFAULT_WORD_CHUNK_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_WORD_CHUNK_METADATA_UPDATE_TIME,
    image_index_start: int = 0,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """
    遍历图片路径，计算 CLIP 向量，并生成与 Word 切片同形的 metadata。

    metadata 与 ``doc_file_utils.build_chunks_for_word_document`` / ``build_chunks_from_parsed_markdown_file``
    对齐字段：``source_file``、``chunk_id``、``file_hash``、``department``、``update_time``、
    ``block_index``、``block_type``；并增加 ``content_basis``、``ocr_text``、``path_raw``（图片专用）。
    """
    image_vectors: list[Any] = []
    metadata_store: list[dict[str, Any]] = []

    for i, image_path in enumerate(image_paths):
        p = Path(image_path)
        img_text_info = image_to_text(p)
        doc_order = image_index_start + i + 1
        metadata_store.append(
            {
                "source_file": p.name,
                "chunk_id": f"img_{doc_order:02d}_ch_01",
                "file_hash": _image_file_md5_hex(p),
                "department": metadata_department,
                "update_time": metadata_update_time,
                "block_index": 1,
                "block_type": "image",
                "content_basis": "image_clip",
                "ocr_text": img_text_info["ocr"],
                "path_raw": str(image_path),
            }
        )
        image_vectors.append(get_image_embedding(p))

    return metadata_store, image_vectors


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


def _cmd_build_text(args: argparse.Namespace) -> None:
    """仅构建文本向量库（Word → Markdown → chunks_json → LangChain FAISS 落盘）。"""
    load_dotenv()
    classified = classify_disney_knowledge_files(_DISNEY_KNOWLEDGE_BASE)
    if args.verbose:
        _print_classification_summary(classified)

    export_doc_and_docx_to_markdown(classified, _DISNEY_KNOWLEDGE_BASE, _PARSED_MD_DIR)
    export_parsed_markdown_chunks_for_doc_paths(
        classified["doc"],
        _DISNEY_KNOWLEDGE_BASE,
        _PARSED_MD_DIR,
        _CHUNKS_JSON_DIR,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        metadata_department=args.department,
        metadata_update_time=args.update_time,
    )

    metadata_store, _vs, _ = build_knowledge_base(
        chunks_json_dir=_CHUNKS_JSON_DIR,
        faiss_save_dir=Path(args.out_dir) if args.out_dir else _DISNEY_FAISS_DIR,
        api_key=args.api_key or None,
        embedding_model=args.embedding_model or None,
        embeddings=None,
        image_paths=None,
    )
    print(
        f"[build-text] 完成：metadata_store={len(metadata_store)}；"
        f"向量库目录={(Path(args.out_dir) if args.out_dir else _DISNEY_FAISS_DIR)}"
    )


def _cmd_build_images(args: argparse.Namespace) -> None:
    """仅构建图片向量库（CLIP image embedding → 原生 FAISS 落盘）。"""
    load_dotenv()
    classified = classify_disney_knowledge_files(_DISNEY_KNOWLEDGE_BASE)
    if args.verbose:
        _print_classification_summary(classified)

    image_abs_paths = [
        _DISNEY_KNOWLEDGE_BASE / rel for rel in classified.get("images", [])
    ]
    if not image_abs_paths:
        print("[build-images] 未发现图片文件，跳过。")
        return

    image_metadata_store, image_vectors = create_images_vector_store(
        image_abs_paths,
        metadata_department=args.department,
        metadata_update_time=args.update_time,
    )
    out_dir = Path(args.out_dir) if args.out_dir else _DISNEY_IMAGE_FAISS_DIR
    persist_image_vectors_to_faiss(
        image_metadata_store,
        image_vectors,
        out_dir,
        index_name=args.index_name,
    )
    print(
        f"[build-images] 完成：images={len(image_vectors)}；"
        f"向量库目录={out_dir}（{args.index_name}.faiss / {args.index_name}.pkl）"
    )


def _cmd_build_all(args: argparse.Namespace) -> None:
    """默认构建：文本 + 图片。"""
    _cmd_build_text(args)
    _cmd_build_images(args)


def build_knowledge_base(
    *,
    chunks_json_dir: Path | None = None,
    faiss_save_dir: Path | None = None,
    image_faiss_save_dir: Path | None = None,
    embeddings: Embeddings | None = None,
    api_key: str | None = None,
    embedding_model: str | None = None,
    image_paths: list[str] | list[Path] | None = None,
) -> tuple[list[dict[str, Any]], Any, list[Any]]:
    """
    读取已切片 ``*_chunks.json``，经 ``doc_file_utils.persist_word_chunks_to_faiss`` 调用
    ``langchain_faiss_store.process_text_with_splitter``（与 ``langchain_rag.process_text_with_splitter`` 同源），
    将向量写入 ``{faiss_save_dir}/index.faiss`` 与 ``index.pkl``。

    返回 ``(metadata_store, faiss_vector_store, image_vectors)``：

    - ``metadata_store``：文本入库分块 metadata（顺序与入库一致）
    - ``faiss_vector_store``：文本向量库（LangChain FAISS）
    - ``image_vectors``：图片向量列表（便于 debug；生产可不保留在内存）

    若传入 ``image_paths``，会同时创建图片向量并写入 ``image_faiss_save_dir``（默认 ``{faiss_save_dir}/images``）。
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

    # 图片索引：单独落盘（与文本 embedding 维度/空间不同，不合并进同一个 index）
    if image_paths:
        img_dir = (
            image_faiss_save_dir
            if image_faiss_save_dir is not None
            else (target / "images")
        )
        image_metadata_store, image_vectors = create_images_vector_store(
            image_paths,
            metadata_department=DEFAULT_WORD_CHUNK_METADATA_DEPARTMENT,
            metadata_update_time=DEFAULT_WORD_CHUNK_METADATA_UPDATE_TIME,
        )
        persist_image_vectors_to_faiss(
            image_metadata_store,
            image_vectors,
            img_dir,
            index_name="images.index",
        )
        print(
            f"[build_knowledge_base] 已写入图片 FAISS：{img_dir}，图片数 {len(image_vectors)}。"
        )
    return metadata_store, vs, image_vectors


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="迪士尼知识库：构建文本/图片向量库到本地 FAISS。",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    def _add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--department",
            type=str,
            default=DEFAULT_WORD_CHUNK_METADATA_DEPARTMENT,
            help="写入 metadata.department（与文本切片一致）",
        )
        p.add_argument(
            "--update-time",
            type=str,
            default=DEFAULT_WORD_CHUNK_METADATA_UPDATE_TIME,
            help="写入 metadata.update_time（与文本切片一致）",
        )
        p.add_argument(
            "--out-dir",
            type=str,
            default="",
            metavar="PATH",
            help="输出目录；文本默认 vectorModels/迪士尼RAG知识库，图片默认其 images 子目录",
        )
        p.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="打印文件分类摘要",
        )

    p_text = sub.add_parser("build-text", help="仅构建文本向量库（Word→FAISS）")
    _add_common_args(p_text)
    p_text.add_argument(
        "--api-key",
        type=str,
        default="",
        help="覆盖 DashScope API Key（也可用环境变量 BAILIAN_API_KEY / DASHSCOPE_API_KEY）",
    )
    p_text.add_argument(
        "--embedding-model",
        type=str,
        default=_EMBEDDING_MODEL,
        help=f"文本 embedding 模型（默认 {_EMBEDDING_MODEL!r}）",
    )
    p_text.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="文本切片 chunk_size（与 doc_file_utils 默认一致）",
    )
    p_text.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help="文本切片 chunk_overlap（与 doc_file_utils 默认一致）",
    )
    p_text.set_defaults(handler=_cmd_build_text)

    p_img = sub.add_parser("build-images", help="仅构建图片向量库（CLIP→FAISS）")
    _add_common_args(p_img)
    p_img.add_argument(
        "--index-name",
        type=str,
        default="images.index",
        help="图片索引文件前缀名（默认 images.index，产物为 .faiss + .pkl）",
    )
    p_img.set_defaults(handler=_cmd_build_images)

    p_all = sub.add_parser("build-all", help="构建文本+图片向量库（默认）")
    _add_common_args(p_all)
    p_all.add_argument(
        "--api-key",
        type=str,
        default="",
        help="覆盖 DashScope API Key（也可用环境变量 BAILIAN_API_KEY / DASHSCOPE_API_KEY）",
    )
    p_all.add_argument(
        "--embedding-model",
        type=str,
        default=_EMBEDDING_MODEL,
        help=f"文本 embedding 模型（默认 {_EMBEDDING_MODEL!r}）",
    )
    p_all.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="文本切片 chunk_size（与 doc_file_utils 默认一致）",
    )
    p_all.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help="文本切片 chunk_overlap（与 doc_file_utils 默认一致）",
    )
    p_all.add_argument(
        "--index-name",
        type=str,
        default="images.index",
        help="图片索引文件前缀名（默认 images.index，产物为 .faiss + .pkl）",
    )
    p_all.set_defaults(handler=_cmd_build_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is not None:
        args.handler(args)
        return
    # 未指定子命令时，默认构建文本 + 图片
    _cmd_build_all(args)


if __name__ == "__main__":
    main()

    # for image_path in classified["images"]:
    #     img_text_info = image_to_text(str(_DISNEY_KNOWLEDGE_BASE / image_path))

    #     print(img_text_info)
