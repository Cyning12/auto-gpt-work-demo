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
import time
import asyncio
from functools import lru_cache
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
from image_file_utils import image_to_text

_DISNEY_KNOWLEDGE_BASE = _PRACTICE_ROOT / "source" / "迪士尼RAG知识库"
_PARSED_MD_DIR = _PRACTICE_ROOT / "doc" / "迪士尼RAG知识库" / "parsed_markdown"
_CHUNKS_JSON_DIR = _PRACTICE_ROOT / "doc" / "迪士尼RAG知识库" / "chunks_json"
# 与 langchain_rag._VECTOR_MODEL_PATH 风格一致：向量库落盘目录
_DISNEY_FAISS_DIR = _PRACTICE_ROOT / "vectorModels" / "迪士尼RAG知识库"
_DISNEY_IMAGE_FAISS_DIR = _DISNEY_FAISS_DIR / "images"
_EMBEDDING_MODEL = "text-embedding-v1"

from PIL import Image
import torch


@lru_cache(maxsize=1)
def _get_clip_model_and_processor():
    """
    懒加载 CLIP（进程内缓存复用）。

    仅在需要**重新编码**图片向量时才加载，避免 inspect-local / 仅文本流程也触发 HF Hub 请求，
    以及命中图片向量缓存时仍加载大模型。
    """
    # 延迟导入，避免 CLI 仅做本地读取时也触发 transformers 初始化链路
    from hf_clip_utils import load_clip_model_and_processor

    print("正在加载 CLIP 模型...")
    model, processor = load_clip_model_and_processor()
    print("CLIP 模型加载成功。")
    return model, processor


def _image_file_md5_hex(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_image_embedding(image_path):
    """获取图片的 Embedding（一维 float32，长度 = CLIP 视觉投影维度，如 512）。"""
    import numpy as np

    image = Image.open(image_path)
    clip_model, clip_processor = _get_clip_model_and_processor()
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = clip_model.get_image_features(**inputs)
    # 新版 transformers：get_image_features 可能返回 BaseModelOutputWithPooling，
    # 若对其做 out[0] 会得到 last_hidden_state (batch, seq, 768)，squeeze 后变成 (seq, 768) 误用。
    if isinstance(out, torch.Tensor):
        feat = out
    else:
        pool = getattr(out, "pooler_output", None)
        if pool is not None:
            feat = pool
        else:
            lhs = out.last_hidden_state
            feat = clip_model.visual_projection(lhs[:, 0, :])
    row = feat[0].detach().cpu().numpy()
    row = np.squeeze(row, axis=None).astype(np.float32, copy=False)
    if row.ndim != 1:
        raise ValueError(
            f"CLIP 图像向量应为 1 维，实际 shape={getattr(row, 'shape', None)}"
        )
    return row


def get_clip_text_embedding_1d(text: str):
    """将用户问句编码为 CLIP 文本侧向量（与图片向量同空间，用于跨模态检索）。"""
    import numpy as np

    clip_model, clip_processor = _get_clip_model_and_processor()
    inputs = clip_processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        out = clip_model.get_text_features(**inputs)
    if isinstance(out, torch.Tensor):
        feat = out
    else:
        pool = getattr(out, "pooler_output", None)
        if pool is not None:
            feat = pool
        else:
            lhs = out.last_hidden_state
            feat = clip_model.text_projection(lhs[:, -1, :])
    row = feat[0].detach().cpu().numpy()
    row = np.squeeze(row, axis=None).astype(np.float32, copy=False)
    if row.ndim != 1:
        raise ValueError(
            f"CLIP 文本向量应为 1 维，实际 shape={getattr(row, 'shape', None)}"
        )
    return row


def similarity_search_text_topk(
    query: str,
    *,
    embeddings: Embeddings,
    text_faiss_dir: str | Path,
    top_k: int = 10,
    index_name: str = "index",
) -> list[dict[str, Any]]:
    """
    文本向量库：基于 DashScope（与建库相同）对 query 做 embed_query，再在 LangChain FAISS 中取 Top-K。

    返回列表元素含 ``rank``、``l2_distance``、``page_content``、``metadata``（便于后续与图片结果分流处理）。
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")

    vs = load_text_faiss_vector_store(
        text_faiss_dir,
        embeddings,
        index_name=index_name,
    )
    k = max(1, int(top_k))
    pairs = vs.similarity_search_with_score(q, k=k)
    rows: list[dict[str, Any]] = []
    for rank, (doc, dist) in enumerate(pairs, start=1):
        meta = dict(doc.metadata) if doc.metadata else {}
        rows.append(
            {
                "rank": rank,
                "l2_distance": float(dist),
                "page_content": doc.page_content,
                "metadata": meta,
            }
        )
    return rows


def similarity_search_image_topk_clip(
    query: str,
    *,
    image_faiss_dir: str | Path,
    top_k: int = 10,
    index_name: str = "images.index",
) -> list[dict[str, Any]]:
    """
    图片向量库：CLIP 文本编码 query，在本地 ``IndexFlatL2`` 上取 Top-K。

    返回列表元素含 ``rank``、``l2_distance``、``metadata``（与建库时 pickle 一致）。
    """
    import numpy as np

    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")

    index, metadata_store = load_image_faiss_and_metadata(
        image_faiss_dir,
        index_name=index_name,
    )
    ntotal = int(getattr(index, "ntotal", 0) or 0)
    if ntotal <= 0:
        return []

    vec = get_clip_text_embedding_1d(q).astype(np.float32, copy=False).reshape(1, -1)
    k = min(max(1, int(top_k)), ntotal)
    dists, ids = index.search(vec, k)
    rows: list[dict[str, Any]] = []
    for rank, (dist, idx) in enumerate(zip(dists[0], ids[0]), start=1):
        ii = int(idx)
        if ii < 0 or ii >= len(metadata_store):
            continue
        rows.append(
            {
                "rank": rank,
                "l2_distance": float(dist),
                "metadata": dict(metadata_store[ii]),
            }
        )
    return rows


def persist_image_vectors_to_faiss(
    metadata_store: list[dict[str, Any]],
    image_vectors: list[Any],
    save_dir: str | Path,
    *,
    index_name: str = "images.index",
    log_write: bool = True,
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

    rows: list[np.ndarray] = []
    for v in image_vectors:
        r = np.asarray(v, dtype=np.float32)
        r = np.squeeze(r)
        if r.ndim != 1:
            raise ValueError(
                f"每条 image 向量应为 1 维，实际 shape={r.shape}（请检查 get_image_embedding 输出）"
            )
        rows.append(r)
    dim0 = int(rows[0].shape[0])
    for i, r in enumerate(rows):
        if int(r.shape[0]) != dim0:
            raise ValueError(
                f"向量维度不一致：第 0 条 dim={dim0}，第 {i} 条 dim={r.shape[0]}"
            )
    vec = np.stack(rows, axis=0)
    if vec.ndim != 2:
        raise ValueError(f"image_vectors 期望堆叠为 (n, d)，实际 ndim={vec.ndim}")

    dim = int(vec.shape[1])
    index = faiss.IndexFlatL2(dim)
    index.add(vec)

    index_path = out_dir / f"{index_name}.faiss"
    meta_path = out_dir / f"{index_name}.pkl"
    faiss.write_index(index, str(index_path))
    with meta_path.open("wb") as f:
        pickle.dump(metadata_store, f)
    if log_write:
        print(
            f"[图片向量] 落盘完成 n={index.ntotal} dim={dim} "
            f"faiss={index_path} meta={meta_path}"
        )
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


def get_or_create_image_faiss_cache(
    image_paths: list[str] | list[Path],
    save_dir: str | Path,
    *,
    index_name: str = "images.index",
    use_cache: bool = True,
    metadata_department: str = DEFAULT_WORD_CHUNK_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_WORD_CHUNK_METADATA_UPDATE_TIME,
    quiet_images: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    """
    获取（优先本地缓存）或创建图片向量库，并返回 ``(faiss_index, metadata_store)``。

    - **use_cache=True**：若 ``{save_dir}/{index_name}.faiss`` 与 ``.pkl`` 存在，则直接读取；
      否则重新编码图片并落盘，再读取返回。
    - **use_cache=False**：强制重新编码并覆盖落盘（适合模型升级/元数据变更/排错）。
    """
    base = Path(save_dir)
    index_path = base / f"{index_name}.faiss"
    meta_path = base / f"{index_name}.pkl"
    can_load = use_cache and index_path.is_file() and meta_path.is_file()
    if can_load:
        if not quiet_images:
            print(f"[图片向量] 命中本地缓存：{index_path} / {meta_path}")
        return load_image_faiss_and_metadata(base, index_name=index_name)

    if not quiet_images:
        reason = "禁用缓存" if not use_cache else "缓存不存在"
        print(f"[图片向量] 重新构建（{reason}）：out={base} index_name={index_name!r}")

    meta, vecs = create_images_vector_store(
        image_paths,
        metadata_department=metadata_department,
        metadata_update_time=metadata_update_time,
        log_vectors=not quiet_images,
    )
    persist_image_vectors_to_faiss(
        meta,
        vecs,
        base,
        index_name=index_name,
        log_write=not quiet_images,
    )
    return load_image_faiss_and_metadata(base, index_name=index_name)


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


def load_text_faiss_index_and_metadata(
    save_dir: str | Path,
    *,
    index_name: str = "index",
) -> tuple[Any, Any, dict[int, str]]:
    """
    **不依赖 Embeddings** 地读取本地文本向量库（与 LangChain `FAISS.save_local` 产物兼容）。

    返回 ``(faiss_index, docstore, index_to_docstore_id)``：
    - ``{save_dir}/{index_name}.faiss``：FAISS 索引
    - ``{save_dir}/{index_name}.pkl``：pickle，内容为 ``(docstore, index_to_docstore_id)``

    用途：仅做本地“是否存在/能读/规模统计”或后续自定义检索（避免因 embeddings 触发 API）。
    """
    import faiss

    base = Path(save_dir)
    index_path = base / f"{index_name}.faiss"
    meta_path = base / f"{index_name}.pkl"
    if not index_path.is_file():
        raise FileNotFoundError(f"文本索引不存在: {index_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"文本 metadata 不存在: {meta_path}")

    index = faiss.read_index(str(index_path))
    with meta_path.open("rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)
    if not isinstance(index_to_docstore_id, dict):
        raise ValueError(
            f"index_to_docstore_id 格式异常，期望 dict，实际 {type(index_to_docstore_id)}"
        )
    if getattr(index, "ntotal", None) is not None and index.ntotal != len(
        index_to_docstore_id
    ):
        raise ValueError(
            f"文本索引与映射数量不一致: index.ntotal={index.ntotal} vs map={len(index_to_docstore_id)}"
        )
    return index, docstore, index_to_docstore_id


def create_images_vector_store(
    image_paths: list[str] | list[Path],
    *,
    metadata_department: str = DEFAULT_WORD_CHUNK_METADATA_DEPARTMENT,
    metadata_update_time: str = DEFAULT_WORD_CHUNK_METADATA_UPDATE_TIME,
    image_index_start: int = 0,
    log_vectors: bool = True,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """
    遍历图片路径，计算 CLIP 向量，并生成与 Word 切片同形的 metadata。

    metadata 与 ``doc_file_utils.build_chunks_for_word_document`` / ``build_chunks_from_parsed_markdown_file``
    对齐字段：``source_file``、``chunk_id``、``file_hash``、``department``、``update_time``、
    ``block_index``、``block_type``；并增加 ``content_basis``、``ocr_text``、``path_raw``（图片专用）。

    ``log_vectors``：是否打印每张图进度（path、dim、L2、OCR 字数、ocr_ms、clip_ms）及落盘摘要。
    """
    import numpy as np

    image_vectors: list[Any] = []
    metadata_store: list[dict[str, Any]] = []
    n = len(image_paths)
    t_batch0 = time.perf_counter()
    if log_vectors and n:
        print(f"[图片向量] 开始 CLIP 编码，共 {n} 张")

    for i, image_path in enumerate(image_paths):
        p = Path(image_path)
        t_ocr0 = time.perf_counter()
        img_text_info = image_to_text(p)
        ocr_ms = (time.perf_counter() - t_ocr0) * 1000.0
        doc_order = image_index_start + i + 1
        chunk_id = f"img_{doc_order:02d}_ch_01"
        metadata_store.append(
            {
                "source_file": p.name,
                "chunk_id": chunk_id,
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
        t_enc0 = time.perf_counter()
        emb = get_image_embedding(p)
        enc_ms = (time.perf_counter() - t_enc0) * 1000.0
        image_vectors.append(emb)
        if log_vectors:
            ocr_n = len((img_text_info.get("ocr") or "").strip())
            l2 = float(np.linalg.norm(emb))
            path_hint = str(p.resolve()) if p.exists() else str(image_path)
            print(
                f"[图片向量] {i + 1}/{n} {chunk_id} file={p.name!r} "
                f"path={path_hint} "
                f"dim={emb.shape[0]} L2={l2:.4f} ocr_chars={ocr_n} "
                f"ocr_ms={ocr_ms:.1f} clip_ms={enc_ms:.1f}"
            )

    if log_vectors and n:
        total_ms = (time.perf_counter() - t_batch0) * 1000.0
        print(
            f"[图片向量] 编码完成，向量条数={len(image_vectors)}，"
            f"CLIP 累计约 {total_ms:.0f} ms（含循环内开销，不含落盘）"
        )

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

    out_dir = Path(args.out_dir) if args.out_dir else _DISNEY_IMAGE_FAISS_DIR
    quiet_images = bool(getattr(args, "quiet_images", False))
    index, meta = get_or_create_image_faiss_cache(
        image_abs_paths,
        out_dir,
        index_name=args.index_name,
        use_cache=not bool(getattr(args, "no_cache", False)),
        metadata_department=args.department,
        metadata_update_time=args.update_time,
        quiet_images=quiet_images,
    )
    print(
        f"[build-images] 完成：images={index.ntotal}；"
        f"向量库目录={out_dir}（{args.index_name}.faiss / {args.index_name}.pkl）"
    )


def _cmd_inspect_local(args: argparse.Namespace) -> None:
    """并发读取本地文本/图片向量库（不重建），汇总输出。"""
    text_dir = Path(args.text_dir) if args.text_dir else _DISNEY_FAISS_DIR
    image_dir = Path(args.image_dir) if args.image_dir else _DISNEY_IMAGE_FAISS_DIR
    text_index_name = args.text_index_name or "index"
    image_index_name = args.image_index_name or "images.index"

    async def _load_text() -> dict[str, Any]:
        idx, docstore, mapping = await asyncio.to_thread(
            load_text_faiss_index_and_metadata,
            text_dir,
            index_name=text_index_name,
        )
        # docstore 多为 InMemoryDocstore，有 _dict；这里做兼容性统计
        ds_len = getattr(docstore, "_dict", None)
        return {
            "ok": True,
            "index_ntotal": getattr(idx, "ntotal", None),
            "index_dim": getattr(idx, "d", None),
            "docstore_size": len(ds_len) if isinstance(ds_len, dict) else None,
            "mapping_size": len(mapping),
            "dir": str(text_dir),
        }

    async def _load_images() -> dict[str, Any]:
        idx, meta = await asyncio.to_thread(
            load_image_faiss_and_metadata,
            image_dir,
            index_name=image_index_name,
        )
        return {
            "ok": True,
            "index_ntotal": getattr(idx, "ntotal", None),
            "index_dim": getattr(idx, "d", None),
            "metadata_size": len(meta),
            "dir": str(image_dir),
        }

    async def _run() -> tuple[dict[str, Any], dict[str, Any]]:
        async def _wrap(coro):
            try:
                return await coro
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return await asyncio.gather(_wrap(_load_text()), _wrap(_load_images()))

    text_res, img_res = asyncio.run(_run())
    print("[inspect-local] 汇总：")
    print(f"- text: {text_res}")
    print(f"- image: {img_res}")


def _print_retrieve_results(
    text_out: dict[str, Any],
    image_out: dict[str, Any],
    *,
    q: str,
    k: int,
    tag: str = "retrieve",
) -> None:
    """打印双路检索结果（文本 + 图片）。"""
    print(f"[{tag}] query={q!r} top_k={k}")
    print(f"[{tag}] --- 文本向量 Top-K（L2 越小越近）---")
    if not text_out.get("ok"):
        print(f"  失败: {text_out.get('error')}")
    else:
        for h in text_out.get("hits") or []:
            meta = h.get("metadata") or {}
            src = meta.get("source_file", "")
            cid = meta.get("chunk_id", "")
            prev = (h.get("page_content") or "")[:120].replace("\n", " ")
            print(
                f"  #{h.get('rank')} L2={h.get('l2_distance'):.4f} "
                f"chunk_id={cid!r} source_file={src!r} preview={prev!r}"
            )
    print(f"[{tag}] --- 图片向量 Top-K（CLIP 文本↔图，L2 越小越近）---")
    if not image_out.get("ok"):
        print(f"  失败: {image_out.get('error')}")
    else:
        for h in image_out.get("hits") or []:
            meta = h.get("metadata") or {}
            print(
                f"  #{h.get('rank')} L2={h.get('l2_distance'):.4f} "
                f"chunk_id={meta.get('chunk_id')!r} file={meta.get('source_file')!r} "
                f"path_raw={meta.get('path_raw')!r}"
            )
    print(f"[{tag}] 汇总：")
    print(f"- text_ok={text_out.get('ok')} n={len(text_out.get('hits') or [])}")
    print(f"- image_ok={image_out.get('ok')} n={len(image_out.get('hits') or [])}")


async def _retrieve_dual_async(
    q: str,
    *,
    embedder: Embeddings | None,
    text_dir: Path,
    image_dir: Path,
    top_k: int,
    text_index_name: str,
    image_index_name: str,
    require_text_key: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """并发：文本 Top-K（可选，无 Key 时跳过）+ 图片 Top-K。"""

    async def _text() -> dict[str, Any]:
        if embedder is None:
            if require_text_key:
                return {
                    "ok": False,
                    "error": "缺少 DashScope API Key，无法做文本向量检索",
                    "hits": [],
                }
            return {
                "ok": False,
                "error": "未配置 API Key，已跳过文本向量检索",
                "hits": [],
            }
        try:
            hits = await asyncio.to_thread(
                similarity_search_text_topk,
                q,
                embeddings=embedder,
                text_faiss_dir=text_dir,
                top_k=top_k,
                index_name=text_index_name,
            )
            return {"ok": True, "hits": hits}
        except Exception as e:
            return {"ok": False, "error": str(e), "hits": []}

    async def _image() -> dict[str, Any]:
        try:
            hits = await asyncio.to_thread(
                similarity_search_image_topk_clip,
                q,
                image_faiss_dir=image_dir,
                top_k=top_k,
                index_name=image_index_name,
            )
            return {"ok": True, "hits": hits}
        except Exception as e:
            return {"ok": False, "error": str(e), "hits": []}

    return await asyncio.gather(_text(), _image())


def _cmd_retrieve(args: argparse.Namespace) -> None:
    """
    RAG 检索（暂不调用大模型）：对用户问句分别做文本向量 Top-K 与 CLIP 图片向量 Top-K，并发执行后汇总打印。
    """
    load_dotenv()
    text_dir = Path(args.text_dir) if args.text_dir else _DISNEY_FAISS_DIR
    image_dir = Path(args.image_dir) if args.image_dir else _DISNEY_IMAGE_FAISS_DIR
    k = int(args.top_k)
    q = (args.query or "").strip()

    key = (args.api_key or "").strip() or _resolve_dashscope_api_key()
    if not key:
        raise ValueError(
            "文本检索需要 DashScope API Key：请设置 BAILIAN_API_KEY 或 DASHSCOPE_API_KEY，"
            "或传入 --api-key（图片侧用 CLIP，不依赖该 Key）。"
        )
    embedder = get_dashscope_embeddings(key, model=args.embedding_model or _EMBEDDING_MODEL)

    text_out, image_out = asyncio.run(
        _retrieve_dual_async(
            q,
            embedder=embedder,
            text_dir=text_dir,
            image_dir=image_dir,
            top_k=k,
            text_index_name=args.text_index_name or "index",
            image_index_name=args.image_index_name or "images.index",
            require_text_key=True,
        )
    )
    _print_retrieve_results(text_out, image_out, q=q, k=k, tag="retrieve")


def _cmd_ask(args: argparse.Namespace) -> None:
    """
    交互式问答入口（不调大模型）：循环读取用户问题，对每问执行与 retrieve 相同的双路 Top-K 检索。
    无子命令运行脚本时默认进入本模式。
    """
    load_dotenv()
    text_dir = Path(args.text_dir) if args.text_dir else _DISNEY_FAISS_DIR
    image_dir = Path(args.image_dir) if args.image_dir else _DISNEY_IMAGE_FAISS_DIR
    k = int(args.top_k)
    text_index_name = args.text_index_name or "index"
    image_index_name = args.image_index_name or "images.index"

    key = (args.api_key or "").strip() or _resolve_dashscope_api_key()
    embedder: Embeddings | None
    if key:
        embedder = get_dashscope_embeddings(key, model=args.embedding_model or _EMBEDDING_MODEL)
    else:
        embedder = None
        print(
            "[ask] 未检测到 DashScope API Key，将跳过文本向量检索，仅执行图片 CLIP 检索。"
        )

    def _one_round(question: str) -> None:
        q = question.strip()
        if not q:
            return
        text_out, image_out = asyncio.run(
            _retrieve_dual_async(
                q,
                embedder=embedder,
                text_dir=text_dir,
                image_dir=image_dir,
                top_k=k,
                text_index_name=text_index_name,
                image_index_name=image_index_name,
                require_text_key=False,
            )
        )
        _print_retrieve_results(text_out, image_out, q=q, k=k, tag="ask")

    preset = (getattr(args, "query", None) or "").strip()
    if preset:
        _one_round(preset)
        return

    print(
        "[ask] 输入问题后回车检索（文本+图片 Top-K）；空行忽略；输入 quit / exit / q 结束。"
    )
    while True:
        try:
            line = input("问题> ")
        except EOFError:
            print("\n[ask] EOF，结束。")
            break
        if line.strip().lower() in ("quit", "exit", "q"):
            print("[ask] 再见。")
            break
        _one_round(line)


def _cmd_build_all(args: argparse.Namespace) -> None:
    """默认构建：文本 + 图片。"""
    # 并发执行：两边都跑完后再汇总；任一失败不阻断另一方。
    async def _run() -> tuple[dict[str, Any], dict[str, Any]]:
        async def _wrap(name: str, fn) -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                await asyncio.to_thread(fn, args)
                return {"ok": True, "task": name, "ms": (time.perf_counter() - t0) * 1000.0}
            except Exception as e:
                return {
                    "ok": False,
                    "task": name,
                    "ms": (time.perf_counter() - t0) * 1000.0,
                    "error": str(e),
                }

        return await asyncio.gather(
            _wrap("build-text", _cmd_build_text),
            _wrap("build-images", _cmd_build_images),
        )

    text_res, img_res = asyncio.run(_run())
    print("[build-all] 汇总：")
    print(f"- text: {text_res}")
    print(f"- image: {img_res}")


def _default_build_all_namespace() -> argparse.Namespace:
    """
    未写子命令时 parse_args 不会带子解析器参数，补全与 build-all 相同的默认值，
    避免 AttributeError（如缺少 verbose / chunk_size）。
    """
    return argparse.Namespace(
        verbose=False,
        department=DEFAULT_WORD_CHUNK_METADATA_DEPARTMENT,
        update_time=DEFAULT_WORD_CHUNK_METADATA_UPDATE_TIME,
        out_dir="",
        api_key="",
        embedding_model=_EMBEDDING_MODEL,
        chunk_size=500,
        chunk_overlap=80,
        index_name="images.index",
        quiet_images=False,
    )


def _default_ask_namespace() -> argparse.Namespace:
    """无子命令时进入 ask 交互模式所需的默认参数。"""
    return argparse.Namespace(
        query="",
        top_k=10,
        text_dir="",
        image_dir="",
        text_index_name="index",
        image_index_name="images.index",
        api_key="",
        embedding_model=_EMBEDDING_MODEL,
    )


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
        _idx, _meta = get_or_create_image_faiss_cache(
            image_paths,
            img_dir,
            index_name="images.index",
            use_cache=True,
            metadata_department=DEFAULT_WORD_CHUNK_METADATA_DEPARTMENT,
            metadata_update_time=DEFAULT_WORD_CHUNK_METADATA_UPDATE_TIME,
            quiet_images=False,
        )
        print(
            f"[build_knowledge_base] 图片 FAISS 已就绪：{img_dir}，图片数 {_idx.ntotal}。"
        )
    return metadata_store, vs, image_vectors


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "迪士尼知识库：默认 ask 交互双路检索；子命令可构建向量库、inspect-local、retrieve 等（不调对话大模型）。"
        ),
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
    p_img.add_argument(
        "--quiet-images",
        action="store_true",
        help="关闭 [图片向量] 逐张编码与落盘日志",
    )
    p_img.add_argument(
        "--no-cache",
        action="store_true",
        help="不使用本地图片向量缓存，强制重新构建并覆盖落盘",
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
    p_all.add_argument(
        "--quiet-images",
        action="store_true",
        help="关闭 [图片向量] 逐张编码与落盘日志",
    )
    p_all.add_argument(
        "--no-cache",
        action="store_true",
        help="不使用本地图片向量缓存，强制重新构建并覆盖落盘",
    )
    p_all.set_defaults(handler=_cmd_build_all)

    p_inspect = sub.add_parser(
        "inspect-local",
        help="并发读取本地 text/image 向量库并汇总（不重建，不调用 embeddings API）",
    )
    p_inspect.add_argument(
        "--text-dir",
        type=str,
        default="",
        metavar="PATH",
        help="文本向量库目录（默认 vectorModels/迪士尼RAG知识库）",
    )
    p_inspect.add_argument(
        "--image-dir",
        type=str,
        default="",
        metavar="PATH",
        help="图片向量库目录（默认 vectorModels/迪士尼RAG知识库/images）",
    )
    p_inspect.add_argument(
        "--text-index-name",
        type=str,
        default="index",
        help="文本 index_name 前缀（默认 index）",
    )
    p_inspect.add_argument(
        "--image-index-name",
        type=str,
        default="images.index",
        help="图片 index_name 前缀（默认 images.index）",
    )
    p_inspect.set_defaults(handler=_cmd_inspect_local)

    p_retrieve = sub.add_parser(
        "retrieve",
        help="对用户问句并发检索：文本向量 Top-K + CLIP 图片向量 Top-K（不调对话大模型）",
    )
    p_retrieve.add_argument("query", type=str, help="用户问题（自然语言）")
    p_retrieve.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="文本与图片各自返回的 Top-K（默认 10）",
    )
    p_retrieve.add_argument(
        "--text-dir",
        type=str,
        default="",
        metavar="PATH",
        help="文本向量库目录（默认 vectorModels/迪士尼RAG知识库）",
    )
    p_retrieve.add_argument(
        "--image-dir",
        type=str,
        default="",
        metavar="PATH",
        help="图片向量库目录（默认 vectorModels/迪士尼RAG知识库/images）",
    )
    p_retrieve.add_argument(
        "--text-index-name",
        type=str,
        default="index",
        help="文本 index 前缀（默认 index）",
    )
    p_retrieve.add_argument(
        "--image-index-name",
        type=str,
        default="images.index",
        help="图片 index 前缀（默认 images.index）",
    )
    p_retrieve.add_argument(
        "--api-key",
        type=str,
        default="",
        help="DashScope API Key（文本 query 向量化用；也可用环境变量）",
    )
    p_retrieve.add_argument(
        "--embedding-model",
        type=str,
        default=_EMBEDDING_MODEL,
        help=f"文本 embedding 模型，须与建库一致（默认 {_EMBEDDING_MODEL!r}）",
    )
    p_retrieve.set_defaults(handler=_cmd_retrieve)

    p_ask = sub.add_parser(
        "ask",
        help="交互式双路检索：循环输入问题（也可单次传入 query）；默认无子命令时进入本模式",
    )
    p_ask.add_argument(
        "query",
        nargs="?",
        default="",
        help="可选：单次提问；省略则进入交互输入",
    )
    p_ask.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="文本与图片各自返回的 Top-K（默认 10）",
    )
    p_ask.add_argument(
        "--text-dir",
        type=str,
        default="",
        metavar="PATH",
        help="文本向量库目录（默认 vectorModels/迪士尼RAG知识库）",
    )
    p_ask.add_argument(
        "--image-dir",
        type=str,
        default="",
        metavar="PATH",
        help="图片向量库目录（默认 vectorModels/迪士尼RAG知识库/images）",
    )
    p_ask.add_argument(
        "--text-index-name",
        type=str,
        default="index",
        help="文本 index 前缀（默认 index）",
    )
    p_ask.add_argument(
        "--image-index-name",
        type=str,
        default="images.index",
        help="图片 index 前缀（默认 images.index）",
    )
    p_ask.add_argument(
        "--api-key",
        type=str,
        default="",
        help="DashScope API Key（文本检索用；无则仅跑图片 CLIP 检索）",
    )
    p_ask.add_argument(
        "--embedding-model",
        type=str,
        default=_EMBEDDING_MODEL,
        help=f"文本 embedding 模型（默认 {_EMBEDDING_MODEL!r}）",
    )
    p_ask.set_defaults(handler=_cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    if getattr(args, "handler", None) is not None:
        args.handler(args)
        return
    # 未指定子命令：默认进入 ask 交互检索（不调大模型）
    _cmd_ask(_default_ask_namespace())


if __name__ == "__main__":
    main()

    # for image_path in classified["images"]:
    #     img_text_info = image_to_text(str(_DISNEY_KNOWLEDGE_BASE / image_path))

    #     print(img_text_info)
