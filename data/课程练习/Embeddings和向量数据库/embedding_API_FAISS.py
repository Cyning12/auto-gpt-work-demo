# 2026-3-25 延伸：将 DashScope 文本向量写入 FAISS（IndexFlatIP + L2 归一化 ≈ 余弦相似度），
# 元数据（文件名、chunk 序号、snippet）单独存 JSON；向量索引默认落盘为 *.faiss（旧版 *.index 仍可读取）。
#
# 依赖：faiss-cpu（见 requirements.txt）。语料切块与批量编码逻辑复用 embedding_API_pickle。
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import faiss
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import embedding_API_pickle as ap

# 课程练习根目录（与 pickle 模块一致）
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent

_FAISS_SUFFIX = "_API_FAISS"
_META_VERSION = 1


def default_faiss_index_path(corpus_folder: str) -> Path:
    """本地 FAISS 二进制索引（标准扩展名 .faiss）。"""
    return (
        _PRACTICE_ROOT
        / "vectorModels"
        / corpus_folder
        / f"{corpus_folder}{_FAISS_SUFFIX}.faiss"
    )


def legacy_faiss_index_path(corpus_folder: str) -> Path:
    """历史文件名 faiss.write_index 常用 .index，仅用于兼容已存在文件。"""
    return (
        _PRACTICE_ROOT
        / "vectorModels"
        / corpus_folder
        / f"{corpus_folder}{_FAISS_SUFFIX}.index"
    )


def resolve_faiss_index_path(corpus_folder: str) -> Path | None:
    """优先 .faiss，否则回退到旧版 .index。"""
    p = default_faiss_index_path(corpus_folder)
    if p.is_file():
        return p
    leg = legacy_faiss_index_path(corpus_folder)
    if leg.is_file():
        return leg
    return None


def mirror_legacy_index_as_faiss(corpus_folder: str) -> Path | None:
    """
    若仅有历史 *.index、尚无 *.faiss，则复制一份为 *.faiss（内容相同，便于统一扩展名）。
    返回新路径；无需复制时返回 None。
    """
    dst = default_faiss_index_path(corpus_folder)
    if dst.is_file():
        return None
    src = legacy_faiss_index_path(corpus_folder)
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def default_faiss_meta_path(corpus_folder: str) -> Path:
    return (
        _PRACTICE_ROOT
        / "vectorModels"
        / corpus_folder
        / f"{corpus_folder}{_FAISS_SUFFIX}.meta.json"
    )


def _faiss_bundle_exists(corpus_folder: str) -> bool:
    return (
        resolve_faiss_index_path(corpus_folder) is not None
        and default_faiss_meta_path(corpus_folder).is_file()
    )


def build_faiss_from_pickle_store(
    corpus_folder: str = "three_kingdoms",
    *,
    pickle_path: Path | None = None,
) -> tuple[Path, Path]:
    """
    从已有的 embedding_API_pickle 产物（*.model）读取向量与条目，写入 FAISS + meta.json，
    不调用 Embedding API，适合已跑通 pickle 建库后只做索引格式迁移。
    """
    p = (
        pickle_path
        if pickle_path is not None
        else ap.default_api_model_path(corpus_folder)
    )
    if not p.is_file():
        raise FileNotFoundError(f"未找到 pickle 向量库: {p}")

    store = ap.load_api_embedding_store(p)
    items_full = store.get("items") or []
    if not items_full:
        raise ValueError(f"pickle 中无 items: {p}")

    vectors = [it["embedding"] for it in items_full]
    mat = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(mat)
    dim = int(mat.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    meta_items = [
        {
            "file": it.get("file", ""),
            "chunk_index": it.get("chunk_index", 0),
            "snippet": it.get("snippet", ""),
        }
        for it in items_full
    ]
    payload = {
        "format_version": _META_VERSION,
        "embedding_model": store.get("embedding_model", ap.EMBEDDING_MODEL),
        "corpus_folder": corpus_folder,
        "source_dir": store.get("source_dir", ""),
        "dim": dim,
        "count": len(meta_items),
        "items": meta_items,
    }

    index_path = default_faiss_index_path(corpus_folder)
    meta_path = default_faiss_meta_path(corpus_folder)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index_path.resolve(), meta_path.resolve()


def ensure_faiss_bundle(
    corpus_folder: str = "three_kingdoms",
    *,
    corpus_dir: Path | None = None,
    encoding: str = "utf-8",
    skip_if_exists: bool = True,
    prefer_pickle_bootstrap: bool = True,
) -> tuple[Path, Path]:
    """
    若 FAISS 产物已存在则跳过。
    否则若存在 pickle 向量库则从中迁移（不调 API）；否则走完整语料 + API 建库。
    """
    if skip_if_exists and _faiss_bundle_exists(corpus_folder):
        mirror_legacy_index_as_faiss(corpus_folder)
        idx = resolve_faiss_index_path(corpus_folder)
        assert idx is not None
        return idx.resolve(), default_faiss_meta_path(corpus_folder).resolve()
    if prefer_pickle_bootstrap and ap.default_api_model_path(corpus_folder).is_file():
        return build_faiss_from_pickle_store(corpus_folder)
    return build_faiss_embedding_index(
        corpus_folder,
        corpus_dir=corpus_dir,
        encoding=encoding,
        skip_if_exists=False,
    )


def build_faiss_embedding_index(
    corpus_folder: str = "three_kingdoms",
    *,
    corpus_dir: Path | None = None,
    encoding: str = "utf-8",
    skip_if_exists: bool = True,
) -> tuple[Path, Path]:
    """
    与 pickle 版相同的语料与切块，批量请求 Embedding 后：
    - 向量矩阵 L2 归一化，写入 faiss.IndexFlatIP（内积即余弦相似度）
    - 元数据写入 *.meta.json（不含向量）

    skip_if_exists=True 时，若 .faiss（或旧版 .index）与 .meta.json 均已存在则跳过 API 调用。
    返回 (index_path, meta_path)。
    """
    index_path = default_faiss_index_path(corpus_folder)
    meta_path = default_faiss_meta_path(corpus_folder)
    if skip_if_exists and _faiss_bundle_exists(corpus_folder):
        mirror_legacy_index_as_faiss(corpus_folder)
        idx = resolve_faiss_index_path(corpus_folder)
        assert idx is not None
        return idx.resolve(), meta_path.resolve()

    src = (
        corpus_dir
        if corpus_dir is not None
        else _PRACTICE_ROOT / "source" / corpus_folder
    )
    files = ap._iter_corpus_files(src.resolve())

    work_units: list[tuple[str, int, str]] = []
    for fp in files:
        text = fp.read_text(encoding=encoding, errors="replace")
        parts = ap._chunk_text(text)
        if not parts:
            continue
        rel = fp.name
        for idx, chunk in enumerate(parts):
            work_units.append((rel, idx, chunk))

    if not work_units:
        raise ValueError(f"目录中无可用文本: {src}")

    texts_only = [t for _, _, t in work_units]
    vectors = ap._embed_texts_batched(texts_only)
    if len(vectors) != len(work_units):
        raise RuntimeError(
            f"返回向量条数与请求不一致: got {len(vectors)}, expected {len(work_units)}"
        )

    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"期望二维向量矩阵，得到 shape={mat.shape}")
    faiss.normalize_L2(mat)

    dim = int(mat.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    meta_items: list[dict] = []
    for (fname, chunk_index, chunk), _vec in zip(work_units, vectors, strict=True):
        chunk_stripped = chunk.strip()
        snippet = (
            chunk_stripped[:240] + ("…" if len(chunk_stripped) > 240 else "")
            if chunk_stripped
            else ""
        )
        meta_items.append(
            {
                "file": fname,
                "chunk_index": chunk_index,
                "snippet": snippet,
            }
        )

    payload = {
        "format_version": _META_VERSION,
        "embedding_model": ap.EMBEDDING_MODEL,
        "corpus_folder": corpus_folder,
        "source_dir": str(src.resolve()),
        "dim": dim,
        "count": len(meta_items),
        "items": meta_items,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return index_path.resolve(), meta_path.resolve()


def load_faiss_bundle(
    corpus_folder: str = "three_kingdoms",
) -> tuple[faiss.Index, dict]:
    """读取 FAISS 索引与 meta JSON；meta['items'] 行序与 FAISS 向量 id 一致。"""
    mirror_legacy_index_as_faiss(corpus_folder)
    ip = resolve_faiss_index_path(corpus_folder)
    mp = default_faiss_meta_path(corpus_folder)
    if ip is None or not mp.is_file():
        raise FileNotFoundError(
            "缺少 FAISS 产物，请先 ensure_faiss_bundle / build_faiss_embedding_index：\n"
            f"  期望索引: {default_faiss_index_path(corpus_folder)} "
            f"或 {legacy_faiss_index_path(corpus_folder)}\n"
            f"  期望元数据: {mp}"
        )
    index = faiss.read_index(str(ip))
    meta = json.loads(mp.read_text(encoding="utf-8"))
    return index, meta


def similarity_topk_faiss(
    index: faiss.Index,
    meta: dict,
    query: str,
    *,
    topk: int = 10,
    query_vector: list[float] | None = None,
) -> list[dict]:
    """
    查询向量 L2 归一化后做 IndexFlatIP 搜索；score 为内积，等价于与库内向量的余弦相似度。
    """
    items = meta.get("items") or []
    n = len(items)
    if n == 0:
        return []

    q = query_vector if query_vector is not None else ap.embed_query_text(query)
    qv = np.asarray(ap._as_float_list(q), dtype=np.float32).reshape(1, -1)
    if qv.shape[1] != index.d:
        raise ValueError(f"查询维度 {qv.shape[1]} 与索引维度 {index.d} 不一致")
    faiss.normalize_L2(qv)

    k = min(topk, n)
    scores, ids = index.search(qv, k)
    row_scores = scores[0].tolist()
    row_ids = ids[0].tolist()

    out: list[dict] = []
    for i, s in zip(row_ids, row_scores, strict=True):
        if i < 0:  # FAISS 不足 k 时可能填 -1
            continue
        it = items[i]
        out.append(
            {
                "file": it.get("file", ""),
                "chunk_index": it.get("chunk_index", 0),
                "score": float(s),
                "snippet": it.get("snippet", ""),
            }
        )
    return out


def print_similarity_topk_faiss(
    index: faiss.Index,
    meta: dict,
    query: str,
    *,
    topk: int = 10,
    query_vector: list[float] | None = None,
    query_display: str | None = None,
) -> None:
    rows = similarity_topk_faiss(
        index, meta, query, topk=topk, query_vector=query_vector
    )
    label = query_display if query_display is not None else query
    print(f"查询: {label!r}")
    print(f"Top {len(rows)}（FAISS IndexFlatIP，归一化后等价余弦）:")
    for i, row in enumerate(rows, 1):
        snip = row.get("snippet") or ""
        if snip:
            snip = snip.replace("\n", " ")
        print(
            f"  {i:2}. score={row['score']:.4f}\t{row['file']!r} "
            f"chunk={row['chunk_index']}" + (f"\n      {snip}" if snip else "")
        )


if __name__ == "__main__":
    corpus = "three_kingdoms"
    reused_before = _faiss_bundle_exists(corpus)
    had_pickle = ap.default_api_model_path(corpus).is_file()
    ip, mp = ensure_faiss_bundle(
        corpus, skip_if_exists=True, prefer_pickle_bootstrap=True
    )
    print(f"FAISS 索引: {ip}")
    print(f"元数据:     {mp}")
    if reused_before:
        print("已复用本地 FAISS（未重新请求语料编码）")
    elif had_pickle:
        print("已写入 FAISS（从 pickle 迁移，未调 Embedding API）")
    else:
        print("已通过 API 编码语料并写入 FAISS + meta.json")

    index, meta = load_faiss_bundle(corpus)
    print(
        f"向量数: {meta.get('count')}, 维度: {meta.get('dim')}, "
        f"模型: {meta.get('embedding_model')}"
    )

    demo_query = ap._DEMO_QUERY_CHANGBAN_ADOU
    print()
    try:
        print_similarity_topk_faiss(index, meta, demo_query, topk=10)
    except RuntimeError as e:
        print(e)
        if meta.get("items"):
            print(
                "\n（离线兜底）用索引中第 1 条向量作查询（需从索引重构向量，"
                "Flat 索引可直接 reconstruct）："
            )
            try:
                qv = index.reconstruct(0).astype(np.float32).reshape(1, -1)
                faiss.normalize_L2(qv)
                # similarity_topk_faiss 期望 list，取第一行
                qlist = qv.reshape(-1).tolist()
                print_similarity_topk_faiss(
                    index,
                    meta,
                    "",
                    topk=10,
                    query_vector=qlist,
                    query_display="<<离线：reconstruct(0) 归一化后检索>>",
                )
            except Exception as ex:  # noqa: BLE001
                print(f"reconstruct 兜底失败: {ex}")
