# FAISS IndexFlatIP 与「向量 id → 元数据」的一体化封装：保存 / 加载 / 检索。
# 元数据以 JSON 存盘（与 embedding_API_FAISS 的 *.meta.json 结构兼容）。
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

_DEFAULT_FORMAT_VERSION = 1


def _as_float32_matrix(vectors: np.ndarray | list) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"期望形状 (n, dim)，得到 {m.shape}")
    return m


class FaissVectorStore:
    """
    同时持有 FAISS 索引与本地元数据：``items[i]`` 与 FAISS 向量 id ``i`` 一一对应（字典语义：
    ``id_to_meta[i] == items[i]``）。
    """

    def __init__(self, index: faiss.Index, meta: dict[str, Any]) -> None:
        self._index = index
        self._meta = dict(meta)
        items = self._meta.get("items") or []
        n = int(index.ntotal)
        if len(items) != n:
            raise ValueError(
                f"元数据条数 {len(items)} 与索引向量数 {n} 不一致（要求按行对齐）"
            )

    @property
    def index(self) -> faiss.Index:
        return self._index

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def items(self) -> list[dict[str, Any]]:
        return list(self._meta.get("items") or [])

    @property
    def dim(self) -> int:
        return int(self._index.d)

    @property
    def embedding_model(self) -> str:
        return str(self._meta.get("embedding_model", ""))

    def __len__(self) -> int:
        return int(self._index.ntotal)

    def metadata_for_id(self, faiss_id: int) -> dict[str, Any]:
        """FAISS 返回的向量 id → 对应元数据行。"""
        if faiss_id < 0 or faiss_id >= len(self.items):
            raise IndexError(f"faiss_id 越界: {faiss_id}")
        return dict(self.items[faiss_id])

    def id_to_meta(self) -> dict[int, dict[str, Any]]:
        """显式构造 id → 元数据 字典（数据大时占用更多内存，仅按需使用）。"""
        return {i: dict(row) for i, row in enumerate(self.items)}

    @classmethod
    def from_embeddings(
        cls,
        vectors: np.ndarray | list,
        items: list[dict[str, Any]],
        *,
        normalize_l2: bool = True,
        embedding_model: str = "",
        corpus_folder: str = "",
        source_dir: str = "",
        format_version: int = _DEFAULT_FORMAT_VERSION,
    ) -> FaissVectorStore:
        """
        用 (n, dim) 向量矩阵与 n 条元数据构建 ``IndexFlatIP``（与练习中「归一化 + 内积 ≈ 余弦」一致）。
        """
        mat = _as_float32_matrix(vectors)
        if mat.shape[0] != len(items):
            raise ValueError(
                f"向量行数 {mat.shape[0]} 与 items 长度 {len(items)} 不一致"
            )
        if normalize_l2:
            faiss.normalize_L2(mat)
        dim = int(mat.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        meta = {
            "format_version": format_version,
            "embedding_model": embedding_model,
            "corpus_folder": corpus_folder,
            "source_dir": source_dir,
            "dim": dim,
            "count": len(items),
            "items": items,
        }
        return cls(index, meta)

    def save(
        self,
        faiss_path: str | Path,
        meta_path: str | Path | None = None,
    ) -> tuple[Path, Path]:
        """
        写入 ``*.faiss`` 与 ``*.meta.json``。
        若未指定 meta_path，则使用 ``<stem>.meta.json``（与 ``foo.faiss`` → ``foo.meta.json`` 规则一致）。
        """
        fp = Path(faiss_path)
        mp = Path(meta_path) if meta_path is not None else fp.with_name(f"{fp.stem}.meta.json")
        fp.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(fp))
        # 落盘时写回 count / dim，与索引一致
        payload = {
            **self._meta,
            "dim": self.dim,
            "count": len(self),
        }
        mp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return fp.resolve(), mp.resolve()

    @classmethod
    def load(
        cls,
        faiss_path: str | Path,
        meta_path: str | Path | None = None,
    ) -> FaissVectorStore:
        fp = Path(faiss_path)
        if not fp.is_file():
            raise FileNotFoundError(f"FAISS 索引不存在: {fp}")
        mp = Path(meta_path) if meta_path is not None else fp.with_name(f"{fp.stem}.meta.json")
        if not mp.is_file():
            raise FileNotFoundError(f"元数据 JSON 不存在: {mp}")
        index = faiss.read_index(str(fp))
        meta = json.loads(mp.read_text(encoding="utf-8"))
        return cls(index, meta)

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        topk: int = 10,
        *,
        normalize_query: bool = True,
    ) -> list[dict[str, Any]]:
        """
        对查询向量做 TopK 检索（IndexFlatIP + 可选 L2 归一化，score 为内积≈余弦）。
        每条结果含 ``faiss_id``、``score`` 及元数据字段。
        """
        n = len(self)
        if n == 0:
            return []
        qv = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if qv.shape[1] != self.dim:
            raise ValueError(f"查询维度 {qv.shape[1]} 与索引维度 {self.dim} 不一致")
        if normalize_query:
            faiss.normalize_L2(qv)
        k = min(topk, n)
        scores, ids = self._index.search(qv, k)
        row_scores = scores[0].tolist()
        row_ids = ids[0].tolist()
        out: list[dict[str, Any]] = []
        for fid, s in zip(row_ids, row_scores, strict=True):
            if fid < 0:
                continue
            row = self.metadata_for_id(fid)
            out.append(
                {
                    "faiss_id": fid,
                    "score": float(s),
                    **row,
                }
            )
        return out

    @classmethod
    def from_practice_corpus(cls, corpus_folder: str = "three_kingdoms") -> FaissVectorStore:
        """加载课程练习默认目录下由 ``embedding_API_FAISS`` 生成的索引与 meta。"""
        import embedding_API_FAISS as ef

        ef.mirror_legacy_index_as_faiss(corpus_folder)
        ip = ef.resolve_faiss_index_path(corpus_folder)
        mp = ef.default_faiss_meta_path(corpus_folder)
        if ip is None or not mp.is_file():
            raise FileNotFoundError(
                f"未找到语料 {corpus_folder!r} 的 FAISS 产物（索引或 meta.json）"
            )
        return cls.load(ip, mp)
