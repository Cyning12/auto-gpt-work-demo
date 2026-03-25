# 2026-3-25 embedding打卡任务一:
# 使用 DashScope TextEmbedding 对指定文件夹（默认 source/{语料文件夹}/）下所有文件逐段编码，
# 将向量表序列化保存到 vectorModels/{语料名}/{语料名}_API.model（与 Word2Vec 产物目录一致）。
#
# 说明：云端 embedding 为「文本段 → 向量」，与 Word2Vec 的「词 → 向量」不同；单文件过长时会按行切块，
# 每条记录含 file、chunk_index、embedding。
from __future__ import annotations

import math
import os
import pickle
import time
from http import HTTPStatus
from pathlib import Path

import dashscope
import requests
from dotenv import load_dotenv

# 课程练习根目录（…/data/课程练习）
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent

_here = Path(__file__).resolve().parent
load_dotenv(_here / ".env")
load_dotenv()
_api_key = (
    os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
).strip()
dashscope.api_key = _api_key or None

EMBEDDING_MODEL = "text-embedding-v3"
# 单次请求条数（避免触达平台上限）
_BATCH_SIZE = 10
# 单条文本字符上限（过长则按行/硬切分块）
_MAX_CHARS_PER_CHUNK = 500
# 批次间轻微间隔，降低限流概率
_BATCH_SLEEP_SEC = 0.05
# 单批请求遇 SSL/断连时的重试：指数退避（秒）
_EMBED_RETRY_DELAYS_SEC = (1.0, 2.0, 4.0, 8.0, 16.0)


def _ensure_api_key() -> None:
    if not dashscope.api_key:
        raise RuntimeError(
            "未配置 API Key：请在环境变量或本目录 .env 中设置 BAILIAN_API_KEY 或 DASHSCOPE_API_KEY"
        )


def _iter_corpus_files(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"语料目录不存在: {corpus_dir}")
    files: list[Path] = []
    for entry in sorted(corpus_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            continue
        if entry.is_file():
            files.append(entry)
    return files


def _chunk_text(text: str, max_chars: int = _MAX_CHARS_PER_CHUNK) -> list[str]:
    """按行累加切块；超长单行再按 max_chars 硬切。"""
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        add = len(line) + (1 if buf else 0)
        if size + add > max_chars and buf:
            chunks.append("\n".join(buf))
            buf = [line]
            size = len(line)
        else:
            size += add
            buf.append(line)
    if buf:
        chunks.append("\n".join(buf))

    out: list[str] = []
    for piece in chunks:
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            for i in range(0, len(piece), max_chars):
                slice_ = piece[i : i + max_chars].strip()
                if slice_:
                    out.append(slice_)
    return out


def _embedding_error_hint(resp) -> str:
    """根据 DashScope 返回的 code 附加简短中文说明。"""
    code = (
        resp.get("code") if isinstance(resp, dict) else getattr(resp, "code", "")
    ) or ""
    if "FreeTierOnly" in str(code) or "AllocationQuota" in str(code):
        return (
            "\n  → 免费额度已用尽：请在阿里云 DashScope/百炼控制台关闭「仅使用免费额度」"
            "或开通按量付费后再调用；仅加载本地 *.model 不会消耗额度，但对「新句子」做相似度仍需一次 Embedding。"
        )
    return ""


def _embed_batch(texts: list[str]) -> list[list[float]]:
    _ensure_api_key()
    if not texts:
        return []
    transient = (
        requests.exceptions.SSLError,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    )
    last_net_err: BaseException | None = None
    for attempt in range(len(_EMBED_RETRY_DELAYS_SEC) + 1):
        try:
            resp = dashscope.TextEmbedding.call(model=EMBEDDING_MODEL, input=texts)
        except transient as e:
            last_net_err = e
            if attempt < len(_EMBED_RETRY_DELAYS_SEC):
                time.sleep(_EMBED_RETRY_DELAYS_SEC[attempt])
                continue
            raise RuntimeError(
                "调用 DashScope TextEmbedding 时 HTTPS/SSL 或连接多次失败（例如 "
                "SSLEOFError / UNEXPECTED_EOF_WHILE_READING）。常见原因：网络抖动、VPN/代理、"
                "公司防火墙或中间人证书。可换网络/关代理后重试；若已有本地 "
                f"{_PRACTICE_ROOT.name}/vectorModels/.../*_API.model，"
                "请使用 build_api_embedding_store(..., skip_if_exists=True) 跳过重新编码。"
            ) from last_net_err
        if resp.status_code != HTTPStatus.OK:
            base = (
                f"TextEmbedding 失败: status={resp.status_code} "
                f"code={resp.code!r} message={resp.message!r}"
            )
            raise RuntimeError(base + _embedding_error_hint(resp))
        output = resp.output or {}
        raw = output.get("embeddings") or []
        ordered = sorted(raw, key=lambda x: x.get("text_index", 0))
        return [item["embedding"] for item in ordered]


def _embed_texts_batched(all_texts: list[str]) -> list[list[float]]:
    result: list[list[float]] = []
    for i in range(0, len(all_texts), _BATCH_SIZE):
        batch = all_texts[i : i + _BATCH_SIZE]
        result.extend(_embed_batch(batch))
        if i + _BATCH_SIZE < len(all_texts):
            time.sleep(_BATCH_SLEEP_SEC)
    return result


def default_api_model_path(corpus_folder: str) -> Path:
    """vectorModels/{corpus_folder}/{corpus_folder}_API.model"""
    return (
        _PRACTICE_ROOT / "vectorModels" / corpus_folder / f"{corpus_folder}_API.model"
    )


def build_api_embedding_store(
    corpus_folder: str = "three_kingdoms",
    *,
    corpus_dir: Path | None = None,
    encoding: str = "utf-8",
    skip_if_exists: bool = True,
) -> Path:
    """
    读取 corpus_dir 下所有普通文件（不递归）；默认 corpus_dir = source/{corpus_folder}/。
    返回保存路径 vectorModels/{corpus_folder}/{corpus_folder}_API.model。

    skip_if_exists=True 时，若目标 .model 已存在则不再请求 API，直接返回该路径。
    """
    out_dir = _PRACTICE_ROOT / "vectorModels" / corpus_folder
    out_path = out_dir / f"{corpus_folder}_API.model"
    if skip_if_exists and out_path.is_file():
        return out_path.resolve()

    src = (
        corpus_dir
        if corpus_dir is not None
        else _PRACTICE_ROOT / "source" / corpus_folder
    )
    files = _iter_corpus_files(src.resolve())

    work_units: list[tuple[str, int, str]] = []
    for fp in files:
        text = fp.read_text(encoding=encoding, errors="replace")
        parts = _chunk_text(text)
        if not parts:
            continue
        rel = fp.name
        for idx, chunk in enumerate(parts):
            work_units.append((rel, idx, chunk))

    if not work_units:
        raise ValueError(f"目录中无可用文本: {src}")

    texts_only = [t for _, _, t in work_units]
    vectors = _embed_texts_batched(texts_only)
    if len(vectors) != len(work_units):
        raise RuntimeError(
            f"返回向量条数与请求不一致: got {len(vectors)}, expected {len(work_units)}"
        )

    items: list[dict] = []
    for (fname, chunk_index, chunk), vec in zip(work_units, vectors, strict=True):
        chunk_stripped = chunk.strip()
        snippet = (
            chunk_stripped[:240] + ("…" if len(chunk_stripped) > 240 else "")
            if chunk_stripped
            else ""
        )
        items.append(
            {
                "file": fname,
                "chunk_index": chunk_index,
                "embedding": vec,
                "snippet": snippet,
            }
        )

    store = {
        "format_version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "corpus_folder": corpus_folder,
        "source_dir": str(src.resolve()),
        "items": items,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path.resolve()


def load_api_embedding_store(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def embed_query_text(text: str) -> list[float]:
    """对单条查询语句请求与语料相同的 embedding 模型，返回一条向量。"""
    vecs = _embed_batch([text.strip()])
    if not vecs:
        raise RuntimeError("查询向量化结果为空")
    return vecs[0]


def _as_float_list(vec) -> list[float]:
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a = _as_float_list(a)
    b = _as_float_list(b)
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def similarity_topk(
    store: dict,
    query: str,
    *,
    topk: int = 10,
    query_vector: list[float] | None = None,
) -> list[dict]:
    """
    用余弦相似度在 store['items'] 里检索与 query 最相近的 topk 条。
    默认会调用 API 对 query 编码；若已持有 query_vector（与 items 同维度），可传入以省一次请求。
    返回每项: file, chunk_index, score, snippet（旧版 model 可能无 snippet）。
    """
    items = store.get("items") or []
    if not items:
        return []
    qv = query_vector if query_vector is not None else embed_query_text(query)
    scored: list[tuple[float, dict]] = []
    qv = _as_float_list(qv)
    for it in items:
        vec = it.get("embedding")
        if not vec:
            continue
        s = _cosine_similarity(qv, vec)
        scored.append(
            (
                s,
                {
                    "file": it.get("file", ""),
                    "chunk_index": it.get("chunk_index", 0),
                    "score": s,
                    "snippet": it.get("snippet", ""),
                },
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:topk]]


# 与 embedding_Word2Vec 中预留查询一致，便于两种 embedding 对照（不在 __main__ 中调用）。
_DEMO_QUERY_CHANGBAN_ADOU = "谁在桃园结义了？"


def demo_query_changban_adou_similarity(
    *,
    corpus_folder: str = "three_kingdoms",
    store: dict | None = None,
    topk: int = 10,
) -> list[dict]:
    """
    预留：对本地 API 向量表检索该问句的 TopK 语料块（默认会调用一次查询向量化 API）。
    未在 __main__ 中调用；问句与 embedding_Word2Vec.demo_query_changban_adou_similarity 相同。
    """
    if store is None:
        store = load_api_embedding_store(default_api_model_path(corpus_folder))
    return similarity_topk(store, _DEMO_QUERY_CHANGBAN_ADOU, topk=topk)


def print_similarity_topk(
    store: dict,
    query: str,
    *,
    topk: int = 10,
    query_vector: list[float] | None = None,
    query_display: str | None = None,
) -> None:
    """
    控制台打印余弦相似度最高的 topk 条（测试用）。
    若提供 query_vector，则不再请求 API；query_display 仅用于展示说明。
    """
    rows = similarity_topk(store, query, topk=topk, query_vector=query_vector)
    label = query_display if query_display is not None else query
    print(f"查询: {label!r}")
    print(f"Top {len(rows)}（余弦相似度）:")
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
    model_path = default_api_model_path(corpus)
    reused = model_path.is_file()
    saved = build_api_embedding_store(corpus, skip_if_exists=True)
    print(f"API 向量表: {saved}")
    print(
        "已复用本地 model（未重新请求语料编码）" if reused else "已构建并写入本地 model"
    )
    data = load_api_embedding_store(saved)
    n = len(data.get("items", []))
    dim = len(data["items"][0]["embedding"]) if n else 0
    print(f"条目数: {n}, 维度: {dim}, 模型: {data.get('embedding_model')}")

    demo_query = _DEMO_QUERY_CHANGBAN_ADOU
    print()
    try:
        print_similarity_topk(data, demo_query, topk=10)
    except RuntimeError as e:
        print(e)
        items = data.get("items") or []
        if items:
            print(
                "\n（离线兜底）使用本地 model 中第 1 条语料向量作为「查询向量」，"
                "不调 API，仅演示与库内各块的相似度排序："
            )
            qv0 = items[0].get("embedding")
            if qv0:
                print_similarity_topk(
                    data,
                    "",
                    topk=10,
                    query_vector=_as_float_list(qv0),
                    query_display="<<离线：与第 1 条语料向量最相近的块>>",
                )
        else:
            print("本地 model 无条目，无法做离线演示。")
