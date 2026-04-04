"""
FAISS × LangChain 知识点演示（本地可运行，无需真实 Embedding API）

涵盖内容：
1. 原生 faiss 模块 vs LangChain 的 FAISS 类（后者不是 faiss 命名空间，没有 IndexFlatIP）
2. LangChain 建库：from_texts（默认 L2）与 MAX_INNER_PRODUCT（内积索引）
3. 检索：similarity_search 无分数；similarity_search_with_score 的分数含义随索引而变
4. （可选）手动 new 空索引再 add_texts，与官方文档一致

运行：在项目根目录执行
  python data/课程练习/RAG技术与应用/faiss_langchain_demo.py
"""

from __future__ import annotations

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import DeterministicFakeEmbedding

# 固定维度的小玩具嵌入：同一文本每次向量相同，便于复现排序
EMBED_DIM = 32
embeddings = DeterministicFakeEmbedding(size=EMBED_DIM)

SAMPLE_TEXTS = [
    "迪士尼乐园门票退换政策",
    "上海迪士尼度假区餐饮指南",
    "星黛露周边商品介绍",
]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_raw_faiss_vs_langchain_class() -> None:
    """原生 faiss 建索引；LangChain 的 FAISS 类不能当 faiss 用。"""
    section("1. 原生 faiss vs LangChain 的 FAISS 类")

    # 正确：Meta faiss 包提供的类型
    idx_l2 = faiss.IndexFlatL2(EMBED_DIM)
    idx_ip = faiss.IndexFlatIP(EMBED_DIM)
    print("原生 faiss.IndexFlatL2:", type(idx_l2))
    print("原生 faiss.IndexFlatIP:", type(idx_ip))

    # 错误：LangChain FAISS 是「向量库封装类」，没有 IndexFlatIP
    try:
        _ = FAISS.IndexFlatIP(EMBED_DIM)  # type: ignore[attr-defined]
    except AttributeError as e:
        print("\n错误写法 FAISS.IndexFlatIP(...) →", e)
    print(
        "\n说明：需要底层索引时，应 `import faiss` 后使用 faiss.IndexFlat*，"
        "或让 FAISS.from_texts / 构造函数内部去创建。"
    )


def demo_from_texts_l2_default() -> None:
    """LangChain 默认 EUCLIDEAN → 内部 IndexFlatL2；分数为距离，越小越近。"""
    section("2. FAISS.from_texts（默认欧氏 / IndexFlatL2）")

    store = FAISS.from_texts(SAMPLE_TEXTS, embeddings)
    query = "门票可以退吗"
    print("查询:", repr(query))
    print("\n--- similarity_search（仅文档，无分数）---")
    for i, doc in enumerate(store.similarity_search(query, k=2), 1):
        print(f"  {i}. {doc.page_content}")

    print("\n--- similarity_search_with_score（第二项为 faiss L2 距离，越小越相似）---")
    for doc, score in store.similarity_search_with_score(query, k=3):
        print(f"  L2距离={score:.4f} | {doc.page_content}")


def demo_from_texts_inner_product() -> None:
    """MAX_INNER_PRODUCT → IndexFlatIP；分数为内积，越大越相似（向量常配合 L2 归一化当 cosine）。"""
    section("3. FAISS.from_texts（内积 / IndexFlatIP）")

    # 不设 normalize_L2：避免 LangChain 对「非欧氏 + normalize_L2」的 UserWarning；
    # 生产上若要用内积近似余弦，通常要求嵌入已是单位向量，或查阅版本文档是否接受 normalize_L2=True（仍会调用 faiss.normalize_L2，仅多一条告警）。
    store = FAISS.from_texts(
        SAMPLE_TEXTS,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    query = "门票可以退吗"
    print("查询:", repr(query))
    print("\n--- similarity_search_with_score（内积，越大越相似）---")
    for doc, score in store.similarity_search_with_score(query, k=3):
        print(f"  内积={score:.4f} | {doc.page_content}")
    print(
        "\n说明：玩具嵌入下 L2 与内积的 Top-k 顺序可能不同；线上应使用真实嵌入，并保持「距离定义」与模型训练/归一化方式一致。"
    )


def demo_manual_empty_then_add() -> None:
    """与 LangChain 文档一致：先 faiss.IndexFlat*，再 FAISS(...) + add_texts。"""
    section("4. 手动创建空索引再 add_texts")

    index = faiss.IndexFlatL2(EMBED_DIM)
    from langchain_community.docstore.in_memory import InMemoryDocstore

    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    ids = store.add_texts(SAMPLE_TEXTS)
    print("写入 id 列表:", ids)
    hits = store.similarity_search("迪士尼餐饮", k=2)
    for doc in hits:
        print("  ", doc.page_content)


def main() -> None:
    demo_raw_faiss_vs_langchain_class()
    demo_from_texts_l2_default()
    demo_from_texts_inner_product()
    demo_manual_empty_then_add()

    section("小结")
    print(
        "- 默认：LangChain FAISS 使用 IndexFlatL2，similarity_search_with_score 返回 L2 距离（越小越好）。\n"
        "- 指定 MAX_INNER_PRODUCT：使用 IndexFlatIP，分数为内积（越大越好）；与余弦的关系取决于向量是否已 L2 归一化。\n"
        "- 业务项目若已用 LangChain RAG，优先用 FAISS.from_texts / load_local；仅当要强控索引类型时再手写 faiss.Index*。"
    )


if __name__ == "__main__":
    main()
