# 2026-3-25 embedding打卡任务一:
# 三国演义embedding
# 使用Gensim中的Word2Vec对 source/{语料文件夹}/ 下文件进行 Word Embedding
# 分析和曹操最相近的词有哪些
# 曹操+刘备-张飞=?
#
# 流程概要（五步，满足「至少四步」）：
#   ---Step1: 枚举 source/{path}/ 下目标文件夹内所有语料文件。
#   ---Step2: 对每个文件按行 jieba 分词，空格连接；分词结果写入与 source 同级的 segmented/{path}/（每源文件对应 *_segmented.txt）。
#   ---Step3: 用 gensim.models.word2vec.PathLineSentences 读取该目录，再 Word2Vec 训练（一行一句、词已空格切分）。
#   ---Step4: 用 wv.most_similar 查询与「曹操」向量最邻近的词。
#   ---Step5: 向量类比 wv.most_similar(positive=[曹操,刘备], negative=[张飞])。
#
# 说明：PathLineSentences 会读取目录下「所有」可打开的文件，故分词结果放在 segmented/{path}/，
# 与 source/{path}/ 原文分离，避免把未分词正文读进模型。
from pathlib import Path

import jieba
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

# 课程练习根目录（…/data/课程练习）
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent


def build_tokenized_corpus(
    folder_name: str = "three_kingdoms",
    *,
    encoding: str = "utf-8",
) -> Path:
    """
    ---Step1: 获取 source/{folder_name}/ 下所有普通文件（不递归子目录、跳过隐藏文件）。
    ---Step2: 对每个文件按行分词（jieba），一行一句、词之间空格；写入与 source 同级的
    segmented/{folder_name}/，文件名规则：{原文件名}_segmented.txt，供 PathLineSentences 独占读取。
    返回分词结果目录 Path。
    """
    src_dir = _PRACTICE_ROOT / "source" / folder_name
    if not src_dir.is_dir():
        raise FileNotFoundError(f"语料目录不存在: {src_dir}")

    out_dir = _PRACTICE_ROOT / "segmented" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in sorted(src_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            continue
        if not entry.is_file():
            continue
        if entry.name.endswith(".seg.txt"):
            continue

        out_path = out_dir / f"{entry.name}_segmented.txt"
        with open(entry, "r", encoding=encoding, errors="replace") as fin, open(
            out_path, "w", encoding=encoding
        ) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                words = [w.strip() for w in jieba.lcut(line) if w.strip()]
                if len(words) < 2:
                    continue
                fout.write(" ".join(words) + "\n")

    return out_dir


def get_embedding(sentences_dir: str | Path) -> Word2Vec:
    """---Step3: PathLineSentences(目录) 迭代句子，再训练 Word2Vec；目录内须为「一行一句、空格分词」文件。"""
    path = Path(sentences_dir).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"需要分词后的语料目录: {path}")
    sentences = PathLineSentences(str(path))
    model = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=5,
        min_count=1,
        workers=4,
    )
    return model


def get_wv_model(corpus_folder: str):
    # ---Step1~2: 源文件来自 source/{corpus_folder}/，分词输出至与 source 同级的 segmented/{corpus_folder}/
    seg_dir = build_tokenized_corpus(corpus_folder)
    print(f"分词语料目录: {seg_dir}")

    # ---Step3: 自文件夹迭代句子并训练（PathLineSentences）
    model = get_embedding(seg_dir)
    model_dir = _PRACTICE_ROOT / "vectorModels" / corpus_folder
    model_dir.mkdir(parents=True, exist_ok=True)
    # gensim SaveLoad 期望 str 或带 write 的文件对象，不能传 Path（会触发 endswith 等错误）
    model.save(str(model_dir / f"{corpus_folder}_word2vec.model"))
    return model


# 与 embedding_API 中预留查询一致，便于两种 embedding 对照（不在 __main__ 中调用）。
_DEMO_QUERY_CHANGBAN_ADOU = "谁在长坂坡救了阿斗？"


def similar_words_for_sentence(
    wv,
    sentence: str,
    *,
    topn: int = 10,
) -> list[tuple[str, float]]:
    """
    用 Word2Vec 对「句子 / 问句」做近邻查询的常用做法：
    jieba 分词 → 丢掉不在词表里的词 → 对剩余词的向量取平均 → similar_by_vector 找最相近的「词」。

    说明：得到的是与整句语义方向接近的「词」，不是另一整句；若要句对句相似度，请用 Doc2Vec，
    或使用 embedding_API.py 里对整段文本的句向量 + 余弦相似度。
    """
    tokens = [w for w in jieba.lcut(sentence) if w.strip() and w in wv]
    if not tokens:
        return []
    vec = np.mean([wv[t] for t in tokens], axis=0)
    return wv.similar_by_vector(vec, topn=topn)


def demo_query_changban_adou_similarity(
    model: Word2Vec | None = None,
    *,
    corpus_folder: str = "three_kingdoms",
    topn: int = 10,
) -> list[tuple[str, float]]:
    """
    预留：对问句分词，取各词向量平均后，用 similar_by_vector 得到最相近的 topn 个词。
    未在 __main__ 中调用；问句与 embedding_API.demo_query_changban_adou_similarity 相同。
    """
    if model is None:
        path = (
            _PRACTICE_ROOT
            / "vectorModels"
            / corpus_folder
            / f"{corpus_folder}_word2vec.model"
        )
        model = Word2Vec.load(str(path))
    return similar_words_for_sentence(model.wv, _DEMO_QUERY_CHANGBAN_ADOU, topn=topn)


def case_three_kingdoms(corpus_folder: str = "three_kingdoms"):

    model = get_wv_model(corpus_folder)
    wv = model.wv
    for name in ("曹操", "刘备", "张飞"):
        if name not in wv:
            print(f"词「{name}」不在词表中（可检查分词是否拆散）。")
            return

    # ---Step4: 基于余弦相似度，查看与目标词「曹操」在向量空间中最邻近的词。
    # 注意：most_similar 只接受词表中的「词」，不能把整句当作一个 key；问句请用 demo_query_changban_adou_similarity。
    print("与「曹操」最相近的词（余弦相似）：")
    for word, score in wv.most_similar("曹操", topn=15):
        print(f"  {word!r}\t{score:.4f}")

    # ---Step5: 向量加减类比——将「曹操+刘备−张飞」合成一个方向，再取最邻近词（king−man+woman 类思路）。
    print("\n「曹操 + 刘备 - 张飞」向量附近词（类比）：")
    for word, score in wv.most_similar(
        positive=["曹操", "刘备"],
        negative=["张飞"],
        topn=10,
    ):
        print(f"  {word!r}\t{score:.4f}")


if __name__ == "__main__":
    case_three_kingdoms("three_kingdoms")
