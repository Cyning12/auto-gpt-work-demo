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
import sys
from pathlib import Path

# 课程练习根目录（.../data/课程练习）；与 langchain_rag 等一致，由此导入拆分的工具模块
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent
if str(_PRACTICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_ROOT))

from disney_classify import classify_disney_knowledge_files
from doc_file_utils import (
    export_doc_and_docx_to_markdown,
    export_parsed_markdown_chunks_for_classified_docs,
)
from hf_clip_utils import load_clip_model_and_processor
from image_file_utils import image_to_text

_DISNEY_KNOWLEDGE_BASE = _PRACTICE_ROOT / "source" / "迪士尼RAG知识库"
_PARSED_MD_DIR = _PRACTICE_ROOT / "doc" / "迪士尼RAG知识库" / "parsed_markdown"
_CHUNKS_JSON_DIR = _PRACTICE_ROOT / "doc" / "迪士尼RAG知识库" / "chunks_json"

# CLIP：默认缓存 ~/.cache/huggingface/hub/；可传 revision= 钉死权重，见 hf_clip_utils 模块说明
print("正在加载 CLIP 模型...")
try:
    clip_model, clip_processor = load_clip_model_and_processor()
    print("CLIP 模型加载成功。")
except Exception as e:
    print(f"加载 CLIP 模型失败，请检查网络连接或 Hugging Face Token。错误: {e}")
    raise SystemExit(1) from e


def _print_classification_summary(data: dict[str, list[str]]) -> None:
    """打印分类摘要与完整 JSON（中文保真）。"""
    print("迪士尼知识库文件分类统计：")
    for k in ("doc", "pdf", "ppt", "images", "other"):
        print(f"- {k}: {len(data[k])} 个")
    print("\n完整分类结果（JSON）：")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_knowledge_base():
    """构建完整的知识库，包括解析、切片、Embedding和索引。"""

    # 1. 处理Word文档

    # 2. 解析文档
    # 3. 提取文本
    # 4. 提取图像
    # 5. 提取表格
    # 6. 提取公式
    # 7. 提取图片
    # 8. 提取视频
    # 9. 提取音频
    # 10. 提取链接


if __name__ == "__main__":
    classified = classify_disney_knowledge_files(_DISNEY_KNOWLEDGE_BASE)
    _print_classification_summary(classified)
    export_doc_and_docx_to_markdown(classified, _DISNEY_KNOWLEDGE_BASE, _PARSED_MD_DIR)
    # 对已导出的 Markdown（由 chunks_to_markdown 生成）还原块并切片，写入 _CHUNKS_JSON_DIR
    export_parsed_markdown_chunks_for_classified_docs(
        classified,
        _DISNEY_KNOWLEDGE_BASE,
        _PARSED_MD_DIR,
        _CHUNKS_JSON_DIR,
    )
    # for image_path in classified["images"]:
    #     result = image_to_text(str(_DISNEY_KNOWLEDGE_BASE / image_path))
    #     print(result)
