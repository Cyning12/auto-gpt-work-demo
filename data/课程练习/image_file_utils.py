"""
图片文件工具：OCR 等；供迪士尼 RAG 等练习复用。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytesseract


def image_to_text(image_path: str | Path) -> dict[str, str]:
    """对图片进行 OCR（chi_sim+eng）。"""
    try:
        image = Image.open(image_path)
        ocr_text = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
        return {"ocr": ocr_text}
    except Exception as e:
        print(f"处理图片失败 {image_path}: {e}")
        return {"ocr": ""}
