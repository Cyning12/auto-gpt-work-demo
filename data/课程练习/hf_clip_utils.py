"""
Hugging Face CLIP 加载工具：统一缓存目录、可选 revision 钉死权重版本、进程内单例复用。

默认缓存仍遵循 Hugging Face 约定（一般为 ``~/.cache/huggingface/hub/``），可用环境变量
``HF_HOME`` / ``HF_HUB_CACHE`` 或参数 ``cache_dir`` 覆盖。

关于「本地同名、版本是否可能不一致」：
- 仅传 ``model_id``（如 ``openai/clip-vit-base-patch32``）时，解析的是 Hub 上该仓库的**默认引用**
  （多为 ``main`` 分支当前提交）。Hub 端若更新默认分支，你**联网** ``from_pretrained`` 可能拉到新提交；
  已缓存的旧 revision 仍留在磁盘，Hub 库会按 **revision** 分子目录存放，一般不会互相覆盖错读。
- 若你**手动**把不同版本的文件混进同一目录、或拷贝仓库时只改了文件名未改内部 ``config.json`` 的提交信息，
  则可能出现「路径同名、权重实际不一致」的混乱；规范做法是始终通过 **同一 ``cache_dir`` + 明确 ``revision``** 加载。
- 需要**可复现、与机器无关**：请传入 ``revision=`` **Git 提交 SHA**（或经你验证的 tag），与本地文件夹叫什么无关，
  只要缓存里存在该 revision 的快照即一致。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import CLIPModel, CLIPProcessor

_logger = logging.getLogger(__name__)

# 课程练习常用；更换模型时请同步改 ``revision`` 或接受 Hub 默认分支漂移
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# 可选：钉死提交（示例占位；请到模型页复制实际 commit SHA 后启用）
# DEFAULT_CLIP_REVISION = "abc123..."  # noqa: ERA001


@lru_cache(maxsize=8)
def _load_clip_pair(
    model_id: str,
    revision_key: str,
    cache_dir_key: str,
    local_files_only: bool,
) -> tuple[object, object]:
    """进程内单例；参数全部用可哈希的 str/bool。"""
    from transformers import CLIPModel, CLIPProcessor

    kwargs: dict = {
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    rev = revision_key if revision_key else None
    if rev is not None:
        kwargs["revision"] = rev
    cd = cache_dir_key if cache_dir_key else None
    if cd is not None:
        kwargs["cache_dir"] = cd

    _logger.info("加载 CLIP：model_id=%s revision=%s local_only=%s", model_id, rev, local_files_only)
    model = CLIPModel.from_pretrained(model_id, **kwargs)
    processor = CLIPProcessor.from_pretrained(model_id, **kwargs)
    return model, processor


def load_clip_model_and_processor(
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple["CLIPModel", "CLIPProcessor"]:
    """
    加载 ``CLIPModel`` + ``CLIPProcessor``（各练习脚本共享同参即复用同一份内存）。

    :param revision: 建议生产/复现实验时传入 Hub **commit SHA**；``None`` 则使用仓库默认引用。
    :param cache_dir: 显式缓存根目录；``None`` 使用 Hugging Face 默认（受 ``HF_HOME`` 等环境变量影响）。
    :param local_files_only: 仅使用已有缓存，不发起下载（离线/CI）。
    """
    revision_key = revision if revision else ""
    cache_dir_key = str(Path(cache_dir).resolve()) if cache_dir is not None else ""
    model, processor = _load_clip_pair(
        model_id,
        revision_key,
        cache_dir_key,
        local_files_only,
    )
    return model, processor  # type: ignore[return-value]


def clear_clip_cache() -> None:
    """单测或需强制换模型时清空进程内 CLIP 单例。"""
    _load_clip_pair.cache_clear()
