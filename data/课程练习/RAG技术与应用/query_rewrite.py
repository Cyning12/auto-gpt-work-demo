from __future__ import annotations

import sys
from pathlib import Path

# 课程练习根目录（.../data/课程练习）；与 langchain_rag / disney_help_rag 一致
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent
if str(_PRACTICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_ROOT))

import os
from dotenv import load_dotenv
import json
import re
from dashscope_generation import call_generation
from utils import generation_first_message

load_dotenv()
_api_key = (
    os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
).strip()

if not _api_key:
    raise RuntimeError(
        "未配置 API Key：请在环境变量或本目录 .env 中设置 BAILIAN_API_KEY 或 DASHSCOPE_API_KEY"
    )


class QueryRewrite:
    def __init__(
        self,
        *,
        model: str = "qwen-turbo",
        api_key: str = _api_key,
    ) -> None:
        self._model = (model or "").strip() or "qwen-turbo"
        self._api_key = (api_key or "").strip()
        if not self._api_key:
            raise RuntimeError(
                "未配置 API Key：请在环境变量或本目录 .env 中设置 BAILIAN_API_KEY 或 DASHSCOPE_API_KEY"
            )

    @staticmethod
    def _clean_one_line_text(s: str) -> str:
        """清洗模型输出为单行文本（去首尾空白/引号/多余换行）。"""
        out = (s or "").strip()
        if not out:
            return ""
        # 容错：部分模型会把结果包在引号里
        out = out.strip().strip('"').strip("'").strip()
        # 保证只取第一行
        out = out.splitlines()[0].strip()
        return out

    @staticmethod
    def _extract_first_json_object(text: str) -> dict:
        """
        从模型输出中提取第一个 JSON 对象并解析。

        兼容：
        - 纯 JSON / 多行 JSON
        - ```json ... ``` 代码块包裹
        - 输出前后有少量解释文字（尽量截取 {...}）
        """
        raw = (text or "").strip()
        if not raw:
            raise ValueError("empty")
        # 去掉 markdown code fence
        raw = re.sub(r"^```(?:json)?\\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\\s*```\\s*$", "", raw).strip()
        # 先尝试整体解析
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # 兜底：截取第一个 {...}
        m = re.search(r"\\{[\\s\\S]*\\}", raw)
        if not m:
            raise ValueError("no_json_object")
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError("json_not_object")
        return obj

    @staticmethod
    def _extract_first_json_array(text: str) -> list:
        """
        从模型输出中提取第一个 JSON 数组并解析。

        兼容：
        - 纯 JSON / 多行 JSON
        - ```json ... ``` 代码块包裹
        - 输出前后有少量解释文字（尽量截取 [...]）
        """
        raw = (text or "").strip()
        if not raw:
            raise ValueError("empty")
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            raise ValueError("no_json_array")
        arr = json.loads(m.group(0))
        if not isinstance(arr, list):
            raise ValueError("json_not_array")
        return arr

    def _call_rewrite_model(
        self,
        *,
        system: str,
        user: str,
        output_format: str = "one_line_text",
    ):
        """
        统一的调用与解析：

        - output_format=\"one_line_text\"：返回单行文本（str）
        - output_format=\"json_object\"：返回 JSON 对象（dict）
        """
        resp = call_generation(
            model=self._model,
            api_key=self._api_key,
            messages=[
                {"role": "system", "content": (system or "").strip()},
                {"role": "user", "content": (user or "").strip()},
            ],
            temperature=0.0,
            result_format="message",
        )
        msg = generation_first_message(resp)
        if msg is None:
            raise RuntimeError(
                "Query 改写失败：未取到有效 message（请检查 API Key/配额/网络）"
            )
        out = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        )
        out_text = (str(out) if out is not None else "").strip()
        if not out_text:
            raise RuntimeError("Query 改写失败：模型返回内容为空")
        fmt = (output_format or "one_line_text").strip().lower()
        if fmt == "json_object":
            return self._extract_first_json_object(out_text)
        if fmt == "json_array":
            return self._extract_first_json_array(out_text)
        out_text = self._clean_one_line_text(out_text)
        if not out_text:
            raise RuntimeError("Query 改写失败：模型返回内容为空")
        return out_text

    def rewrite_context_dependent_query(
        self, current_query: str, conversation_history: str = ""
    ) -> str:
        """上下文依赖型 Query 改写（仅输出 1 行改写结果，不输出解释）。"""
        q = (current_query or "").strip()
        hist = (conversation_history or "").strip()
        if not q:
            return ""

        system = (
            "你是查询改写器（Query Rewriter）。你的任务是把用户当前问题改写为自洽的独立问题。\n"
            "规则：\n"
            "1) 只输出一行“改写后的问题”，不要解释、不要加前缀、不要加引号。\n"
            "2) 若当前问题不依赖上下文，必须原样输出当前问题（逐字一致）。\n"
            "3) 若依赖上下文，补全必要信息（对象/地点/时间/指代等），使其不看历史也能理解。\n"
        )
        user = (
            "【对话历史】\n"
            f"{hist}\n\n"
            "【当前问题】\n"
            f"{q}\n\n"
            "请输出：改写后的问题（仅一行）。"
        )
        return self._call_rewrite_model(system=system, user=user)

    def rewrite_ambiguous_reference_query(
        self,
        current_query: str,
        conversation_history: str = "",
    ) -> str:
        """模糊指代型Query改写"""
        q = (current_query or "").strip()
        hist = (conversation_history or "").strip()
        if not q:
            return ""
        instruction = """
你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出问题中 "都"、"它"、"这个"等模糊指代词具体指向的对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的查询。
"""

        user = f"""
### 对话历史/上下文信息 ###
{hist}

### 原始问题 ###
{q}

请输出：改写后的查询（仅一行）。
"""
        return self._call_rewrite_model(system=instruction, user=user)

    def rewrite_comparative_query(self, query, context_info):
        """对比型Query改写"""

        instruction = """
你是一个查询分析专家。请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。
"""

        user = f"""
### 对话历史/上下文信息 ###
{context_info}

### 原始问题 ###
{query}

请输出：改写后的对比性查询（仅一行）。
"""
        return self._call_rewrite_model(system=instruction, user=user)

    def rewrite_multi_intent_query(self, query):
        """多意图型Query改写 - 分解查询"""
        q = (query or "").strip()
        if not q:
            return []
        instruction = """
你是一个任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。以JSON数组格式输出。
"""

        prompt = f"""
### 原始问题 ###
{q}

请只输出 JSON 数组本身（必须能被 json.loads 解析），不要使用 markdown 代码块，不要输出任何解释文字。
例如：["问题1", "问题2", "问题3"]
"""
        arr = self._call_rewrite_model(
            system=instruction,
            user=prompt,
            output_format="json_array",
        )
        out: list[str] = []
        for item in arr:
            if item is None:
                continue
            s = str(item).strip()
            s = self._clean_one_line_text(s)
            if s:
                out.append(s)
        # 兜底：至少返回原问题
        return out or [q]

    def rewrite_rhetorical_query(self, current_query, conversation_history):
        """反问型Query改写"""
        q = (current_query or "").strip()
        hist = (conversation_history or "").strip()
        if not q:
            return ""
        instruction = (
            "你是一个沟通理解大师。请将用户的反问/情绪化表述改写为一个中立、客观、"
            "可以直接用于知识库检索的问句。\n"
            "规则：\n"
            "1) 只输出一行改写后的问句，不要分析、不要解释、不要加任何前缀。\n"
            "2) 若需要补全指代，请结合对话历史补全。\n"
        )
        user = f"""
【对话历史】
{hist}

【当前问题】
{q}

请输出：改写后的中立问句（仅一行）。
"""
        return self._call_rewrite_model(system=instruction, user=user)

    def auto_rewrite_query(self, query, conversation_history="", context_info=""):
        """自动识别Query类型并进行改写"""
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含"还有"、"其他"等需要上下文理解的词汇
2. 对比型 - 包含"哪个"、"比较"、"更"、"哪个更好"、"哪个更"等比较词汇
3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
4. 多意图型 - 包含多个独立问题，用"、"或"？"分隔
5. 反问型 - 包含"不会"、"难道"等反问语气
说明：如果同时存在多意图型、模糊指代型，优先级为多意图型>模糊指代型

请返回JSON格式的结果：
{
    "query_type": "查询类型",
    "rewritten_query": "改写后的查询",
    "confidence": "置信度(0-1)"
}
"""

        prompt = f"""
### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 上下文信息 ###
{context_info}

### 原始查询 ###
{query}

### 分析结果 ###
"""
        instruction2 = (
            instruction.strip()
            + "\n\n"
            + "要求：只输出 JSON 对象本身，不要使用 markdown 代码块，不要输出任何解释文字。"
        )
        try:
            obj = self._call_rewrite_model(
                system=instruction2, user=prompt, output_format="json_object"
            )
            qt = str(obj.get("query_type") or "").strip() or "未知类型"
            rq = str(obj.get("rewritten_query") or "").strip() or str(query).strip()
            try:
                cf = float(obj.get("confidence"))
            except Exception:
                cf = 0.5
            return {
                "query_type": qt,
                "rewritten_query": rq,
                "confidence": max(0.0, min(1.0, cf)),
            }
        except Exception:
            return {
                "query_type": "未知类型",
                "rewritten_query": str(query).strip(),
                "confidence": 0.5,
            }

    def auto_rewrite_and_execute(self, query, conversation_history="", context_info=""):
        """自动识别Query类型并进行改写，然后根据类型调用相应的改写方法"""
        # 首先进行自动识别
        result = self.auto_rewrite_query(query, conversation_history, context_info)

        # 根据识别结果调用相应的改写方法
        query_type = result.get("query_type", "")

        if "上下文依赖" in query_type:
            final_result = self.rewrite_context_dependent_query(
                query, conversation_history
            )
        elif "对比" in query_type:
            final_result = self.rewrite_comparative_query(
                query, context_info or conversation_history
            )
        elif "模糊指代" in query_type:
            final_result = self.rewrite_ambiguous_reference_query(
                query, conversation_history
            )
        elif "多意图" in query_type:
            final_result = self.rewrite_multi_intent_query(query)
        elif "反问" in query_type:
            final_result = self.rewrite_rhetorical_query(query, conversation_history)
        else:
            # 对于其他类型，返回自动识别的改写结果
            final_result = result.get("rewritten_query", query)

        return {
            "original_query": query,
            "detected_type": query_type,
            "confidence": result.get("confidence", 0.5),
            "rewritten_query": final_result,
            "auto_rewrite_result": result,
        }


def main():
    query_rewrite = QueryRewrite()

    def _print_case(title: str, payload: dict) -> None:
        print(f"\n{title}")
        print(f"- original_query: {payload.get('original_query')}")
        print(f"- detected_type: {payload.get('detected_type')}")
        print(f"- confidence: {payload.get('confidence')}")
        print(f"- rewritten_query: {payload.get('rewritten_query')}")

    # 一个不同类型的 query 对应一个示例方法（caseX）
    def case1_context_dependent() -> None:
        conversation_history = """
用户: 我想了解一下上海迪士尼乐园的最新项目。
助手: 上海迪士尼乐园最新推出了“疯狂动物城”主题园区，包含沉浸式街区与互动体验。
用户: 这个园区有什么游乐设施？
助手: 目前有若干互动体验与主题设施，具体以园区公告为准。
""".strip()
        query = "还有其他设施吗？"
        r = query_rewrite.auto_rewrite_and_execute(
            query, conversation_history=conversation_history
        )
        _print_case("case1 上下文依赖型", r)

    def case2_ambiguous_reference() -> None:
        conversation_history = """
用户: 我想了解一下“疯狂动物城”主题园区。
助手: 好的，你想了解它的哪些方面？
""".strip()
        query = "这个园区有什么游乐设施？"
        r = query_rewrite.auto_rewrite_and_execute(
            query, conversation_history=conversation_history
        )
        _print_case("case2 模糊指代型", r)

    def case3_comparative() -> None:
        context_info = """
对比对象候选：疯狂动物城主题园区、明日世界、探险岛。
""".strip()
        query = "哪个园区更好玩？"
        r = query_rewrite.auto_rewrite_and_execute(query, context_info=context_info)
        _print_case("case3 对比型", r)

    def case4_multi_intent() -> None:
        query = "我想了解门票价格、有哪些必玩项目？以及怎么去？"
        r = query_rewrite.auto_rewrite_and_execute(query)
        _print_case("case4 多意图型（分解为数组）", r)

    def case5_rhetorical() -> None:
        conversation_history = """
用户: 我排队排了很久。
助手: 抱歉让你久等了，请问你想了解排队时间还是快速通行方案？
""".strip()
        query = "难道就没有更快的办法吗？"
        r = query_rewrite.auto_rewrite_and_execute(
            query, conversation_history=conversation_history
        )
        _print_case("case5 反问型", r)

    def case6_other() -> None:
        query = "上海迪士尼乐园的最新项目是什么？"
        r = query_rewrite.auto_rewrite_and_execute(query)
        _print_case("case6 其他（不改/轻改）", r)

    case1_context_dependent()
    case2_ambiguous_reference()
    case3_comparative()
    case4_multi_intent()
    case5_rhetorical()
    case6_other()


#     conversation_history = """
# 用户: "我想了解一下上海迪士尼乐园的最新项目。"
# AI: "上海迪士尼乐园最新推出了'疯狂动物城'主题园区，这里有朱迪警官和尼克狐的互动体验。"
# 用户: "这个园区有什么游乐设施？"
# AI: "'疯狂动物城'园区目前有疯狂动物城警察局、朱迪警官训练营和尼克狐的冰淇淋店等设施。"
# """
#     current_query = "还有其他设施吗？"
#     print(
#         query_rewrite.rewrite_context_dependent_query(
#             current_query,
#             conversation_history,
#         )
#     )


if __name__ == "__main__":
    main()
