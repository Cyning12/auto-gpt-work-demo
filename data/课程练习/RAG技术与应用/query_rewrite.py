from __future__ import annotations

import sys
from pathlib import Path

# 课程练习根目录（.../data/课程练习）；与 langchain_rag / disney_help_rag 一致
_PRACTICE_ROOT = Path(__file__).resolve().parent.parent
if str(_PRACTICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PRACTICE_ROOT))

import os
import logging
from dotenv import load_dotenv
import json
import re
from dashscope_generation import (
    call_dashscope_chat,
    call_generation,
    call_generation_can_search,
    chat_answer_text,
)
from utils import generation_first_message
from web_search import search_web

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
        err: str = ""

        try:
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
                final_result = self.rewrite_rhetorical_query(
                    query, conversation_history
                )
            else:
                # 对于其他类型，返回自动识别的改写结果
                final_result = result.get("rewritten_query", query)
        except Exception as e:
            # Demo 兜底：网络/配额/解析异常时不阻断流程
            err = str(e)
            final_result = result.get("rewritten_query", query)

        out = {
            "original_query": query,
            "detected_type": query_type,
            "confidence": result.get("confidence", 0.5),
            "rewritten_query": final_result,
            "auto_rewrite_result": result,
        }
        if err:
            out["execute_error"] = err
        return out

    def auto_web_search_rewrite(self, query, conversation_history="", context_info=""):
        """自动识别并改写为联网搜索查询（仅给出是否需要搜索 + 查询改写 + 搜索策略，不实际联网）。"""
        # 第一步：识别是否需要联网搜索
        search_analysis = self.identify_web_search_needs(
            query, conversation_history=conversation_history, context_info=context_info
        )

        if not bool(search_analysis.get("need_web_search", False)):
            return {
                "need_web_search": False,
                "reason": str(
                    search_analysis.get("reason") or "查询不需要联网搜索"
                ).strip(),
                "confidence": float(search_analysis.get("confidence") or 0.7),
                "original_query": query,
            }

        # 第二步：改写查询
        rewritten_result = self.rewrite_for_web_search(
            query, conversation_history=conversation_history, context_info=context_info
        )

        # 第三步：生成搜索策略
        search_strategy = self.generate_search_strategy(
            query,
            rewritten_query=rewritten_result.get("rewritten_query", query),
        )

        return {
            "need_web_search": True,
            "search_reason": str(search_analysis.get("search_reason") or "").strip(),
            "confidence": float(search_analysis.get("confidence") or 0.7),
            "original_query": query,
            "rewritten_query": str(
                rewritten_result.get("rewritten_query") or query
            ).strip(),
            "search_keywords": list(rewritten_result.get("search_keywords") or []),
            "search_intent": str(rewritten_result.get("search_intent") or "").strip(),
            "suggested_sources": list(rewritten_result.get("suggested_sources") or []),
            "search_strategy": dict(search_strategy or {}),
        }

    @staticmethod
    def format_web_search_context(tavily_response: dict) -> str:
        """
        将 Tavily search response 格式化为可直接喂给 LLM 的上下文文本。
        这里先不做内容筛选，仅做轻量排版与截断，避免 token 爆炸。
        """
        if not isinstance(tavily_response, dict):
            return ""
        results = tavily_response.get("results") or []
        if not isinstance(results, list) or not results:
            return ""

        blocks: list[str] = []
        for i, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            # 不做筛选，只做长度控制
            if len(content) > 800:
                content = content[:800] + "..."
            blocks.append(
                "\n".join(
                    [
                        f"[{i}] {title}" if title else f"[{i}]",
                        f"URL: {url}" if url else "URL: （空）",
                        f"摘要: {content}" if content else "摘要: （空）",
                    ]
                )
            )
        return "\n\n".join(blocks).strip()

    def answer_with_web_search(
        self,
        *,
        user_question: str,
        tavily_response: dict,
        model: str | None = None,
    ) -> str:
        """
        format_context -> call_dashscope_chat -> answer
        说明：不做内容筛选，直接把搜索结果上下文交给 LLM 整合回答。
        """
        q = (user_question or "").strip()
        if not q:
            return ""
        ctx = self.format_web_search_context(tavily_response)
        if not ctx:
            raise RuntimeError("搜索结果为空，无法生成回答")

        sys_prompt = (
            "你是一个智能助手，负责根据联网搜索结果回答用户问题。\n"
            "回答原则：\n"
            "1) 只基于“搜索结果”里的事实回答，不要编造。\n"
            "2) 信息不足时，明确说明不足，并指出你缺少哪类信息。\n"
            "3) 给出结论后，可用 2-5 条要点补充细节。\n"
            "4) 涉及时效信息（活动/票价/开放时间等），请说明信息可能随时间变化。\n"
            "5) 尽量在答案末尾给出引用来源（URL 列表或编号）。\n"
        )
        user_prompt = (
            f"用户问题：{q}\n\n"
            "搜索结果（可引用编号）：\n"
            f"{ctx}\n\n"
            "请基于以上搜索结果回答。"
        )

        m = (model or "").strip() or os.getenv("DASHSCOPE_CHAT_MODEL") or self._model
        resp = call_dashscope_chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=m,
            api_key=self._api_key,
        )
        return chat_answer_text(resp)

    def identify_web_search_needs(
        self,
        query: str,
        *,
        conversation_history: str = "",
        context_info: str = "",
    ) -> dict:
        """
        判断是否需要联网搜索（面向“缺少最新信息/实时数据/外部事实核验”等场景）。

        返回 JSON 对象：
        {
          "need_web_search": true/false,
          "search_reason": "...",   # need_web_search=true 时必填
          "reason": "...",          # need_web_search=false 时必填
          "confidence": 0.0-1.0
        }
        """
        q = (query or "").strip()
        hist = (conversation_history or "").strip()
        ctx = (context_info or "").strip()
        if not q:
            return {
                "need_web_search": False,
                "reason": "空查询无需联网搜索",
                "confidence": 1.0,
            }
        system = (
            "你是一个联网搜索需求判定器。你的任务是判断用户问题是否必须联网搜索才能可靠回答。\n"
            "必须联网搜索的典型：实时/最新（今天/本周/2026/最近）、价格/库存/开放时间/活动排期、"
            "政策变更、新闻、需要核验的外部事实。\n"
            "不需要联网搜索的典型：常识、对话内已给出信息、用户只是在改写/澄清意图。\n"
            "输出要求：只输出严格 JSON 对象，不要解释文字。\n"
        )
        user = f"""
【对话历史】
{hist}

【上下文信息】
{ctx}

【用户问题】
{q}

请输出 JSON：
{{
  "need_web_search": true/false,
  "search_reason": "需要搜索时填写，说明缺什么外部信息",
  "reason": "不需要搜索时填写，说明为什么不需要",
  "confidence": 0.0
}}
"""
        try:
            obj = self._call_rewrite_model(
                system=system, user=user, output_format="json_object"
            )
            need = bool(obj.get("need_web_search"))
            try:
                cf = float(obj.get("confidence"))
            except Exception:
                cf = 0.7
            out = {
                "need_web_search": need,
                "confidence": max(0.0, min(1.0, cf)),
            }
            if need:
                out["search_reason"] = str(obj.get("search_reason") or "").strip()
                if not out["search_reason"]:
                    out["search_reason"] = "需要联网获取最新/实时信息"
            else:
                out["reason"] = str(obj.get("reason") or "").strip()
                if not out["reason"]:
                    out["reason"] = "该问题可在本地知识库/常识范围内回答"
            return out
        except Exception:
            # 兜底：用启发式判断（包含“最新/今天/本周/价格/活动”等词就倾向需要）
            cues = (
                "最新",
                "最近",
                "今天",
                "本周",
                "价格",
                "票价",
                "活动",
                "开放时间",
                "排期",
                "新闻",
            )
            if any(c in q for c in cues):
                return {
                    "need_web_search": True,
                    "search_reason": "问题包含明显的“最新/实时/价格/活动”信息需求",
                    "confidence": 0.6,
                }
            return {
                "need_web_search": False,
                "reason": "问题不强依赖外部实时信息",
                "confidence": 0.6,
            }

    def rewrite_for_web_search(
        self,
        query: str,
        *,
        conversation_history: str = "",
        context_info: str = "",
    ) -> dict:
        """
        将原始 query 改写为更适合搜索引擎的查询，并产出关键词/意图/建议来源。

        返回 JSON：
        {
          "rewritten_query": "...",
          "search_keywords": ["...", "..."],
          "search_intent": "...",
          "suggested_sources": ["..."]
        }
        """
        q = (query or "").strip()
        hist = (conversation_history or "").strip()
        ctx = (context_info or "").strip()
        if not q:
            return {
                "rewritten_query": "",
                "search_keywords": [],
                "search_intent": "",
                "suggested_sources": [],
            }
        system = (
            "你是搜索 Query 改写器。把用户问题改写为适合搜索引擎检索的中文查询。\n"
            "要求：\n"
            "1) 只输出严格 JSON 对象，不要解释文字。\n"
            "2) rewritten_query 要尽量包含地点/对象/时间范围等限定。\n"
            "3) search_keywords 给 3-8 个关键词，尽量去冗余。\n"
        )
        user = f"""
【对话历史】
{hist}

【上下文信息】
{ctx}

【原始问题】
{q}

请输出 JSON：
{{
  "rewritten_query": "...",
  "search_keywords": ["..."],
  "search_intent": "一句话说明要搜什么",
  "suggested_sources": ["官网", "权威媒体", "票务平台", "地图/点评平台"]
}}
"""
        try:
            obj = self._call_rewrite_model(
                system=system, user=user, output_format="json_object"
            )
            rq = str(obj.get("rewritten_query") or "").strip() or q
            kws = obj.get("search_keywords") or []
            if not isinstance(kws, list):
                kws = [str(kws)]
            kws2: list[str] = []
            for x in kws:
                s = self._clean_one_line_text(str(x))
                if s:
                    kws2.append(s)
            intent = str(obj.get("search_intent") or "").strip()
            srcs = obj.get("suggested_sources") or []
            if not isinstance(srcs, list):
                srcs = [str(srcs)]
            srcs2 = [self._clean_one_line_text(str(x)) for x in srcs if str(x).strip()]
            return {
                "rewritten_query": rq,
                "search_keywords": kws2,
                "search_intent": intent,
                "suggested_sources": srcs2,
            }
        except Exception:
            return {
                "rewritten_query": q,
                "search_keywords": [],
                "search_intent": "",
                "suggested_sources": [],
            }

    def generate_search_strategy(
        self,
        query: str,
        *,
        rewritten_query: str = "",
    ) -> dict:
        """
        给出搜索策略（不联网）：关键词拆分、平台建议、时间范围。

        返回 dict：
        {
          "primary_keywords": [...],
          "extended_keywords": [...],
          "search_platforms": [...],
          "time_range": "..."
        }
        """
        q = (query or "").strip()
        rq = (rewritten_query or "").strip()
        base = rq or q
        if not base:
            return {
                "primary_keywords": [],
                "extended_keywords": [],
                "search_platforms": [],
                "time_range": "",
            }
        system = (
            "你是搜索策略助手。请为搜索引擎查询生成可执行的搜索策略。\n"
            "输出要求：只输出严格 JSON 对象，不要解释文字。\n"
        )
        user = f"""
【查询】
{base}

请输出 JSON：
{{
  "primary_keywords": ["..."],
  "extended_keywords": ["..."],
  "search_platforms": ["Google", "Bing", "百度", "官网", "社交媒体"],
  "time_range": "例如：最近7天/2026年/不限"
}}
"""
        try:
            obj = self._call_rewrite_model(
                system=system, user=user, output_format="json_object"
            )
            pk = obj.get("primary_keywords") or []
            ek = obj.get("extended_keywords") or []
            sp = obj.get("search_platforms") or []
            tr = str(obj.get("time_range") or "").strip()
            if not isinstance(pk, list):
                pk = [str(pk)]
            if not isinstance(ek, list):
                ek = [str(ek)]
            if not isinstance(sp, list):
                sp = [str(sp)]
            return {
                "primary_keywords": [
                    self._clean_one_line_text(str(x)) for x in pk if str(x).strip()
                ],
                "extended_keywords": [
                    self._clean_one_line_text(str(x)) for x in ek if str(x).strip()
                ],
                "search_platforms": [
                    self._clean_one_line_text(str(x)) for x in sp if str(x).strip()
                ],
                "time_range": tr,
            }
        except Exception:
            return {
                "primary_keywords": [],
                "extended_keywords": [],
                "search_platforms": [],
                "time_range": "",
            }


def query_rewrite_demo() -> None:
    """Demo：自动识别 Query 类型并执行对应改写策略（case1..case6）。"""
    query_rewrite = QueryRewrite()

    def _print_case(title: str, payload: dict) -> None:
        print(f"\n{title}")
        print(f"- original_query: {payload.get('original_query')}")
        print(f"- detected_type: {payload.get('detected_type')}")
        print(f"- confidence: {payload.get('confidence')}")
        print(f"- rewritten_query: {payload.get('rewritten_query')}")

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


def auto_web_search_rewrite_demo() -> None:
    """Demo：判断是否需要联网搜索，并打印结构化日志（需要时可实际联网搜索）。"""
    qr = QueryRewrite()

    def _print_result(title: str, result2: dict) -> None:
        print(f"\n{title}")
        if bool(result2.get("need_web_search")):
            print("✓ 需要联网搜索")
            print(f"  搜索原因: {result2.get('search_reason')}")
            print(f"  置信度: {result2.get('confidence')}")
            print(f"  改写查询: {result2.get('rewritten_query')}")
            print(f"  搜索关键词: {result2.get('search_keywords')}")
            print(f"  搜索意图: {result2.get('search_intent')}")
            print(f"  建议来源: {result2.get('suggested_sources')}")
            ss = result2.get("search_strategy") or {}
            print("  搜索策略:")
            print(f"    - 主要关键词: {ss.get('primary_keywords')}")
            print(f"    - 扩展关键词: {ss.get('extended_keywords')}")
            print(f"    - 搜索平台: {ss.get('search_platforms')}")
            print(f"    - 时间范围: {ss.get('time_range')}")
            # 可选：真的去搜（Tavily）
            enabled = str(
                os.getenv("WEB_SEARCH_ENABLED") or "1"
            ).strip().lower() not in (
                "0",
                "false",
                "off",
            )
            if enabled:
                q2 = str(result2.get("rewritten_query") or "").strip()
                kw = result2.get("search_keywords") or []
                query_for_search = q2 or (
                    " ".join([str(x) for x in kw if str(x).strip()]) or ""
                )
                if query_for_search:
                    try:
                        data = search_web(
                            query_for_search,
                            max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS") or 5),
                            search_depth=str(os.getenv("WEB_SEARCH_DEPTH") or "basic"),
                            time_range=(
                                os.getenv("WEB_SEARCH_TIME_RANGE") or ""
                            ).strip()
                            or None,
                            start_date=(
                                os.getenv("WEB_SEARCH_START_DATE") or ""
                            ).strip()
                            or None,
                            end_date=(os.getenv("WEB_SEARCH_END_DATE") or "").strip()
                            or None,
                            include_answer=False,
                            include_raw_content=False,
                        )
                        results = data.get("results") or []
                        print(f"  搜索结果: {len(results)} 条（Tavily）")
                        for i, item in enumerate(results[:3], start=1):
                            title2 = (item.get("title") or "").strip()
                            url2 = (item.get("url") or "").strip()
                            snippet = (item.get("content") or "").strip()
                            snippet = snippet[:140] + (
                                "..." if len(snippet) > 140 else ""
                            )
                            print(f"    [{i}] {title2}")
                            print(f"        {url2}")
                            if snippet:
                                print(f"        {snippet}")
                        # LLM 整合搜索结果输出最终回答
                        try:
                            ans = qr.answer_with_web_search(
                                user_question=str(
                                    result2.get("original_query") or query_for_search
                                ),
                                tavily_response=data,
                            )
                            print("  --- LLM Answer ---")
                            print(ans)
                        except Exception as e2:
                            print(f"  生成回答失败: {e2}")
                    except Exception as e:
                        print(f"  搜索失败: {e}")
                else:
                    print("  搜索跳过: 改写查询/关键词为空")
            else:
                print("  搜索跳过: WEB_SEARCH_ENABLED=0")
        else:
            print("✗ 不需要联网搜索")
            print(f"  原因: {result2.get('reason')}")
            print(f"  置信度: {result2.get('confidence')}")
            print(f"  原始查询: {result2.get('original_query')}")

    def case1_need_search() -> None:
        query = "2026年上海迪士尼最近有什么活动？票价有变化吗？"
        r = qr.auto_web_search_rewrite(query)
        _print_result("case1（应需要联网）", r)

    def case2_no_search() -> None:
        conversation_history = """
用户: 我想了解一下“疯狂动物城”主题园区。
助手: 目前园区包含沉浸式街区与互动体验。
""".strip()
        query = "这个园区有什么游乐设施？"
        r = qr.auto_web_search_rewrite(query, conversation_history=conversation_history)
        _print_result("case2（应不需要联网）", r)

    case1_need_search()
    case2_no_search()


def auto_web_search_rewrite_demo_with_search_function_call():
    """
    Demo：让 LLM 通过 Function Calling 自动触发 web_search 工具，再基于工具结果回答。

    说明：
    - 本 Demo 不走 auto_web_search_rewrite 的“判定/改写/策略”链路，直接让模型自主决定是否调用 web_search。
    - 日志：`dashscope_generation.call_generation_can_search` 内部会打印 tool_call 相关日志（需 INFO 级别）。
    """
    print("\n=== auto_web_search_rewrite_demo_with_search_function_call ===")
    # Demo 默认打开 INFO 日志，确保能看到 functioncall/web_search 的调用日志
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    question = "2026年上海迪士尼最近有什么活动？票价有变化吗？请给出来源链接。"

    system = (
        "你是一个带工具的智能助手。\n"
        "当用户问题涉及最新/实时/价格/活动/开放时间等外部信息时，你必须先调用 web_search 工具获取资料；\n"
        "如果不需要搜索，也要说明原因。\n"
        "回答要求：\n"
        "1) 先给结论，再给要点。\n"
        "2) 给出来源（URL 或编号）。\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    resp, trace = call_generation_can_search(
        model=os.getenv("DASHSCOPE_CHAT_MODEL") or "qwen-turbo",
        messages=messages,
        api_key=(os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
        or None,
        search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS") or 6),
        search_depth=str(os.getenv("WEB_SEARCH_DEPTH") or "advanced"),
    )
    print("✓ 对话完成")
    print(f"- trace_messages: {len(trace)}")
    print("---- Answer ----")
    print(chat_answer_text(resp))


def main():
    # 先跑 query rewrite demo，再跑 web search rewrite demo
    # query_rewrite_demo()
    # auto_web_search_rewrite_demo()
    auto_web_search_rewrite_demo_with_search_function_call()


if __name__ == "__main__":
    main()
