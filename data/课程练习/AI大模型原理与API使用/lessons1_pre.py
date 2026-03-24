import os
import sys
from pathlib import Path
import dashscope
from dotenv import load_dotenv
from dashscope.api_entities.dashscope_response import Role
import json
import random
from collections.abc import Mapping

# 与 utils 同位于「课程练习」目录下：…/课程练习/utils.py
_practice_root = Path(__file__).resolve().parents[1]
if str(_practice_root) not in sys.path:
    sys.path.insert(0, str(_practice_root))
from utils import (
    generation_first_message,
    message_function_call,
    pick,
    prompt_back_or_exit,
)


# 优先加载「本脚本所在目录」的 .env，避免只在仓库根目录有配置时读不到
_here = Path(__file__).resolve().parent
load_dotenv(_here / ".env")
load_dotenv()
api_key = (os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "").strip()
dashscope.api_key = api_key or None

# 覆盖 SDK 启动时从环境读到的 base（修复空字符串 / 错填其它厂商 URL）
# dashscope.base_http_api_url = _resolve_dashscope_http_base()


def get_model(model_name: str, prompt: str, messages: list[dict]):
    response = dashscope.Generation.call(
        model=model_name,
        prompt=prompt,
        messages=messages,
        result_format="message",
    )
    return response


# case1:
# 舆情分析
def case1():
    review = "这款音效特别好，给你意想不到的音质。"
    messages = [
        {
            "role": Role.SYSTEM,
            "content": "你是一名舆论分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向。",
        },
        {"role": Role.USER, "content": review},
    ]

    response = get_model(
        model_name="qwen-turbo",
        prompt=review,
        messages=messages,
    )
    print(response)
    prompt_back_or_exit()


# case2:
# 天气预报
# 编写你的天气函数
# 为了掩饰流程，这里指定了天气的温度，实际上可以哟调用 搞得接口获取实时天气
# 这里可以先用每个城市的固定天气进行返回，查看大模型的调用情况


class UnitEnum:
    CELSIUS = "摄氏度"
    FAHRENHEIT = "华氏度"


def get_current_weather(location: str, unit: UnitEnum = UnitEnum.CELSIUS):
    # 这是一个模拟的天气数据，实际需要调用对应API
    temperature = -1
    if "大连" in location or "dalian" in location.lower():
        temperature = 10
    elif "北京" in location or "beijing" in location.lower():
        temperature = 20
    elif "上海" in location or "shanghai" in location.lower():
        temperature = 20
    elif "广州" in location or "guangzhou" in location.lower():
        temperature = 20
    elif "深圳" in location or "shenzhen" in location.lower():
        temperature = 20
    weather_info = {
        "location": location,
        "temperature": temperature,
        "forecast": ["晴天", "威风"],
        "unit": unit.name,
    }
    return json.dumps(weather_info)


def get_case2_response(messages: list[dict]):
    try:
        functions: list[dict] = [
            {
                "name": "get_current_weather",
                "description": "获取当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称,如：北京、上海、广州、深圳、大连等",
                        },
                        "unit": {
                            "type": "string",
                            "description": "温度单位,如：摄氏度、华氏度",
                        },
                    },
                },
                "required": ["location"],
            }
        ]
        response = dashscope.Generation.call(
            model="qwen-turbo",
            messages=messages,
            result_format="message",
            functions=functions,
        )
        return response
    except Exception as e:
        print(e)
        return None


def case2():
    messages = [
        {
            "role": Role.SYSTEM,
            "content": "你是一名天气预报员，帮我查询城市的天气，回复请用一个词语：晴天 或者 阴天 或者 雨天 或者 雪天。",
        },
    ]
    while True:
        location = input(
            "请输入城市名称: 如：北京、上海、广州、深圳、大连等（输入 back 返回菜单，exit 退出程序）: "
        )
        cmd = location.strip().lower()
        if cmd == "exit":
            print("再见")
            sys.exit(0)
        if cmd == "back":
            return
        messages.append(
            {
                "role": Role.USER,
                "content": location,
            }
        )
        response = get_case2_response(messages)
        message = generation_first_message(response)
        if message is None:
            print("第一次查询失败，请重试", response)
            continue
        print("response=", response)
        messages.append(message)
        print("messages=", messages)

        function_call = message_function_call(message)
        if function_call:
            function_name = (
                function_call.get("name")
                if isinstance(function_call, dict)
                else function_call.name
            )
            raw_args = (
                function_call.get("arguments")
                if isinstance(function_call, dict)
                else function_call.arguments
            )
            if isinstance(raw_args, str):
                function_args = json.loads(raw_args) if raw_args else {}
            else:
                function_args = raw_args if isinstance(raw_args, dict) else {}
            print("function_name=", function_name)
            print("function_args=", function_args)
            print("--------------------------------")
            if function_name == "get_current_weather":
                tool_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                print("tool_response=", tool_response)
                print("--------------------------------")
                messages.append(
                    {
                        "role": Role.FUNCTION,
                        "name": function_name,
                        "content": tool_response,
                    }
                )

                response = get_case2_response(messages)

                message = generation_first_message(response)
                if message is None:
                    print("工具调用失败，请重试")
                    continue
                messages.append(message)
                print("messages=", messages)

            else:
                print("不支持的函数调用")

        print("最终答案：", generation_first_message(response).content)
        print("--------------------------------")


# case3:
# 表格提取
# 表格提取与理解是工作中的场景人物，需要使用多模态模型,调用方法与普通对话模型不同
# 需要使用 MultiModalConversation 方法
# 参数 messages 是一个列表，列表中每个元素是一个字典，字典中包含 role 和 content 键
# role 可以是 Role.USER 或 Role.ASSISTANT
# content 可以是字符串或字典
# 字典中包含 image 和 text 键
# image 是图片的 URL
# text 是文本


def get_case3_response(messages: list[dict]):
    try:
        response = dashscope.MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages,
        )
        return response
    except Exception as e:
        print(e)
        return None


def case3():
    content = [
        {"image": "https://aiwucai.oss-cn-huhehaote.aliyuncs.com/pdf_table.jpg"},
        {"text": "这是一个表格图片，帮我提取里面的内容，输出JSON格式"},
    ]
    messages = [
        {"role": Role.USER, "content": content},
    ]
    response = get_case3_response(messages)
    print("response=", response.output)
    prompt_back_or_exit()


# case4:
# 1、告警内容理解。根据输入的告警信息，结合第三方接口数据，判断当前的异常情况（告警对象、异常模式）；
# 2、分析方法建议。根据当前告警内容，结合应急预案、运维文档和大语言模型自有知识，形成分析方法的建议；
# 3、分析内容自动提取。根据用户输入的分析内容需求，调用多种第三方接口获取分析数据，并进行总结；
# 4、处置方法推荐和执行。根据当前上下文的故障场景理解，结合应急预案和第三方接口，形成推荐处置方案，待用户确认后调用第三方接口进行执行。


# 通过第三方接口获取数据库服务器状态
def get_current_status():
    # 生成连接数数据
    connections = random.randint(10, 100)
    # 生成CPU使用率数据
    cpu_usage = round(random.uniform(1, 100), 1)
    # 生成内存使用率数据
    memory_usage = round(random.uniform(10, 100), 1)
    status_info = {
        "连接数": connections,
        "CPU使用率": f"{cpu_usage}%",
        "内存使用率": f"{memory_usage}%",
    }
    return json.dumps(status_info, ensure_ascii=False)


# 封装模型相应函数
def get_case4_response(messages: list[dict]):
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_status",
                    "description": "调用监控系统接口，获取当前数据库服务器性能指标，包括：连接数、CPU使用率、内存使用率",
                    "parameters": {},
                    "required": [],
                },
            }
        ]
        response = dashscope.Generation.call(
            model="qwen-turbo",
            messages=messages,
            tools=tools,
            result_format="message",
        )
        return response
    except Exception as e:
        print(e)
        return None


def case4():
    while True:
        content = input("请输入告警信息（输入 back 返回菜单，exit 退出程序）: ")
        cmd = content.strip().lower()
        if cmd == "exit":
            print("再见")
            sys.exit(0)
        if cmd == "back":
            return
        messages = [
            {
                "role": Role.SYSTEM,
                "content": "我是运维分析师，用户会告诉我们告警内容。我会基于告警内容，判断当前的异常情况（告警对象、异常模式）",
            },
            {"role": Role.USER, "content": content},
        ]
        response = get_case4_response(messages)
        response_message = response.output.choices[0].message
        tool_call = response_message.tool_calls[0]
        print("tool_call=", tool_call)
        if tool_call:
            print("tool_call.function=", tool_call["function"])
            # pick(obj, key, default) 的第三个参数是「缺 key 时的默认值」，不是嵌套字段名。
            # 应先取出 function 子对象，再取 name / arguments。
            call_function = pick(tool_call, "function") or {}
            print("call_function=", call_function)
            function_call_name = pick(call_function, "name")
            function_call_args_raw = pick(call_function, "arguments", "{}")
            print("function_call_name=", function_call_name)
            print("function_call_args_raw=", function_call_args_raw)

            if isinstance(function_call_args_raw, str):
                arguments_json = (
                    json.loads(function_call_args_raw)
                    if function_call_args_raw.strip()
                    else {}
                )
            elif isinstance(function_call_args_raw, Mapping):
                arguments_json = dict(function_call_args_raw)
            else:
                arguments_json = {}

            if not function_call_name or function_call_name not in globals():
                print("未知或缺失的函数名:", function_call_name)
                continue
            function = globals()[function_call_name]
            tool_response = function(**arguments_json)

            # 百炼/OpenAI 规范：tool 消息必须紧跟在「带 tool_calls 的 assistant」之后。
            # 之前只 append 了 tool，没有 append  assistant，会报：
            # messages with role "tool" must be a response to a preceeding message with "tool_calls"
            assistant_content = pick(response_message, "content")
            if assistant_content is None:
                assistant_content = ""
            tool_calls_list = pick(response_message, "tool_calls")
            messages.append(
                {
                    "role": Role.ASSISTANT,
                    "content": assistant_content,
                    "tool_calls": tool_calls_list,
                }
            )
            tool_info = {
                "role": "tool",
                "tool_call_id": pick(tool_call, "id"),
                "name": function_call_name,
                "content": tool_response,
            }
            print("tool_info=", tool_info)
            messages.append(tool_info)

            final_response = get_case4_response(messages)
            print("final_response=", final_response)

        # print("response=", response)


def _print_main_menu() -> None:
    print(
        """
========== 百炼 API 练习 ==========
  1 / case1   舆情分析（单次示例）
  2 / case2   天气预报（Function Call）
  3 / case3   表格图片多模态提取
  4 / case4   运维告警 + Tools

主菜单输入 exit 退出程序。
进入某一 case 后输入 back 可返回本菜单。
===================================="""
    )


def main() -> None:
    while True:
        _print_main_menu()
        choice = input("请选择 case: ").strip().lower()
        if choice in ("exit", "quit", "q"):
            print("再见")
            break
        if choice in ("1", "case1"):
            case1()
        elif choice in ("2", "case2"):
            case2()
        elif choice in ("3", "case3"):
            case3()
        elif choice in ("4", "case4"):
            case4()
        else:
            print("无效输入，请输入 1～4、case1～case4 或 exit。\n")


if __name__ == "__main__":
    main()
