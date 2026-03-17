## 项目简介

这是一个基于 **ReAct（Reason + Act）** 思路实现的轻量 Demo：让大模型通过“思考 → 调用工具 → 观察结果 → 再思考”的循环，完成**文件检索、Excel 数据分析、文档问答、邮件生成**等任务。

- **Demo 仓库**：[`https://github.com/Cyning12/auto-gpt-work-demo`](https://github.com/Cyning12/auto-gpt-work-demo)
- **学习文章**：`data/2026-03-17-react-agent-learning.md`

---

## 快速开始

### 1）创建虚拟环境（推荐）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2）配置环境变量（使用 SiliconFlow）

本项目已将 `ChatModelFactory` **默认改为走 SiliconFlow 的 OpenAI 兼容接口**。

如果你还没有账号，可以通过我的邀请链接注册：[`https://cloud.siliconflow.cn/i/2Ouh86B4`](https://cloud.siliconflow.cn/i/2Ouh86B4)

在项目根目录创建 `.env`（不要提交到 git）：

```
SILICONFLOW_API_KEY=sk-xxxx

# 可选：自定义模型名与 API Base
# SILICONFLOW_MODEL=Pro/moonshotai/Kimi-K2.5
# SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1
```

> 说明：本项目默认**只需要** `SILICONFLOW_API_KEY` 即可运行。

### 3）安装依赖

```bash
pip install -r requirements.txt
```

### 4）运行

```bash
python main.py
```

然后按提示输入问题，例如：

- 9月份的销售额是多少
- 销售总额最大的产品是什么
- 对比8月和9月销售情况，写一份报告

---

## 数据说明

默认演示数据在 `data/`：

- `data/2023年8月-9月销售记录.xlsx`
- `data/供应商名录.xlsx`
- `data/供应商资格要求.pdf`

---

## 常见问题

### Q1：读取 Excel/PDF 报缺依赖怎么办？

常见依赖：

- `.xlsx`：`openpyxl`
- `.pdf`：`PyMuPDF`

可直接安装：

```bash
pip install openpyxl PyMuPDF
```

---

## GitHub Actions（`.github/workflows/`）说明

`.github/workflows/` 用于配置 **GitHub Actions**：当你 push 代码时自动触发 CI/CD 工作流（例如：跑测试、构建、部署到服务器等）。

本仓库的 demo 版本**不需要**任何自动部署流程；如果你看到历史遗留的工作流文件，直接删除即可，避免误触发。
