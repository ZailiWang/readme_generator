# readme_generator 框架说明

## 项目简介

readme_generator 是一个自动化 README 生成与远程验证的流水线框架，支持多模型、多阶段处理，适用于大规模模型仓库的文档生成、校验和 PR 自动化。

核心特性：
- 多阶段流水线（input_parser、model_search、readme_generation、remote_execution、post_remote_refine、github_pr）
- 支持本地与远程（SSE/RESTful）模型校验
- 可扩展的 CrewAI agent 体系
- 丰富的输入模式（文本、URL、GitHub 仓库等）

## 依赖环境

- Python >= 3.8
- 推荐使用虚拟环境（venv/conda/pipenv）
- 依赖包见 requirements.txt
- 需安装 crewai 及相关依赖

```bash
# 创建虚拟环境（可选）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

如需本地开发 crewai，可参考其官方文档：https://github.com/joaomdmoura/crewAI

## 快速上手（Quick Start）

1. 启动后端服务（FastAPI）

```bash
cd readme_generator
python src/main.py
```

2. 通过 API 触发端到端流程

推荐使用 POST /api/run 或 /api/run/stream，输入示例：

```json
{
  "generation_mode": "reference",
  "input_text": "Generate README for selected models",
  "model_list": ["qwen3-1.7b"],
  "github_url": [""],
  "ssh_config": {
    "request_url": "http://127.0.0.1:8000/legacy_test",
    "request_stream": false
  },
  "stages": [
    "input_parser",
    "model_search",
    "readme_generation",
    "remote_execution"
  ]
}
```

3. 查看结果
- 直接返回 JSON，包含各阶段产出与统计信息
- 详细执行日志见终端输出

## 主要阶段说明

1. **input_parser**：解析输入文本或 URL，抽取关键信息
2. **model_search**：检索目标模型及其元数据
3. **readme_generation**：生成 README.md 及相关代码片段
4. **remote_execution**：将生成内容发送到远端服务校验
5. **post_remote_refine**：根据远端反馈自动修正/精炼文档
6. **github_pr**：自动生成/更新 PR，推送到目标仓库

可通过 stages 字段灵活选择/跳过阶段

## 关键配置说明

- **ssh_config**：远端服务连接参数（request_url、hostname、port、endpoint 等）
- **remote_payload**：远端请求补充信息（generation_mode、source_urls、metadata 等）
- **model_list/model_id_list**：待处理模型名/ID
- **family_md/family_index_js**：生成的 README/代码内容

## 典型输入输出

- 输入：见 Quick Start 示例
- 输出：每阶段产出均写入全局内存，最终通过 API 返回

## 常见问题与排查

1. **依赖缺失/安装失败**
   - 检查 Python 版本与 requirements.txt
2. **远端请求失败**
   - 检查 ssh_config 配置与远端服务可用性
3. **阶段卡住**
   - 检查是否有断点调试语句（如 pdb.set_trace）
4. **内容缺失/流程异常**
   - 检查前置阶段产出是否完整（如 family_md、model_list 等）

## 目录结构简述

- src/
  - main.py：API 服务入口
  - readme_generator/：核心流水线与 agent 实现
    - tools/：各阶段工具实现
    - crews/：CrewAI agent 及任务编排
    - ...
- requirements.txt：依赖列表
- frontend/：可选前端页面

## 参考/致谢
- CrewAI: https://github.com/joaomdmoura/crewAI
- FastAPI: https://fastapi.tiangolo.com/

---
如需详细开发文档或二次开发建议，请查阅 src/readme_generator/tools/README.md 及各阶段源码。
