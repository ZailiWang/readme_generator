# Remote Execution Agent 使用说明

本文档面向项目使用者，说明 remote_exec_tool.py 在整个 README 生成流水线中的角色、触发方式、输入要求和结果读取方法。

## 1. 这个 Agent 是做什么的

Remote Execution Agent 是流程中的 remote_execution 阶段，用于把上一阶段生成的内容发送到远端执行服务做校验。

你可以把它理解为：

1. 读取当前 run 的上下文（模型列表、README 内容、配置）。
2. 组装远端请求 payload。
3. 调用远端接口执行校验。
4. 把结果写回全局内存，供后续阶段使用。

## 2. 它在整体流程中的位置

默认阶段顺序如下：

1. input_parser
2. model_search
3. readme_generation
4. remote_execution
5. post_remote_refine
6. github_pr

其中 remote_execution 是 readme_generation 之后、post_remote_refine 之前的关键校验步骤。

## 3. 实际执行方式（重要）

虽然项目里有 RemoteExecutionCrew，但在主流程执行 remote_execution 阶段时，代码走的是“直接执行”路径，而不是依赖 Agent LLM 推理。

也就是说在当前实现里：

1. 主流程先注入 RemoteExecutionTool.global_memory。
2. 然后直接调用 RemoteExecutionTool 中的方法完成远端请求。
3. 结果写回 GLOBAL_MEMORY。

这种方式可以降低 LLM 不可用时该阶段失败的风险。

## 4. 运行前你需要准备什么

Remote Execution Agent 的输入都来自 GLOBAL_MEMORY。最关键的是以下几类：

1. 远端连接配置
   ssh_config（request_url 或 hostname/port/endpoint）

2. 模式与请求结构
   remote_payload（generation_mode、metadata、source_urls 等）

3. 模型信息
   model_list、model_id_list、model_url_list、github_url

4. 待校验内容
   legacy 模式通常依赖 family_md 与 family_index_js
   url_source 模式通常依赖 source_urls

5. 结果容器
   execution_result、executed_command、fail_reason_list（会被阶段更新）

## 5. 两种运行模式

### 5.1 legacy

适合本地流程已经拿到 family_md 和 family_index_js 的场景。

行为：

1. 按模型逐个发送请求。
2. 每个模型保存一条执行结果。

### 5.2 url_source

适合只提供 URL，让远端自行拉取 md/js 源文件的场景。

行为：

1. 一次请求携带 model_list 和 source_urls。
2. 结果按单条写入（index=0）。

## 6. 如何触发这个 Agent

你通常不需要单独调用 remote_exec_tool.py。推荐通过流程入口触发：

1. 使用 API 运行全流程或按阶段运行。
2. 确保 stages 包含 remote_execution（或不跳过该阶段）。

最常见接口：

1. POST /api/run
2. POST /api/run/stream
3. POST /api/start + POST /api/next（逐阶段）

## 7. 最小请求示例（触发到 remote_execution）

下面示例通过后端 API 跑全流程，核心是带上 ssh_config 和 mode 对应输入。

```json
{
  "generation_mode": "reference",
  "input_text": "Generate README for selected models",
  "model_list": ["qwen3-1.7b"],
  "github_url": [""],
  "ssh_config": {
    "request_url": "http://127.0.0.1:8000/legacy_test",
    "request_stream": false,
    "request_payload": {}
  },
  "stages": [
    "input_parser",
    "model_search",
    "readme_generation",
    "remote_execution"
  ]
}
```

如果你使用 url_source 模式，请改为提供 source_md_url/source_js_url（或 github_*_folder_url）并让 generation_mode 命中 url_source 分支。

## 8. 这个阶段的产出是什么

remote_execution 阶段完成后，会把以下结果写回 GLOBAL_MEMORY：

1. executed_command
   每个模型对应一次远端执行命令标识。

2. execution_result
   远端返回结果（字符串化存储）。

3. fail_reason_list
   对应索引的失败原因（成功时为空）。

主流程返回中还会包含一个简化统计：

1. mode
2. stored_count
3. failed_count

## 9. 常见问题

### 9.1 阶段卡住或无响应

检查 remote_exec_tool.py 中 validate_payload 是否还保留 pdb.set_trace()。在非交互环境会导致阻塞。

### 9.2 报 Missing request_url or hostname

说明 ssh_config 不完整。至少提供 request_url，或 hostname + port + endpoint。

### 9.3 legacy 模式直接失败

通常是 family_md、family_index_js、model_id_list 缺失。先确认 readme_generation 阶段已产出 family_* 内容。

### 9.4 url_source 返回空结果

优先检查 source_urls、request_stream，以及远端服务是否按预期返回 JSON/SSE。

## 10. 相关代码

1. 远端执行实现
   src/readme_generator/tools/remote_exec_tool.py

2. RemoteExecution Crew 定义
   src/readme_generator/crews/remote_execution_crew.py

3. 主流程阶段调度与 direct 执行入口
   src/readme_generator/crew.py

4. API 入口（/api/run, /api/run/stream）
   src/main.py
