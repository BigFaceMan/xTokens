# xTokens Serve API 设计

## 1. 背景

xTokens 需要提供面向用户的 Web Server，用于接收 OpenAI-compatible HTTP 请求，并将请求转换为推理引擎可处理的内部请求。

Web Server 不应直接参与模型执行、请求调度、continuous batching、KV cache 管理或 CUDA 资源管理。服务层和推理引擎之间需要建立稳定且足够小的抽象边界，使以下目标能够同时成立：

- HTTP/OpenAI 协议可以独立演进；
- 推理引擎可以在没有 Web Server 的情况下独立测试和使用；
- Web Server 可以通过 Fake Engine 在无 GPU 环境中完整测试；
- 同一套服务逻辑可以连接 in-process Engine 或独立 Engine 进程；
- 客户端断开、请求取消和服务关闭能够正确传递到推理引擎；
- 后续可以增加 gRPC、Python SDK 或其他协议，而不修改 Engine Core。

本设计参考 vLLM 0.26.0 的以下调用链：

```text
FastAPI route
  -> OpenAI serving adapter
  -> EngineClient
  -> AsyncLLM
  -> EngineCoreClient
  -> EngineCore
```

相关参考实现：

| 职责 | vLLM 文件 |
| --- | --- |
| HTTP 路由 | `vllm/entrypoints/openai/chat_completion/api_router.py` |
| OpenAI 协议转换 | `vllm/entrypoints/openai/chat_completion/serving.py` |
| Engine 抽象接口 | `vllm/engine/protocol.py` |
| 异步推理客户端 | `vllm/v1/engine/async_llm.py` |
| In-process/Multiprocess 切换 | `vllm/v1/engine/core_client.py` |
| App 与 Engine 生命周期组装 | `vllm/entrypoints/openai/api_server.py` |

xTokens 借鉴其中的薄路由、异步流式接口、依赖注入、取消传播和可替换 Engine Client，但不直接复制 vLLM 已经较宽的 `EngineClient` 接口。

---

## 2. 设计目标

### 2.1 功能目标

第一阶段提供以下 HTTP API：

```text
GET  /live
GET  /ready
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

生成接口需要同时支持：

- 流式响应；
- 非流式响应；
- OpenAI-compatible request/response；
- request ID；
- usage 统计；
- stop、temperature、top-p、top-k 和 max tokens 等基础采样参数；
- 客户端断开后的请求取消；
- Engine 错误到 HTTP/OpenAI 错误的稳定映射。

### 2.2 架构目标

- Web 层只处理 HTTP transport concern；
- OpenAI 协议转换与 FastAPI 路由分离；
- Engine Core 不依赖 FastAPI、Pydantic HTTP DTO 或 OpenAI 类型；
- Server 仅依赖稳定的推理接口，不依赖 scheduler、model runner 或 KV cache；
- Local Engine 和 RPC Engine 实现相同接口；
- 数据面和控制面接口分离；
- 首版即定义清晰的流式输出、backpressure、取消和关闭语义。

### 2.3 非目标

首版不计划实现：

- 多模型动态装载；
- LoRA 在线加载；
- Prefix cache 管理 API；
- profiling 管理 API；
- elastic data parallel；
- 权重在线更新；
- 多节点 Engine transport；
- 完整复制所有 OpenAI API；
- 在 Engine 接口中暴露内部 scheduler、tokenizer、renderer 或 model config 对象。

这些能力可以后续通过独立控制面接口增加，不应提前扩展核心推理接口。

---

## 3. 总体架构

```text
                         HTTP boundary
                              │
                              ▼
┌─────────────────────────────────────────────────┐
│ x_tokens/server                                 │
│ FastAPI、鉴权、校验、request ID、SSE、断连检测  │
└───────────────────────┬─────────────────────────┘
                        │ OpenAI request/response
                        ▼
┌─────────────────────────────────────────────────┐
│ x_tokens/serving                                │
│ OpenAI -> canonical request                     │
│ chat template、模型解析、usage、finish reason   │
└───────────────────────┬─────────────────────────┘
                        │ GenerateRequest/EngineEvent
                        ▼
┌─────────────────────────────────────────────────┐
│ x_tokens/engine/api.py                          │
│ 最小、稳定、与 FastAPI/OpenAI 无关的接口        │
└───────────────────────┬─────────────────────────┘
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
 LocalEngineClient              RpcEngineClient
 in-process                     multiprocess/remote
           │                         │
           └────────────┬────────────┘
                        ▼
               scheduler/model/KV cache
```

核心依赖方向为：

```text
server -> serving -> engine.api
                       ▲
                       │
                engine implementation
```

禁止形成以下依赖：

```text
engine -> server
engine -> FastAPI
engine -> OpenAI Pydantic models
scheduler -> HTTP Request
model runner -> StreamingResponse
```

`server/cli.py` 是 composition root，允许同时依赖 Server 和具体 Engine 实现，负责根据配置组装系统。除 composition root 外，Server 代码只依赖 `engine.api` 中定义的抽象。

---

## 4. 模块规划

```text
x_tokens/
├── engine/
│   ├── __init__.py
│   ├── api.py                 # 稳定的 Engine 数据面接口
│   ├── types.py               # GenerateRequest/EngineEvent
│   ├── async_engine.py        # 实际异步推理实现
│   ├── scheduler.py
│   └── clients/
│       ├── __init__.py
│       ├── local.py           # LocalEngineClient
│       └── rpc.py             # 后续添加
│
├── serving/
│   ├── __init__.py
│   ├── generation.py          # 协议无关的 generation service
│   ├── models.py              # served model registry
│   └── renderer.py            # chat template/tokenizer frontend
│
├── server/
│   ├── __init__.py
│   ├── app.py                 # create_app()
│   ├── config.py
│   ├── cli.py                 # composition root
│   ├── errors.py
│   └── openai/
│       ├── __init__.py
│       ├── protocol.py        # Pydantic HTTP DTO
│       ├── routes.py          # 薄路由
│       ├── adapter.py         # OpenAI 与 canonical DTO 转换
│       └── sse.py
│
└── __main__.py

tests/
├── engine/
├── serving/
└── server/
    ├── fake_engine.py
    ├── test_models.py
    ├── test_completions.py
    ├── test_chat_completions.py
    ├── test_streaming.py
    └── test_cancellation.py
```

模块职责如下：

| 模块 | 职责 |
| --- | --- |
| `server` | HTTP、SSE、鉴权、中间件、HTTP 错误、断连检测 |
| `server.openai` | OpenAI HTTP schema 和 wire format |
| `serving` | OpenAI 到内部请求的转换、prompt rendering、响应聚合 |
| `engine.api` | Server 与 Engine 之间的稳定接口 |
| `engine.clients.local` | 将稳定接口适配到同进程 Async Engine |
| `engine.clients.rpc` | 将稳定接口适配到独立 Engine 进程 |
| Engine Core | 调度、batching、采样、模型执行、KV cache |

---

## 5. Engine 接口

### 5.1 最小数据面接口

初期使用 `typing.Protocol` 定义接口，避免基类携带实现细节：

```python
from collections.abc import AsyncIterator
from typing import Protocol


class InferenceClient(Protocol):
    def generate(
        self,
        request: "GenerateRequest",
    ) -> AsyncIterator["EngineEvent"]:
        """Submit one request and stream its outputs."""
        ...

    async def abort(self, request_id: str) -> None:
        """Abort an admitted or running request."""
        ...

    async def health(self) -> "EngineHealth":
        """Return engine readiness and liveness information."""
        ...

    async def close(self) -> None:
        """Release engine-side resources."""
        ...
```

接口约束：

- `generate()` 返回异步迭代器；
- 每个请求必须有全局唯一的 `request_id`；
- `abort()` 必须幂等；
- `close()` 必须可重复调用；
- Engine 不接受 FastAPI `Request`；
- Engine 不返回 `JSONResponse`、`StreamingResponse` 或 OpenAI DTO；
- Engine 错误通过内部异常或 `ErrorEvent` 表达；
- transport 切换不得改变调用方语义。

### 5.2 Canonical request

内部请求使用与 HTTP 协议无关的数据结构：

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    request_id: str
    model: str
    prompt: str | tuple[int, ...]
    sampling: SamplingParams
    priority: int = 0
```

后续可以按需增加以下字段，但应避免直接透传任意 OpenAI request body：

- `seed`；
- `stop_token_ids`；
- `ignore_eos`；
- `logprobs`；
- `trace_context`；
- multimodal input reference。

参数应在进入 Engine 前完成协议校验和 canonicalization。

### 5.3 Engine event

Engine 通过事件流返回增量结果：

```python
@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    token_id: int
    text: str


@dataclass(frozen=True, slots=True)
class FinishedEvent:
    request_id: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    request_id: str
    message: str
    retryable: bool = False


EngineEvent = TokenEvent | FinishedEvent | ErrorEvent
```

事件流约束：

- `TokenEvent` 可以出现零次或多次；
- 正常完成必须产生且只产生一个 `FinishedEvent`；
- 终止事件之后不得再产生 token；
- 请求失败可以抛出内部异常或产生一个终止性的 `ErrorEvent`，具体实现需在编码阶段统一；
- `finish_reason` 使用内部稳定枚举，OpenAI adapter 负责映射为 `stop`、`length`、`content_filter` 等外部值；
- usage 由 Engine 返回的实际 token 数计算，不依赖 Web Server 对字符串的估算。

### 5.4 数据面与控制面分离

不将所有 Engine 功能放入 `InferenceClient`。未来需要时拆分为：

```python
class InferenceClient(Protocol):
    # generate / abort
    ...


class EngineStatus(Protocol):
    # health / model information / capacity
    ...


class EngineAdmin(Protocol):
    # sleep / wake / reset cache / profile / load LoRA
    ...
```

OpenAI Web Server 默认只持有 `InferenceClient` 和只读的 `EngineStatus`，管理 API 单独鉴权和部署。

---

## 6. Serving 层

Serving 层位于 HTTP schema 和 Engine API 之间，负责业务协议转换，不处理 HTTP transport。

主要职责：

- 校验请求中的 model；
- 将 OpenAI sampling 参数转换为内部 `SamplingParams`；
- 将 chat messages 渲染为 prompt；
- 创建 request ID；
- 调用 `InferenceClient.generate()`；
- 将 `EngineEvent` 转换成 completion/chat chunk；
- 聚合非流式响应；
- 生成 usage 和 finish reason；
- 保证未完成流被关闭时调用 `abort()`。

建议的主要 service：

```text
CompletionService
ChatCompletionService
ModelService
```

Router 不直接操作 tokenizer，也不直接调用 scheduler。

### 6.1 Prompt rendering

Chat template 和 tokenization 是容易导致 Server/Engine 重新耦合的部分。为此定义独立 frontend abstraction：

```python
from collections.abc import Sequence
from typing import Protocol


class PromptRenderer(Protocol):
    async def render_chat(
        self,
        model: str,
        messages: Sequence["ChatMessage"],
    ) -> "RenderedPrompt":
        ...
```

MVP 阶段可以返回字符串：

```python
@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
```

当 tokenizer 与 scheduler 接入后，可以演进为：

```python
@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    token_ids: tuple[int, ...]
    text: str | None = None
```

`ChatCompletionService` 的组合关系为：

```text
ChatCompletionService
├── PromptRenderer
├── ModelRegistry
└── InferenceClient
```

Engine API 不暴露 renderer、tokenizer 或完整 model config 对象。

---

## 7. Web Server 层

### 7.1 职责

Web Server 只处理 transport concern：

- HTTP 请求解析；
- Pydantic schema 校验；
- API key；
- request ID header；
- JSON/SSE 序列化；
- client disconnect 检测；
- HTTP 状态码和错误格式；
- trace header 提取；
- access log、metrics 和 CORS；
- readiness/liveness endpoint。

Web Server 不处理：

- scheduling；
- continuous batching；
- KV cache；
- CUDA；
- 模型执行；
- Engine token queue；
- 采样算法。

### 7.2 薄 Router

Router 只负责从 app state/dependency injection 获取 service，并选择 JSON 或 SSE 响应：

```python
@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
) -> Response:
    service = get_chat_service(raw_request)
    result = await service.create(request)

    if isinstance(result, ChatCompletionResponse):
        return JSONResponse(result.model_dump())

    return StreamingResponse(
        result,
        media_type="text/event-stream",
    )
```

OpenAI adapter 负责将内部结果序列化为 wire format。Router 不包含 prompt rendering、sampling conversion 或 response aggregation 逻辑。

### 7.3 SSE 格式

流式响应采用：

```text
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

响应示例：

```text
data: {"id":"cmpl-...","object":"text_completion","choices":[...]}

data: {"id":"cmpl-...","object":"text_completion","choices":[...]}

data: [DONE]

```

要求：

- 每条 event 以 `data:` 开始；
- event 之间使用两个换行符分隔；
- 正常结束必须发送 `data: [DONE]`；
- `stream_options.include_usage=true` 时发送 usage chunk；
- Engine 失败且响应尚未开始时返回标准 JSON error；
- 响应已经开始后发生失败时，通过流式 error event 结束，并记录服务端错误；
- SSE serializer 不应依赖 Engine Core 类型之外的实现细节。

---

## 8. OpenAI-compatible API

### 8.1 Models

```text
GET /v1/models
```

返回当前配置的 served model。首版可以只支持一个模型，但返回格式保持 OpenAI-compatible。

### 8.2 Completions

```text
POST /v1/completions
```

基础请求：

```json
{
  "model": "model-name",
  "prompt": "hello",
  "max_tokens": 32,
  "temperature": 1.0,
  "top_p": 1.0,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

### 8.3 Chat completions

```text
POST /v1/chat/completions
```

基础请求：

```json
{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "max_completion_tokens": 32,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

首版支持基础 text content。tool call、多模态 content 和 reasoning content 后续扩展。

### 8.4 Usage

最终 usage 格式：

```json
{
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 32,
    "total_tokens": 42
  }
}
```

usage 应以 Engine 返回的实际 token 数为准。非流式响应直接携带 usage；流式响应仅在请求启用 `include_usage` 时发送 usage chunk。

### 8.5 错误格式

外部错误保持稳定格式：

```json
{
  "error": {
    "message": "The requested model does not exist",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

初步错误映射：

| 内部错误 | HTTP 状态码 | OpenAI error type |
| --- | ---: | --- |
| Schema validation | 400/422 | `invalid_request_error` |
| Model not found | 404 | `invalid_request_error` |
| Invalid sampling params | 400 | `invalid_request_error` |
| Engine overloaded | 429/503 | `server_error` |
| Engine unavailable | 503 | `server_error` |
| Engine internal error | 500 | `server_error` |
| Authentication failure | 401 | `authentication_error` |

内部异常信息不得直接向用户泄露堆栈、文件路径或敏感配置。

---

## 9. 请求流程

### 9.1 流式 completion

```text
Client
  -> POST /v1/completions
  -> FastAPI validates HTTP DTO
  -> CompletionService validates model and sampling params
  -> build GenerateRequest
  -> InferenceClient.generate()
  -> Engine admits request
  -> scheduler/model emits EngineEvent
  -> CompletionService converts event to OpenAI chunk
  -> SSE serializer writes data frames
  -> usage chunk
  -> [DONE]
```

### 9.2 非流式 completion

```text
Client
  -> POST /v1/completions stream=false
  -> same request conversion
  -> consume complete EngineEvent stream
  -> aggregate text, finish reason and usage
  -> return one JSON response
```

非流式请求仍使用相同的 Engine streaming API，避免 Engine 同时维护两套执行路径。

### 9.3 Chat completion

```text
messages
  -> PromptRenderer
  -> RenderedPrompt
  -> GenerateRequest
  -> InferenceClient.generate()
  -> EngineEvent
  -> chat delta/full response
```

---

## 10. 取消和断连语义

推理请求可能长时间占用 GPU。HTTP 客户端断开后，Server 必须停止对应 Engine 请求。

建议由 Serving 层提供统一包装：

```python
async def generate_with_abort(
    client: InferenceClient,
    request: GenerateRequest,
) -> AsyncIterator[EngineEvent]:
    completed = False
    try:
        async for event in client.generate(request):
            yield event
            if isinstance(event, FinishedEvent):
                completed = True
    finally:
        if not completed:
            await client.abort(request.request_id)
```

必须处理：

- 浏览器或 SDK 主动断开；
- Uvicorn response task 被取消；
- SSE generator 被关闭或垃圾回收；
- response 序列化抛出异常；
- Engine stream 抛出异常；
- Server shutdown；
- 客户端消费速度过慢。

取消语义要求：

- `abort(request_id)` 幂等；
- 请求尚未 admission 时也可以取消；
- scheduler queue 中的请求可移除；
- running request 在下一个安全点停止；
- 已完成请求调用 `abort()` 不报错；
- Server 不等待一个无法结束的输出流后才执行 abort；
- 请求资源、输出队列和 KV cache 最终都必须释放。

---

## 11. Backpressure

Engine 输出速度可能高于 HTTP 客户端消费速度。不能让一个慢客户端阻塞全局 Engine output loop。

建议：

- 每个请求维护独立的有界输出队列；
- Engine 全局 output handler 只进行轻量分发；
- 队列大小通过配置控制；
- 队列长期满时执行明确策略，而不是无限增长；
- 首版可以取消持续阻塞的慢请求；
- 记录 queue wait、blocked duration 和 cancellation reason；
- terminal event 只能写入一次；
- request cleanup 必须从正常完成、错误、取消三条路径统一进入。

首版不建议静默丢弃 token，因为这会产生内容不完整但表面成功的响应。

---

## 12. 生命周期与依赖注入

### 12.1 App factory

不使用隐式全局 Engine。通过 app factory 和 lifespan 注入依赖：

```python
def create_app(
    config: ServerConfig,
    engine_factory: EngineFactory,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with engine_factory.create() as engine:
            renderer = build_renderer(config)
            app.state.completion_service = CompletionService(
                engine=engine,
            )
            app.state.chat_service = ChatCompletionService(
                engine=engine,
                renderer=renderer,
            )
            yield

    return FastAPI(lifespan=lifespan)
```

测试可以直接注入 `FakeInferenceClient`，无需加载模型或 GPU。

### 12.2 启动顺序

```text
load and validate config
  -> construct Engine Client
  -> load model/start Engine worker
  -> Engine health check
  -> construct renderer and services
  -> mark ready
  -> start accepting inference requests
```

Engine 未 ready 前：

- `/live` 可以返回成功；
- `/ready` 返回 503；
- 推理请求不应进入 scheduler。

### 12.3 关闭顺序

```text
mark not ready
  -> stop accepting new inference requests
  -> drain or abort in-flight requests
  -> close serving streams
  -> close Engine Client
  -> stop scheduler and worker processes
  -> release model and KV cache
```

关闭策略应可配置：

- `drain`：等待请求完成，达到 timeout 后 abort；
- `abort`：立即停止所有请求。

### 12.4 健康检查

```text
GET /live
```

只检查 Web 进程和 event loop 是否存活，不执行 GPU 操作。

```text
GET /ready
```

检查：

- Engine 已启动；
- 模型已加载；
- Engine 未进入 fatal error 状态；
- Server 当前接受新请求。

---

## 13. Local 与 RPC Engine

“解耦”分为代码层和进程层两个阶段。

### 13.1 第一阶段：代码层解耦

```text
FastAPI
  -> InferenceClient Protocol
  -> LocalEngineClient
  -> AsyncEngine
```

特点：

- Web 和 Engine 位于同一进程；
- 无 RPC 序列化开销；
- 首先稳定接口和行为；
- 可以使用 Fake Engine 测试全部 HTTP 逻辑；
- 实现和调试成本较低。

### 13.2 第二阶段：进程层解耦

```text
Web process
  -> RpcEngineClient
  -> UDS/ZMQ
  -> Engine worker process
  -> AsyncEngine
```

Server 代码不因 transport 改变：

```python
engine: InferenceClient

if config.engine_mode == "local":
    engine = LocalEngineClient(...)
else:
    engine = RpcEngineClient(...)
```

### 13.3 Transport message

进程分离后至少需要：

```text
SubmitRequest
AbortRequest
TokenOutput
FinishedOutput
EngineError
HealthRequest
HealthResponse
ShutdownRequest
```

每条消息至少包含：

```text
protocol_version
message_type
request_id
payload
```

Transport 约束：

- 不跨进程传输 Python exception 对象；
- 不跨进程传输 FastAPI/Pydantic HTTP DTO；
- 不跨进程传输 tokenizer、renderer 或 model 对象；
- 消息协议需要显式版本；
- transport failure 与 request failure 分开表达；
- Web 进程退出后，Engine 能够清理该 client 的未完成请求；
- Engine worker 异常退出时，所有 pending stream 必须收到终止错误。

---

## 14. 部署约束

### 14.1 Local Engine 模式

Local 模式下一个 Web worker 会加载一个 Engine 和一份模型。禁止直接配置多个 Uvicorn worker：

```text
1 Web worker : 1 Engine
```

否则类似以下命令可能为每个 worker 重复加载模型并造成 GPU OOM：

```bash
uvicorn x_tokens.server.app:app --workers 4
```

Local 模式应强制或校验：

```text
workers = 1
```

### 14.2 RPC Engine 模式

Remote/RPC 模式可以使用：

```text
N Web workers : 1 or N Engine workers
```

多个 Web worker 通过 `RpcEngineClient` 共享 Engine 服务，由 Engine 层统一调度。

### 14.3 配置边界

Server 配置示例：

```text
host
port
api_key
served_model_name
engine_mode
request_timeout
shutdown_timeout
max_request_body_size
cors
access_log
```

Engine 配置示例：

```text
model_path
device
dtype
tensor_parallel_size
max_model_len
max_num_seqs
kv_cache_size
scheduler policy
```

Server 不解析和修改 Engine 算法配置；composition root 负责将各自配置传给对应组件。

---

## 15. 可观测性

Web 和 Engine 使用相同 `request_id` 关联日志、trace 和 metrics。

建议首批指标：

- request count；
- HTTP error count；
- active requests；
- request queue time；
- time to first token；
- inter-token latency；
- end-to-end latency；
- prompt/completion token count；
- abort count；
- client disconnect count；
- Engine error count；
- output queue blocked duration；
- readiness state。

日志约束：

- 默认不记录完整 prompt 和 generated text；
- 不记录 API key；
- 错误日志包含 request ID；
- Engine fatal error 与单请求错误使用不同级别和错误类型；
- HTTP access log 与 Engine request log 可以通过 request ID 关联。

---

## 16. 测试策略

### 16.1 Fake Engine

Server 测试不依赖 GPU：

```python
class FakeInferenceClient:
    async def generate(self, request):
        yield TokenEvent(request.request_id, 1, "Hello")
        yield TokenEvent(request.request_id, 2, " world")
        yield FinishedEvent(request.request_id, "stop", 3, 2)

    async def abort(self, request_id):
        ...
```

Fake Engine 应支持配置：

- 固定 token 输出；
- token 间延迟；
- admission failure；
- 流中错误；
- 永不结束的请求；
- 记录 abort 调用；
- health 状态切换。

### 16.2 首批测试

1. `/v1/models` 返回 configured model；
2. completion 非流式响应正确；
3. chat completion 非流式响应正确；
4. completion SSE chunk 格式正确；
5. chat SSE delta 格式正确；
6. 流最后产生 `[DONE]`；
7. `include_usage` 返回实际 token 统计；
8. HTTP 断开触发 `abort()`；
9. Engine error 映射为 OpenAI error；
10. 不存在的 model 返回 404；
11. 非法 sampling params 返回 400/422；
12. application shutdown 关闭 Engine；
13. Engine 未 ready 时 `/ready` 返回 503；
14. 非流式请求复用 streaming Engine API；
15. `abort()` 重复调用不失败；
16. 慢客户端不会阻塞其他请求；
17. Engine 包不 import `fastapi` 或 `x_tokens.server`。

### 16.3 架构测试

通过静态测试保护依赖方向：

```text
x_tokens.engine    must not import x_tokens.server
x_tokens.engine    must not import fastapi
x_tokens.engine    must not import OpenAI HTTP DTO
x_tokens.server    must not import scheduler/model runner/KV cache
```

### 16.4 Benchmark 兼容性

使用仓库中的 `benchmarks/test_serve` 做端到端验证。该 benchmark 已按外部 OpenAI-compatible Server 设计，不会导入或启动 Engine，适合验证真正的服务边界。

重点验证：

- `/v1/models` 自动发现模型；
- `/v1/completions`；
- `/v1/chat/completions`；
- SSE parsing；
- TTFT、ITL、E2EL；
- `stream_options.include_usage`；
- request rate 和 max concurrency。

---

## 17. 实施计划

### Phase 1：接口和 Server 骨架

建议 commit：

```text
feat(server): add decoupled inference API
```

范围：

- 添加 FastAPI、Pydantic、Uvicorn 测试/运行依赖；
- 定义 `InferenceClient`；
- 定义 canonical request/event types；
- 实现 Fake Engine；
- 实现 `create_app()` 和 lifespan；
- 实现 `/live`、`/ready`、`/v1/models`；
- 实现 `/v1/completions`；
- 实现 SSE serializer；
- 添加 Server 单元测试。

验收标准：

- 不加载 GPU 即可运行全部 Server 测试；
- Engine 包不依赖 FastAPI；
- completion stream 可以被 `benchmarks/test_serve` 消费；
- 客户端取消会触发 Fake Engine 的 `abort()`。

### Phase 2：Chat completions

建议 commit：

```text
feat(server): add OpenAI chat completions
```

范围：

- Chat HTTP schema；
- `PromptRenderer`；
- chat template；
- chat stream/non-stream adapter；
- finish reason 和 usage；
- benchmark compatibility。

### Phase 3：接入真实 Engine

建议 commit：

```text
feat(engine): connect async engine to server
```

范围：

- `LocalEngineClient`；
- scheduler request submission；
- per-request output queue；
- output handler；
- abort；
- health；
- graceful shutdown；
- Engine integration tests。

### Phase 4：进程分离

建议 commit：

```text
feat(engine): add multiprocess engine client
```

范围：

- Engine worker；
- versioned transport protocol；
- `RpcEngineClient`；
- process monitoring；
- pending request failure propagation；
- independent Web/Engine lifecycle；
- multi-Web-worker deployment tests。

---

## 18. 关键决策

### 决策 1：Engine 使用异步事件流作为唯一生成接口

理由：

- 自然支持 token streaming；
- 非流式响应可以在 Serving 层聚合；
- 避免维护两套 Engine 执行路径；
- 便于传播取消和错误。

### 决策 2：OpenAI DTO 不进入 Engine

理由：

- Engine 可以被 Python SDK、gRPC 或离线任务复用；
- OpenAI API 演进不会影响 scheduler；
- Engine 单元测试不需要 Web 依赖。

### 决策 3：Renderer 独立于 Engine Client

理由：

- 避免通过 Engine Client 暴露 tokenizer 和 model config；
- chat template 是 serving/frontend concern；
- 后续 renderer 可以独立部署或替换。

### 决策 4：首版先做代码解耦，再做进程解耦

理由：

- 先稳定接口和行为；
- 避免过早引入 RPC、进程监控和序列化复杂度；
- Local 和 Fake Client 可以快速验证设计；
- 稳定接口后增加 RPC 不需要重写 Server。

### 决策 5：数据面和控制面分离

理由：

- 保持 `InferenceClient` 足够小；
- 避免复制 vLLM 大型 EngineClient 的演进结果；
- 管理能力可以拥有独立权限和部署策略。

---

## 19. 最终约束总结

xTokens Serve API 采用：

```text
FastAPI Router
    ↓
OpenAI Adapter
    ↓
Generation Service
    ↓
InferenceClient
    ↓
Local/RPC Engine
```

实现时必须长期保持以下约束：

1. Web 层不 import scheduler、model runner 或 KV cache；
2. Engine 层不 import FastAPI 或 OpenAI HTTP schema；
3. Local 与 RPC Engine 实现同一个最小推理接口；
4. 非流式响应复用 Engine 的异步事件流；
5. 请求断连和流关闭最终必须触发 Engine cleanup；
6. Local Engine 模式只允许一个 Web worker；
7. Engine 内部对象不通过接口直接暴露给 Server；
8. 新增管理能力优先进入独立控制面接口，而不是扩展数据面接口。
