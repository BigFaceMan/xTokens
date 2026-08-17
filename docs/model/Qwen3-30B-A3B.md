# Qwen3-30B-A3B 模型简介

> 模型权重已下载至服务器 `/model`（16 个 safetensors 分片，共约 57GB，bf16）。
> 官方仓库：[Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)，技术报告：[arXiv:2505.09388](https://arxiv.org/abs/2505.09388)。

Qwen3-30B-A3B 是 Qwen3 系列中的 MoE（Mixture-of-Experts）模型：总参数量 30.5B，但每个 token 仅激活 3.3B 参数，兼顾了容量与推理效率。模型原生支持**思考模式（thinking）与非思考模式（non-thinking）无缝切换**，可在同一套权重内完成复杂推理（数学/代码）与高效通用对话两种场景。

---

## 1. 模型架构

### 1.1 基本信息

| 项目 | 值 |
| ------ | ----- |
| 类型 | Causal Language Model（Decoder-only） |
| 模型类型标识 | `qwen3_moe`（`Qwen3MoeForCausalLM`） |
| 总参数量 | 30.532B（非 Embedding 29.9B） |
| 激活参数量（每 token） | 3.353B |
| 层数 | 48 |
| hidden size | 2048 |
| Attention | GQA：32 个 Q 头 / 4 个 KV 头，head_dim = 128（Q 头总输出维度 4096，与 hidden_size 2048 不同，见 1.2） |
| MoE | 128 个专家，每 token 激活 8 个，**无共享专家** |
| 词汇表 | 151936 |
| 原生上下文长度 | 32,768 tokens |
| YaRN 扩展上下文 | 131,072 tokens |
| 精度 | bfloat16 |
| 激活函数 | SiLU（SwiGLU） |

### 1.2 逐层结构

每一层由 **Attention + MoE MLP** 组成（前后各接 RMSNorm）：

```text
input_layernorm → self_attn → post_attention_layernorm → mlp(router + experts)
```

#### Attention（GQA）

- `q_proj`: 2048 → 4096（32 头 × 128），`k_proj` / `v_proj`: 2048 → 512（4 头 × 128），`o_proj`: 4096 → 2048

> **注意 head_dim 的计算**：head_dim = **q_proj 输出维度 / Q 头数** = 4096 / 32 = **128**，而非 `hidden_size / Q 头数`（2048 / 32 = 64）。Qwen3-30B-A3B 在 Attention 中额外做了一次升维：hidden_size 是 2048，但 `q_proj` 将输出扩到 4096，使每头维度为 128。因此本模型 `hidden_size ≠ q_proj 输出维度`，不能用 `hidden_size / num_attention_heads` 直接计算 head_dim。

- Q/K 输出后各接逐头 RMSNorm（`q_norm` / `k_norm`，维度 128）
- RoPE：`rope_theta = 1e6`，`head_dim = 128`，原生位置编码支持 32K 上下文
- 单层 Attention 参数量 ≈ 18.9M

#### MoE MLP

- 路由：`mlp.gate`，形状 [128, 2048]（每 token 取 top-8，`norm_topk_prob=true` 对 top-k 概率做归一化）
- 每个专家为 SwiGLU FFN：`gate_proj`/`up_proj` 2048→768，`down_proj` 768→2048
- **无 shared expert**（本机权重中不存在 `shared_expert` 张量；config 中的 `intermediate_size: 6144` 未被实际使用）
- 单层专家参数量：128 × 3.15M ≈ 604M，其中每 token 仅激活 8/128
- 路由负载均衡：`router_aux_loss_coef = 0.001`（训练期 auxiliary loss）

#### 其余

- `embed_tokens` / `lm_head`：151936 × 2048 ≈ 311M × 2，`tie_word_embeddings = false`（不共享）
- 最终 `model.norm`（RMSNorm，2048）

### 1.3 参数分布

| 模块 | 参数量 | 占比 | 每 token 是否激活 |
| ------ | -------- | ------ | ------------------ |
| Embedding + LM Head | 0.62B | 2.0% | 是 |
| Attention（48 层合计） | 0.91B | 3.0% | 是 |
| Router（48 层合计） | 12.6M | ~0.04% | 是 |
| MoE 专家（128 × 48） | 28.99B | 94.9% | 仅 8/128 ≈ 1.81B |
| **合计** | **30.532B** | 100% | **激活 3.353B** |

> 注意：94.9% 的参数在专家中，这是 MoE 显存占用与带宽优化的关键 —— 权重必须全部常驻显存，但推理算力/带宽只消耗激活部分。

### 1.4 思考模式

- `enable_thinking=True`（默认）：输出 `<think>...</think>` 推理块 + 最终回答；采样建议 `T=0.6, top_p=0.95, top_k=20`，**不要用 greedy**
- `enable_thinking=False`：直接回答，对齐 Qwen2.5-Instruct 行为；建议 `T=0.7, top_p=0.8`
- 多轮对话中可用 `/think`、`/no_think` 软切换（仅当 `enable_thinking=True` 时生效）
- 历史消息中不应包含 thinking 内容（chat template 已处理，实现方需遵循）

---

## 2. 显存分析

### 2.1 权重显存

权重显存 = 参数量 × 每参数字节数（无量化时不含任何压缩）：

| 精度 | 每参数 | 权重显存 |
| ------ | -------- | ---------- |
| bf16 / fp16（官方权重） | 2 B | **61.1 GB** |
| fp8（W8A8 量化） | 1 B | 30.5 GB |
| INT4（GPTQ/AWQ） | 0.5 B | 15.3 GB |

> 本机 `/model` 磁盘占用约 57GB，与 61.1GB 的差异来自分片对齐与索引文件计算口径，实际加载为 61.1GB。

### 2.2 KV Cache

KV cache 大小 = `2（K+V）× layers × kv_heads × head_dim × bytes`：

```text
2 × 48 × 4 × 128 × 2 B = 98,304 B ≈ 96 KB / token
```

| 上下文长度（单请求） | KV cache |
| --------------------- | ---------- |
| 4,096 | 384 MB |
| 8,192 | 768 MB |
| 32,768（原生上限） | 3.07 GB |
| 131,072（YaRN） | 12.6 GB |

KV cache 与并发请求数成正比：如 32 并发 × 32K 上下文 ≈ 98GB，远大于权重本身，长上下文场景下 KV 是显存主导项。

### 2.3 激活显存（Activations）

- **Decode（逐 token 生成）**：激活显存很小（约 B × hidden × 若干 buffer，MB 级），主要由权重带宽而非显存决定
- **Prefill（长输入）**：需使用 FlashAttention 类 kernel 避免物化 B×H×L×L 的注意力分数矩阵（L=32K 时该矩阵单层即达 ~67GB）；开启 recompute 后激活可压到几 GB 级
- MoE 相比同规模 Dense 模型激活更低（每层仅 8 个专家的中间结果）

### 2.4 部署显存估算公式

```text
总显存 ≈ 权重 + KV(96KB × 总上下文tokens) + 激活 + 10% 冗余
```

#### 本服务器环境：10 × NVIDIA RTX A6000（48GB）

| 方案 | 权重 | 剩余给 KV+激活 | 可支撑规模（示例） |
| ------ | ------ | ---------------- | ------------------- |
| bf16，TP=2 | 61.1GB | ~35GB | 2 卡内跑 ~365K tokens 总 KV（如 8 并发 × 32K） |
| fp8，单卡 | 30.5GB | ~17.5GB | 单卡 ~180K tokens 总 KV |
| INT4，单卡 | 15.3GB | ~32GB | 单卡 ~330K tokens 总 KV，权重省下的空间给长上下文 |

> 权重读取带宽估算（decode 吞吐上限）：激活参数 3.353B × 2B ≈ **6.7GB/token**。A6000 显存带宽 768GB/s，单卡纯带宽上限 ≈ 768 / 6.7 ≈ **114 tok/s**（未计 expert 并行、量化与计算重叠等优化；TP 切分专家权重后单卡读取量降低，吞吐可进一步提升）。

---

## 3. 训练方式

### 3.1 预训练（Pretraining）

存在独立 Base 检查点（`Qwen/Qwen3-30B-A3B-Base`），预训练为**从零开始**，语料规模约 **36T tokens**、覆盖 **119 种语言**（约为 Qwen2.5 语料 18T 的两倍）。数据构建手段：

- Web 爬取 + PDF 类文档：用 Qwen2.5-VL 微调模型做 PDF 文本抽取，再用 Qwen2.5 提升抽取文本质量
- 合成数据：Qwen2.5-Math 生成数学内容、Qwen2.5-Coder 生成代码数据，并用 rule-based 校验过滤

预训练分三阶段（对全部 Qwen3 模型统一执行）：

| 阶段 | 内容 | Token 规模 |
| ------ | ------ | ----------- |
| S1 通用阶段 | 大规模通用语料（多语言、多领域） | > 30T |
| S2 推理阶段（退火） | 高质量 STEM/推理/代码为主的高质量语料 | ~5T |
| S3 长上下文阶段 | 长文本语料，将上下文从 32K 训练扩展到更长 | 数百亿 |

### 3.2 后训练（Post-training）

Instruct 版本在 Base 之上经过多阶段后训练：

1. **长 CoT 冷启动（Long-CoT Cold Start）**：构造高质量长思维链 SFT 数据，激活模型 reasoning 能力
2. **推理强化学习（Reasoning RL）**：GRPO 在线 RL，配合基于规则的奖励（数学、代码等可自动判分任务）
3. **思考/非思考模式融合（Thinking Mode Fusion, SFT）**：引入非思考模式的通用指令数据，实现单模型内两种模式切换
4. **通用 RL（General RL）**：用 LLM-as-Judge / Arena 式奖励优化对齐、指令遵循与 agent 能力

#### Strong-to-Weak Distillation（30B-A3B 特有）

- Qwen3 后训练对轻量模型采用强到弱蒸馏管线，**Qwen3-30B-A3B 是其中唯一的 MoE 模型**
- 以 Qwen3-32B 或 Qwen3-235B-A22B 为 teacher，结合 **off-policy（教师离线生成数据）与 on-policy（在线采样 + 教师打分）** 两种知识迁移方式，将大模型的推理与对齐能力压缩到 3.3B 激活参数上

### 3.3 MoE 训练要点

- 训练期使用全局 batch 负载均衡 loss（config `router_aux_loss_coef = 0.001`）抑制路由坍缩，保证 128 个专家被均匀使用
- 推理阶段 `norm_topk_prob = true`，top-8 路由概率归一化后加权求和专家输出
- 技术报告未披露 30B-A3B 的具体训练集群与 FLOPs 细节，整体训练范式与 Qwen3 系列一致（预训练三阶段 + SFT/DPO/RLVR 后训练）

---

### 参考

- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [Qwen3: Think Deeper, Act Faster（官方博客）](https://qwenlm.github.io/blog/qwen3/)
- [Qwen/Qwen3-30B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B)
