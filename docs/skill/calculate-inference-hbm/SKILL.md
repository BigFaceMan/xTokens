---
name: calculate-inference-hbm
description: Calculate, profile, and explain per-GPU HBM usage, KV-cache capacity, and active-sequence limits for LLM inference deployments with or without KV cache. Use when sizing GPU deployments, estimating or measuring maximum concurrency or batch size, checking whether a model/workload fits, validating Hugging Face device placement, or diagnosing HBM-related OOM and latency risk from hardware, model, parallelism, scheduler, and workload parameters.
---

# Calculate Inference HBM

为 LLM 推理部署输出可审计的逐 GPU HBM breakdown 和并发限制。始终区分配置上限、实测 OOM
上限、安全建议和满足服务 SLO 的并发。

## 收集输入

优先读取用户给出的模型配置、部署配置、executor 实现和 profiler 结果，不要仅凭参数量猜测。

| 类别 | 必需参数 |
| --- | --- |
| 硬件 | 物理 GPU ID/UUID、型号、逐卡 total/free HBM 和现有进程 |
| 权重 | 总/逐卡实际权重 bytes、dtype、量化元数据 |
| 模型 | layers、KV/query heads、head dimension、hidden size、vocabulary size |
| 部署 | `CUDA_VISIBLE_DEVICES`、TP/PP/EP 或 Accelerate layer sharding、offload 策略 |
| 执行 | 是否保存 KV、attention backend、activation/logits/KV dtype、logits 范围 |
| Workload | `L_pad`、prompt/输出/预留长度、长度分布、目标并发 |
| 调度 | `max_num_seqs`、token budget、KV block size、chunked prefill |
| 实测/SLO | warmup 后 free HBM、峰值、TTFT、TPOT、吞吐量、P99 latency |

缺少参数时继续计算可靠部分，并列出假设和影响方向。逐卡计算后取最紧约束；禁止简单累加多卡
HBM。

## 读取参考资料

读取 [references/hbm-analysis.md](references/hbm-analysis.md) 的第 1～3、7～8 节。然后按路径读取：

- 完整上下文重算或 `use_cache=False`：读取第 4 节。
- prefill 后持久保存 K/V：读取第 5 节。
- Qwen3-30B-A3B：额外读取第 6 节。

## 执行 GPU Preflight

本地 GPU 和模型可用且任务允许测量时，先执行只读检查：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

显式设置目标 `CUDA_VISIBLE_DEVICES`。不要终止其他任务，也不要在显存占用不稳定的 GPU 上宣称
得到有效容量。记录 visible index 与物理 GPU ID 的映射。

## 有 KV cache

1. 使用每卡实际本地 layers 和 KV heads；不要无条件除以 TP size。
2. 将 paged cache 长度向上对齐到 block size。
3. 从逐卡 HBM 扣除权重、runtime、prefill/decode 峰值、workspace 和安全余量。
4. 对每张卡运行 `scripts/calculate_capacity.py with-kv` 并取最小值。
5. 再应用 scheduler、token budget 和 SLO 限制。

```bash
python scripts/calculate_capacity.py with-kv \
  --hbm-gib 48 --weight-gib 30.5 --runtime-gib 2 \
  --activation-gib 2 --margin-gib 2 \
  --local-layers 48 --local-kv-heads 4 --head-dim 128 --kv-bytes 2 \
  --context-tokens 4096 --block-size 16
```

## 无 KV cache：优先实测

### 1. 确认执行语义

检查 executor 和加载后的模型，而不是只看磁盘 config：

- 是否每步对 `L_pad=max(L_j)` 的完整上下文 forward；
- 是否显式 `use_cache=False`；
- 是否在 lm_head 后才切片，导致完整 `[B,L_pad,V]` logits storage 仍存活；
- 加载后的 `model.config._attn_implementation`；
- `model.hf_device_map`、逐卡参数 bytes，以及是否出现 CPU/disk/meta offload。

### 2. 运行 profiler

只在目标 GPU 空闲且允许逼近 OOM 边界时运行。以下命令以 Skill 目录为当前目录；否则根据
`SKILL.md` 解析脚本绝对路径。

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/profile_no_kv.py \
  --model /model \
  --seq-len 4096 \
  --max-batch 16 \
  --dtype bfloat16 \
  --device-map auto \
  --input-mode random \
  --local-files-only \
  --output /tmp/no-kv-profile.json
```

Profiler 必须在加载后确认纯 GPU placement。返回 `invalid_placement` 时停止：CPU/meta offload
结果不能代表指定 GPU 部署。若搜索 ceiling 未 OOM，只能报告成功 lower bound；提高
`--max-batch` 后重新测量，才能得到相邻成功/OOM 的硬边界。

随机 token 只能改善 MoE 路由覆盖，最终还要用真实 prompt、padding 和长度分布复测。

### 3. 计算实测容量

```bash
python scripts/calculate_capacity.py without-kv-profile \
  --profile-json /tmp/no-kv-profile.json \
  --margin-gib 2 \
  --scheduler-max-seqs 15
```

安全余量从 warmup 后每张卡的真实 free HBM 扣除。使用成功 case 中最大的
`peak_delta/batch` 作为逐卡保守斜率，且建议值不得超过已经成功实测的 batch。
若输出 `recommended_safe_limit_sampled=false`，再以该 batch 直接复测；即使内存通常随 batch
单调增长，也不要把范围内插值描述成该点已采样。

### 4. 无法实测时使用静态估算

只有在没有可用 GPU、无法加载模型或用户只要求初步规划时，才运行
`scripts/calculate_capacity.py without-kv`。明确标注 activation multiplier、attention 和
logits 生命周期是假设，结果不是安全最大并发。

```bash
python scripts/calculate_capacity.py without-kv \
  --hbm-gib 48 --weight-gib 30.5 --runtime-gib 2 --workspace-gib 1 \
  --margin-gib 2 --seq-len 4096 --hidden-size 2048 \
  --activation-bytes 2 --activation-multiplier 8 \
  --vocab-size 151936 --logits-bytes 2 --attention efficient
```

仅当 executor 确实只计算最后位置 logits 时添加 `--last-token-logits`。eager attention 必须提供
`--num-query-heads`。

## 输出四层并发结论

始终输出：

| 字段 | 含义 |
| --- | --- |
| `configured_limit` | scheduler 的 `max_num_seqs` |
| `measured_hard_limit` | 与 OOM 相邻的最大成功 batch；未测到相邻 OOM 时为 null |
| `recommended_safe_limit` | 扣除真实 free HBM 余量且不超过成功范围的建议值 |
| `recommended_safe_limit_sampled` | 建议 batch 是否作为独立 case 被直接测量 |
| `slo_limit` | 满足 TTFT/TPOT/吞吐/P99 的并发；未做服务 benchmark 时为 null |

同时给出输入与假设、逐卡 placement/HBM、`L_pad`、logits storage、batch/step time、aggregate
output tokens/s、置信度和限制。区分 `max_num_seqs` 与固定 batch：前者只是活跃序列调度上限。

若任一 GPU 静态显存不够，报告 `does_not_fit`。若 placement 无效，报告
`invalid_placement`。若 no-KV 未实测 OOM 边界，禁止使用“精确最大并发”。最终生产配置不得高于
`recommended_safe_limit`，并应继续由服务 SLO 收缩。
