# xTokens 开发与协作规范

## 背景
高效、简洁、易用的LLM推理系统



## Commit 规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 格式：

```
<type>(<scope>): <subject>
```

### 合法 type

| type | 用途 | 示例 |
|------|------|------|
| `feat` | 新增功能 | `feat(engine): support continuous batching` |
| `fix` | 修复 bug | `fix(kernel): fix attention mask out-of-bounds` |
| `docs` | 仅文档变更 | `docs: update README architecture` |
| `refactor` | 代码重构，不改变外部行为 | `refactor(sampler): extract base sampler class` |
| `test` | 增加/修改测试 | `test(tokenizer): add BPE merge cases` |
| `perf` | 性能优化（推理引擎核心关注点） | `perf(kernel): use CUDA Graph to cut launch overhead` |
| `build` | 构建系统或外部依赖变更 | `build: upgrade torch to 2.6` |
| `ci` | CI 配置与脚本变更 | `ci: add GPU unit test job` |
| `style` | 格式调整，不影响逻辑 | `style: unify header include order` |
| `chore` | 杂项维护（非 src/test 的改动） | `chore: update .gitignore` |
| `revert` | 回滚某次提交 | `revert: roll back continuous batching` |

### type 怎么选（重点：perf vs refactor vs feat）

按**这次改动的主要目的**选 type，而不是按 diff 长什么样：

- 目的是**让代码更好读/好维护**，行为不变 → `refactor`
- 目的是**更快/更省内存**，行为不变 → `perf`（哪怕 diff 看起来像重构）
- 目的是**新增用户可见能力**，行为变化 → `feat`（提速不算新功能，新增开关/API 才算）

重叠时按主要意图选一个，次要效果写进 body。例如：重构调度器顺带提速 → `refactor(engine): refactor scheduler`，body 里注明吞吐提升。

### scope（可选，建议使用）

模块名放在括号内，便于按模块过滤历史：

- `engine`：调度/执行主流程
- `kernel`：CUDA kernel
- `model`：模型结构与权重加载
- `tokenizer`：分词器
- `sampler`：采样逻辑
- `kv-cache`：KV cache 管理
- `quant`：量化
- `server`：HTTP/API 层
- `cli`：命令行入口
- `docs` / `ci` / `build`

### subject 要求

- **全部用英文（纯 ASCII）**：subject 和 body 都避免非 ASCII 字符，防止终端乱码（中文说明写进 PR 描述或代码注释）
- 用祈使句（如 `add`、`fix`），不要用过去式（`added`、`fixed`）
- 首字母小写，不超过 50 字符，结尾不加句号

### 完整 commit message 结构

```text
<type>(<scope>): <subject>

<body>        # 解释为什么这么做、怎么做的
<footer>      # 破坏性变更 / 关联 issue
```

- **破坏性变更**：type 后加 `!`（如 `feat!: ...`），并在 footer 中写 `BREAKING CHANGE: ...`
- 一次 commit 只做一件事，避免混入无关改动

### 常用命令

```bash
git commit -m "feat(engine): support continuous batching"
git commit -m "fix(kernel): fix attention mask out-of-bounds" -m "pass seq_len explicitly instead of implicit inference"
```

## 分支约定

- `main`：稳定分支，始终可构建
- `feat/<描述>` / `fix/<描述>`：功能与修复分支

## 代码风格

- C++（kernel 层）：遵循项目 clang-format 配置
- Python（engine/server 层）：遵循 PEP 8，用 ruff 检查
- 新增代码需附带对应 `test/` 用例
