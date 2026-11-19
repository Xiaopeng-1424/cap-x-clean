# Molmo2 在昇腾 vLLM 上的 Pointing 错误调查

## 调查背景

第一阶段调查见 [molmo.md](molmo.md)。该阶段已经证明：

- 坐标错误存在于模型原始输出中
- 客户端 parser、x/y 顺序、0–1000 尺度和 overlay 不是主因
- 服务端输入与官方 Hugging Face processor 基本一致
- 官方 Transformers 后端能在同一 PNG 上正确 pointing
- 远端 vLLM 0.20.2 后端产生严重空间误差

随后确认 Molmo2 实际部署在昇腾 NPU 上。这一信息显著改变了根因优先级。

## 结论摘要

昇腾与 NVIDIA 的差异很可能与本问题直接相关，但原因不是“硬件坐标系不同”，而是不同硬件使用了不同的 attention 后端：

- NVIDIA vLLM 可以使用支持 multimodal Prefix-LM 的 TritonAttention 或 FlexAttention。
- 昇腾使用 `vllm-ascend` 自定义 NPU attention。
- 在与远端 core `vLLM 0.20.2` 对应的 `vllm-ascend 0.20.2rc1` 中，
  model runner 虽然计算了图像 prefix range，
  但 Ascend attention 没有消费该 range。
- 实际 attention metadata 被初始化为普通因果注意力，NPU kernel 走 `sparse_mode=3` causal 路径。

这意味着 Molmo2 需要的“图像 token 内双向、文本 token 因果”的混合注意力没有真正生效。

该缺口能解释当前现象：模型仍能识别物体和生成合理描述，但 pointing、counting 和精细空间定位显著退化。

## 不是硬件坐标系问题

同一个 `(x,y)` 输出在任何硬件上的语义都相同：

```text
pixel_x = x / 1000 * image_width
pixel_y = y / 1000 * image_height
```

NPU 不会把 x/y 自动交换，也不会定义另一套 0–1000 坐标系。

如果两个后端实现数学等价，BF16 数值差异通常只会造成小幅 logits 或 token 变化，不应稳定地产生几十到上百像素的结构性空间畸变。

本问题更像是模型执行语义不等价，而非一般浮点噪声。

## Molmo2 为什么需要 Prefix-LM

Molmo2 的文本 transformer 不是对所有输入统一使用普通 causal mask。

官方 Transformers 实现会根据 `token_type_ids` 构造额外 mask：
当 query token 和 key/value token 都属于图像块时，允许双向注意力。

官方源码中的核心逻辑为：

```python
is_image_block = (
    token_type_ids[batch_idx, q_idx] == 1
) & (
    token_type_ids_at_kv_idx == 1
)

# Bidirectional attention for image tokens.
return is_image_block & is_image_block
```

随后将该 mask 与 causal mask 做 OR：

```python
mask_kwargs["or_mask_function"] = token_type_ids_mask_function(...)
causal_mask_mapping = create_causal_mask(**mask_kwargs)
```

参考：

- [官方 Molmo2 `token_type_ids_mask_function`](https://huggingface.co/allenai/Molmo2-8B/blob/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b/modeling_molmo2.py#L1086-L1111)
- [官方模型在 prefill 中加入图像双向 mask](https://huggingface.co/allenai/Molmo2-8B/blob/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b/modeling_molmo2.py#L1515-L1544)

这类 mask 的语义是：

```text
图像块内部：双向
图像/文本之外：保持自回归因果约束
```

如果把整段图像 token 当成普通 causal sequence，较早的图像 token 无法访问后续图像 token，经过语言 transformer 后的视觉表示将不再等价于训练时的表示。

pointing 对视觉 token 的空间关系特别敏感，因此比一般描述任务更容易暴露该问题。

## 上游 vLLM 对 Prefix-LM 的要求

Molmo2 的 vLLM 支持 PR 明确说明：

> Molmo2 uses a prefix language modeling formulation and requires
> TritonAttention-based implementation for correct and efficient execution.

参考 [vLLM PR #30997](https://github.com/vllm-project/vllm/pull/30997)。

NVIDIA 常用的上游 vLLM 后端有完整实现。

### TritonAttention

Triton 后端明确声明支持 multimodal prefix：

- [Triton backend `supports_mm_prefix()`](https://github.com/vllm-project/vllm/blob/bc150f50299199599673614f80d12a196f377655/vllm/v1/attention/backends/triton_attn.py#L360-L366)

它会从 metadata 读取 `mm_prefix_range_tensor`，并传入 attention kernel：

- [读取并传入 mm-prefix range](https://github.com/vllm-project/vllm/blob/bc150f50299199599673614f80d12a196f377655/vllm/v1/attention/backends/triton_attn.py#L608-L636)

### FlexAttention

FlexAttention 也明确声明支持，并构造：

```text
causal mask OR image-prefix bidirectional mask
```

参考：

- [Flex backend `supports_mm_prefix()`](https://github.com/vllm-project/vllm/blob/bc150f50299199599673614f80d12a196f377655/vllm/v1/attention/backends/flex_attention.py#L93-L105)
- [构造 prefix-LM mask](https://github.com/vllm-project/vllm/blob/bc150f50299199599673614f80d12a196f377655/vllm/v1/attention/backends/flex_attention.py#L519-L568)

上游 TritonAttention 的 prefix 实现也曾出现过“能运行、描述看似正常，
但 hidden states 不正确”的 bug，并由
[PR #30974](https://github.com/vllm-project/vllm/pull/30974) 修复。
这进一步说明 Prefix-LM mask 的细节会直接影响模型正确性。

## `vllm-ascend 0.20.2rc1` 源码审计

远端 `/version` 只返回 vLLM core `0.20.2`。它不显示插件版本；与之对应的公开 Ascend tag 是 `v0.20.2rc1`，commit：

```text
367b8e62da799870a7476ce34f5f7658589a8aad
```

### 1. Model runner 知道模型需要 mm-prefix

初始化 attention backend 时传入：

```python
use_mm_prefix=(
    self.model_config is not None
    and self.model_config.is_mm_prefix_lm
)
```

参考：

- [model_runner_v1.py#L342-L350](https://github.com/vllm-project/vllm-ascend/blob/367b8e62da799870a7476ce34f5f7658589a8aad/vllm_ascend/worker/model_runner_v1.py#L342-L350)

### 2. Ascend backend selector 没有根据 `use_mm_prefix` 选择专用实现

`get_attn_backend_cls()` 只根据 MLA、sparse、compress 等字段选择 backend，没有检查 `attn_selector_config.use_mm_prefix`：

- [platform.py#L606-L632](https://github.com/vllm-project/vllm-ascend/blob/367b8e62da799870a7476ce34f5f7658589a8aad/vllm_ascend/platform.py#L606-L632)

最终仍选择普通 `AscendAttentionBackend`。

### 3. Common attention metadata 被固定为 causal

构造 metadata 时明确传入：

```python
causal=True
```

参考：

- [model_runner_v1.py#L2856-L2878](https://github.com/vllm-project/vllm-ascend/blob/367b8e62da799870a7476ce34f5f7658589a8aad/vllm_ascend/worker/model_runner_v1.py#L2856-L2878)

Molmo2 需要的不是全局 non-causal，而是“全局 causal + 图像块内部双向”。单一 `causal` 布尔值不足以表达这一需求。

### 4. Model runner 计算了 `mm_prefix_range`

代码会遍历 multimodal features，提取图像 embedding range：

```python
img_doc_range = pos_info.extract_embeds_range()
image_doc_ranges.extend(img_doc_range)
...
_metadata.mm_prefix_range = req_doc_ranges
```

参考：

- [model_runner_v1.py#L3011-L3029](https://github.com/vllm-project/vllm-ascend/blob/367b8e62da799870a7476ce34f5f7658589a8aad/vllm_ascend/worker/model_runner_v1.py#L3011-L3029)

但这里使用了：

```python
# type: ignore[attr-defined]
```

因为 Ascend attention metadata 类型本身没有声明该字段。

### 5. Ascend attention 完全没有消费该 range

对 tag `v0.20.2rc1` 执行源码搜索：

```bash
git grep -l 'mm_prefix_range' v0.20.2rc1 -- vllm_ascend
```

结果只有：

```text
vllm_ascend/worker/model_runner_v1.py
```

没有任何 attention backend 或 NPU kernel 引用该字段。

当前 `vllm-ascend` main 分支也仍然只有 model runner 一个引用文件。

换言之，`mm_prefix_range` 在 Ascend 路径中是写入后无人读取的 metadata。

### 6. NPU attention 最终走普通 causal kernel

当 `attn_metadata.causal` 为 true 且没有 sliding window 时，代码调用：

```python
torch_npu.npu_fused_infer_attention_score(
    ...,
    atten_mask=attn_metadata.attn_mask,
    sparse_mode=3,
)
```

参考：

- [attention_v1.py#L1109-L1157](https://github.com/vllm-project/vllm-ascend/blob/367b8e62da799870a7476ce34f5f7658589a8aad/vllm_ascend/attention/attention_v1.py#L1109-L1157)

`AttentionMaskBuilder` 对 generation runner 返回普通 split-fuse 上三角 mask，
也没有根据 image range 修改：

- [attention_mask.py#L75-L79](https://github.com/vllm-project/vllm-ascend/blob/367b8e62da799870a7476ce34f5f7658589a8aad/vllm_ascend/attention/attention_mask.py#L75-L79)

## 为什么描述正常、pointing 错误

缺失图像块双向 attention 不一定让模型完全失效：

- vision encoder 在进入语言模型前已经提取了局部视觉特征
- assistant token 在生成时仍可以读取所有先前图像 token
- 大尺度物体、颜色和场景类别仍可能被识别

但语言 transformer 内部的图像 token 表示不再与训练时一致：

- 较早的图像 token 看不到较后的图像 token
- 多 crop、空间位置和对象关系不能按训练时方式融合
- 精细 counting、grounding 和坐标生成更容易退化

这与现有实验一致：

- 圆的颜色和大致位置可被描述
- x 有时接近正确
- y 和跨区域位置出现严重、非统一压缩
- 小把手的语义描述正确，但坐标落到木架、碗或柜体下方
- 不能用固定 offset 修复

## vLLM-Ascend 已有准确率报告

`vllm-ascend` 的 Molmo2 支持 issue 仍为 open：

- [Issue #7318: Verify / Support Molmo2-8B](https://github.com/vllm-project/vllm-ascend/issues/7318)

其中有直接报告：

```text
vllm-ascend 0.17.0rc1 + CountBench: 72
Transformers + CountBench:             94
```

这说明 Molmo2 在 Ascend vLLM 上出现过可复现的准确率大幅下降。

对应的 [PR #8228](https://github.com/vllm-project/vllm-ascend/pull/8228)
截至调查时仍未合并。该 PR：

- 增加 Molmo2 文档和 E2E 配置
- 修复一个 slot-mapping runtime fallback
- 使用更新的 vLLM 0.25.1 和 `enforce_eager=True`
- 在 MMMU 上报告约 `0.5344`

但该 PR 没有实现 `mm_prefix_range` 的 NPU attention 消费，也没有测试 pointing/grounding。因此：

> MMMU 通过不能证明 Molmo2 pointing 在 Ascend 上正确。

一般 VQA benchmark 对 image-token attention 误差可能不如坐标生成敏感。

## 与 BF16/FP32 调查的关系

第一阶段发现：

- 官方 checkpoint 配置为 FP32
- 官方 vision 和 pooling attention 设置 `float32_attention=true`
- 远端服务以 BF16 执行
- vLLM Molmo2 自定义 vision attention 未完整保留官方 FP32 upcast 语义

这些仍然可能放大误差，但在确认 Ascend Prefix-LM 缺口后，原因优先级应调整为：

1. Ascend attention 未应用 image-prefix bidirectional mask
2. BF16 与官方 FP32/`float32_attention` 差异
3. vLLM 自定义 Molmo2 vision/weight mapping 差异
4. 仓库重复 prompt 前缀
5. 其他数值或 kernel 精度差异

因此，仅把 `--dtype` 改为 `float32` 很可能仍无法修复缺失的 attention mask。

## 如何最终确认

至少执行下面两个交叉实验。

### 实验 A：同一昇腾设备运行官方 Transformers

保持：

- 同一模型目录
- 同一 PNG
- 同一 prompt
- greedy decoding
- 尽可能相同 dtype

若官方 Transformers 在昇腾上能正确输出五圆和把手坐标，则可排除昇腾硬件本身，定位到 `vllm-ascend`。

### 实验 B：同一模型在 NVIDIA vLLM 上运行

要求启动日志确认选择了支持 mm-prefix 的：

```text
TRITON_ATTN
```

或 FlexAttention。

若 NVIDIA vLLM BF16 正确，而 Ascend vLLM BF16 错误，则基本坐实 Ascend attention backend。

### 推荐完整矩阵

| 平台 | 后端 | 精度 | 预期用途 |
| --- | --- | --- | --- |
| 昇腾 | 官方 Transformers | BF16/FP32 | 排除硬件本身 |
| 昇腾 | vLLM-Ascend | BF16 | 当前失败基线 |
| 昇腾 | vLLM-Ascend + mm-prefix 修复 | BF16 | 验证修复 |
| NVIDIA | 官方 Transformers | FP32/auto | 已知正确基准 |
| NVIDIA | vLLM Triton/Flex | BF16 | 验证上游 Prefix-LM 路径 |

每个组合都应先跑五圆静态校准，再跑 LIBERO 把手。不要先用大型通用 VQA benchmark 代替 pointing correctness test。

## 部署环境信息采集

远端 `/version` 只显示 vLLM core，
不显示 Ascend 插件、`CANN` 或 torch-npu 版本。部署机上应记录：

```bash
python - <<'PY'
from importlib.metadata import version

for name in ["vllm", "vllm-ascend", "torch", "torch-npu"]:
    try:
        print(name, version(name))
    except Exception as exc:
        print(name, exc)
PY

npu-smi info
```

同时保存：

- 完整启动命令
- NPU 型号，例如 910B、A2、A3 或 310P
- `CANN` 版本
- `--dtype`
- `--enforce-eager` 是否启用
- tensor parallel size
- attention backend 启动日志
- 是否使用本地修改的 `vllm-ascend`

如果 core 是 `0.20.2`，还需确认插件是否为 `0.20.2rc1`，不能只依据 `/version` 假设二者一致。

## `enforce_eager` 的作用边界

PR #8228 的准确率测试使用：

```text
enforce_eager=True
```

这可以绕开部分 ACL graph、Dynamo 或 runtime compatibility 问题，因此值得作为诊断项测试。

但 eager mode 不会自动生成缺失的 Prefix-LM hybrid mask。
若 attention backend 仍不读取 `mm_prefix_range`，
仅设置 `--enforce-eager` 不足以保证 pointing 正确。

## 临时生产方案

在 Ascend pointing correctness 通过前，优先级如下：

1. 使用已验证正确的官方 Transformers 服务。
2. 使用支持 Triton/Flex Prefix-LM 的 NVIDIA vLLM 服务。
3. 将 Ascend vLLM 仅用于已有 correctness benchmark 覆盖的非 grounding 请求。
4. 不要在机器人控制中对 Ascend vLLM 坐标增加经验 offset。

如果必须继续使用 Ascend vLLM，应在服务启动后运行静态校准 gate；不通过就拒绝注册为 point detector。

## Ascend 修复方向

正确修复不是简单设置：

```text
is_mm_prefix_lm = false
```

这只会明确关闭模型需要的语义，可作为诊断对照，但不是解决方案。

Ascend backend 需要在 prefill 阶段实现：

```text
final_mask = causal_mask OR image_block_bidirectional_mask
```

实现时应考虑：

- 每个 request 可能有多个图像 range
- packed batch 中每个 request 的局部索引
- chunked prefill
- prefix cache
- eager 与 ACL graph 两条路径
- 单图、多图和视频
- NPU attention operator 对 arbitrary mask 的 layout、dtype 和 sparse mode 要求
- decode 阶段仍保持 causal

最低测试集应包括：

1. 与官方 Transformers 比较第一层和末层 hidden states
2. 五圆 image pointing
3. 多图 pointing
4. 视频 pointing/tracking
5. CountBench
6. MMMU
7. eager 与 graph mode
8. BF16 与 FP32 对照

只有一般文本生成或 MMMU 通过，不足以认定 Molmo2 在 Ascend 上完整支持。

## 最终判断

结合第一阶段的输入等价性实验和本阶段的源码审计，当前最合理的解释是：

> Molmo2 请求在 Ascend vLLM 中完成了正确的图像预处理和 embedding 插入，
> 但语言 transformer 的 NPU attention 没有应用图像 token 所需的双向
> Prefix-LM mask，因而保留了粗粒度识图能力，
> 却破坏了 pointing 所需的精细空间关系。

在取得部署机精确版本和完成跨平台 A/B 前，这仍应表述为高置信度根因，而非已经由上游正式确认的最终 bug。但它比坐标解析、resize/crop 或统一偏移假设更符合全部现有证据。
