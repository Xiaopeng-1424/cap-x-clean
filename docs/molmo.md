# Molmo2 Pointing 坐标错误调查

## 调查范围

本文记录对 LIBERO 场景中 Molmo2 定位“黑色三抽屉柜的中间灰色竖直把手”失败问题的第一阶段调查。调查重点包括：

- Molmo2 原始输出格式与坐标语义
- 客户端 prompt 包装
- 坐标解析、归一化尺度和 x/y 顺序
- 图像 resize、padding、crop 与 image-token metadata
- vLLM OpenAI-compatible 服务输入
- 坐标 overlay
- 两组 100-trial 输出的分布
- 官方 Transformers 与远端 vLLM 的同图对照

调查时没有修改 Molmo 集成源码。

## 结论摘要

已证实：

1. 仓库的 Molmo2 核心坐标解析和像素换算是正确的。
2. 客户端没有在请求前执行 resize 或 crop。
3. 错误坐标已经出现在模型返回的原始 `<points coords="...">` 文本中，不是 overlay 把正确坐标画错了。
4. 对同一 PNG，远端 vLLM 的 token IDs、image-token 顺序、
   `image_token_pooling` 和官方 Hugging Face processor 基本完全一致。
5. 同一 Molmo2-8B checkpoint 通过官方 Transformers 后端能够准确指向目标；远端 vLLM 服务则产生大幅、非统一的空间误差。
6. `init_molmo()` 会把完整的 `Point to ...` 再包装成 `Point at Point to ...`。这是确定的客户端 bug，但移除后仍然定位错误，因此不是根因。

第一阶段能确定的故障边界是：

> 问题主要发生在预处理完成之后的服务端模型执行路径，而不是客户端坐标 mapping。

## 被调查的输出

主要输出文件：

```text
outputs/human_oracle_code/google_gemini-3.1-pro-preview/
├── libero_goal_swap_0/
│   ├── oracle_code.py
│   ├── summaries.txt
│   └── molmo_100_trials_points_on_first_frame.png
└── libero_goal_swap_0_offset/
    ├── oracle_code.py
    ├── summaries.txt
    └── molmo_offset_100_trials_overlaid_on_original.png
```

两张 overlay 均为 `800×512`。图中中间把手中心约为 `(309,187)`；
`libero_goal_swap_0_offset/oracle_code.py` 使用的参考 fallback 也是：

```python
fallback_middle = np.array([0.386 * width, 0.365 * height])
```

在 `800×512` 图像上对应 `(308.8,186.88)`。

## 客户端链路

主链路集中在 `capx/integrations/vision/molmo.py`：

```text
PIL image
  -> PNG data URL
  -> /v1/chat/completions
  -> choices[0].message.content
  -> _parse_points()
  -> convert_to_pixel_coordinates()
  -> (x, y) pixel coordinate
```

LIBERO 包装层只执行 ndarray 到 PIL 的转换，没有额外几何变换：

```python
image = Image.fromarray(image_array)
```

`_image_to_data_url()` 将当前 PIL 图像原样保存为 PNG。客户端没有执行
resize、letterbox、center crop 或坐标逆变换。

## Molmo2 坐标格式

Molmo2 图像 pointing 的典型输出为：

```text
<points coords="1 1 380 369">middle handle</points>
```

字段含义为：

```text
image_or_frame_id, object_id, x, y, [object_id, x, y, ...]
```

其中：

- `x`、`y` 范围为 0–1000
- x 表示水平方向，向右增加
- y 表示垂直方向，向下增加
- 图像输出的第一个字段通常是 1-based image index

像素换算应为：

```text
pixel_x = x / 1000 * image_width
pixel_y = y / 1000 * image_height
```

仓库实现 `capx/integrations/vision/molmo.py:43-53,119-132` 正是按这一规则处理：

```python
# Skip image/frame id, then consume (object_id, x, y) triplets.
x, y = nums[i + 1], nums[i + 2]

pixel_x = int(x / norm_scale * width)
pixel_y = int(y / norm_scale * height)
```

例如：

```text
<points coords="1 1 362 511">...</points>
```

映射到 `800×512` 图像为：

```text
x = 362 / 1000 * 800 = 289.6
y = 511 / 1000 * 512 = 261.632
```

这与 overlay 中的点一致。因此没有发现：

- x/y 颠倒
- width/height 混用
- 0–100 与 0–1000 混用
- 10 倍尺度错误

官方格式与示例见 [Molmo2-8B 模型卡](https://huggingface.co/allenai/Molmo2-8B/blob/main/README.md)。

## 原始模型文本已经错误

100-trial 日志中的原始输出已经位于错误区域。例如：

```text
<points coords="1 1 362 511">...</points>
```

得到 `(290,262)` 左右，而目标约为 `(309,187)`。

另一条低簇输出为：

```text
<points coords="1 1 362 586">...</points>
```

得到 `(290,300)` 左右，距离目标更远。

因此错误发生在 `_parse_points()` 之前。修改 overlay、交换 x/y 或手工加固定偏移都不能解释或修复根因。

## 官方 Transformers 与远端服务 A/B

使用同一张 LIBERO PNG、同一 checkpoint 和等价 prompt，对官方 Transformers 后端进行测试。

### LIBERO 把手

参考位置：约 `(309,187)`。

| 后端与 prompt | 原始归一化坐标 | 像素坐标 | 参考误差 |
| --- | ---: | ---: | ---: |
| 官方 Transformers，短 prompt | `(380,369)` | `(304.0,188.9)` | 约 5.2 px |
| 官方 Transformers，精简 prompt | `(379,366)` | `(303.2,187.4)` | 约 5.6 px |
| 远端 vLLM，原样短 prompt | `(362,514)` | `(289.6,263.2)` | 约 78.7 px |
| 远端 vLLM，原样精简 prompt | `(354,501)` | `(283.2,256.5)` | 约 74.2 px |
| 远端 vLLM，仓库包装后的短 prompt | `(357,586)` | `(285.6,300.0)` | 约 115.5 px |

官方后端能够在同一输入图上正确定位小目标，说明 Molmo2 checkpoint 本身具有完成该任务的能力。

### 五圆合成图

构造 `512×512` 白底图，在四角和中心放置五个已知圆心：

```text
red    true = 150,150
blue   true = 850,150
green  true = 150,850
purple true = 850,850
black  true = 500,500
```

坐标均为 0–1000 归一化坐标。

| 目标 | 官方 Transformers | 远端 vLLM |
| --- | ---: | ---: |
| red | `(149,149)` | `(146,262)` |
| blue | `(852,149)` | `(805,280)` |
| green | `(150,850)` | `(146,584)` |
| purple | `(852,851)` | `(808,583)` |
| black | `(500,500)` | `(468,535)` |

结果：

- 官方 Transformers 平均像素误差：约 `0.60 px`
- 远端 vLLM 平均像素误差：约 `85.34 px`

远端服务可以正确描述五个圆的位置和颜色，但生成坐标出现严重空间畸变。这说明问题不是单纯的目标语义混淆。

不同宽高比、不同目标位置的误差并不一致。例如某些左侧目标接近正确，而同一行右侧目标严重偏移。因此不应拟合单一 x/y offset 或全局仿射补偿。

## 服务端预处理等价性检查

通过 vLLM 暴露的 `/v1/chat/completions/render` 取得服务端实际渲染输入，
并使用官方 Molmo2 processor 对同一 PNG 和 prompt 本地重放。

### 五圆图

以下内容逐位一致：

- 完整 `input_ids`
- `image_tokens`
- `image_token_pooling`
- 转为 BF16 后的 `pixel_values`

### LIBERO 图

服务端和本地 processor 的形状一致：

```text
input_ids:           [1, 1009]
image_tokens:        [982]
image_token_pooling: [955, 4]
pixel_values:        [7, 729, 588]
```

比较约 300 万个预处理值：

```text
总元素：              3,000,564
不同元素：                   53
不同占比：              0.0018%
最大绝对差：            0.0078125
```

这些差异属于 BF16 舍入量级，不足以解释几十到上百像素的结构性误差。

该实验排除了以下主要假设：

- 客户端发送错图
- vLLM 使用了不同图像尺寸
- resize/padding/crop 参数不一致
- image-token 顺序不一致
- `image_token_pooling` 几何重排不一致
- chat prompt tokenization 不一致

## 图像与文本 content 顺序

对同一请求测试：

```json
[text, image]
```

和：

```json
[image, text]
```

五圆图输出完全相同；LIBERO 输出只出现 3 个归一化 y 单位的微小变化。因此 content 顺序不是主要原因。

## Prompt 包装 bug

`capx/integrations/vision/molmo.py:187-190` 无条件执行：

```python
prompt = f"Point at {obj}"
```

而两个 oracle 已传入完整指令：

```text
Point to the small handle ...
```

实际请求变成：

```text
Point at Point to the small handle ...
```

同图实测：

```text
包装后：357,586 -> (285.6,300.0)
原样：  362,514 -> (289.6,263.2)
```

重复前缀显著改变输出模态，因此必须修复。不过去掉包装后仍离目标约 79 px，所以不能把全部错误归因于 prompt。

推荐的接口语义是二选一：

1. `det_fn()` 接收完整 prompt，并完全原样传递；或
2. `det_fn()` 只接收名词短语，并由内部统一添加 pointing 指令。

不应同时支持两种语义却无条件拼接。

## 两组 100-trial 分布

### `libero_goal_swap_0`

100 个归一化坐标统计：

```text
x mean / median: 363.98 / 362
x range:         354–382
y mean / median: 527.68 / 511
y range:         504–598
```

按 `y=550` 分簇：

```text
主簇 y < 550:   79
低簇 y >= 550:  21
```

最高频坐标包括：

```text
362,511  17 次
362,514  14 次
362,586   9 次
```

像素均值约为 `(291.2,270.2)`。

### `libero_goal_swap_0_offset`

```text
x mean / median: 363.92 / 362
x range:         354–380
y mean / median: 510.41 / 511
y range:         498–579
```

按 `y=550` 分簇：

```text
主簇 y < 550:   99
低簇 y >= 550:   1
```

像素均值约为 `(291.1,261.3)`。

长 prompt 基本消除了较低的错误簇，但主簇仍离真实把手约 75 px。

### 两组关系

- 两组只有 33 个多重集合意义上的完全相同坐标。
- 对每个 offset 点寻找第一组最近点，97/100 在 5 px 内。
- 两组 x 分布几乎相同。

因此两组属于同一个错误分布；长 prompt 只是改变了模态权重，没有修复坐标系统。

## 配对首帧分析

对两组各 100 个 `video_turn_00.mp4` 提取首帧：

- 每组内部 100 帧都不是逐位完全相同，环境存在轻微渲染或状态变化。
- 相同 trial 编号的两组首帧平均像素 MAE 约 `0.659`。
- phase correlation 棃测到的几何平移通常小于 `0.02 px`。
- 首帧差异与配对坐标距离的相关系数约 `0.13`。

所以两个目录的配对场景在几何上相同。坐标分布变化主要由 prompt/执行路径造成，不是相机帧整体移动。

## Offset 运行的 reward 不能证明 Molmo 变准

`libero_goal_swap_0_offset/oracle_code.py` 会先检查 Molmo 原始点是否落在柜体限定区域内。100 次日志均为：

```text
accepted: False
```

随后实际采用固定 fallback：

```text
(0.386 * width, 0.365 * height) ~= (309,187)
```

因此 offset 运行约 48% 的 reward 主要来自 fallback，
而不是 Molmo pointing 改善。overlay 中画出的仍是被拒绝的原始 Molmo 点。

## 第一阶段的服务端执行嫌疑

服务端 `/version` 返回：

```text
vLLM 0.20.2
```

官方 Molmo2 checkpoint 的 `config.json` 包含：

```json
{
  "dtype": "float32",
  "vit_config": {
    "float32_attention": true
  },
  "adapter_config": {
    "float32_attention": true
  }
}
```

官方 Transformers vision attention 会按配置将 Q/K 转为 FP32；
SDPA 非 autocast 路径还会将 V 转为 FP32。
远端 render 数据则显示模型视觉输入以 BF16 进入执行路径。

与此同时，vLLM 0.20.2 的 Molmo2 自定义实现没有在 `VitConfig` 和 `AdapterConfig` 中保留 `float32_attention`，vision/pooling attention 也没有官方实现中的显式 FP32 upcast。检查到 vLLM 0.24.0 时，该模型文件仍存在这一差异。

因此第一阶段将以下因素列为服务端候选：

- 模型被从官方 FP32 配置降为 BF16
- vLLM vision attention 未保留官方 `float32_attention` 语义
- vLLM 自定义 attention、Prefix-LM 或权重映射与官方实现不等价
- 服务端本地权重与官方 checkpoint 不一致或损坏

仅凭第一阶段实验，能够确定是“服务端执行路径”，但不能只靠输出进一步区分上述机制。

相关上游资料：

- [Molmo2 vLLM 支持 PR #30997](https://github.com/vllm-project/vllm/pull/30997)
- [Molmo2 image-token metadata 修复 PR #42162](https://github.com/vllm-project/vllm/pull/42162)
- [经典 Molmo pointing patch 重排错误 issue #26451](https://github.com/vllm-project/vllm/issues/26451)
- [经典 Molmo 修复 PR #26563](https://github.com/vllm-project/vllm/pull/26563)

PR #42162 在 vLLM 0.20.2 发布后合入，进入后续版本。
它主要修复 MolmoWeb 风格 processor 配置下的 token/embedding 数量不一致。
当前标准 Molmo2-8B 的 token metadata 已与官方 processor 匹配，
因此它不是本次标准图像输入的直接解释。

## 仓库中的次要健壮性问题

### Molmo2 坐标缺少范围与有限值检查

Molmo2 分支会直接接受解析到的浮点数，没有验证：

```text
isfinite(x), isfinite(y), 0 <= x <= 1000, 0 <= y <= 1000
```

malformed 数值还可能直接触发 `ValueError`。

### 边界可能越界一像素

当前换算：

```python
int(1000 / 1000 * width) == width
```

但合法像素最大索引是 `width - 1`。用于数组索引或 SAM point prompt 前应 clamp。

### 多点结果只取第一个

`_parse_points()` 支持多个点，但 detector 固定使用：

```python
convert_to_pixel_coordinates(...)[0]
```

如果模型返回多个实例，第一个点未必是调用者期望的目标。

### 失败判断无法识别 `(None, None)`

某些调用方使用：

```python
if None not in result.values():
```

但 values 是 tuple；`(None,None)` 本身不等于 `None`，导致失败值仍进入 overlay 或下游逻辑。

### 测试没有 correctness assertion

`tests/integrations/test_molmo_integration.py` 目前主要是联网调用和打印，没有断言：

- prompt 是否原样传递
- Molmo2 单点/多点解析
- 0–1000 尺度
- 非方图 x/y 顺序
- malformed 和越界输入
- 静态校准图正确性

## 不建议的修复

以下措施可能改变输出，但不能解决根因：

- 给所有 y 加固定负偏移
- 交换 x/y
- 把 1000 改成 100
- 依据单张 LIBERO 图拟合仿射变换
- 单纯增加更多否定描述
- 只看最终 overlay，不记录原始模型文本

合成校准已经证明误差随位置和宽高比变化，不是统一几何偏移。

## 推荐验证矩阵

应在同一 checkpoint、同一 PNG、同一 greedy decoding 参数下比较：

| 后端 | 精度 | 目的 |
| --- | --- | --- |
| 官方 Transformers | FP32/auto | 官方基准 |
| 官方 Transformers | BF16 | 隔离精度影响 |
| vLLM | FP32 | 隔离 vLLM 算子/attention 影响 |
| vLLM | BF16 | 当前部署对照 |
| 更新版本 vLLM | BF16 与 FP32 | 验证上游修复 |

每次部署至少运行：

1. 五圆静态校准图
2. `800×512` LIBERO 原始 PNG
3. 224、378、512 等不同尺寸
4. wide、square、tall 三种宽高比
5. exact prompt 与仓库包装 prompt

建议将五圆校准做成服务启动后的 correctness gate：若任一圆心超过预设误差阈值，则不允许该服务用于机器人 point prompt。

## 最终判断

第一阶段调查足以排除客户端坐标 mapping 和 overlay 为主要原因。最关键证据是：

> 官方 Transformers 在同一图上返回正确点，而远端 vLLM 在收到几乎完全相同的预处理张量后返回错误点。

因此应优先验证服务端 attention、精度、权重加载和具体硬件插件，而不是继续修改客户端坐标公式。
