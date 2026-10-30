# Pika H3 Flex 集成方案与实施记录

日期：2026-10-30。状态：代码已接入；真实部署推理由用户后续测试。

## 范围与最终决定

在现有 H3 Turbo implementation 中新增 **Pika H3 Flex (6-step)**，并通过原有 refinement LoRA 选择器提供同一选项。按任务自动选择 FL2VA 或 Ref2VA 适配器，沿用统一模型管理、队列、构图、输出与偏好持久化。

按用户最新要求，不依据尚未验证的兼容性禁用 native latent upscale、Split Upscale、Regenerate、attention、VAE、缓存、Semantic Bridge 或其他现有输入/输出功能，也不自动改变用户选择的尺寸和时长。现有功能原有的校验继续生效。这里的“可选择、可构图”不代表已经实测通过，也不代表已证实不兼容。

默认模型、Turbo、预设和预加载清单保持原有行为。无需新 backend、自定义采样节点或安装 Pika 的 Diffusers 推理包。

## 已核实的上游契约

官方工作流固定版本：`Comfy-Org/workflow_templates@abb9cf6477b59c52dbe234c67f82318dadd872b2`。

- [T2V 工作流](https://github.com/Comfy-Org/workflow_templates/blob/abb9cf6477b59c52dbe234c67f82318dadd872b2/templates/video_pika_h3_flex_t2v.json)
- [FL2VA 工作流](https://github.com/Comfy-Org/workflow_templates/blob/abb9cf6477b59c52dbe234c67f82318dadd872b2/templates/video_pika_h3_flex_fl2va.json)
- [Ref2VA 工作流](https://github.com/Comfy-Org/workflow_templates/blob/abb9cf6477b59c52dbe234c67f82318dadd872b2/templates/video_pika_h3_flex_ref2va.json)

上述工作流使用 `LoraLoaderModelOnly`、strength 1.0、`BasicGuider`、`KSamplerSelect(lcm)`、`ManualSigmas` 和 `SamplerCustomAdvanced`。T2V/FL2VA 的节点在 JSON subgraph 内。

原始时间结点为 `1000/750/500/250/100/40/0`，video/audio shift 为 12/3。官方 Comfy video sigma 列表为：

```text
1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0
```

该列表已经过 video shift，不能再次对列表变换。模型上的 `MiniMaxH3SigmaShift(12, 3)` 设置双模态时间映射，与固定列表各司其职。现有 Comfy pin `0df1646a9cd71d4c990a87014fb37797a2956ead` 已包含这些节点，未改动部署依赖版本。

已阅读该 pin 的 [LCM 实现](https://github.com/Comfy-Org/ComfyUI/blob/0df1646a9cd71d4c990a87014fb37797a2956ead/comfy/k_diffusion/sampling.py)、[AV sampling](https://github.com/Comfy-Org/ComfyUI/blob/0df1646a9cd71d4c990a87014fb37797a2956ead/comfy/model_sampling.py) 和 [H3 sigma patch](https://github.com/Comfy-Org/ComfyUI/blob/0df1646a9cd71d4c990a87014fb37797a2956ead/comfy_extras/nodes_minimax_h3.py)。LCM 按非终点 sigma 调用模型并重新混入噪声；末尾 sigma=0 不再加噪。源码核对不能替代真实执行测试。

## 权重来源和可复现下载

实施采用官方工作流直接引用的 `Comfy-Org/MiniMax-H3` 镜像，而非最初方案中的 Pika 仓库 Comfy 文件。固定 revision：`912b6e38b2092ec4d2e86d24183e0d59ea059d48`。

| 模型 key | HF 路径 | SHA256 |
| --- | --- | --- |
| `pika_flex_fl2va_lora` | `loras/pika_h3_flex_fl2va_v1_bf16.safetensors` | `cc5453aeecd6189468409003fe4cf1b57e52ff3a830a0a154b87decbe07358b7` |
| `pika_flex_ref2va_lora` | `loras/pika_h3_flex_ref2va_v1_bf16.safetensors` | `de3745ceb7d8d5c20898dd8fa3a05f1c0b00089ccd55297b2a8bab95c766b0e0` |

已通过 HF metadata 核对 revision、路径、文件大小和哈希，并以小范围 HTTP 读取比较两种来源的 safetensors header：416 个 tensor 的名称、形状、dtype、数据偏移完全一致，metadata 不同。未下载完整权重，未宣称 tensor payload 数值相同。格式说明见 [Pika Comfy README](https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/3ff166447d5edb1406107ab8370c42000ce173e4/comfy/README.md)。

下载模型时只获取当前任务所需的一份 LoRA。`ModelSpec.revision` 贯穿 metadata 查询、下载和 manifest 身份；同一仓库的 main 与固定 revision 分开处理，不影响其他模型。新下载的固定权重进行实际 SHA256 校验；失败时不覆盖已有目标文件。旧模型配置缺少新字段时使用官方文件名默认值。

## 构图与 native latent upscale

基础生成始终使用 Pika 完整六步 schedule。UI 显示固定步数并禁用无效的通用 scheduler 调整；API 同样解析为固定 schedule，记录请求值与有效值的差异。

native latent upscale 沿用现有低分辨率生成、视频 latent 放大、refinement 和原始音轨保留流程：

- **Same as generation**：保留 Pika 模型、LCM 和 Pika schedule 的低噪声尾段。
- 显式选择 **Pika** 作为 refinement LoRA：按任务加载对应 Pika 适配器，并从 Pika 固定六步 schedule 截取 refinement 尾段。
- Pika 基础生成后选择另一种 Turbo refiner：refinement 使用所选适配器的 sampler、shift 和自身步数的 schedule 尾段。
- Regenerate 保持既有的独立三步 schedule 和 Euler sampler。

这些组合已接入供真实场景测试；未预判图像质量、性能或量化兼容性。后续实测再决定是否需要专用采样、loader 或调参。

执行快照记录 Pika 的 LoRA 文件、来源 revision、哈希、strength、sampler、sigma 列表和 video/audio shift。UI 与任务状态注明真实部署 GPU 验证待进行。

## 代码与验证

代码覆盖模型清单和配置读取、按模式下载、Turbo policy、有效设置、FL2VA/Ref2VA 共用构图、refinement、UI 展示、偏好恢复和执行元数据。

针对性 CPU 测试检查：按模式 LoRA 路由、固定 sigma 和双模态 shift、已有功能保持可用、native upscale 完整基础采样和独立 refinement 路径、缺失节点错误、固定版本 metadata 分离、下载哈希失败和旧 manifest 处理。原有 workflow 快照用于检查未引入 Pika 的图保持稳定。没有运行完整测试集。

真实部署后可依次试 T2V、首尾帧、Ref2VA、native upscale 与不同 refinement、量化模型、attention/VAE 和缓存组合。记录实际输出和异常后再迭代。未执行本地模型推理或远端部署，不对未测组合宣称通过或失败。
