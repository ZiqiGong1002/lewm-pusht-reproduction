# 单张 12GB GPU 上的 LeWorldModel PushT 复现

[English](README.md) · [上游原始 README](UPSTREAM_README.md)

本仓库记录了 [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312) 在 PushT 任务上的独立复现：使用官方代码在单张 12GB GPU 上从零训练，并用官方评测脚本与官方 checkpoint 对比。

| | |
|---|---|
| 上游 | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)，提交 `8edfeb3`（2026-05-22），MIT 许可证 |
| 改动 | 两处训练配置与 `.gitignore`，作为上游之上的单独提交（`git log 8edfeb3..HEAD`）；本 README 替换了原 README，原文保留为 [`UPSTREAM_README.md`](UPSTREAM_README.md) |
| 不包含 | 数据集、模型权重、训练日志 |

## 结果概要

| 评测题量 | 复现模型 | 官方 checkpoint |
|---|---|---|
| 50（默认） | 90.0 | 92.0 |
| 200 | 87.5 | — |
| 500 | 85.8 | 83.8 |

规划成功率（%）。两者互有高低，差距约 2 个百分点，在该评测协议下复现模型与官方 checkpoint 处于同一水平。结果来自单个训练种子，不能据此判断哪个模型更优。

## 实验设置

**硬件与软件。** NVIDIA RTX 4080 Laptop GPU（12GB），Windows + WSL2（Ubuntu），驱动 616.56；Python 3.10；PyTorch 2.12.1、torchvision 0.27.1、transformers 5.12.1、Lightning 2.6.5、stable-worldmodel 0.1.1、h5py 3.16.0、hdf5plugin 7.0.0。

**训练配置改动。**

| 设置 | 上游 | 本复现 | 依据 |
|---|---|---|---|
| 数据文件 | `pusht_expert_train.lance` | `pusht_expert_train.h5` | 数据集以 `.h5` 格式发布 |
| `loader.batch_size` | 128 | 64 | batch size 128 超出 12GB 显存 |
| `trainer.accumulate_grad_batches` | — | 2 | 保持等效 batch size 128 |
| `trainer.max_epochs` | 100 | 10 | 论文附录 E 称 10 个 epoch 已足够 |

其余部分（模型、损失、优化器、数据管线）均与上游配置一致。

**评测协议。** 官方 `eval.py`，seed 42，goal offset 25，规划 budget 50，`world.max_episode_steps=100`。固定 seed 时评测结果是确定的。

## 结果

**训练。** 10 个 epoch 训练耗时约 26 小时。最终训练集预测损失为 0.0049，验证集为 0.0030。SIGReg 项在整个训练过程中稳定在约 1.28，未见表征塌缩迹象。

**与论文对比。** 论文报告的 PushT 成功率为 96.0 ± 2.83，基于一组固定的 50 条轨迹、3 个训练种子取平均（附录 G，Table 5）。该轨迹集未公开，因此上表结果不能与论文数值直接比较。在本评测协议下，官方 checkpoint 自身的成功率为 92.0 和 83.8，表明与 96.0 的差距主要源于评测协议。

## 复现中发现的问题

以下问题均在上游提交 `8edfeb3` 上遇到并在本地解决，尚未向上游报告。本仓库配置中仅处理了第 1 项，其余为环境或文档问题。

1. **数据配置与发布格式不一致。** `config/train/data/pusht.yaml` 指向 `pusht_expert_train.lance`，而数据集以 `.h5` 发布。
2. **解压命令与发布文件不一致。** 上游 README 使用 `tar --zstd -xvf archive.tar.zst`，而发布的是单个 `.h5.zst` 文件，需要用 `zstd -d` 解压。
3. **默认数据目录不一致。** 上游 README 称 `$STABLEWM_HOME` 默认为 `~/.stable-wm/`，而 stable-worldmodel 0.1.1 实际使用 `~/.stable_worldmodel/`。
4. **缺少可选依赖时报错具有误导性。** 未安装 `hdf5plugin` 时，数据加载报 "No format detected"，底层 ImportError 被隐藏。
5. **文档中的 `policy=` 格式不适用于自训权重。** 上游 README 针对 `_object.ckpt` checkpoint，而训练产出的是 `weights_epoch_N.pt`；可用的写法为 `policy=lewm/weights_epoch_10.pt`。
6. **checkpoint 下载失败时无明确提示。** Hugging Face 下载失败时（如网络受限）仅提示 "no pt file found"。手动下载 `weights.pt` 与 `config.json` 后可正常使用。
7. **参数命名与 transformers 5.x 不兼容。** 官方权重使用 transformers < 5 的 ViT 参数命名，在 5.x 下出现 missing/unexpected keys。已将参数按 5.x 规范重命名（如 `query → q_proj`、`intermediate.dense → mlp.fc1`），上文官方 checkpoint 的结果即基于转换后的权重。

## 复现步骤

```bash
# 环境（详见 UPSTREAM_README.md）
uv venv --python=3.10 && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]" hdf5plugin

# 数据：从 Hugging Face collection 下载 pusht_expert_train.h5.zst
zstd -d pusht_expert_train.h5.zst -o ~/.stable_worldmodel/datasets/pusht_expert_train.h5

# 训练（上述硬件约 26 小时）
python train.py data=pusht

# 在 500 题上评测第 10 个 epoch 的权重
python eval.py --config-name=pusht.yaml policy=lewm/weights_epoch_10.pt \
    world.max_episode_steps=100 eval.num_eval=500
```

评测官方 checkpoint 时，将 `policy` 设为转换后的官方权重（见问题 7）。

## 引用

模型与训练代码出自 LeWM 作者（[论文](https://arxiv.org/abs/2603.19312)、[代码](https://github.com/lucas-maes/le-wm)），请引用原始工作：

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## 许可证

MIT，沿用上游仓库，见 [LICENSE](LICENSE)。
