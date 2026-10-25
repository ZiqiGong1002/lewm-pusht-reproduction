# 用单张 12GB GPU 复现 LeWM（PushT）

[English](README.md) · [上游原始 README](UPSTREAM_README.md)

本 fork 记录了对 [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312) 在 PushT 任务上的独立复现：用官方代码从零训练，再用官方 `eval.py` 与官方 checkpoint 对比。

- **上游**：[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)，提交 `8edfeb3`（2026-05-22），代码与历史未改动，只替换了 README。
- **本人的改动**：两处训练配置、`.gitignore` 和本 README（官方原 README 保留为 [`UPSTREAM_README.md`](UPSTREAM_README.md)），均为上游之上的单独提交（`git log 8edfeb3..HEAD`）。
- **不包含**：数据集、模型权重、训练日志。

## 结论速览

| 评测题量 | 本人从零训练的模型 | 官方 checkpoint |
|---|---|---|
| 50（默认） | 90.0 | 92.0 |
| 200 | 87.5 | — |
| 500 | 85.8 | 83.8 |

成功率（%）。两者互有高低，差距在 2 个点左右，可视为在这套评测下与官方处于同一水平。只训练了一个种子，这些数字不能说明哪个模型更好。

## 环境与配置

### 硬件与软件

- NVIDIA RTX 4080 Laptop GPU（12GB），Windows + WSL2 (Ubuntu)，驱动 616.56
- Python 3.10（uv venv）
- torch 2.12.1、torchvision 0.27.1、transformers 5.12.1、lightning 2.6.5、stable-worldmodel 0.1.1、h5py 3.16.0、hdf5plugin 7.0.0

### 相对上游的训练改动

| 设置 | 上游 | 本次 | 原因 |
|---|---|---|---|
| 数据文件 | `pusht_expert_train.lance` | `pusht_expert_train.h5` | HF 上发布的是 `.h5` |
| `loader.batch_size` | 128 | 64 | 12GB 显存放不下 128 |
| `trainer.accumulate_grad_batches` | — | 2 | 保持等效 batch 128 |
| `trainer.max_epochs` | 100 | 10 | 论文附录 E 称 10 个 epoch 已足够 |

模型、损失、优化器和数据管线其余部分均与上游一致。

## 复现步骤

```bash
# 环境（详见 UPSTREAM_README.md）
uv venv --python=3.10 && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]" hdf5plugin

# 数据：从 HF collection 下载 pusht_expert_train.h5.zst，然后
zstd -d pusht_expert_train.h5.zst -o ~/.stable_worldmodel/datasets/pusht_expert_train.h5

# 训练（上述 GPU 约 26 小时）
python train.py data=pusht

# 用第 10 个 epoch 的权重评测 500 题
python eval.py --config-name=pusht.yaml policy=lewm/weights_epoch_10.pt \
    world.max_episode_steps=100 eval.num_eval=500
```

评测设置：seed 42，goal_offset 25，规划 budget 50。固定 seed 时评测结果是确定的。评官方 checkpoint 时，把 `policy` 换成转换后的官方权重路径（见问题 7）。

## 结果

**训练**：10 个 epoch 约 26 小时。结束时 train pred_loss 0.0049，val pred_loss 0.0030；SIGReg 项全程稳定在约 1.28，未见表征塌缩。

**评测**：见上方速览表。与论文对比时请注意：

- 论文报告的 PushT 成绩是 96.0 ± 2.83，来自一组固定的 50 条轨迹、3 个训练种子取平均（附录 G，Table 5）。这组轨迹未公开，因此上表不能与论文数字直接比较。
- 在本评测设置下，官方 checkpoint 自身是 92.0 / 83.8，说明与 96.0 的差距主要来自评测口径。

## 复现中遇到的问题

以下问题均在上游提交 `8edfeb3` 上遇到并在本地绕过，**尚未向上游报告**。只有第 1 项改在了本 fork 的配置里，其余是环境或文档问题，在此记录。

1. **数据配置与发布格式不一致**：`config/train/data/pusht.yaml` 写的是 `pusht_expert_train.lance`，HF 上发布的是 `.h5`。
2. **解压命令与下载文件不对应**：上游 README 用 `tar --zstd -xvf archive.tar.zst`，实际下载的是单个 `.h5.zst` 文件，需要用 `zstd -d`。
3. **默认数据目录不一致**：上游 README 写 `$STABLEWM_HOME` 默认为 `~/.stable-wm/`，stable-worldmodel 0.1.1 实际使用 `~/.stable_worldmodel/`。（严格说是上游 README 与该库之间的出入。）
4. **缺少可选依赖时报错误导**：没装 `hdf5plugin` 时，加载数据报 "No format detected"，底层的 ImportError 不会显示。
5. **文档里的 `policy=` 写法不适用于自训权重**：上游 README 让 `policy` 指向 `_object.ckpt`；训练产出的是 `weights_epoch_N.pt`，实际可用的写法是 `policy=lewm/weights_epoch_10.pt`。
6. **checkpoint 下载失败没有明确报错**：HF 下载失败时（例如网络受限），只提示 "no pt file found"；手动下载 `weights.pt` 和 `config.json` 后可正常使用。
7. **官方权重使用 transformers < 5 的参数命名**：在 transformers 5.x 下加载会报 missing/unexpected keys。本人把 ViT 参数重命名为 5.x 的名称（例如 `query → q_proj`、`intermediate.dense → mlp.fc1`），上表中官方 checkpoint 的数字即用转换后的权重评得。

## 致谢

模型与训练代码均出自 LeWM 作者（[论文](https://arxiv.org/abs/2603.19312)、[代码](https://github.com/lucas-maes/le-wm)）。如使用，请引用：

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## 许可证

MIT，沿用上游，见 [LICENSE](LICENSE)。
