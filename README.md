# Reproduction of LeWorldModel on PushT with a Single 12 GB GPU

[中文版](README.zh-CN.md) · [Original upstream README](UPSTREAM_README.md)

This repository documents an independent reproduction of [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312) on the PushT task. The model is trained from scratch with the official code on a single 12 GB GPU and compared with the official checkpoint using the official evaluation script.

| | |
|---|---|
| Upstream | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), commit `8edfeb3` (2026-05-22), MIT License |
| Modifications | Two training-configuration changes and a `.gitignore`, committed separately on top of upstream (`git log 8edfeb3..HEAD`); this README replaces the original, which is kept as [`UPSTREAM_README.md`](UPSTREAM_README.md) |
| Not included | Datasets, model weights and training logs |

## Summary of results

| Evaluation episodes | Reproduced model | Official checkpoint |
|---|---|---|
| 50 (default) | 90.0 | 92.0 |
| 200 | 87.5 | — |
| 500 | 85.8 | 83.8 |

Planning success rate (%). The two models differ by about 2 points in either direction, so the reproduced model performs on par with the official checkpoint under this evaluation protocol. The results come from a single training seed and do not establish that either model is superior.

## Setup

**Hardware and software.** NVIDIA RTX 4080 Laptop GPU (12 GB), Windows with WSL2 (Ubuntu), driver 616.56; Python 3.10; PyTorch 2.12.1, torchvision 0.27.1, transformers 5.12.1, Lightning 2.6.5, stable-worldmodel 0.1.1, h5py 3.16.0, hdf5plugin 7.0.0.

**Changes to the training configuration.**

| Setting | Upstream | This reproduction | Rationale |
|---|---|---|---|
| Dataset file | `pusht_expert_train.lance` | `pusht_expert_train.h5` | The dataset is published in `.h5` format |
| `loader.batch_size` | 128 | 64 | Batch size 128 exceeds 12 GB of GPU memory |
| `trainer.accumulate_grad_batches` | — | 2 | Preserves the effective batch size of 128 |
| `trainer.max_epochs` | 100 | 10 | Appendix E of the paper reports 10 epochs as sufficient |

All other components (model, loss, optimiser and data pipeline) follow the upstream configuration.

**Evaluation protocol.** Official `eval.py` with seed 42, goal offset 25, planning budget 50 and `world.max_episode_steps=100`. With a fixed seed, evaluation is deterministic.

## Results

**Training.** Training for 10 epochs took approximately 26 hours. Final prediction loss was 0.0049 on the training set and 0.0030 on the validation set. The SIGReg term remained stable at approximately 1.28 throughout training, with no indication of representation collapse.

**Comparison with the paper.** The paper reports a success rate of 96.0 ± 2.83 on PushT, measured on a fixed set of 50 trajectories and averaged over three training seeds (Appendix G, Table 5). This trajectory set is not publicly available, so the results above are not directly comparable to the reported value. Under the present protocol the official checkpoint itself reaches 92.0 and 83.8, which indicates that the difference from 96.0 is attributable mainly to the evaluation protocol.

## Issues identified during reproduction

The following issues were encountered on upstream commit `8edfeb3` and resolved locally. They have not yet been reported upstream. Only issue 1 is addressed in this repository's configuration; the remaining issues concern the environment or documentation.

1. **Dataset configuration does not match the published format.** `config/train/data/pusht.yaml` refers to `pusht_expert_train.lance`, whereas the dataset is published as `.h5`.
2. **Decompression command does not match the published file.** The upstream README uses `tar --zstd -xvf archive.tar.zst`, whereas the published file is a single `.h5.zst` archive that requires `zstd -d`.
3. **Inconsistent default data directory.** The upstream README states that `$STABLEWM_HOME` defaults to `~/.stable-wm/`, whereas stable-worldmodel 0.1.1 uses `~/.stable_worldmodel/`.
4. **Misleading error for a missing optional dependency.** Without `hdf5plugin`, dataset loading fails with "No format detected" and the underlying ImportError is suppressed.
5. **Documented `policy=` format does not apply to trained weights.** The upstream README refers to `_object.ckpt` checkpoints, whereas training produces `weights_epoch_N.pt`; the working form is `policy=lewm/weights_epoch_10.pt`.
6. **Silent checkpoint download failure.** When the Hugging Face download fails (e.g. on a restricted network), the only message is "no pt file found". Manually downloading `weights.pt` and `config.json` resolves the issue.
7. **Parameter naming incompatible with transformers 5.x.** The official weights use transformers < 5 ViT parameter names, which produce missing and unexpected keys under transformers 5.x. The parameters were renamed to the 5.x convention (e.g. `query → q_proj`, `intermediate.dense → mlp.fc1`); the official-checkpoint results above use these converted weights.

## Reproducibility

```bash
# Environment (see UPSTREAM_README.md for details)
uv venv --python=3.10 && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]" hdf5plugin

# Data: download pusht_expert_train.h5.zst from the Hugging Face collection
zstd -d pusht_expert_train.h5.zst -o ~/.stable_worldmodel/datasets/pusht_expert_train.h5

# Training (approximately 26 h on the hardware above)
python train.py data=pusht

# Evaluation of the epoch-10 weights on 500 episodes
python eval.py --config-name=pusht.yaml policy=lewm/weights_epoch_10.pt \
    world.max_episode_steps=100 eval.num_eval=500
```

To evaluate the official checkpoint, set `policy` to the converted official weights (see issue 7).

## Citation

The model and training code are by the LeWM authors ([paper](https://arxiv.org/abs/2603.19312), [code](https://github.com/lucas-maes/le-wm)). Please cite the original work:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT, inherited from the upstream repository. See [LICENSE](LICENSE).
