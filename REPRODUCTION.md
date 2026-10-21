# Reproducing LeWM on PushT with a single 12 GB GPU

[中文版](REPRODUCTION.zh-CN.md)

This fork records an independent reproduction of [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312) on the PushT task. I trained the model from scratch with the official code and compared it with the official checkpoint using the official `eval.py`.

- **Upstream:** [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) at commit `8edfeb3` (2026-05-22). Its code and history are unchanged.
- **My changes:** two training-config edits, a `.gitignore`, a note at the top of `README.md` and these documents, in separate commits on top of upstream (`git log 8edfeb3..HEAD`).
- **Not included:** datasets, model weights and training logs.

## TL;DR

| Episodes evaluated | My model (trained from scratch) | Official checkpoint |
|---|---|---|
| 50 (default) | 90.0 | 92.0 |
| 200 | 87.5 | — |
| 500 | 85.8 | 83.8 |

Success rate (%). The two models are within about 2 points of each other in both directions, so the reproduction is on par with the official checkpoint under this setup. This is a single training seed, so the numbers do not show that either model is better.

## Setup

### Hardware and software

- NVIDIA RTX 4080 Laptop GPU (12 GB), Windows + WSL2 (Ubuntu), driver 616.56
- Python 3.10 (uv venv)
- torch 2.12.1, torchvision 0.27.1, transformers 5.12.1, lightning 2.6.5, stable-worldmodel 0.1.1, h5py 3.16.0, hdf5plugin 7.0.0

### Training changes versus upstream

| Setting | Upstream | This run | Reason |
|---|---|---|---|
| Dataset file | `pusht_expert_train.lance` | `pusht_expert_train.h5` | HF publishes `.h5` |
| `loader.batch_size` | 128 | 64 | 128 does not fit in 12 GB |
| `trainer.accumulate_grad_batches` | — | 2 | keeps the effective batch at 128 |
| `trainer.max_epochs` | 100 | 10 | paper appendix E reports 10 epochs as sufficient |

Model, loss, optimizer and data pipeline otherwise follow the upstream config.

## Reproduce

```bash
# environment (details in the upstream instructions below)
uv venv --python=3.10 && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]" hdf5plugin

# data: download pusht_expert_train.h5.zst from the HF collection, then
zstd -d pusht_expert_train.h5.zst -o ~/.stable_worldmodel/datasets/pusht_expert_train.h5

# train (about 26 h on the GPU above)
python train.py data=pusht

# evaluate the epoch-10 weights on 500 episodes
python eval.py --config-name=pusht.yaml policy=lewm/weights_epoch_10.pt \
    world.max_episode_steps=100 eval.num_eval=500
```

Evaluation settings: seed 42, goal_offset 25, planning budget 50. With a fixed seed the evaluation is deterministic. For the official checkpoint, point `policy` at the converted official weights (see issue 7).

## Results

**Training.** 10 epochs took about 26 hours. At the end, the train prediction loss was 0.0049 and the validation prediction loss 0.0030. The SIGReg term stayed around 1.28 throughout, with no sign of representation collapse.

**Evaluation.** See the TL;DR table. When comparing with the paper:

- The paper reports 96.0 ± 2.83 on PushT, measured on a fixed set of 50 trajectories and averaged over 3 training seeds (appendix G, Table 5). That trajectory set is not public, so the table above cannot be compared with it directly.
- Under this evaluation setup the official checkpoint scores 92.0 / 83.8, which suggests the gap to 96.0 is mostly due to the evaluation protocol.

## Issues found while reproducing

Problems I hit on upstream commit `8edfeb3` and worked around locally. **None has been reported upstream yet.** Only issue 1 is changed in this fork's config; the others are environment or documentation issues and are recorded here.

1. **Dataset config does not match the published format.** `config/train/data/pusht.yaml` names `pusht_expert_train.lance`, while HF publishes `.h5`.
2. **Decompression command does not match the download.** The README uses `tar --zstd -xvf archive.tar.zst`, but the download is a single `.h5.zst` file, which needs `zstd -d`.
3. **Default data directory differs.** The README says `$STABLEWM_HOME` defaults to `~/.stable-wm/`; stable-worldmodel 0.1.1 uses `~/.stable_worldmodel/`. (Strictly a mismatch between this README and the library.)
4. **A missing optional dependency gives a misleading error.** Without `hdf5plugin`, loading the dataset fails with "No format detected" and the underlying ImportError is not shown.
5. **The documented `policy=` form does not cover trained weights.** The README points `policy` at an `_object.ckpt` checkpoint. Training writes `weights_epoch_N.pt`; the form that worked was `policy=lewm/weights_epoch_10.pt`.
6. **Checkpoint download fails silently.** When the HF download fails (for example on a restricted network), the only message is "no pt file found". Downloading `weights.pt` and `config.json` by hand works.
7. **Official weights use transformers < 5 parameter names.** Under transformers 5.x, loading them reports missing and unexpected keys. I renamed the ViT parameters to the 5.x names (for example `query → q_proj`, `intermediate.dense → mlp.fc1`) and used the converted weights for the official-checkpoint numbers above.

## Acknowledgements

All model and training code is by the LeWM authors. If you use it, please cite their paper (BibTeX in the upstream README).

## License

MIT, inherited from upstream. See [LICENSE](LICENSE).
