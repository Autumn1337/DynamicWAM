![DynamicWAM](assets/dynamicwam-github-banner-light.png)

DynamicWAM is an exact-time motion-conditioned world-action model for dynamic
bimanual manipulation. It augments rendered history flow with numeric motion
tokens that preserve displacement magnitude, elapsed simulator time, velocity,
and acceleration.

[Project page](https://dynamicwam.github.io/) ·
[Checkpoint](https://huggingface.co/KhalilGao/DynamicWAM/tree/925cbb7aef5033c924f809ae87479d39fe9f76ff) ·
[Apache-2.0 license](LICENSE)

## Method

DynamicWAM uses two complementary motion representations:

- **History flow** preserves where and in which direction the scene moved.
- **Exact-time motion tokens** preserve how far and how quickly it moved.

The released model conditions on four history-flow maps and four exact-time
motion tokens; a 12-layer compact video expert and a 12-layer action expert
exchange information through layer-wise joint attention.

## Results — DOMINO Level 1

All 35 clean Level-1 tasks, 100 episodes per task, unseen instructions, native
synchronous protocol with 16 committed joint-position actions per observation:

| System | Motion input | Success | SR | MS |
|---|---|---:|---:|---:|
| WAM reference | Current observation | 795 / 3,500 | 22.71 | 38.32 |
| History-Flow WAM | Four history-flow maps | 953 / 3,500 | 27.23 | 41.62 |
| **DynamicWAM** | History flow + exact-time motion tokens | **1,337 / 3,500** | **38.20** | **53.16** |

These rows are complete systems with different training trajectories, not a
matched one-variable ablation.

## Installation

Linux x86_64 with an NVIDIA GPU, Python 3.10–3.12. The dependency graph is
locked in `uv.lock`; FlashAttention installs separately because its wheel must
match the local CUDA and GPU architecture:

```bash
git clone https://github.com/Autumn1337/DynamicWAM.git
cd DynamicWAM
uv sync --extra dev
uv pip install flash-attn==2.8.3.post1 --no-build-isolation
```

## Checkpoint and external assets

Weights, simulator assets, and datasets stay outside Git; every revision and
SHA-256 is pinned in `manifests/`.

```bash
hf download KhalilGao/DynamicWAM \
  --revision "925cbb7aef5033c924f809ae87479d39fe9f76ff" \
  --include "external/checkpoints/DynamicWAM_full.pt" \
  --include "configs/absolute_motion_v2.yaml" \
  --local-dir .
uv run python scripts/verify_checkpoints.py

uv run python scripts/prepare_external.py wan --purpose inference
uv run python scripts/bootstrap_domino.py
uv run python scripts/prepare_external.py robotwin-assets
uv run python scripts/prepare_external.py curobo-source
```

## Evaluation

DOMINO evaluation runs in a separate Python 3.10 environment because the
simulator and CuRobo require a different PyTorch/CUDA stack:

```bash
python3.10 -m venv external/robotwin
external/robotwin/bin/pip install -r environments/evaluation.txt
external/robotwin/bin/pip install -e . --no-deps

TORCH_CUDA_ARCH_LIST=9.0 \
  external/robotwin/bin/pip install -e external/curobo \
  --no-build-isolation --no-deps

external/robotwin/bin/python scripts/prepare_external.py domino-runtime
external/robotwin/bin/python scripts/eval_domino.py
```

## Training

The canonical profile is `configs/absolute_motion_v2.yaml`. Training prompts
are generated from the pinned DOMINO instruction generator (`seen` pool);
evaluation generates `unseen` instructions at runtime. Fresh training also
needs the History-Flow WAM initialization named under `paths.base_checkpoint`,
which is not part of the minimal release.

```bash
uv run python scripts/prepare_external.py wan \
  --purpose language --purpose packing --purpose training

uv run python scripts/collect_domino.py
uv run python scripts/precompute_language.py
uv run python scripts/convert_domino.py
uv run python scripts/precompute_motion.py
uv run python scripts/pack_dataset.py
scripts/train_all.sh
```

`scripts/train.py` exposes the three stages individually: video-expert
distillation, action-expert training, and joint fine-tuning.

## License

Project-owned code is released under the [Apache License 2.0](LICENSE).
Vendored WAN-derived files and the DOMINO patch retain their upstream
licenses. CuRobo is an external dependency under NVIDIA's non-commercial
research/evaluation terms; the project license does not override that
restriction.
