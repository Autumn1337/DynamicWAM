# DynamicWAM

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

The compact video expert and action expert exchange information through
layer-wise joint attention. The released model uses four history-flow maps,
four exact-time motion tokens, a 12-layer compact WAN expert, and a 12-layer
action expert.

## DOMINO Level 1

The reported evaluation uses all 35 clean DOMINO Level-1 tasks, 100 accepted
episodes per task, unseen instructions, and the native synchronous protocol
with 16 committed joint-position actions per observation.

| System | Motion input | Success | SR | MS |
|---|---|---:|---:|---:|
| WAM reference | Current observation | 795 / 3,500 | 22.71 | 38.32 |
| History-Flow WAM | Four history-flow maps | 953 / 3,500 | 27.23 | 41.62 |
| **DynamicWAM** | History flow + exact-time motion tokens | **1,337 / 3,500** | **38.20** | **53.16** |

These rows are complete systems with different training trajectories. They are
not a matched one-variable causal ablation.

## Released checkpoint

The [Hugging Face repository](https://huggingface.co/KhalilGao/DynamicWAM/tree/925cbb7aef5033c924f809ae87479d39fe9f76ff)
contains exactly the final checkpoint and its matching configuration:

| File | Description |
|---|---|
| `external/checkpoints/DynamicWAM_full.pt` | Stage-3 checkpoint at 40,000 steps |
| `configs/absolute_motion_v2.yaml` | Training, inference, and DOMINO evaluation configuration |

Checkpoint identity:

```text
size:   1,977,857,939 bytes
sha256: 7c0dfc44a785ea1f6bd1f833f09dcadc2e470dadb1ba5508fa98918e147671d7
```

The checkpoint embeds the action-normalization statistics required for
inference; no additional statistics file is needed.

Download both files into a clone:

```bash
hf download KhalilGao/DynamicWAM \
  --revision "925cbb7aef5033c924f809ae87479d39fe9f76ff" \
  --include "external/checkpoints/DynamicWAM_full.pt" \
  --include "configs/absolute_motion_v2.yaml" \
  --local-dir .

uv run python scripts/verify_checkpoints.py
```

## Installation

The supported runtime is Linux x86_64 with an NVIDIA GPU and Python 3.10–3.12.
The exact dependency graph is recorded in `uv.lock`.

```bash
git clone https://github.com/Autumn1337/DynamicWAM.git
cd DynamicWAM
uv sync --extra dev
```

FlashAttention is installed separately because its wheel must match the local
PyTorch, CUDA, and GPU architecture:

```bash
uv pip install flash-attn==2.8.3.post1 --no-build-isolation
```

## External assets

Model weights, simulator assets, datasets, and generated outputs are excluded
from Git. Their revisions and SHA-256 hashes are pinned in
`manifests/external_assets.json`.

For inference:

```bash
uv run python scripts/prepare_external.py wan --purpose inference
uv run python scripts/bootstrap_domino.py
uv run python scripts/prepare_external.py robotwin-assets
uv run python scripts/prepare_external.py curobo-source
```

For training, also download the WAN assets used by language preprocessing,
packing, and distillation:

```bash
uv run python scripts/prepare_external.py wan \
  --purpose language \
  --purpose packing \
  --purpose training
```

## Data and training

The canonical profile is `configs/absolute_motion_v2.yaml`. Generated data,
flow caches, language embeddings, checkpoints, and logs stay under ignored
`data/`, `external/`, and `outputs/` paths.

Language preprocessing reads the collected clean `scene_info.json` files and
uses the pinned DOMINO instruction generator directly. The training prompts are
therefore derived from the same source as the benchmark instead of being copied
into this repository. Training uses DOMINO's `seen` instruction pool; evaluation
continues to generate `unseen` instructions at runtime.

Together with the pinned upstream assets, the two-file public release is
sufficient for loading and evaluating the final model. Fresh training also needs
the History-Flow WAM initialization and its action statistics named under
`paths.base_checkpoint` and `paths.base_action_stats`; those intermediate
artifacts are intentionally not mirrored in the minimal Hugging Face release.

```bash
uv run python scripts/collect_domino.py
uv run python scripts/precompute_language.py
uv run python scripts/convert_domino.py
uv run python scripts/precompute_motion.py
uv run python scripts/pack_dataset.py
scripts/train_all.sh
```

The training schedule is:

1. compact video-expert distillation;
2. action-expert training;
3. joint video-action fine-tuning.

Individual stages are available through `scripts/train.py`.

## Evaluation

DOMINO evaluation uses a separate Python 3.10 environment because the simulator
and CuRobo runtime require a different PyTorch/CUDA stack:

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

The evaluator validates the checkpoint, WAN assets, DOMINO source, RoboTwin
assets, CuRobo source/runtime, and the configured execution protocol before
launching the 35-task suite.

## Repository layout

```text
configs/                 canonical training and evaluation profiles
environments/            simulator evaluation dependencies
manifests/               checkpoint and external-asset identities
scripts/                 stable data, training, and evaluation entrypoints
src/dynamicwam/          DynamicWAM implementation
third_party/domino/      evaluated DOMINO source patch and upstream license
```

## License

Project-owned code is released under the [Apache License 2.0](LICENSE).
Vendored WAN-derived files and the DOMINO patch retain their bundled upstream
licenses. CuRobo is an external dependency under NVIDIA's non-commercial
research/evaluation terms; the project license does not override that
restriction.
