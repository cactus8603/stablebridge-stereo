# StableBridge Stereo Submission Runtime

Submission-only runtime for the frozen **CroCo-StableBridge** method on the
RoCo-Spring stereo track. This repository contains the code needed to turn
Spring/RobustSpring test RGB pairs into official disparity submission files.

> This is an inference and packaging release. It intentionally excludes
> controller training, teacher/cache construction, OOF experiments, ablations,
> paper analysis, datasets, checkpoints, predictions, and historical outputs.

## What is included

- Frozen three-action StableBridge router and dual-depth evidence model.
- CroCo-Stereo adapter plus the pinned upstream CroCo source.
- Resumable prediction runner for left/right disparity references.
- Prediction-tree validation and official Spring packaging orchestration.
- Focused tests, source provenance, and per-file SHA-256 checksums.

```text
RGB stereo pairs
  -> StableBridge Core-3 route (identity / denoise / sharpen)
  -> frozen CroCo-Stereo inference
  -> .dsp5 prediction tree
  -> official Spring packer
  -> disp1_submission.hdf5 / disp1_robustness.hdf5
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `tracks/stereo/run_stablebridge_submission.py` | Generate `.dsp5` predictions |
| `tracks/stereo/run_submission_pipeline.py` | Resumable inference + packaging workflow |
| `tracks/stereo/package_submission.py` | Validate/package existing predictions |
| `tracks/stereo/croco_stablebridge/` | Frozen StableBridge runtime |
| `vendor/croco/` | Pinned upstream CroCo source |
| `SOURCE_MANIFEST.json` | Canonical-source provenance |
| `SHA256SUMS` | Bundle integrity manifest |

## Requirements

- Linux, Python 3.12
- NVIDIA GPU with a compatible CUDA 12.4 runtime
- PyTorch 2.6.0 and torchvision 0.21.0
- Dependencies in `environment/requirements.txt`

Example environment setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r environment/requirements.txt
```

## External assets

Assets are deliberately not committed to this repository.

| Asset | Required identity |
| --- | --- |
| CroCo-Stereo Spring checkpoint | SHA-256 `123273a722a58134c6816efe5122c427815c92f82896be4d484e6d21b52da575` |
| StableBridge deployment bundle | `FROZEN_METHOD.json`, `fixed-evidence.pt`, and six ensemble heads |
| Data | Spring and RobustSpring test images |
| Packers | Official `disp1_subsampling` / `disp1_robust_subsampling` and HDF5 runtime |

The StableBridge bundle validates every artifact against the SHA-256 values in
`FROZEN_METHOD.json`. You may place checkpoints anywhere and pass explicit CLI
paths. Alternatively, set `ROCO_SPRING_CHECKPOINT_ROOT`.

## Generate predictions

Run from the repository root:

```bash
PYTHONPATH=. python tracks/stereo/run_stablebridge_submission.py \
  --output-root /path/to/predictions \
  --clean-test-root /path/to/spring/test \
  --robust-test-root /path/to/robust_spring \
  --checkpoint /path/to/crocostereo_finetune_spring.pth \
  --croco-root vendor/croco \
  --bundle /path/to/stablebridge/FROZEN_METHOD.json \
  --variants clean
```

The historical frozen clean configuration is retained at
`tracks/stereo/configs/submission-sb0-stablebridge-clean.json`. Its absolute
paths are provenance only; replace them for a new machine.

## Package an official submission

Validate first without writing an HDF5 package:

```bash
PYTHONPATH=. python tracks/stereo/package_submission.py \
  --predictions-root /path/to/predictions \
  --clean-test-root /path/to/spring/test \
  --robust-test-root /path/to/robust_spring \
  --tools-root /path/to/official/subsampling_tools \
  --runtime-root /path/to/hdf5/runtime \
  --output-dir /path/to/submission \
  --mode clean \
  --validation-level full
```

Add `--execute` only after validation passes. Use `--mode robust` or
`--mode all` only for the corresponding frozen evaluation protocol.

## Verification

```bash
sha256sum -c SHA256SUMS
PYTHONPATH=. pytest -q tests/stereo_submission
```

The focused tests require neither a GPU nor model checkpoints. Full end-to-end
verification additionally requires the external assets above.

## Licensing and data

No blanket license is granted for the first-party research code by this
repository. The vendored CroCo code retains its upstream license and notices in
`vendor/croco/LICENSE`, `vendor/croco/NOTICE`, and
`vendor/croco/NOTICE_CHECKPOINTS`. Dataset, checkpoint, and benchmark-tool terms
remain those of their respective providers; none of those assets are
redistributed here.

