# Compute

Measurements used NVIDIA RTX 3090 GPUs (24 GiB), driver 550.163.01 and
PyTorch 2.6.0+cu124. Each stage can run serially on one GPU. Cached-feature
fitting also supports CPU; image preparation and inference require CUDA with
BF16 support. CroCo uses the bundled PyTorch RoPE implementation.

## Training and data preparation

| Workload | GPUs | Wall time | Peak allocated / reserved |
| --- | ---: | ---: | ---: |
| Targets for five treatments, 43,920 forwards | 4 | 69 min | — |
| Feature extraction | 4 | 30 min | — |
| Nested scene-held-out fitting, 216 heads | 1 | 43.49 min | — |
| Six deployment heads, 80 epochs, batch 512 | 1 | 152.17 s | 68.65 / 92.00 MiB |
| Preparation of four examples, targets and features | 1 | 44.70 s | 5.442 / 7.211 GiB |

The first three timings cover complete reference runs. The last two used the
supplied scripts on a shared host. Head fitting uses cached features without
loading CroCo. The four-example preparation includes model loading and does
not measure full-dataset elapsed time.

## Inference

Submission prediction uses FP32 CroCo and BF16 selector features. A forward
pass on four 1080 × 1920 pairs, producing eight disparity maps, took 410.41 s
and used 6.821 GiB peak allocated memory on a shared GPU. This excludes image
decoding and selector routing.

The paper's BF16 audit used batch size one, overlap 0.9, one warm-up pair and
eight measured clean pairs, excluding image decoding:

| Component | Median time per pair | Peak allocated memory |
| --- | ---: | ---: |
| Five-crop selector | 1.219 s | 2.590 GiB |
| CroCo full-resolution forward | 89.293 s | 2.115 GiB |
| Selector + CroCo | 90.584 s | — |

The selector and CroCo memory peaks were measured separately. All memory values
are PyTorch measurements and exclude CUDA context memory and other processes.
Training and preparation record time and memory in `resources.json` for the
current invocation; prediction `run.json` records the last disparity batch.

## CPU and storage

Use four CPU numerical threads and budget about 16 GiB of host RAM; these are
planning allowances. Spring archives total 40.35 GiB compressed and RobustSpring
archives 101.94 GiB. Downloads are deleted after extraction.

Full submission inference writes 42,000 float32 1080 × 1920 maps, about 324 GiB
before compression. Allow at least 500 GiB for datasets, predictions and packing,
with additional extraction headroom as needed.
