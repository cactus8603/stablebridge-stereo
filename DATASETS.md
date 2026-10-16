# Datasets

Sample paths, seeds and scene folds are listed in `configs/samples.jsonl`.
Download locations and archive sizes are in `configs/datasets.json`.

| Stage | Data |
| --- | --- |
| Selector fitting and calibration | Spring train: 36 scenes, four frames per scene, 144 stereo pairs. Clean plus 20 local corruptions at three severities gives 8,784 examples, each scored on a 352×704 center crop. |
| Target construction | The same Spring RGB and left disparity. Three deployed treatments and two reference treatments give 43,920 crop forwards. |
| Local validation | Six outer folds with six scenes each; five inner folds for each outer training set. Outer scenes are excluded from head fitting and threshold selection. |
| Deployment fitting | All 36 development scenes after threshold calibration. |
| Clean submission | Spring test RGB: 1,000 stereo pairs, both disparity references. |
| Robust submission | RobustSpring test RGB: 20 corruption categories × 1,000 pairs, plus the clean maps required by the packer. |

Dataset records: [Spring](https://doi.org/10.18419/DARUS-3376) and
[RobustSpring](https://doi.org/10.18419/DARUS-5047).

Frames are sorted by filename and sampled at indices `floor(j*N/5)`, for
j = 1, 2, 3, 4. The local corruption renderer in `stablebridge/corruptions.py`
includes approximations of motion blur and weather and differs from the official
RobustSpring renderer. Training disparity is used for target construction and
depth-aware corruptions; inference takes RGB images only.

`prepare.py` uses FP32 target prediction, BF16 encoder features, FP16 prefix
quantization and FP32 feature reduction. The two reference treatments contribute
to the paper's five-choice normalization denominator.

## CroCo checkpoint ancestry

The Spring-fine-tuned CroCo-Stereo checkpoint inherits the following datasets.

- **Cross-view pretraining:** HM3D, ScanNet, Replica and ReplicaCAD through
  Habitat; ARKitScenes, MegaDepth, 3DStreetView and IndoorVL. IndoorVL includes
  Hyundai Department Store and Gangnam Station imagery.
- **Supervised stereo training:** CREStereo, SceneFlow, ETH3D low-res,
  Middlebury 2005/2006/2014/2021/v3, and Booster.
- **Upstream validation:** SceneFlow `test1of100_finalpass` and
  `test1of100_cleanpass`, ETH3D `subval`, Middlebury `subval`/`subval_full`,
  and Booster `subval_balanced`.
- **Fine-tuning:** Spring `train`, eight epochs from the main CroCo-Stereo model.

The ancestry follows the upstream
[stereo recipes](https://github.com/naver/croco/blob/5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb/stereoflow/README.MD),
[pretraining instructions](https://github.com/naver/croco/blob/5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb/datasets/crops/README.MD)
and bundled `vendor/croco/NOTICE_CHECKPOINTS`. Full IndoorVL training pairs are
not publicly distributed. Dataset and checkpoint licenses remain those of their
providers.
