# Study II: NAFNetMobile as a front end for YOLO detectors

This fork of [MILab-PKU/MaskDCPT](https://github.com/MILab-PKU/MaskDCPT) contains the code and results of Study II of the paper "Enhance, Restore, or Adapt? An Empirical Study of YOLO Object Detection under Adverse Imaging Conditions".
Study II distills the MaskDCPT NAFNet 5D model into NAFNetMobile and tests the distilled network as a preprocessing stage for YOLOv8s, YOLO11s, and YOLO26s on RTTS and BDD100K.
It also repeats the Study I matrix of five classical enhancement operators on BDD100K with the paired bootstrap.

## Contents

| Path | Purpose | Paper |
|---|---|---|
| `basicsr/archs/nafnet_mobile_arch.py` | NAFNetMobile with batch normalization, the bounded gate, and folding for deployment | Section 3.4, Table 2 |
| `basicsr/losses/feature_distill_loss.py` | Feature distillation loss with learned adapters | Section 3.4 |
| `scripts/distill_mobile.py` | Distillation of the S and M students from the NAFNet 5D teacher | Table 6 |
| `scripts/export_torchscript.py` | Folding and TorchScript export of M for the latency measurements | Section 4.3.1 |
| `scripts/rtts_to_yolo.py` | RTTS annotations to YOLO labels | Section 3.6 |
| `scripts/bdd_to_yolo.py` | BDD100K validation annotations to YOLO labels with COCO class IDs and the rain, snow, and night subsets | Section 3.6 |
| `scripts/restore_folder.py` | Restoration of an image folder with M | Section 3.4 |
| `scripts/rtts_bootstrap.py` | Detector evaluation with the paired bootstrap over images | Table 8, Figure 5 |
| `scripts/enhance.py` | CLAHE, gamma correction, DCP, single-scale Retinex, and MSRCR with the parameters of Section 3.3 | Section 3.3 |
| `scripts/enhancement_bootstrap.sh` | Enhancement of BDD100K with every operator and the paired bootstrap for three detectors and four subsets | Section 4.2, Table A6 |
| `scripts/qualitative.py` | Selection of the example images and drawing of the qualitative figures | Section 4.5, Figures 7 and 8 |
| `scripts/pipeline_latency.py` | Latency of the detector, of M, and of the whole pipeline | Table 9 |
| `scripts/study2.sh` | All steps above in order | — |
| `options/local/` | Restoration test configs and YOLO dataset configs | — |
| `results/yolo/*.json` | Bootstrap and latency results used in the paper | Tables 8, 9, and A6 |
| `tests/` | Unit tests for the new architecture, loss, export, operators, parallel bootstrap, and box matching | — |

## Environment

The experiments used two conda environments on one NVIDIA RTX 4090.

- Restoration: the MaskDCPT environment from `environment.yaml` (Python 3.10, PyTorch 2.0.1 with CUDA 11.8, OpenCV 4.8).
- Detection: Python 3.11, PyTorch 2.5.1 with CUDA 12.1, Ultralytics 8.4.175, OpenCV 5.0.

The runner takes both interpreters from environment variables:

```bash
RESTORE_PYTHON=~/anaconda3/envs/maskdcpt/bin/python DETECT_PYTHON=~/anaconda3/envs/yolo/bin/python scripts/study2.sh
```

Add `TRAIN=1` to repeat the distillation, which takes about 13 hours on an RTX 4090.
Without it, the runner expects the released weights in `experiments/mobile_v4_feat/`.

The classical operator matrix runs separately with the same variables and expects the original BDD100K subset prepared by `scripts/study2.sh`:

```bash
RESTORE_PYTHON=~/anaconda3/envs/maskdcpt/bin/python DETECT_PYTHON=~/anaconda3/envs/yolo/bin/python scripts/enhancement_bootstrap.sh
```

It writes about 48 GB of PNG images, and its bootstrap takes about five hours with 32 CPU workers.

The qualitative figures need the restored and CLAHE images from both runners.
`select` picks the examples into `results/yolo/qualitative.json`, and the other two commands draw Figures 7 and 8 into the given folder:

```bash
~/anaconda3/envs/yolo/bin/python scripts/qualitative.py select
~/anaconda3/envs/yolo/bin/python scripts/qualitative.py appearance figures
~/anaconda3/envs/yolo/bin/python scripts/qualitative.py detections figures
```

## Weights

The release [Study II weights](https://github.com/eminaev/MaskDCPT/releases) contains:

- `nafnet_mobile_M.pth` and `nafnet_mobile_S.pth`, the final students (v4 in Table 6);
- `nafnet_mobile_M_fused.ts`, the folded M traced at 1280 × 720.

Put them into `experiments/mobile_v4_feat/`.
The teacher `nafnet_maskdcpt_5d.pth` comes from the [MaskDCPT weights](https://huggingface.co/Jiakui/MaskDCPT) and goes into `pretrained_models/`.
YOLO weights go into `pretrained_models/` as `yolov8s.pt`, `yolo11s.pt`, and `yolo26s.pt`; Ultralytics downloads them on first use.

## Data

The datasets are not redistributed here because of their licenses.
The scripts expect the following layout under `datasets/`.

| Folder | Content | Source used in the paper |
|---|---|---|
| `train/noise/clean` | BSD400 | MaskDCPT training sets |
| `train/rain/{input,target}` | RainTrainL | MaskDCPT training sets |
| `train/haze/{hazy,clear}` | RESIDE OTS, without the 45 scenes that also appear in SOTS | [Anbinh93/image_restoration](https://huggingface.co/datasets/Anbinh93/image_restoration) |
| `train/lowlight/{low,high}` | LOL training pairs | [okita-souji/LOLv1](https://huggingface.co/datasets/okita-souji/LOLv1) |
| `train/blur/{input,target}` | GoPro training pairs | [laoduanaaa/Image_Restoration_Datasets](https://huggingface.co/datasets/laoduanaaa/Image_Restoration_Datasets) |
| `derain`, `denoise`, `dehaze`, `lowlight`, `deblur` | 5D test sets, see `options/local/test_NAFNet_5d.yml` | MaskDCPT test sets |
| `rtts_raw` | RTTS images and COCO-format annotations | [fuhao7i/RTTS](https://huggingface.co/datasets/fuhao7i/RTTS) |
| `bdd100k_raw/samples.json` | BDD100K 2020 detection labels of the validation set | [dgural/bdd100k](https://huggingface.co/datasets/dgural/bdd100k) |
| `bdd100k_zip/extracted/bdd100k/images/100k/val` | BDD100K validation images | [hirundo-io/bdd100k-val](https://huggingface.co/datasets/hirundo-io/bdd100k-val) |

BDD100K boxes of pedestrian, rider, and other person map to COCO person; bicycle, car, motorcycle, bus, train, truck, and traffic light map to the COCO classes of the same name; traffic sign, other vehicle, and trailer are dropped.
This leaves 151,222 boxes in 10,000 images.

## Results

Each bootstrap file holds the full-set metrics of both inputs and the mean and 95% interval of the mAP50 difference, overall (`gap.all`) and per COCO class.
`bootstrap.json` and `latency.json` belong to YOLOv8s; the other files carry the detector name.
Files named `bdd_enhance_<operator>_<detector>_<subset>.json` hold the classical operator matrix; their DCP omits guided-filter refinement of the transmission.
`qualitative.json` names the selected images and counts, for each subset, the images on which the treatment raised or lowered the number of YOLOv8s true positives; these counts cover only images with 1 to 30 annotated objects.
The bootstrap draws all resamples from one generator before scoring them in parallel, so results do not depend on the number of workers.

## License

The files added in this fork, listed in the table above, are released under the GNU Affero General Public License v3.0 (`LICENSE`).
They import Ultralytics, which uses the same license.
The original MaskDCPT code carries no license from its authors and keeps their terms; please refer to the [upstream repository](https://github.com/MILab-PKU/MaskDCPT) for it.
