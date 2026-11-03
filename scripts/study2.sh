#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
restore=${RESTORE_PYTHON:-python}
detect=${DETECT_PYTHON:-python}
weights=experiments/mobile_v4_feat
if [ "${TRAIN:-0}" = 1 ]; then
    "$restore" scripts/distill_mobile.py --iters 200000 --features 0.5 --out "$weights"
fi
"$restore" basicsr/test.py -opt options/local/test_NAFNet_5d.yml
"$restore" basicsr/test.py -opt options/local/test_NAFNetMobile_M_5d.yml
"$restore" basicsr/test.py -opt options/local/test_NAFNetMobile_S_5d.yml
"$restore" scripts/export_torchscript.py --weights "$weights/nafnet_mobile_M.pth" --out "$weights/nafnet_mobile_M_fused.ts"
"$detect" scripts/rtts_to_yolo.py
"$detect" scripts/bdd_to_yolo.py
for set in rtts bdd100k; do
    source=datasets/$set/$([ "$set" = rtts ] && echo hazy || echo original)
    "$restore" scripts/restore_folder.py --weights "$weights/nafnet_mobile_M.pth" --source "$source/images" --target "datasets/$set/restored_m/images"
    rm -rf "datasets/$set/restored_m/labels"
    cp -r "$source/labels" "datasets/$set/restored_m/labels"
done
for subset in all rainy snowy night; do
    sed 's/\.jpg$/.png/' "datasets/bdd100k/original/$subset.txt" > "datasets/bdd100k/restored_m/$subset.txt"
done
mkdir -p results/yolo
for detector in yolov8s yolo11s yolo26s; do
    suffix=$([ "$detector" = yolov8s ] && echo "" || echo "_$detector")
    "$detect" scripts/rtts_bootstrap.py --weights "pretrained_models/$detector.pt" --out "results/yolo/bootstrap$suffix.json"
    "$detect" scripts/pipeline_latency.py --detector "pretrained_models/$detector.pt" --out "results/yolo/latency$suffix.json"
    "$detect" scripts/pipeline_latency.py --detector "pretrained_models/$detector.pt" --images datasets/bdd100k/original/images --out "results/yolo/bdd_latency_$detector.json"
    for subset in all rainy snowy night; do
        "$detect" scripts/rtts_bootstrap.py --weights "pretrained_models/$detector.pt" --base "options/local/bdd_original_$subset.yaml" --treated "options/local/bdd_restored_m_$subset.yaml" --out "results/yolo/bdd_bootstrap_${detector}_$subset.json"
    done
done
