#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
restore=${RESTORE_PYTHON:-python}
detect=${DETECT_PYTHON:-python}
mkdir -p results/yolo
for operator in clahe gamma dcp retinex msrcr; do
    if [ ! -f "datasets/bdd100k/$operator/all.txt" ]; then
        "$restore" scripts/enhance.py --operator "$operator" --source datasets/bdd100k/original/images --target "datasets/bdd100k/$operator/images"
        rm -rf "datasets/bdd100k/$operator/labels"
        cp -r datasets/bdd100k/original/labels "datasets/bdd100k/$operator/labels"
        for subset in all rainy snowy night; do
            sed 's/\.jpg$/.png/' "datasets/bdd100k/original/$subset.txt" > "datasets/bdd100k/$operator/$subset.txt"
        done
    fi
    for subset in all rainy snowy night; do
        sed "s#^path: .*#path: datasets/bdd100k/$operator#; s#^train: .*#train: $subset.txt#; s#^val: .*#val: $subset.txt#" "options/local/bdd_original_$subset.yaml" > "options/local/bdd_${operator}_$subset.yaml"
    done
done
for operator in clahe gamma dcp retinex msrcr; do
    for detector in yolov8s yolo11s yolo26s; do
        for subset in all rainy snowy night; do
            out="results/yolo/bdd_enhance_${operator}_${detector}_$subset.json"
            [ -f "$out" ] || "$detect" scripts/rtts_bootstrap.py --weights "pretrained_models/$detector.pt" --base "options/local/bdd_original_$subset.yaml" --treated "options/local/bdd_${operator}_$subset.yaml" --out "$out"
        done
    done
done
