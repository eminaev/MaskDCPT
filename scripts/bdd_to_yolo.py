import argparse
import json
import os

COCO = {"pedestrian": 0, "rider": 0, "other person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "train": 6, "truck": 7, "traffic light": 9}
SUBSETS = {"all": lambda sample: True, "rainy": lambda sample: sample["weather"] == "rainy", "snowy": lambda sample: sample["weather"] == "snowy", "night": lambda sample: sample["time"] == "night"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="datasets/bdd100k_raw/samples.json")
    parser.add_argument("--images", default="datasets/bdd100k_zip/extracted/bdd100k/images/100k/val")
    parser.add_argument("--out", default="datasets/bdd100k/original")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    samples = []
    boxes = 0
    for record in json.load(open(args.samples))["samples"]:
        name = os.path.basename(record["filepath"])
        target = os.path.join(args.out, "images", name)
        if not os.path.exists(target):
            os.link(os.path.join(args.images, name), target)
        with open(os.path.join(args.out, "labels", os.path.splitext(name)[0] + ".txt"), "w") as handle:
            for detection in (record.get("detections") or {}).get("detections", []):
                if detection["label"] not in COCO:
                    continue
                left, top, span, tall = detection["bounding_box"]
                left, top, span, tall = max(left, 0.0), max(top, 0.0), min(left + span, 1.0) - max(left, 0.0), min(top + tall, 1.0) - max(top, 0.0)
                if span <= 0 or tall <= 0:
                    print(f"Box {detection['bounding_box']} of image {name} lies outside its frame, dropping it")
                    continue
                handle.write(f"{COCO[detection['label']]} {left + span / 2:.6f} {top + tall / 2:.6f} {span:.6f} {tall:.6f}\n")
                boxes += 1
        samples.append({"name": name, "weather": (record.get("weather") or {}).get("label"), "time": (record.get("timeofday") or {}).get("label")})
    for subset, test in SUBSETS.items():
        with open(os.path.join(args.out, f"{subset}.txt"), "w") as handle:
            handle.writelines(f"./images/{sample['name']}\n" for sample in samples if test(sample))
    print(f"wrote {len(samples)} label files with {boxes} boxes")


if __name__ == "__main__":
    main()
