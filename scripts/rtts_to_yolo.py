import argparse
import json
import os

import cv2

COCO = {"person": 0, "bicycle": 1, "car": 2, "motorbike": 3, "bus": 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/rtts_raw")
    parser.add_argument("--out", default="datasets/rtts/hazy")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    boxes = 0
    for part in ["trainval", "test"]:
        record = json.load(open(os.path.join(args.raw, "coco", f"instances_{part}2017.json")))
        kinds = {category["id"]: COCO[category["name"]] for category in record["categories"]}
        lines = {image["id"]: [] for image in record["images"]}
        for annotation in record["annotations"]:
            lines[annotation["image_id"]].append(annotation)
        for image in record["images"]:
            source = os.path.join(args.raw, "VOCdevkit/VOC2007/JPEGImages", image["file_name"])
            height, width = cv2.imread(source).shape[:2]
            if (height, width) != (image["height"], image["width"]):
                print(f"Image {image['file_name']} is {width}x{height} but annotation says {image['width']}x{image['height']}, clipping its boxes")
            target = os.path.join(args.out, "images", image["file_name"])
            if not os.path.exists(target):
                os.link(source, target)
            with open(os.path.join(args.out, "labels", os.path.splitext(image["file_name"])[0] + ".txt"), "w") as handle:
                for annotation in lines[image["id"]]:
                    left, top = max(annotation["bbox"][0], 0), max(annotation["bbox"][1], 0)
                    span, tall = min(annotation["bbox"][0] + annotation["bbox"][2], width) - left, min(annotation["bbox"][1] + annotation["bbox"][3], height) - top
                    if span <= 0 or tall <= 0:
                        print(f"Box {annotation['bbox']} of image {image['file_name']} lies outside its {width}x{height} frame, dropping it")
                        continue
                    handle.write(f"{kinds[annotation['category_id']]} {(left + span / 2) / width:.6f} {(top + tall / 2) / height:.6f} {span / width:.6f} {tall / height:.6f}\n")
                    boxes += 1
    print(f"wrote {len(os.listdir(os.path.join(args.out, 'labels')))} label files with {boxes} boxes")


if __name__ == "__main__":
    main()
