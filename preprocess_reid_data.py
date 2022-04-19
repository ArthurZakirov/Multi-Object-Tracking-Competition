import os
import sys
import json
from tqdm import tqdm
import argparse
import glob
import re
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from src.detector.utils import full_image_box
from src.detector.object_detector import init_detector
from src.utils.file_utils import ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obj_detect_config_path",
    default="config\\obj_detect\\maskrcnn_from_external_proposals.json",
)
parser.add_argument("--raw_data_dir", default="data/market/query")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


def main():
    config = json.load(open(args.obj_detect_config_path, "r"))
    obj_detect = init_detector(**config)
    # obj_detect = torch.load(args.obj_detect_path)
    obj_detect.eval()

    output_dir = args.raw_data_dir + "_masked"
    raw_image_paths = glob.glob(os.path.join(args.raw_data_dir, "*.jpg"))
    output_image_paths = [
        os.path.join(output_dir, file_name)
        for file_name in os.listdir(args.raw_data_dir)
    ]
    if args.resume:
        start = len(os.listdir(output_dir))
    else:
        start = 0

    raw_image_paths = raw_image_paths[start:]
    output_image_paths = output_image_paths[start:]

    remove_count = 0
    for img_count, (output_image_path, raw_image_path) in tqdm(
        enumerate(zip(output_image_paths, raw_image_paths), start=1),
        total=len(output_image_paths),
        desc="mask images...",
    ):
        image = TF.to_tensor(Image.open(raw_image_path).convert("RGB"))

        with torch.no_grad():
            det = obj_detect.predict_on_external_proposals(
                images=[image], proposals=[full_image_box(image).unsqueeze(0)]
            )[0]

        try:
            masked_image = image * det["masks"].squeeze()

            # # if det["labels"] != 1:
            # fig = visualize_detection(image, det)
            # plt.show()

            ensure_dir(output_image_path)
            TF.to_pil_image(masked_image).save(output_image_path)

        except:
            remove_count += 1

    print(
        f"\nFinished!\n {remove_count}/{img_count} images have been removed due to errors."
    )


if __name__ == "__main__":
    main()
