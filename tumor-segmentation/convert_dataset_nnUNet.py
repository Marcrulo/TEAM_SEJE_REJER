import os
import shutil
import argparse
import json
from glob import glob

import imageio.v2 as imageio
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert cancer patient + healthy control PET images into nnU-Net raw format"
    )
    p.add_argument(
        "--data_root", required=True,
        help="Root of your data, containing 'patients/imgs', 'patients/labels', and 'controls/imgs'"
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Where to write nnUNet_raw/DatasetXXX_Name"
    )
    p.add_argument(
        "--dataset_id", default="100",
        help="3-digit ID for this dataset (e.g. 100 → Dataset100_Name)"
    )
    p.add_argument(
        "--dataset_name", default="TumorSeg",
        help="Name for this dataset (e.g. 'TumorSeg')"
    )
    return p.parse_args()

def make_dirs(base):
    for sub in ("imagesTr", "labelsTr"):
        d = os.path.join(base, sub)
        os.makedirs(d, exist_ok=True)

def save_single_channel(img_path, dst_path):
    # Read image and ensure single-channel uint8
    img = imageio.imread(img_path)
    # images from data_2025 are RGBA, take the first channel
    if img.ndim > 2:
        img = img[..., 0]

    img = img.astype(np.uint8)
    imageio.imwrite(dst_path, img)

def copy_patients(data_root, out_base):
    imgs_dir = os.path.join(data_root, "patients", "imgs")
    lbls_dir = os.path.join(data_root, "patients", "labels")
    for img_path in sorted(glob(os.path.join(imgs_dir, "*.png"))):
        fname = os.path.basename(img_path)
        case = os.path.splitext(fname)[0]
        
        # Write image as case_0000.png ensuring single channel
        dst_img = os.path.join(out_base, "imagesTr", f"{case}_0000.png")
        save_single_channel(img_path, dst_img)

        idx = case.split("_")[-1]
        seg_name = f"segmentation_{idx}.png"
        seg_path = os.path.join(lbls_dir, seg_name)
        if not os.path.isfile(seg_path):
            raise FileNotFoundError(f"Missing segmentation for {case}: {seg_path}")
        dst_seg = os.path.join(out_base, "labelsTr", f"{case}.png")
        
        # Read, binarize (>0 -> 1), and save
        mask = imageio.imread(seg_path)
        if mask.ndim > 2:
            mask = mask[..., 0]
        bin_mask = (mask > 0).astype(np.uint8)
        imageio.imwrite(dst_seg, bin_mask)

def copy_controls(data_root, out_base):
    ctr_dir = os.path.join(data_root, "controls", "imgs")
    for img_path in sorted(glob(os.path.join(ctr_dir, "*.png"))):
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]
        case = base
        
        # Write control image
        dst_img = os.path.join(out_base, "imagesTr", f"{case}_0000.png")
        save_single_channel(img_path, dst_img)
        
        # create blank segmentation
        img = imageio.imread(img_path)
        if img.ndim > 2:
            img = img[..., 0]
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        dst_seg = os.path.join(out_base, "labelsTr", f"{case}.png")
        imageio.imwrite(dst_seg, mask)

def write_dataset_json(out_base, n_cases, file_ending=".png"):
    d = {
        "channel_names": {"0": "MIP-PET"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": n_cases,
        "file_ending": file_ending
    }
    with open(os.path.join(out_base, "dataset.json"), "w") as f:
        json.dump(d, f, indent=2)

def main():
    args = parse_args()
    ds_folder = os.path.join(
        args.output_dir,
        f"Dataset{args.dataset_id.zfill(3)}_{args.dataset_name}"
    )
    make_dirs(ds_folder)
    copy_patients(args.data_root, ds_folder)
    copy_controls(args.data_root, ds_folder)
    n_cases = len(glob(os.path.join(ds_folder, "imagesTr", "*_0000.png")))
    print(f"→ total cases: {n_cases}")
    write_dataset_json(ds_folder, n_cases, file_ending=".png")
    print("Conversion complete!")
    print(f"  • imagesTr → {os.path.join(ds_folder, 'imagesTr')}")
    print(f"  • labelsTr → {os.path.join(ds_folder, 'labelsTr')}")
    print(f"  • dataset.json → {os.path.join(ds_folder, 'dataset.json')}")

if __name__ == "__main__":
    main()
