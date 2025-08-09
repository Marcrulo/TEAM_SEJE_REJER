import os
from pathlib import Path
import cv2
import numpy as np
import albumentations as A

def main(imgs_dir: Path, masks_dir: Path, out_imgs_dir: Path, out_masks_dir: Path):
    """
    Read each image/mask pair, apply transforms, and save to output directories preserving format.
    """
    # Define your augmentation pipeline
    transforms = A.Compose([
        # geometric flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # salt & pepper (your original)
        A.SaltAndPepper(p=0.5),
    ])

    # Make sure output directories exist
    out_imgs_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over images
    for img_path in sorted(imgs_dir.glob('*.png')):
        stem = img_path.stem  # e.g. 'patient_000'
        # extract the numeric ID from 'patient_000'
        parts = stem.split('_')
        if len(parts) < 2:
            print(f"Warning: unexpected image name '{stem}', skipping")
            continue
        idx = parts[-1]  # '000'
        mask_name = f"segmentation_{idx}.png"
        mask_path = masks_dir / mask_name
        if not mask_path.exists():
            print(f"Warning: mask '{mask_name}' not found for image '{stem}', skipping")
            continue

        # Read image and mask
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        # Ensure single-channel mask
        if mask.ndim == 3:
            mask = mask[..., 0]

        # Apply transforms (albumentations expects HxWxC)
        augmented = transforms(image=img, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        # If mask is single-channel, convert back to grayscale image
        if mask_aug.ndim == 2:
            mask_to_save = mask_aug
        else:
            mask_to_save = mask_aug[..., 0]

        # Save as same format (PNG)
        img_out_path = out_imgs_dir / f"{stem}.png"
        mask_out_path = out_masks_dir / f"segmentation_{idx}.png"
        cv2.imwrite(str(img_out_path), img_aug)
        cv2.imwrite(str(mask_out_path), mask_to_save)

    print("Augmentation and save complete.")

if __name__ == '__main__':
   imgs_dir = Path('/tumor-segmentation/data_2025/patients/imgs')
   masks_dir = Path('/tumor-segmentation/data_2025/patients/labels')
   out_imgs_dir = Path('/tumor-segmentation/data_2025/patients_aug/imgs')
   out_masks_dir = Path('/tumor-segmentation/data_2025/patients_aug/labels')
   main(imgs_dir, masks_dir, out_imgs_dir, out_masks_dir)
