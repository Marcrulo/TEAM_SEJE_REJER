# nnUNet Setup, Training and Inference

Folder `/tumor-segmentation/training-nnUNet` is cloned from the original nnUNet repository:  
[https://github.com/MIC-DKFZ/nnUNet.git](https://github.com/MIC-DKFZ/nnUNet.git)

Installation instructions for nnUNet can be found here:  
`/tumor-segmentation/training-nnUNet/documentation/installation_instructions.md`

---

## Setup

Set the relevant path variables by running:

```bash
source /tumor-segmentation/setup_nnunet.sh
```

---

## Training

Run the following commands to train models using **5-fold cross-validation** on the provided dataset:

```bash
python convert_dataset_nnUNet.py --data_root /tumor-segmentation/data_2025 --output_dir /tumor-segmentation/nnUNet_raw
```

```bash
nnUNetv2_plan_and_preprocess -d 400 --verify_dataset_integrity
```

```bash
nnUNetv2_train 400 2d 0 --npz -tr nnUNetTrainer_450epochs
```

```bash
nnUNetv2_train 400 2d 1 --npz -tr nnUNetTrainer_450epochs
```

```bash
nnUNetv2_train 400 2d 2 --npz -tr nnUNetTrainer_450epochs
```

```bash
nnUNetv2_train 400 2d 3 --npz -tr nnUNetTrainer_450epochs
```

```bash
nnUNetv2_train 400 2d 4 --npz -tr nnUNetTrainer_450epochs
```

```bash
nnUNetv2_find_best_configuration 400 -c 2d -tr nnUNetTrainer_450epochs
```
## Inference

The trained models can be downloaded here: 
[https://drive.google.com/drive/folders/1on6A4-uxbwoaA6ILsL_qptTeWs7qUps7?usp=sharing]([https://drive.google.com/drive/folders/1on6A4-uxbwoaA6ILsL_qptTeWs7qUps7?usp=sharing)

Once the model folder is downloaded, run the following command for inference:
```bash
python /tumor-segmentation/inference/app_nnunet_opt_post.py
```
Change line 21 to the real path to the downloaded model.