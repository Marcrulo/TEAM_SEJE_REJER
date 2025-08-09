import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from utils import decode_request, encode_request, validate_segmentation
from datetime import datetime
import imageio
import pickle
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing  # ← import the helper


load_dotenv()

# Configuration
SECRET_API_KEY = os.getenv("API_KEY")
MODEL_FOLDER = os.getenv(
    "MODEL_FOLDER",
    "/nnUNet_results/Dataset400_TumorSegWithAugm/nnUNetTrainer_450epochs__nnUNetPlans__2d"
)
FOLDS = [0,1,2,3,4]#'all'  # use all folds
CHECKPOINT = os.getenv("CHECKPOINT_NAME", "checkpoint_best.pth")
DEVICE = torch.device(
    os.getenv("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
)

# ————————————————————————————————
# Load postprocessing funcs once
pp_pkl = os.path.join(
    MODEL_FOLDER,
    "crossval_results_folds_0_1_2_3_4",
    "postprocessing.pkl"
)
with open(pp_pkl, "rb") as f:
    pp_fns, pp_fn_kwargs = pickle.load(f)
# ————————————————————————————————

# Instantiate predictor once
def create_predictor():
    predictor = nnUNetPredictor(
        tile_step_size=1.0,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=DEVICE
    )
    predictor.initialize_from_trained_model_folder(
        MODEL_FOLDER,
        use_folds=FOLDS,
        checkpoint_name=CHECKPOINT
    )
    return predictor

app = Flask(__name__)

@app.route("/")
def root():
    return "nnU-Net Tumor Segmentation Service"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    
    req_dto = TumorPredictRequestDto(**request.get_json())
    
    # decode_request returns HxWxC (PNG) -> take single channel
    img = decode_request(req_dto) # shape (H, W, 3)
    img_2d = img[..., 0].astype(np.float32)  #(H,W)

    # Prepare 4D input: (cases=1, channels=1, H, W)
    raw_data = img_2d[None, None, ...]

    # Seed props so preprocessor knows how to undo transforms
    props = {'spacing': np.array([1.0, 1.0, 1.0], dtype=float), 'original_shape': img_2d.shape}

    # Preprocess: data_pp shape (1, 1, H, W)
    data_pp, _, props = preproc.run_case_npy(
        raw_data, None, props,
        plans_manager, cfg, predictor.dataset_json
    )

    # Predict logits (no weight reload)
    input_tensor = torch.from_numpy(data_pp).to(DEVICE)
    logits = predictor.predict_logits_from_preprocessed_data(input_tensor)

    probs = torch.sigmoid(logits).cpu().numpy()[0][0]   #(H, W)

    seg = (probs <= 0.49).astype(np.uint8)
    
    # Apply postprocessing functions
    seg_pp = apply_postprocessing(seg, pp_fns, pp_fn_kwargs)
    seg_pp = (seg_pp * 255).astype(np.uint8) 

    # HxWx3 mask
    seg_pp = np.stack([seg_pp, seg_pp, seg_pp], axis=-1)

    validate_segmentation(img, seg_pp)

    resp_dto = TumorPredictResponseDto(
        img=encode_request(seg_pp)
    )
    return jsonify(resp_dto.model_dump())

if __name__ == '__main__':
    predictor = create_predictor()
    print(f"Loaded nnU-Net model from {MODEL_FOLDER} on {DEVICE} with folds={FOLDS}")

    # Instantiate preprocessor once
    plans_manager = predictor.plans_manager
    cfg = predictor.configuration_manager
    preproc = cfg.preprocessor_class(verbose=False)
    
    port = int(os.getenv('PORT', '4957'))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )
