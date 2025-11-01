import os
import re
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.ops import box_convert
import pycocotools.mask as mask_util
import supervision as sv
from pathlib import Path
from torchvision.ops import box_convert
import csv
from utils.avss_utils import (
    mask_iou,metric_s_for_null_batch,Eval_Fmeasure,
)
from ipdb import set_trace
import torch.nn.functional as F



JSONL_PATH = "epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/checkpoint-551/inference_cot_ref_avs_test_s_bs/inference_results.jsonl"

TEST_NAME = JSONL_PATH.split("/")[-2].split('_')[4] + "_" + JSONL_PATH.split("/")[-2].split('_')[5] # test_s / test_u / test_n
print(">>>>>>>>>> Test type: ", TEST_NAME)

MEDIA_DIR = "./REFAVS/media"    
GT_MASK_DIR = "./REFAVS/gt_mask"        
    
# R2AVSBench 
META_CSV_PATH = "./R2AVSBench/R2AVSBench_metadata.csv"  
# RefAVSBench reference using R2AVSBench videos
# META_CSV_PATH = "./R2AVSBench/RefAVSBenchRef_R2AVSBenchVideo.csv"  

print('meta csv: ', META_CSV_PATH)


uid_to_fid = {}
with open(META_CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row["uid"]
        fid = int(row["fid"])
        uid_to_fid[uid] = fid

BOX_THRESHOLD = 0.1 #
TEXT_THRESHOLD = 0.25 
USE_ORI_REF_TEXT = True # not change, cannot be True when using USE_F_OBJECT
USE_F_OBJECT = False # not change
SAVE_PRED_MASK = True # may change
COUNT_VIDEO = 0
if USE_F_OBJECT:
    OUTPUT_DIR = os.path.join(os.path.dirname(JSONL_PATH), "R2AVSBench_video_batch_infer_f_object")
    metric_log_path = os.path.join(os.path.dirname(JSONL_PATH), f"f_object_eval_results_v2_boxthre{BOX_THRESHOLD}_textthre{TEXT_THRESHOLD}.json")
else:
    OUTPUT_DIR = os.path.join(os.path.dirname(JSONL_PATH), "R2AVSBench_video_batch_infer_s_object")
    metric_log_path = os.path.join(os.path.dirname(JSONL_PATH), f"s_object_eval_results_v2_boxthre{BOX_THRESHOLD}_textthre{TEXT_THRESHOLD}_v2.json")

if USE_ORI_REF_TEXT:
    OUTPUT_DIR = os.path.join(os.path.dirname(JSONL_PATH), "R2AVSBench_butOriref_video_batch_infer")
    metric_log_path = os.path.join(os.path.dirname(JSONL_PATH), f"R2AVSBench_butOriref_eval_results_v2_boxthre{BOX_THRESHOLD}_textthre{TEXT_THRESHOLD}v2.json")


os.makedirs(OUTPUT_DIR, exist_ok=True)

metric_log = {}

if os.path.exists(metric_log_path):
    try:
        with open(metric_log_path, "r") as f:
            loaded_data = json.load(f)
            if isinstance(loaded_data, dict):
                for key, value in loaded_data.items():
                    if key not in ["overall_average_iou", "overall_average_fscore"]:
                        metric_log[key] = value
                print(f"Loaded existing evaluation results from {metric_log_path}")
            else:
                print(f"Warning: Existing file at {metric_log_path} is not in expected dictionary format. Starting with empty results.")
                metric_log = {}
    except json.JSONDecodeError:
        print(f"Warning: Could not decode existing JSON file at {metric_log_path}. Starting with empty results.")
        metric_log = {}
    except Exception as e:
        print(f"An unexpected error occurred while loading {metric_log_path}: {e}. Starting with empty results.")
        metric_log = {}


def extract_tagged_text(text, tag):
    """从 <tag> 中提取文本"""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else "Error"

def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


gdino_model = load_model(model_config_path=GROUNDING_DINO_CONFIG, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT, device=DEVICE)
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)


print("[Warning] using original ref text")
with open(META_CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row["uid"]
        ref_text = row["exp"]
        split = row["split"]
        if split != TEST_NAME: 
            continue
        if uid in metric_log:
            continue

        print(uid)
        print(ref_text)

        vid = uid.rsplit("_", 2)[0]
        frame_dir = os.path.join(MEDIA_DIR, vid, "frames")
        # if SAVE_PRED_MASK and COUNT_VIDEO < 20:
        if SAVE_PRED_MASK:
            save_dir = os.path.join(OUTPUT_DIR, uid)
            os.makedirs(save_dir, exist_ok=True)

        pred_masks = []
        pred_logits_for_fmeasure = [] 
        gt_masks = []

        for img_id in range(10):
            img_path = os.path.join(frame_dir, f"{img_id}.jpg")
            fid = uid_to_fid.get(uid, None)
            if fid is None:
                continue
            gt_mask_path = os.path.join(GT_MASK_DIR, vid, f"fid_{fid}", f"{img_id:05d}.png")

            if not os.path.exists(img_path) or not os.path.exists(gt_mask_path):
                continue

            image_source, image = load_image(img_path)
            image_source = cv2.resize(image_source, (224, 224))
            image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB) 
            h, w, _ = image_source.shape

            if USE_ORI_REF_TEXT:
                boxes, confidences, labels = predict(
                    model=gdino_model,
                    image=image,
                    caption=ref_text.lower().strip(), # remove '.'
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device=DEVICE,
                )
            else:
                boxes, confidences, labels = predict(
                    model=gdino_model,
                    image=image,
                    caption=ref_text.lower().strip() + ".",
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device=DEVICE,
                )

            boxes = boxes * torch.Tensor([w, h, w, h])
            boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

            current_pred_mask = torch.zeros((1, 224, 224), dtype=torch.uint8)
            current_pred_logit = torch.zeros((1, 224, 224), dtype=torch.float32)

            if boxes.shape[0] == 0:
                pred_masks.append(current_pred_mask)
                pred_logits_for_fmeasure.append(current_pred_logit)
                gt = np.array(Image.open(gt_mask_path).convert("L").resize((224, 224))).astype(bool)
                gt_masks.append(torch.from_numpy(gt).unsqueeze(0).to(torch.uint8))
                continue
            else:
                boxes = boxes[:1]  # top-1

            sam2_predictor.set_image(image_source)
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )

            if masks is None or len(masks) == 0:
                pred_masks.append(current_pred_mask)
                pred_logits_for_fmeasure.append(current_pred_logit) 
                gt = np.array(Image.open(gt_mask_path).convert("L").resize((224, 224))).astype(bool)
                gt_masks.append(torch.from_numpy(gt).unsqueeze(0).to(torch.uint8))
                continue

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if masks.ndim == 4:
                masks = np.squeeze(masks, axis=1)
            elif masks.ndim == 3 and masks.shape[0] == 1:
                masks = masks
            else:
                raise ValueError(f"Unexpected mask shape: {masks.shape}")

            mask = masks[0].astype(np.uint8)
            pred_masks.append(torch.from_numpy(mask).unsqueeze(0))

            processed_logits = current_pred_logit
            if logits is not None:
                if isinstance(logits, torch.Tensor):
                    temp_logits = logits.cpu().squeeze(1)
                elif isinstance(logits, np.ndarray):
                    temp_logits = torch.from_numpy(logits).float().squeeze(1)
                else:
                    print(f"Warning: Unexpected type for logits: {type(logits)}. Appending zero logit.")
                    temp_logits = None

                if temp_logits is not None:
                    if temp_logits.ndim == 2:
                        temp_logits = temp_logits.unsqueeze(0).unsqueeze(0)
                    elif temp_logits.ndim == 3 and temp_logits.shape[0] == 1:
                        temp_logits = temp_logits.unsqueeze(0) 
                    
                    processed_logits_tensor = F.interpolate(
                        temp_logits,
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).to(torch.float32)

                    processed_logits_tensor = torch.sigmoid(processed_logits_tensor)
                
                
            pred_logits_for_fmeasure.append(processed_logits_tensor.to(torch.float32))

            gt = np.array(Image.open(gt_mask_path).convert("L").resize((224, 224))).astype(bool)
            gt_masks.append(torch.from_numpy(gt).unsqueeze(0).to(torch.uint8))

            # if SAVE_PRED_MASK and COUNT_VIDEO < 20:
            if SAVE_PRED_MASK:
                detections = sv.Detections(
                    xyxy=boxes,
                    mask=np.array([mask], dtype=bool),
                    class_id=np.array([0])
                )
                annotated = sv.BoxAnnotator().annotate(image_source.copy(), detections)
                annotated = sv.MaskAnnotator().annotate(annotated, detections)
                annotated = sv.LabelAnnotator().annotate(annotated, detections, labels=[ref_text])
                cv2.imwrite(os.path.join(save_dir, f"{img_id}_vis.jpg"), annotated)
            
        COUNT_VIDEO += 1


        pred_masks = [t if t.dim() == 3 else t.unsqueeze(0) for t in pred_masks]
        gt_masks = [t if t.dim() == 3 else t.unsqueeze(0) for t in gt_masks]
        pred_logits_for_fmeasure = [t if t.dim() == 3 else t.unsqueeze(0) for t in pred_logits_for_fmeasure]
        
        if not pred_masks or not gt_masks:
            print(f"Warning: No valid frames processed for uid: {uid}. Skipping evaluation for this UID.")
            mean_iou = 0.0
            fscore = 0.0
        else:
            pred_masks_tensor = torch.cat(pred_masks, dim=0)
            gt_masks_tensor = torch.cat(gt_masks, dim=0)
            pred_logits_tensor_for_fmeasure = torch.cat(pred_logits_for_fmeasure, dim=0)

            if TEST_NAME == 'test_n': # null subset
                if ref_text == 'null':
                    mean_iou = 0.0
                else:
                    mean_iou = metric_s_for_null_batch(pred_masks_tensor)
                fscore = -999 
            else:
                mean_iou = mask_iou(pred_masks_tensor, gt_masks_tensor) 
                fscore = Eval_Fmeasure(pred=pred_logits_tensor_for_fmeasure.cpu(),gt=gt_masks_tensor.cpu())
        

        print(f"uid: {uid}, mean_iou: {mean_iou}, fscore: {fscore}")
        f_mean_iou, s_mean_iou = 0, 0
        f_fscore, s_fscore = 0, 0
        if USE_F_OBJECT:
            f_mean_iou = mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou
            f_fscore = fscore.item() if isinstance(fscore, torch.Tensor) else fscore
        else:
            s_mean_iou = mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou
            s_fscore = fscore.item() if isinstance(fscore, torch.Tensor) else fscore
    
        metric_log[uid] = {
            "vid": vid,
            "ref": ref_text,
            # "f_object": f_object,
            # "s_object": s_object,
            "f_mean_iou": f_mean_iou,
            "s_mean_iou": s_mean_iou,
            "f_fscore": f_fscore,
            "s_fscore": s_fscore
        }

        with open(metric_log_path, "w") as f:
            json.dump(metric_log, f, indent=4)

valid_ious = []
valid_fscores = []

for item_data in metric_log.values():
    if USE_F_OBJECT:
        iou_val = item_data.get("f_mean_iou")
        fscore_val = item_data.get("f_fscore")
    else:
        iou_val = item_data.get("s_mean_iou")
        fscore_val = item_data.get("s_fscore")
    
    if isinstance(iou_val, (int, float)) and iou_val >= 0:
        valid_ious.append(iou_val)
    if isinstance(fscore_val, (int, float)) and fscore_val >= 0:
        valid_fscores.append(fscore_val)

average_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
average_fscore = sum(valid_fscores) / len(valid_fscores) if valid_fscores else 0.0
print(f"\nAverage IoU over all samples: {average_iou:.4f}")
print(f"Average F-score over all samples: {average_fscore:.4f}")

final_results = {
    "overall_average_iou": f"{average_iou:.4f}",
    "overall_average_fscore": f"{average_fscore:.4f}",
    **metric_log 
}

with open(metric_log_path, "w") as f:
    json.dump(final_results, f, indent=4)

print(f"save to: {metric_log_path}")
set_trace()