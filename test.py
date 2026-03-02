import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from utils import setup_seed, cos_sim
from model.moe_adapter import MoECLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS, DATA_PATH
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    group_indices_by_variant,
    metrics_eval,
    calcuate_metric_pixel,
)
import warnings
import re
import sys
import cv2
import time

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_epoch_number(filename):
    match = re.search(r'moe_epoch_(\d+)\.pth', filename)
    return int(match.group(1)) if match else -1

def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader: 
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features

def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    results = {"cls_names": [], "imgs_masks": [], "anomaly_maps": [], "path": []}
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]              
        class_name = input_data["class_name"]            
        assert len(set(class_name)) == 1, "mixed class not supported"

        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        epoch_text_feature = class_text_embeddings

        patch_features, det_feature, _, _ = model(image)

        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())

        patch_preds = []
        for f in patch_features:
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)

    
        anomaly_maps_b = []
        for f in patch_features:
            patch_anomaly_scores = 100.0 * torch.matmul(f, epoch_text_feature) 
            B, L, C = patch_anomaly_scores.shape
            H = int(np.sqrt(L))
            patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)  
            patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2          
            temp = F.interpolate(
                patch_pred.unsqueeze(1), size=img_size, mode="bilinear", align_corners=True
            )                                                                    
            anomaly_maps_b.append(temp)
        anomaly_maps_b = torch.cat(anomaly_maps_b, dim=1).sum(1).cpu().numpy()  

        
        base_path = DATA_PATH[dataset]
        gt_b = input_data["mask"].cpu().numpy()                                   
        for b in range(len(file_name)):
            gt = gt_b[b]
            if gt.ndim == 3 and gt.shape[0] == 1:
                gt = gt[0]
            gt = np.nan_to_num(gt)
            gt = (gt > 0.5).astype(np.uint8)
            gt = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

            am = anomaly_maps_b[b]

            results["cls_names"].append(class_name[b])
            results["imgs_masks"].append(gt)
            results["anomaly_maps"].append(am)
            results["path"].append(os.path.join(base_path, file_name[b]))
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, file_names, results


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument(
        "--no_use_paa",
        action="store_false",
        help="Whether to use patch average aggregation (true/false)",
    )
    parser.add_argument(
        "--seg_proj_sharing_strategy",
        type=str,
        choices=["separate", "shared"],
        default="shared",
        help="Projector sharing strategy for patch average aggregation",
    )
    # MoE controls
    parser.add_argument("--moe_r", type=int, default=8)
    parser.add_argument("--moe_lora_alpha", type=int, default=16)
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument(
        "--moe_layers",
        type=str,
        default="5,11,17,23",
        help="Comma-separated layer indices for MoE application",
    )
    parser.add_argument("--no_use_fofs", action="store_false", help="Use fixed-A LoRA partitioning (default: True)")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--visual_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--save_path", default="ckpt/baseline")
    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_path, "test.log"), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    # parse moe_layers string to list[int]
    try:
        moe_layers = [int(x.strip()) for x in args.moe_layers.split(',') if x.strip() != ""]
    except Exception:
        moe_layers = [5, 11, 17, 23]

    model = MoECLIP(
        clip_model=clip_model,
        use_paa=args.no_use_paa,
        seg_proj_sharing_strategy=args.seg_proj_sharing_strategy,
        image_adapt_weight=args.image_adapt_weight,
        moe_r=args.moe_r,
        moe_lora_alpha=args.moe_lora_alpha,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_layers=moe_layers,
        use_fofs=args.no_use_fofs,
        relu=args.relu,
    ).to(device)
    model.eval()
    # load checkpoints if exists
    files = sorted(
        glob(args.save_path + "/moe_epoch_*.pth"),
        key=extract_epoch_number
    )

    logger.info("-----------------------------------------------")
    logger.info(f"dataset: {args.dataset}")
    logger.info("-----------------------------------------------")
    for checkpoint_path in files:
        logger.info(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "text_adapter" in checkpoint: 
            model.text_adapter.load_state_dict(checkpoint["text_adapter"])
            adapt_text = True
        else:
            adapt_text = False

        if "image_adapter" in checkpoint:
            model.image_adapter.load_state_dict(checkpoint["image_adapter"])

        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info(f"Successfully loaded model from epoch {test_epoch} from {checkpoint_path}")
        logger.info("-----------------------------------------------")
        
        # load dataset
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        image_datasets = get_dataset(
            args.dataset,
            args.img_size,
            None,
            args.shot,
            "test",
            logger=logger,
        )
        with torch.no_grad():
            if adapt_text:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device
                )
            else:
                text_embeddings = get_adapted_text_embedding(
                    clip_model, args.dataset, device
                )
        # ========================================================
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )

        combined_results = {"cls_names": [], "imgs_masks": [], "anomaly_maps": [], "path": []}
        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            # ========================================================
            # testing
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names, results = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                )
            # ========================================================
            variant_indices = group_indices_by_variant(file_names)
            if args.visualize:
                if args.dataset == "DTD-Synthetic" and len(variant_indices) > 1:
                    for variant_name, idx_list in variant_indices.items():
                        v_results = {
                            "cls_names": [results["cls_names"][i] for i in idx_list],
                            "imgs_masks": [results["imgs_masks"][i] for i in idx_list],
                            "anomaly_maps": [results["anomaly_maps"][i] for i in idx_list],
                            "path": [results["path"][i] for i in idx_list],
                        }
                        calcuate_metric_pixel(v_results, [variant_name], args)
                else:
                    calcuate_metric_pixel(results, [class_name], args)

            
            if args.dataset == "DTD-Synthetic" and len(variant_indices) > 1:
                for variant_name, idx_list in variant_indices.items():
                    idx_arr = np.array(idx_list)
                    v_masks = masks[idx_arr]
                    v_labels = labels[idx_arr]
                    v_preds = preds[idx_arr]
                    v_preds_image = preds_image[idx_arr]
                    v_result = metrics_eval(
                        v_masks,
                        v_labels,
                        v_preds,
                        v_preds_image,
                        variant_name,
                        domain=DOMAINS[args.dataset],
                    )
                    df.loc[len(df)] = Series(v_result)
            else:
                class_result_dict = metrics_eval(
                    masks,
                    labels,
                    preds,
                    preds_image,
                    class_name,
                    domain=DOMAINS[args.dataset],
                )
                df.loc[len(df)] = Series(class_result_dict)
        numeric_cols = ["pixel AUC", "pixel AP", "image AUC", "image AP"]
        df.loc[len(df)] = df[numeric_cols].mean()
        df.loc[len(df) - 1, "class name"] = "Average"
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()