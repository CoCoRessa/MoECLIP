import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from datetime import datetime
import secrets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import ipdb
from utils import setup_seed
from model.moe_adapter import MoECLIP
from model.clip import create_model
from dataset import get_dataset
from forward_utils import (
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, save_path: str):
    
    moe_adapter_weights = {}
    for key, value in model.state_dict().items():
        if 'mlp.gate_' in key or 'mlp.experts_' in key:
            moe_adapter_weights[key] = value
    
    checkpoint = {
        "epoch": epoch + 1,
        "optimizer_state_dict": optimizer.state_dict(),
        "text_adapter": model.text_adapter.state_dict(),
        "image_adapter": model.image_adapter.state_dict(),
    }
    
    ckp_path = os.path.join(save_path, f"moe_epoch_{epoch + 1}.pth")
    torch.save(checkpoint, ckp_path)
    last_ckp_path = os.path.join(save_path, "moe_last.pth")
    torch.save(checkpoint, last_ckp_path)
    
    print(f"Successfully saved checkpoint for epoch {epoch + 1} to {last_ckp_path}")

def train_adapter(
    adapted_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    end_epoch: int,
    dataset_name: str,
    img_size: int,
    logger: logging.Logger,
    balance_loss_lambda: float,
    etf_loss_lambda: float, 
):

    for epoch in range(start_epoch, end_epoch):
        logger.info(f"training epoch {epoch+1}:")

        loss_list = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch}")
        for step, input_data in enumerate(pbar):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            class_names = input_data["class_name"]

            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )
            
            patch_features, det_feature, aux_loss, special_loss = adapted_model(image)
            loss = 0.0
            det_feature = det_feature.unsqueeze(1)  # (B,1,D)
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            loss += F.cross_entropy(cls_preds, label)

            for f in patch_features:
                # f: (B, patch_num, D)
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)  # (B,C,H,W)
                # segmentation loss
                loss += calculate_seg_loss(patch_preds, mask)

            loss += aux_loss * balance_loss_lambda
            loss += special_loss * etf_loss_lambda
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            loss_item = loss.item()
            loss_list.append(loss_item)
            pbar.set_postfix({"loss": loss_item})

        avg_epoch_loss = np.mean(loss_list)
        logger.info(f"Average loss for epoch {epoch+1}: {avg_epoch_loss}")
        
        save_checkpoint(
            model=adapted_model, 
            optimizer=optimizer, 
            epoch=epoch, 
            save_path=save_path
        )
        
    return adapted_model


def main():
    
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true", help="use relu after projection")
    # MoE hyperparameters
    parser.add_argument("--moe_r", type=int, default=8, help="LoRA rank r for MoE experts")
    parser.add_argument("--moe_lora_alpha", type=int, default=16, help="LoRA alpha for MoE experts")
    parser.add_argument("--moe_num_experts", type=int, default=4, help="Number of experts in MoE")
    parser.add_argument("--moe_top_k", type=int, default=2, help="Top-k experts to route to")
    parser.add_argument("--no_use_fofs", action="store_false", help="Use fixed-A LoRA partitioning (default: True)")
    parser.add_argument("--balance_loss_lambda", type=float, default=0.01, help="Weight for auxiliary (load balancing) loss")
    parser.add_argument("--etf_loss_lambda", type=float, default=0.01, help="Weight for ETF Loss")
    parser.add_argument(
        "--moe_layers",
        type=str,
        default="5,11,17,23",
        help="Comma-separated layer indices where MoE is applied (e.g., '5,11,17,23')",
    )
    # training
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="full_shot",
        choices=["few_shot", "full_shot"],
    )
    parser.add_argument("--shot", type=int, default=32, help="number of shots (0 means full shot)")
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=20, help="epochs for stage2")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    # hyper-parameters
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    # spatial aggregation controls
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
        help="Projector sharing strategy when using spatial aggregation",
    )
    args = parser.parse_args()
    # ========================================================

    setup_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    logger.propagate = False
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    log_file = os.path.join(args.save_path, "train.log")
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    logger.info("args: %s", vars(args))

    # set device
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
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
        use_fofs=args.no_use_fofs,
        moe_lora_alpha=args.moe_lora_alpha,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_layers=moe_layers,
        relu=args.relu,
    ).to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    params_to_train = []
    
    for param in model.text_adapter.parameters():
        param.requires_grad = True
    params_to_train.append({"params": model.text_adapter.parameters()})    

    image_params = []
    if args.no_use_fofs:
        for name, param in model.image_adapter.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False 
            else:
                param.requires_grad = True
                image_params.append(param)

        params_to_train.append({"params": image_params})

        for name, param in model.named_parameters():
            if "lora_A" in name:
                print(f"{name}: requires_grad={param.requires_grad}")
    else:
        for param in model.image_adapter.parameters():
            param.requires_grad = True
        params_to_train.append({"params": model.image_adapter.parameters()})
    
    optimizer = torch.optim.Adam(
        params_to_train,
        lr=args.lr,
        betas=(0.5, 0.999),
    )

    image_scheduler = MultiStepLR(optimizer, milestones=[16000, 32000], gamma=0.5)
    # ========================================================
    # load checkpoints if exists        
    start_epoch = 0
    last_ckp_path = os.path.join(args.save_path, "moe_last.pth")
    if os.path.exists(last_ckp_path):
        logger.info(f"Resuming from checkpoint: {last_ckp_path}")
        checkpoint = torch.load(last_ckp_path, map_location=device)
        
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        
        logger.info(f"Resumed from epoch {start_epoch}.")
    else:
        logger.info("No checkpoint found, starting from scratch.")
        
    # ========================================================
    # load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    logger.info("loading dataset ...")
    text_dataset, image_dataset = get_dataset(
        args.dataset,
        args.img_size,
        args.training_mode,
        args.shot,
        "train",
        logger,
    )
    
    logger.info("loading image adaptation dataset ...")
    all_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )
    # ========================================================
    # training
    #torch.cuda.empty_cache()

    model = train_adapter(
        adapted_model=model,
        train_loader=all_dataloader,
        optimizer=optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=start_epoch,
        dataset_name=args.dataset,
        end_epoch=args.epoch,
        save_path=args.save_path,
        img_size=args.img_size,
        logger=logger,
        balance_loss_lambda=args.balance_loss_lambda,
        etf_loss_lambda=args.etf_loss_lambda,
    )


if __name__ == "__main__":
    main()
