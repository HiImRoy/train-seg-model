import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.deepcrack_dataset import DeepCrackDataset
import glob
from tqdm import tqdm
import cv2
import logging
import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# --- 模型导入 ---
from models.unet_pytorch import UNet

# --- 辅助函数：日志记录器 ---
def get_logger(output_dir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    log_path = output_dir / f"{name}.log"
    fh = logging.FileHandler(log_path, mode='a' if os.path.exists(log_path) else 'w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# --- 辅助函数：模型分析 ---
def log_parameter_summary(model, log, input_size=(3, 448, 448)):
    log.info("--- Model Summary ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total Parameters: {total_params / 1e6:.2f} M")
    log.info(f"  - Trainable: {trainable_params / 1e6:.2f} M")
    log.info("-------------------------\
")

# --- 辅助函数：绘图 ---
def save_plots(log_df, output_dir):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    plt.plot(log_df['epoch'], log_df['train_loss'], marker='o', linestyle='-', label='Train Loss')
    if 'val_loss' in log_df.columns:
        plt.plot(log_df['epoch'], log_df['val_loss'], marker='x', linestyle='--', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'loss_curve.png')
    plt.close()
    
    metrics_to_plot = ['ODS_F1', 'OIS_F1', 'ODS_mIoU', 'ODS_Foreground_IoU', 'ODS_Precision', 'ODS_Recall']
    plt.figure(figsize=(12, 8))
    for metric in metrics_to_plot:
        if metric in log_df.columns:
            plt.plot(log_df['epoch'], log_df[metric], marker='o', linestyle='-', label=metric)
    plt.title('Validation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(output_dir / 'metrics_curve.png')
    plt.close()

# --- 辅助函数：保存最佳蒙版 ---
def save_best_masks(model, device, val_dataset, output_dir, best_threshold, best_miou):
    log = logging.getLogger('experiment')
    log.info(f"Saving binarized masks for best model (ODS_mIoU: {best_miou:.4f}) using threshold: {best_threshold:.4f}...")
    model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    for f in glob.glob(str(output_dir / '*.png')):
        os.remove(f)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Saving Best Binarized Masks"):
            x, target = batch['image'].to(device), batch['label']
            pred = model(x)

            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_mask_bin = (pred_prob > best_threshold).astype(np.uint8) * 255

            original_image_path = batch['A_paths'][0]
            base_name = os.path.splitext(os.path.basename(original_image_path))[0]
            
            gt_mask_uint8 = (target.squeeze().cpu().numpy() * 255).astype(np.uint8)
            gt_rgb = cv2.cvtColor(gt_mask_uint8, cv2.COLOR_GRAY2BGR)
            pred_rgb = cv2.cvtColor(pred_mask_bin, cv2.COLOR_GRAY2BGR)
            
            stitched_image = np.hstack((gt_rgb, pred_rgb))
            cv2.putText(stitched_image, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(stitched_image, 'Prediction', (gt_rgb.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            compare_save_path = output_dir / f'{base_name}_compare.png'
            cv2.imwrite(str(compare_save_path), stitched_image)
            
    log.info(f"Best binarized masks saved to: {output_dir}")

# --- 辅助函数：为指定图片生成并保存掩膜 ---
def save_inference_mask(model, device, image_path, output_dir, epoch, threshold, image_size):
    if not os.path.exists(image_path):
        logging.getLogger('experiment').warning(f"Inference image not found at {image_path}, skipping single image inference.")
        return

    model.eval()
    img_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = img_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()

    pred_mask_bin = (pred_prob > threshold).astype(np.uint8) * 255
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = output_dir / f'{base_name}_epoch_{epoch + 1}.png'
    cv2.imwrite(str(save_path), pred_mask_bin)

# --- 损失函数 ---
def bce_dice_loss(pred, mask, smooth=1e-5):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred_sig = torch.sigmoid(pred)
    pred_flat = pred_sig.contiguous().view(-1)
    mask_flat = mask.contiguous().view(-1)
    intersection = (pred_flat * mask_flat).sum()
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + mask_flat.sum() + smooth)
    dice = 1 - dice_coeff
    return 0.83 * bce + 0.17 * dice

# --- 评估逻辑 ---
def get_statistics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    return tp, fp, fn

def calculate_metrics_from_disk(epoch_preds_dir: Path) -> dict:
    pred_files = sorted(list(epoch_preds_dir.glob("*_pred.png")))
    gt_files = sorted(list(epoch_preds_dir.glob("*_gt.png")))

    if not pred_files or not gt_files: return {}

    all_prob_maps = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in pred_files]
    all_gt_masks = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in gt_files]

    thresholds = np.arange(0.0, 1.0, 0.01)
    image_f1_scores, image_miou_scores, image_fg_iou_scores, image_precision_scores, image_recall_scores = [np.zeros((len(all_prob_maps), len(thresholds))) for _ in range(5)]

    gt_masks_binary = [(gt > 127).astype(np.uint8) for gt in all_gt_masks]

    for img_idx, (prob_map_uint8, gt_mask) in enumerate(zip(all_prob_maps, gt_masks_binary)):
        prob_map_float = prob_map_uint8.astype(np.float32) / 255.0
        for thresh_idx, t in enumerate(thresholds):
            pred_mask = (prob_map_float > t).astype(np.uint8)
            tp, fp, fn = get_statistics(pred_mask, gt_mask)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            image_f1_scores[img_idx, thresh_idx] = 2 * (precision * recall) / (precision + recall + 1e-8)
            image_precision_scores[img_idx, thresh_idx] = precision
            image_recall_scores[img_idx, thresh_idx] = recall

            tn = np.sum((pred_mask == 0) & (gt_mask == 0))
            iou_foreground = tp / (tp + fp + fn + 1e-8)
            image_miou_scores[img_idx, thresh_idx] = (iou_foreground + (tn / (tn + fp + fn + 1e-8))) / 2
            image_fg_iou_scores[img_idx, thresh_idx] = iou_foreground

    avg_miou_per_threshold = np.mean(image_miou_scores, axis=0)
    best_ods_miou_idx = np.argmax(avg_miou_per_threshold)
    
    return {
        "ODS_F1": np.mean(image_f1_scores, axis=0)[best_ods_miou_idx], 
        "OIS_F1": np.mean(np.max(image_f1_scores, axis=1)), 
        "ODS_Precision": np.mean(image_precision_scores, axis=0)[best_ods_miou_idx],
        "ODS_Recall": np.mean(image_recall_scores, axis=0)[best_ods_miou_idx],
        "ODS_mIoU": avg_miou_per_threshold[best_ods_miou_idx],
        "ODS_Foreground_IoU": np.mean(image_fg_iou_scores, axis=0)[best_ods_miou_idx],
        "Best_Threshold": thresholds[best_ods_miou_idx]
    }

# --- 模型构建工厂 ---
def build_model(args):
    log = logging.getLogger('experiment')
    log.info(f"Building model: {args.model_name}")
    if args.model_name == 'UNet':
        model = UNet(n_channels=3, n_classes=1)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    return model

# --- 主函数 ---
def main(args):
    # --- 实验设置 ---
    if args.resume_path and not os.path.exists(args.resume_path):
        print(f"Warning: resume_path {args.resume_path} does not exist. Starting from scratch.")
        args.resume_path = None

    if args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location='cpu', weights_only=False)
        output_dir = Path(checkpoint['output_dir'])
    else:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dataset_name = Path(args.dataroot).name
        output_dir = Path(args.output_dir) / args.model_name / dataset_name / f"{cur_time}"

    weights_dir = output_dir / 'weights'
    masks_dir = output_dir / 'best_epoch_masks'
    plots_dir = output_dir / 'plots'
    raw_preds_dir = output_dir / 'raw_predictions'
    inference_dir = output_dir / 'inference_samples'
    tb_log_dir = output_dir / 'tensorboard_logs'
    for d in [weights_dir, masks_dir, plots_dir, raw_preds_dir, inference_dir, tb_log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_log_dir))
    log = get_logger(output_dir, 'experiment')
    log.info(f"Experiment started: {output_dir.name}")
    log.info(f"All results will be saved to: {output_dir}")
    log.info("--- Hyperparameters ---")
    for arg, value in sorted(vars(args).items()):
        log.info(f"{arg}: {value}")
    log.info("-----------------------\
")

    # --- 数据加载 ---
    args.load_width = args.image_size
    args.load_height = args.image_size
    
    args.phase = 'train'
    train_dataset = DeepCrackDataset(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    log.info(f"Found {len(train_dataset)} training images.")

    args.phase = 'test'
    val_dataset = DeepCrackDataset(args)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    log.info(f"Found {len(val_dataset)} validation images.")

    # --- 梯度累积设置 ---
    accumulation_steps = args.accumulation_steps if args.accumulation_steps > 0 else (args.target_batch_size // args.batch_size)
    log.info(f"Effective batch size: {args.batch_size * accumulation_steps}")

    # --- 模型、优化器、调度器设置 ---
    device = torch.device("cuda")
    model = build_model(args).to(device)

    log.info("Creating optimizer...")
    optim = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    # --- 断点续训逻辑 ---
    start_epoch, best_miou, log_data = 0, 0.0, []
    if args.resume_path:
        log.info(f"Resuming from checkpoint: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        log.info(f"Resumed from epoch {start_epoch}. Best mIoU so far: {best_miou:.4f}")
        if os.path.exists(output_dir / 'training_log.csv'):
            log_data = pd.read_csv(output_dir / 'training_log.csv').to_dict('records')

    if start_epoch == 0:
        try:
            log_parameter_summary(model, log, input_size=(3, args.image_size, args.image_size))
        except Exception as e:
            log.error(f"[CRITICAL] Failed to generate model summary. Error: {e}", exc_info=True)

    # --- 训练循环 ---
    log.info("--- Starting Training ---")
    for epoch in range(start_epoch, args.epoch):
        log.info(f"\n===== Epoch {epoch + 1}/{args.epoch} ======")
        
        model.train()
        total_train_loss = 0
        optim.zero_grad()
        
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Training]", leave=True)
        for i, batch in enumerate(train_loop):
            x, target = batch['image'].to(device), batch['label'].to(device)
            pred = model(x)
            
            loss = bce_dice_loss(pred, target) / accumulation_steps
            loss.backward()
            
            total_train_loss += loss.item() * accumulation_steps

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optim.step()
                optim.zero_grad()
                
            train_loop.set_postfix(loss=loss.item() * accumulation_steps)
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        log.info(f"Average Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Meta/learning_rate', scheduler.get_last_lr()[0], epoch)
        
        scheduler.step()

        # --- 验证 ---
        model.eval()
        total_val_loss = 0
        epoch_preds_dir = raw_preds_dir / f"epoch_{epoch + 1}"
        epoch_preds_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch + 1} [Validation]", leave=True)
            for i, batch in enumerate(val_loop):
                x, target = batch['image'].to(device), batch['label'].to(device)
                pred = model(x)
                
                loss = bce_dice_loss(pred, target)
                total_val_loss += loss.item()

                pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()
                base_name = os.path.splitext(os.path.basename(batch['A_paths'][0]))[0]
                cv2.imwrite(str(epoch_preds_dir / f"{base_name}_pred.png"), (pred_prob * 255).astype(np.uint8))
                cv2.imwrite(str(epoch_preds_dir / f"{base_name}_gt.png"), (target.squeeze().cpu().numpy() * 255).astype(np.uint8))

                if i < 4:
                    writer.add_image(f'Validation/Prediction_{i}', np.expand_dims(pred_prob, 0), epoch)
                    writer.add_image(f'Validation/GroundTruth_{i}', target.squeeze().cpu().numpy(), epoch, dataformats='HW')

        avg_val_loss = total_val_loss / len(val_dataloader)
        log.info(f"Average Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        metrics = calculate_metrics_from_disk(epoch_preds_dir)
        if metrics:
            log.info(f"Validation Metrics: mIoU={metrics['ODS_mIoU']:.4f}, ODS={metrics['ODS_F1']:.4f}")
            for key, value in metrics.items(): writer.add_scalar(f'Metrics/{key}', value, epoch)
            log_data.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, **metrics})
            
            save_inference_mask(model, device, args.inference_image_path, inference_dir, epoch, metrics['Best_Threshold'], args.image_size)

            if (current_miou := metrics['ODS_mIoU']) > best_miou:
                best_miou = current_miou
                log.info(f"*** New best ODS_mIoU: {best_miou:.4f} at epoch {epoch + 1}. Saving best checkpoint... ***")
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(), 'best_miou': best_miou, 'output_dir': str(output_dir)}, 
                           weights_dir / 'best_checkpoint.pth', _use_new_zipfile_serialization=False)
                save_best_masks(model, device, val_dataset, masks_dir, metrics['Best_Threshold'], best_miou)
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 'best_miou': best_miou, 'output_dir': str(output_dir)}, 
                   weights_dir / 'last_checkpoint.pth', _use_new_zipfile_serialization=False)

    writer.close()
    log.info("\n--- Training Finished ---")
    
    log_df = pd.DataFrame(log_data)
    if not log_df.empty:
        log_df.to_csv(output_dir / 'training_log.csv', index=False)
        log.info(f"Training log saved to {output_dir / 'training_log.csv'}")
        save_plots(log_df, plots_dir)
        log.info(f"Loss and metrics plots saved in {plots_dir}")
        best_epoch_stats = log_df.loc[log_df['ODS_mIoU'].idxmax()]
        log.info("\n" + "="*30 + " Best Metrics Summary " + "="*30)
        log.info(f"Best model was achieved at Epoch {int(best_epoch_stats['epoch'])}")
        for metric, value in best_epoch_stats.items():
            if metric != 'epoch':
                log.info(f"  - {metric:<20}: {value:.4f}")
        log.info("="*82 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("UNet Training Script")
    parser.add_argument("--dataroot", type=str, default='DeepCrack82', help="Path to the dataset root directory.")
    parser.add_argument("--inference_image_path", type=str, default="./DeepCrack82/DeepCrack_11123.jpg", help="Path for single image inference.")
    parser.add_argument("--model_name", type=str, default="UNet", help="Name of the model (UNet).")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to resume training from.")
    parser.add_argument('--output_dir', type=str, default='./results', help="Root directory for results.")
    parser.add_argument("--epoch", type=int, default=75, help="Total training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate.")
    parser.add_argument("--batch_size", default=6, type=int, help="VRAM batch size.")
    parser.add_argument("--target_batch_size", default=12, type=int, help="Effective batch size.")
    parser.add_argument("--accumulation_steps", default=0, type=int, help="Gradient accumulation steps (0 for auto).")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--image_size", type=int, default=448, help="Input image size.")
    parser.add_argument("--no_flip", action='store_true', help="Disable flip data augmentation.")
    parser.add_argument("--use_augment", action='store_true', help="Enable affine transform data augmentation.")
    
    args = parser.parse_args()
    main(args)
