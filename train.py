import os
from xml.parsers.expat import model
import yaml
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from datasets.dataset import ChangeDetectionDataset
from models.siamese_unetpp import SiameseUNetPP
from losses.losses import CombinedLoss
from utils.metrics import SegmentationMetrics
from utils.seed import set_seed


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device
):
    model.train()

    running_loss = 0.0

    pbar = tqdm(loader, desc='Training')

    for batch in pbar:
        pre = batch['pre_image'].to(device)
        post = batch['post_image'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=False):
            preds = model(pre, post)
            if torch.isnan(preds).any():
                # print("NaN detected in predictions")
                continue
            loss = criterion(preds, mask)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })

    epoch_loss = running_loss / len(loader)

    return epoch_loss


def validate(
    model,
    loader,
    criterion,
    metrics,
    device,
    threshold=0.5
):
    model.eval()

    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')

        for batch in pbar:
            pre = batch['pre_image'].to(device)
            post = batch['post_image'].to(device)
            mask = batch['mask'].to(device)

            preds = model(pre, post)

            if torch.isnan(preds).any():
                preds = torch.nan_to_num(
                    preds,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0
                )

            loss = criterion(preds, mask)

            running_loss += loss.item()

            preds = torch.sigmoid(preds)

            # print(preds.min(), preds.max())

            preds = preds.cpu().numpy()
            mask = mask.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(mask)

    val_loss = running_loss / len(loader)

    if len(all_preds) == 0:
        return 999, {
            'iou': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metric_results = metrics.compute(
        all_preds,
        all_targets
    )

    return val_loss, metric_results


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_iou,
    path
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou
    }

    torch.save(checkpoint, path)


def main():

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    os.makedirs(config['save_dir'], exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Using device: {device}')

    # =========================
    # DATASETS
    # =========================

    train_dataset = ChangeDetectionDataset(
        root_dir=config['train_dir'],
        image_size=config['image_size'],
        augment=True
    )

    val_dataset = ChangeDetectionDataset(
        root_dir=config['val_dir'],
        image_size=config['image_size'],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================

    model = SiameseUNetPP(
        encoder_name=config['encoder_name'],
        encoder_weights=config['encoder_weights']
    ).to(device)

    # =========================
    # LOSS
    # =========================

    criterion = CombinedLoss()

    # =========================
    # OPTIMIZER
    # =========================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # =========================
    # SCHEDULER
    # =========================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )

    # =========================
    # AMP
    # =========================

    scaler = GradScaler(
        'cuda',
        enabled=config['use_amp']
    )

    # =========================
    # METRICS
    # =========================

    metrics = SegmentationMetrics(
        threshold=config['threshold']
    )

    best_iou = 0.0

    print('\nStarting Training...\n')

    # =========================
    # TRAINING LOOP
    # =========================

    for epoch in range(config['epochs']):

        print('=' * 60)
        print(f'Epoch {epoch + 1}/{config["epochs"]}')
        print('=' * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device
        )

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            metrics,
            device,
            threshold=config['threshold']
        )

        scheduler.step()

        print('\nTraining Summary')
        print('-' * 30)

        print(f'Train Loss : {train_loss:.4f}')
        print(f'Val Loss   : {val_loss:.4f}')

        print(f'IoU        : {val_metrics["iou"]:.4f}')
        print(f'Precision  : {val_metrics["precision"]:.4f}')
        print(f'Recall     : {val_metrics["recall"]:.4f}')
        print(f'F1 Score   : {val_metrics["f1"]:.4f}')

        # =========================
        # SAVE BEST MODEL
        # =========================

        if val_metrics['iou'] > best_iou:

            best_iou = val_metrics['iou']

            best_model_path = os.path.join(
                config['save_dir'],
                'best_model.pth'
            )

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_iou,
                best_model_path
            )

            print(f'\nBest model saved at: {best_model_path}')
            print(f'Best IoU: {best_iou:.4f}')

        # =========================
        # SAVE LATEST MODEL
        # =========================

        latest_model_path = os.path.join(
            config['save_dir'],
            'latest_model.pth'
        )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_iou,
            latest_model_path
        )

    print('\nTraining Complete.')
    print(f'Best Validation IoU: {best_iou:.4f}')


if __name__ == '__main__':
    main()