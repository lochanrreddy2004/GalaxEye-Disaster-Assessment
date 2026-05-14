import os
import yaml
import cv2
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.dataset import ChangeDetectionDataset
from models.siamese_unetpp import SiameseUNetPP


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(preds, masks):

    preds = preds.flatten()
    masks = masks.flatten()

    tp = np.sum((preds == 1) & (masks == 1))

    fp = np.sum((preds == 1) & (masks == 0))

    fn = np.sum((preds == 0) & (masks == 1))

    iou = tp / (tp + fp + fn + 1e-6)

    precision = tp / (tp + fp + 1e-6)

    recall = tp / (tp + fn + 1e-6)

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-6)
    )

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ==========================================
# LOAD CONFIG
# ==========================================

def load_config():

    with open(
        'configs/config.yaml',
        'r'
    ) as f:

        config = yaml.safe_load(f)

    return config


# ==========================================
# MAIN
# ==========================================

def main():

    config = load_config()

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print(f"\nUsing device: {device}")

    # ======================================
    # DATASET
    # ======================================

    test_dataset = ChangeDetectionDataset(
        root_dir=config['test_dir'],
        image_size=config['image_size'],
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers']
    )

    # ======================================
    # MODEL
    # ======================================

    model = SiameseUNetPP(
        encoder_name=config['encoder_name'],
        encoder_weights=None
    )

    checkpoint_path = os.path.join(
        config['save_dir'],
        'best_model.pth'
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    model = model.to(device)

    model.eval()

    print("\nBest model loaded.")

    # ======================================
    # TESTING
    # ======================================

    threshold = config['threshold']

    all_preds = []

    all_masks = []

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    with torch.no_grad():

        pbar = tqdm(test_loader)

        for idx, batch in enumerate(pbar):

            pre = batch['pre_image'].to(device)

            post = batch['post_image'].to(device)

            masks = batch['mask'].to(device)

            preds = torch.sigmoid(
                model(pre, post)
            )

            preds = (
                preds > threshold
            ).float()

            preds_np = (
                preds.cpu()
                .numpy()
                .astype(np.uint8)
            )

            masks_np = (
                masks.cpu()
                .numpy()
                .astype(np.uint8)
            )

            all_preds.append(preds_np)

            all_masks.append(masks_np)

            # ==================================
            # SAVE PREDICTIONS
            # ==================================

            pred_img = (
                preds_np[0, 0] * 255
            ).astype(np.uint8)

            gt_img = (
                masks_np[0, 0] * 255
            ).astype(np.uint8)

            cv2.imwrite(
                f"outputs/pred_{idx}.png",
                pred_img
            )

            cv2.imwrite(
                f"outputs/gt_{idx}.png",
                gt_img
            )

    all_preds = np.concatenate(all_preds)

    all_masks = np.concatenate(all_masks)

    metrics = calculate_metrics(
        all_preds,
        all_masks
    )

    # ======================================
    # FINAL RESULTS
    # ======================================

    print("\n===================================")
    print("TEST RESULTS")
    print("===================================")

    print(
        f"IoU        : "
        f"{metrics['iou']:.4f}"
    )

    print(
        f"Precision  : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score   : "
        f"{metrics['f1']:.4f}"
    )


if __name__ == "__main__":

    main()