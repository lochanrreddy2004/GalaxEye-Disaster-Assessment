import torch
from tqdm import tqdm

from src.utils.metrics import calculate_metrics


def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    total_metrics = {
        "iou": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
    }

    with torch.no_grad():

        loop = tqdm(loader)

        for images, masks in loop:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            total_loss += loss.item()

            metrics = calculate_metrics(outputs, masks)

            for key in total_metrics:
                total_metrics[key] += metrics[key]

    avg_loss = total_loss / len(loader)

    for key in total_metrics:
        total_metrics[key] /= len(loader)

    return avg_loss, total_metrics