import yaml
import torch
import os

from torch.utils.data import DataLoader

from src.data.dataset import ChangeDetectionDataset
from src.models.unet import ChangeDetectionModel
from src.losses.losses import BCEDiceLoss

from src.training.train_one_epoch import train_one_epoch
from src.training.validate import validate

from src.utils.visualization import save_prediction_visualizations

def main():

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_dataset = ChangeDetectionDataset(
        root_dir=config["train_dir"],
        image_size=config["image_size"],
        train=True
    )

    val_dataset = ChangeDetectionDataset(
        root_dir=config["val_dir"],
        image_size=config["image_size"],
        train=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    model = ChangeDetectionModel().to(device)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )

    scaler = torch.cuda.amp.GradScaler()

    best_iou = 0

    os.makedirs(config["save_dir"], exist_ok=True)

    for epoch in range(config["epochs"]):

        print(f"Epoch {epoch+1}/{config['epochs']}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler
        )
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        print(
            f"Epoch {epoch+1} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        scheduler.step()

        

        if val_metrics["iou"] > best_iou:

            best_iou = val_metrics["iou"]

            torch.save(
                model.state_dict(),
                os.path.join(
                    config["save_dir"],
                    "best_model.pth"
                )
            )

            save_prediction_visualizations(
                model=model,
                loader=val_loader,
                device=device,
                save_dir="outputs/visualizations",
                num_samples=3
            )

            print("Best model saved!")


if __name__ == "__main__":
    main()