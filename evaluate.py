import yaml
import torch

from torch.utils.data import DataLoader

from src.data.dataset import ChangeDetectionDataset
from src.models.unet import ChangeDetectionModel
from src.losses.losses import BCEDiceLoss
from src.training.validate import validate


def main():

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    test_dataset = ChangeDetectionDataset(
        root_dir=config["test_dir"],
        image_size=config["image_size"],
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    model = ChangeDetectionModel().to(device)

    model.load_state_dict(
        torch.load("outputs/checkpoints/best_model.pth")
    )

    criterion = BCEDiceLoss()

    test_loss, test_metrics = validate(
        model,
        test_loader,
        criterion,
        device
    )

    print("Test Results")
    print(test_metrics)


if __name__ == "__main__":
    main()