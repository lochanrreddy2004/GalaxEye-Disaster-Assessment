import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def denormalize_image(image):
    """
    Converts normalized tensor back to displayable image.
    """

    image = image.cpu().numpy().transpose(1, 2, 0)

    eo = image[:, :, :3]
    sar = image[:, :, 3]

    eo_mean = np.array([0.485, 0.456, 0.406])
    eo_std = np.array([0.229, 0.224, 0.225])

    eo = (eo * eo_std) + eo_mean
    eo = np.clip(eo, 0, 1)

    sar = (sar * 0.25) + 0.5
    sar = np.clip(sar, 0, 1)

    return eo, sar


def save_prediction_visualizations(
    model,
    loader,
    device,
    save_dir="outputs/visualizations",
    num_samples=5,
    threshold=0.5
):
    """
    Saves prediction visualization grids.
    """

    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    saved = 0

    with torch.no_grad():

        for images, masks in tqdm(loader):

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            preds = torch.sigmoid(outputs)
            preds = (preds > threshold).float()

            batch_size = images.shape[0]

            for i in range(batch_size):

                eo_img, sar_img = denormalize_image(images[i])

                gt_mask = masks[i].squeeze().cpu().numpy()
                pred_mask = preds[i].squeeze().cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                axes[0].imshow(eo_img)
                axes[0].set_title("EO Pre-Event")
                axes[0].axis("off")

                axes[1].imshow(sar_img, cmap="gray")
                axes[1].set_title("SAR Post-Event")
                axes[1].axis("off")

                axes[2].imshow(gt_mask, cmap="gray")
                axes[2].set_title("Ground Truth")
                axes[2].axis("off")

                axes[3].imshow(pred_mask, cmap="gray")
                axes[3].set_title("Prediction")
                axes[3].axis("off")

                plt.tight_layout()

                save_path = os.path.join(
                    save_dir,
                    f"prediction_{saved}.png"
                )

                plt.savefig(save_path)
                plt.close()

                saved += 1

                if saved >= num_samples:
                    return


def visualize_single_prediction(
    model,
    dataset,
    device,
    index=None,
    threshold=0.5
):
    """
    Visualize one random prediction interactively.
    """

    model.eval()

    if index is None:
        index = random.randint(0, len(dataset) - 1)

    image, mask = dataset[index]

    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        pred = torch.sigmoid(output)
        pred = (pred > threshold).float()

    pred = pred.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    eo_img, sar_img = denormalize_image(image)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(eo_img)
    axes[0].set_title("EO Pre-Event")
    axes[0].axis("off")

    axes[1].imshow(sar_img, cmap="gray")
    axes[1].set_title("SAR Post-Event")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    axes[3].imshow(pred, cmap="gray")
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()