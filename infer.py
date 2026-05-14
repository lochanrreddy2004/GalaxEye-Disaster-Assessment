import os
import cv2
import yaml
import torch
import numpy as np

from models.siamese_unetpp import SiameseUNetPP


def preprocess_image(image_path, image_size):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f'Could not read image: {image_path}')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (image_size, image_size)
    )

    image = image.astype(np.float32) / 255.0

    image = torch.tensor(image).permute(2, 0, 1)

    image = image.unsqueeze(0)

    return image


def postprocess_prediction(prediction, threshold):

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > threshold).float()

    prediction = prediction.squeeze().cpu().numpy()

    prediction = (prediction * 255).astype(np.uint8)

    return prediction


def save_prediction(prediction, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, prediction)


def load_model(config, device):

    model = SiameseUNetPP(
        encoder_name=config['encoder_name'],
        encoder_weights=None
    ).to(device)

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

    model.eval()

    print(f'Loaded model from: {checkpoint_path}')

    return model


def predict(model, pre_image, post_image):

    with torch.no_grad():

        prediction = model(
            pre_image,
            post_image
        )

    return prediction


def main():

    # =========================
    # LOAD CONFIG
    # =========================

    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Using device: {device}')

    # =========================
    # LOAD MODEL
    # =========================

    model = load_model(config, device)

    # =========================
    # INPUT IMAGES
    # =========================

    pre_image_path = 'sample_pre.png'
    post_image_path = 'sample_post.png'

    # =========================
    # PREPROCESS
    # =========================

    pre_image = preprocess_image(
        pre_image_path,
        config['image_size']
    ).to(device)

    post_image = preprocess_image(
        post_image_path,
        config['image_size']
    ).to(device)

    # =========================
    # INFERENCE
    # =========================

    prediction = predict(
        model,
        pre_image,
        post_image
    )

    # =========================
    # POSTPROCESS
    # =========================

    prediction = postprocess_prediction(
        prediction,
        config['threshold']
    )

    # =========================
    # SAVE OUTPUT
    # =========================

    output_path = 'outputs/prediction.png'

    save_prediction(
        prediction,
        output_path
    )

    print(f'Prediction saved to: {output_path}')


if __name__ == '__main__':
    main()