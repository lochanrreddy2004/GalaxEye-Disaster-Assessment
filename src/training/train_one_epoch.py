from tqdm import tqdm
import torch


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):

    model.train()

    running_loss = 0.0

    loop = tqdm(loader)

    for images, masks in loop:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)