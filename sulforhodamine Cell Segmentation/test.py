"""Train a small U-Net on GFP images + label masks, export to TorchScript."""
import random
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

from Algorithm import read_grayscale


def normalize(image):
    # Same normalization must be used at inference time.
    low, high = np.percentile(image, (1, 99.8))
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1.5).astype(np.float32)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, crop=256, samples=2000):
        self.data = [(normalize(read_grayscale(i)), tifffile.imread(m) > 0) for i, m in pairs]
        self.crop, self.samples = crop, samples

    def __len__(self):
        return self.samples

    def __getitem__(self, _):
        image, mask = random.choice(self.data)
        r = random.randint(0, image.shape[0] - self.crop)
        c = random.randint(0, image.shape[1] - self.crop)
        x = image[r:r + self.crop, c:c + self.crop]
        y = mask[r:r + self.crop, c:c + self.crop].astype(np.float32)
        k = random.randint(0, 3)                                # rotation
        x, y = np.rot90(x, k), np.rot90(y, k)
        if random.random() < 0.5:                               # flip
            x, y = x[:, ::-1], y[:, ::-1]
        x = x ** random.uniform(0.7, 1.4) * random.uniform(0.8, 1.2)  # intensity
        return (torch.from_numpy(x.copy())[None], torch.from_numpy(y.copy())[None])


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        self.down = nn.ModuleList()
        prev = 1
        for c in ch:
            self.down.append(block(prev, c))
            prev = c
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.dec.append(block(2 * c, c))
            prev = c
        self.head = nn.Conv2d(prev, 1, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)  # logits


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    return 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)


def predict(model, image):
    """Cell probability for a full image, padded to a multiple of 8."""
    x = normalize(image)
    h, w = x.shape
    padded = np.pad(x, ((0, -h % 8), (0, -w % 8)), mode="reflect")
    with torch.no_grad():
        logits = model(torch.from_numpy(padded)[None, None])
    return torch.sigmoid(logits)[0, 0, :h, :w].numpy()


if __name__ == "__main__":
    import time

    from scipy import ndimage as ndi

    from Algorithm import relabel_min_area, save_results

    here = Path(__file__).resolve().parent
    pairs = [  # (image, mask) — threshold masks from Algorithm.py; correct them by hand for real training
        (here / "../cell1/Duplicate1 gfp for OnlyIge.jpg",
         here / "../cell1/segmentation_results/Duplicate1 gfp for OnlyIge_threshold_labels.tif"),
        (here / "../cell2/duplicate2_gfp.jpg",
         here / "../cell2/segmentation_results/duplicate2_gfp_threshold_labels.tif"),
    ]
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    model = UNet().to(device)
    loader = torch.utils.data.DataLoader(
        CropDataset(pairs, samples=800), batch_size=8, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        start, total = time.time(), 0.0
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch}: loss {total / len(loader):.4f} ({time.time() - start:.0f}s)")

    model.eval().cpu()
    torch.save(model.state_dict(), here / "unet_gfp.pth")
    torch.jit.trace(model, torch.zeros(1, 1, 256, 256)).save(str(here / "unet_gfp.pt"))
    print(f"Saved {here / 'unet_gfp.pth'} and {here / 'unet_gfp.pt'}")

    for image_path, mask_path in pairs:
        image = read_grayscale(image_path)
        labels, _ = ndi.label(predict(model, image) > 0.5)
        labels = relabel_min_area(labels.astype(np.uint32), 100)
        save_results(image, labels, image_path.parent / "segmentation_results", f"{image_path.stem}_unet")
        truth = tifffile.imread(mask_path) > 0
        intersection = np.logical_and(labels > 0, truth).sum()
        union = np.logical_or(labels > 0, truth).sum()
        print(f"{image_path.name}: {int(labels.max())} cells, IoU vs threshold mask {intersection / union:.3f}")
