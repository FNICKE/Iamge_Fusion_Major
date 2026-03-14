#!/usr/bin/env python
"""
Train / Fine-tune EMMA on MSRS dataset (IR + Visible).
======================================================
Prerequisites:
  1. Download MSRS dataset: https://github.com/Linfeng-Tang/MSRS
     Place IR images in: backend/emma/dataprocessing/MSRS_train/ir/
     Place VI images in: backend/emma/dataprocessing/MSRS_train/vi/
  2. Install: pip install torch einops h5py scikit-image tqdm kornia
  3. (Optional) Download Ai.pth, Av.pth from EMMA Google Drive for equivariant loss:
     https://drive.google.com/drive/folders/1Zb9NDW4lZh_jCdv1BIqI1wbYmGSmbpUp

Usage (from project root):
  python -m scripts.train_emma prepare   # Create H5 from MSRS
  python -m scripts.train_emma train     # Train U-Fuser
"""

import os
import sys

# Resolve backend directory
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_PROJECT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

os.chdir(_BACKEND)


def prepare_data():
    """Create H5 training file from MSRS dataset."""
    import h5py
    import numpy as np
    from tqdm import tqdm
    from skimage.io import imread

    base = os.path.join(_BACKEND, "emma", "dataprocessing")
    ir_dir = os.path.join(base, "MSRS_train", "ir")
    vi_dir = os.path.join(base, "MSRS_train", "vi")

    if not os.path.isdir(ir_dir) or not os.path.isdir(vi_dir):
        print("ERROR: Create folders and add MSRS images:")
        print(f"  {ir_dir}")
        print(f"  {vi_dir}")
        print("Download from: https://github.com/Linfeng-Tang/MSRS")
        sys.exit(1)

    def get_imgs(d):
        out = []
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    out.append(os.path.join(root, f))
        return sorted(out)

    ir_files = get_imgs(ir_dir)
    vi_files = get_imgs(vi_dir)
    if len(ir_files) != len(vi_files):
        print("ERROR: IR and VI image counts must match.")
        sys.exit(1)

    img_size, stride = 128, 200

    def rgb2y(img):
        return img[0:1] * 0.299 + img[1:2] * 0.587 + img[2:3] * 0.114

    def im2patch(img, win, s):
        c, h, w = img.shape
        patches = []
        for i in range(0, h - win + 1, s):
            for j in range(0, w - win + 1, s):
                patches.append(img[:, i : i + win, j : j + win])
        return np.stack(patches, axis=-1) if patches else np.zeros((c, win, win, 0), dtype=np.float32)

    def low_contrast(patch, thresh=0.1):
        lo, hi = np.percentile(patch, [10, 90])
        return (hi - lo) / (hi + 1e-8) < thresh

    os.makedirs(os.path.join(_BACKEND, "emma", "data"), exist_ok=True)
    h5_path = os.path.join(_BACKEND, "emma", "data", f"MSRS_train_{img_size}_{stride}.h5")

    with h5py.File(h5_path, "w") as h5f:
        h5_ir = h5f.create_group("ir_patchs")
        h5_vis = h5f.create_group("vis_patchs")
        n = 0
        for i in tqdm(range(len(ir_files)), desc="Preparing"):
            vis = imread(vi_files[i]).astype(np.float32).transpose(2, 0, 1) / 255.0
            vis_y = rgb2y(vis)
            ir = imread(ir_files[i]).astype(np.float32)
            if ir.ndim == 3:
                ir = ir[:, :, 0]
            ir = ir[None, :, :] / 255.0

            ir_p = im2patch(ir, img_size, stride)
            vis_p = im2patch(vis_y, img_size, stride)
            for j in range(ir_p.shape[-1]):
                if low_contrast(ir_p[0, :, :, j]) or low_contrast(vis_p[0, :, :, j]):
                    continue
                h5_ir.create_dataset(str(n), data=ir_p[:, :, :, j])
                h5_vis.create_dataset(str(n), data=vis_p[:, :, :, j])
                n += 1

    print(f"Created {h5_path} with {n} patches.")


def train():
    """Train EMMA U-Fuser."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    try:
        from emma.nets.Ufuser import Ufuser
    except ImportError:
        from nets.Ufuser import Ufuser

    # Simplified training without Ai/Av (equivariant loss) - still effective
    h5_path = os.path.join(_BACKEND, "emma", "data", "MSRS_train_128_200.h5")
    if not os.path.isfile(h5_path):
        print("Run 'python -m scripts.train_emma prepare' first.")
        sys.exit(1)

    import h5py

    class H5Dataset(torch.utils.data.Dataset):
        def __init__(self, path):
            self.path = path
            with h5py.File(path, "r") as f:
                self.keys = list(f["ir_patchs"].keys())

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, i):
            with h5py.File(self.path, "r") as f:
                ir = torch.from_numpy(np.array(f["ir_patchs"][self.keys[i]])).float()
                vis = torch.from_numpy(np.array(f["vis_patchs"][self.keys[i]])).float()
            return ir, vis

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ufuser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loader = DataLoader(H5Dataset(h5_path), batch_size=4, shuffle=True, num_workers=0)
    epochs = 50

    model.train()
    for ep in range(epochs):
        total = 0
        for ir, vis in loader:
            ir, vis = ir.to(device), vis.to(device)
            out = model(ir, vis)
            loss = torch.nn.functional.l1_loss(out, ir) + torch.nn.functional.l1_loss(out, vis)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {ep+1}/{epochs} loss={total/len(loader):.4f}")

    out_path = os.path.join(_BACKEND, "emma", "models", "EMMA_finetuned.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "train"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "prepare":
        prepare_data()
    else:
        train()
