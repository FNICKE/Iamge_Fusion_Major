#!/usr/bin/env python
"""
Train / Fine-tune IR+Visible fusion on your own image pairs.
================================================================
Place your IR and Visible image pairs in:
  backend/emma/dataprocessing/ir_vis_train/ir/   (IR images)
  backend/emma/dataprocessing/ir_vis_train/vi/   (Visible images)

Filenames must match (e.g. scene1.png in both folders = one pair).

Usage (from project root):
  python -m scripts.train_ir_vis

Uses EMMA architecture, trains on your data for cleaner fusion on your scenes.
"""

import os
import sys

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_PROJECT, "backend")
sys.path.insert(0, _BACKEND)
os.chdir(_BACKEND)


def main():
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from glob import glob
    from tqdm import tqdm

    ir_dir = os.path.join(_BACKEND, "emma", "dataprocessing", "ir_vis_train", "ir")
    vi_dir = os.path.join(_BACKEND, "emma", "dataprocessing", "ir_vis_train", "vi")

    if not os.path.isdir(ir_dir) or not os.path.isdir(vi_dir):
        print("Create folders and add paired images:")
        print(f"  {ir_dir}")
        print(f"  {vi_dir}")
        print("Filenames must match (e.g. scene1.png in both).")
        sys.exit(1)

    def get_pairs():
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        pairs = []
        for ext in exts:
            for ir_f in glob(os.path.join(ir_dir, ext)):
                name = os.path.basename(ir_f)
                vi_f = os.path.join(vi_dir, name)
                if os.path.isfile(vi_f):
                    pairs.append((ir_f, vi_f))
        return pairs

    pairs = get_pairs()
    if len(pairs) == 0:
        print("No matching IR/VI pairs found. Check filenames.")
        sys.exit(1)

    print(f"Found {len(pairs)} image pairs.")

    try:
        from emma.nets.Ufuser import Ufuser
    except ImportError:
        print("EMMA module not found. Install: pip install torch einops")
        sys.exit(1)

    class IRVIDataset(Dataset):
        def __init__(self, pairs, patch_size=128, stride=100):
            self.pairs = pairs
            self.ps = patch_size
            self.stride = stride
            self.patches = []
            for ir_p, vi_p in tqdm(pairs, desc="Loading patches"):
                try:
                    ir = np.array(
                        __import__("PIL.Image").Image.open(ir_p).convert("L"),
                        dtype=np.float32,
                    ) / 255.0
                    vi = np.array(
                        __import__("PIL.Image").Image.open(vi_p).convert("RGB"),
                        dtype=np.float32,
                    )
                    vi_gray = (
                        0.299 * vi[:, :, 0] + 0.587 * vi[:, :, 1] + 0.114 * vi[:, :, 2]
                    ) / 255.0
                    h, w = ir.shape
                    for i in range(0, h - self.ps + 1, self.stride):
                        for j in range(0, w - self.ps + 1, self.stride):
                            self.patches.append(
                                (
                                    ir[i : i + self.ps, j : j + self.ps],
                                    vi_gray[i : i + self.ps, j : j + self.ps],
                                )
                            )
                except Exception as e:
                    print(f"Skip {ir_p}: {e}")

        def __len__(self):
            return len(self.patches)

        def __getitem__(self, i):
            ir, vi = self.patches[i]
            ir = torch.FloatTensor(ir).unsqueeze(0)
            vi = torch.FloatTensor(vi).unsqueeze(0)
            return ir, vi

    ds = IRVIDataset(pairs)
    if len(ds) == 0:
        print("No valid patches extracted.")
        sys.exit(1)

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ufuser().to(device)

    # Load pretrained EMMA if exists
    ckpt = os.path.join(_BACKEND, "emma", "models", "EMMA.pth")
    if os.path.isfile(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("Loaded pretrained EMMA, fine-tuning...")

    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    epochs = 20

    for ep in range(epochs):
        total = 0
        for ir, vi in loader:
            ir, vi = ir.to(device), vi.to(device)
            out = model(ir, vi)
            loss = torch.nn.functional.l1_loss(out, ir) + torch.nn.functional.l1_loss(
                out, vi
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {ep+1}/{epochs} loss={total/len(loader):.4f}")

    out_dir = os.path.join(_BACKEND, "emma", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "EMMA_irvis_finetuned.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved to {out_path}")
    print("To use: set model_path in emma_fuse or ir_vis_clean_fuse.")


if __name__ == "__main__":
    main()
