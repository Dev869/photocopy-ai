"""Embed proxy JPEGs with SigLIP2 on MPS -> index/embeddings.npy + index/ids.json."""
import glob, json, os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip2-base-patch16-384"


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    return model, processor, device


def embed_paths(paths, model, processor, device, batch_size=32):
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = getattr(feats, "pooler_output", feats)
        out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        print(f"  {min(i + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out)


if __name__ == "__main__":
    paths = sorted(glob.glob("proxies/*.jpg"))
    assert paths, "no proxies — run proxies.py first"
    model, processor, device = load_model()
    embs = embed_paths(paths, model, processor, device)
    os.makedirs("index", exist_ok=True)
    np.save("index/embeddings.npy", embs)
    json.dump([os.path.splitext(os.path.basename(p))[0] for p in paths],
              open("index/ids.json", "w"))
    print(f"indexed {len(paths)} images, dim={embs.shape[1]}, device={device}")
