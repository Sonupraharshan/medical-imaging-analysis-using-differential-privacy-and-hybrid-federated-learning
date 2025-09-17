#!/usr/bin/env python3
import os, argparse, numpy as np, cv2
from glob import glob
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from torchvision import models, transforms
import torch
import torch.nn as nn
from openslide import OpenSlide

PATCH_SIZE = 1000
THUMB_SIZE = (64,64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_preproc = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def is_tissue(arr, white_thresh=240, min_frac=0.05):
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return (gray < white_thresh).mean() > min_frac

def get_densenet_extractor(version='densenet121'):
    model = getattr(models, version)(pretrained=True)
    features = model.features
    net = nn.Sequential(features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(DEVICE)
    net.eval()
    return net

def extract_patches(slide_path, out_dir, patch_size=PATCH_SIZE, stride=None):
    os.makedirs(out_dir, exist_ok=True)
    slide = OpenSlide(slide_path)
    level = 0
    w,h = slide.level_dimensions[level]
    step = patch_size if stride is None else stride
    idx=0
    for y in range(0, h-patch_size+1, step):
        for x in range(0, w-patch_size+1, step):
            patch = slide.read_region((x,y), level, (patch_size,patch_size)).convert("RGB")
            arr = np.array(patch)
            if is_tissue(arr):
                np.savez_compressed(os.path.join(out_dir,f"patch_{idx}.npz"), img=arr)
                idx+=1
    return idx

def build_mosaic_and_feats(patch_dir, out_bag_path, n_clusters=50, sample_frac=0.10, densenet_ver='densenet121'):
    files = [os.path.join(patch_dir,f) for f in os.listdir(patch_dir) if f.endswith('.npz')]
    if len(files)==0:
        return False
    thumbs=[]
    for f in files:
        arr = np.load(f)['img']
        thumbs.append(cv2.resize(arr, THUMB_SIZE).reshape(-1))
    thumbs = np.vstack(thumbs).astype('float32')
    k = min(n_clusters, len(thumbs))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=256).fit(thumbs)
    labels = km.labels_
    selected=[]
    for c in range(k):
        inds = np.where(labels==c)[0]
        take = max(1, int(len(inds)*sample_frac))
        chosen = np.random.choice(inds, take, replace=False)
        selected.extend([files[i] for i in chosen])
    patches = [np.load(p)['img'] for p in selected]
    patches = np.stack(patches)
    # featurize
    net = get_densenet_extractor(densenet_ver)
    feats=[]
    with torch.no_grad():
        for p in patches:
            x = _preproc(p).unsqueeze(0).to(DEVICE)
            f = net(x)
            feats.append(f.cpu().numpy().squeeze())
    feats = np.vstack(feats)
    np.savez_compressed(out_bag_path, feats=feats)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tmp_dir", default="tmp_patches")
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--sample_frac", type=float, default=0.10)
    parser.add_argument("--densenet", default="densenet121")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    label_map = {}
    if args.labels_csv and os.path.exists(args.labels_csv):
        import pandas as pd
        df = pd.read_csv(args.labels_csv)
        for _, r in df.iterrows():
            label_map[str(r['slide_id'])] = int(r['label'])

    wsi_paths = sorted(glob(os.path.join(args.wsi_dir, "*")))
    for wsi in wsi_paths:
        slide_id = os.path.splitext(os.path.basename(wsi))[0]
        print("Processing", slide_id)
        patch_dir = os.path.join(args.tmp_dir, slide_id)
        n = extract_patches(wsi, patch_dir, patch_size=args.patch_size)
        if n==0:
            print("No patches for", slide_id); continue
        out_bag = os.path.join(args.out_dir, f"bag_{slide_id}.npz")
        ok = build_mosaic_and_feats(patch_dir, out_bag, n_clusters=args.n_clusters, sample_frac=args.sample_frac, densenet_ver=args.densenet)
        if not ok:
            continue
        # attach label if available
        if slide_id in label_map:
            feats = np.load(out_bag)['feats']
            np.savez_compressed(out_bag, feats=feats, label=label_map[slide_id])
        else:
            feats = np.load(out_bag)['feats']
            np.savez_compressed(out_bag, feats=feats, label=-1)
        # optional: cleanup patches to save disk
        try:
            import shutil
            shutil.rmtree(patch_dir)
        except Exception:
            pass

if __name__ == "__main__":
    main()
