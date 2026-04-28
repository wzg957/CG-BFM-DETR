import os
import torch
import torch.nn as nn
import numpy as np
import math
import collections
from pycocotools.coco import COCO
from sklearn.mixture import GaussianMixture
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ================= Configuration Area =================
ANN_FILE = '/data/zegao/coco_visdrone/annotations/aitodv2_trainval.json' 
IMG_DIR = '/data/zegao/coco_visdrone/images/trainval'
D_MODEL = 256
BATCH_SIZE = 256 
FREQ_RATIO = 0.05  


def get_spatial_vector(box_i, box_j):
    xi, yi, wi, hi = box_i
    xj, yj, wj, hj = box_j
    dx = (xj + wj/2 - (xi + wi/2)) / wi
    dy = (yj + hj/2 - (yi + hi/2)) / hi
    dw = math.log(wj / wi + 1e-6)
    dh = math.log(hj / hi + 1e-6)
    return [dx, dy, dw, dh]

class CropDataset(Dataset):
    def __init__(self, crop_list, transform):
        self.crop_list = crop_list
        self.transform = transform

    def __len__(self): return len(self.crop_list)

    def __getitem__(self, idx):
        img_path, bbox = self.crop_list[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            x, y, w, h = [int(v) for v in bbox]
            crop = img.crop((max(0, x), max(0, y), min(img.width, x+w), min(img.height, y+h)))
            if crop.size[0] < 2 or crop.size[1] < 2: return torch.zeros(3, 64, 64)
            return self.transform(crop)
        except:
            return torch.zeros(3, 64, 64)

def main():
    print("1. Initialize feature extractor (ResNet-18 -> 256D)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = models.resnet18(pretrained=True)
    extractor = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(), nn.Linear(512, D_MODEL))
    extractor = extractor.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    coco = COCO(ANN_FILE)
    img_ids = coco.getImgIds()
    cat2idx = {cat_id: i for i, cat_id in enumerate(sorted(coco.getCatIds()))}
    
    
    FREQ_THRESHOLD = int(FREQ_RATIO * len(img_ids))
    print(f"-> Total number of images in dataset |D| = {len(img_ids)}")
    print(f"-> High-frequency threshold Nfreq calculated according to the paper's formula = {FREQ_THRESHOLD}")
    
    # === Stage 1: Mine high-frequency co-occurrence pairs ===
    pair_counts = collections.defaultdict(int)
    for img_id in tqdm(img_ids, desc="Calculate category co-occurrence frequency"):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if len(anns) < 2: continue
        cat_counts = collections.Counter(cat2idx[ann['category_id']] for ann in anns)
        for c_i, count_i in cat_counts.items():
            for c_j, count_j in cat_counts.items():
                if c_i == c_j and count_i < 2: continue
                pair_counts[(c_i, c_j)] += 1

    valid_pairs = sorted(list(set(pair for pair, count in pair_counts.items() if count > FREQ_THRESHOLD)))
    P = len(valid_pairs)
    print(f" -> Found {P} pairs of high-frequency co-occurrence combinations.\n")
    
    # === Stage 2: Collect data strictly by Pair ===
    pair_relations = collections.defaultdict(list)
    pair_crops = collections.defaultdict(list)
    
    for img_id in tqdm(img_ids, desc="Extract spatial features"): 
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if len(anns) < 2: continue
        img_path = os.path.join(IMG_DIR, coco.loadImgs(img_id)[0]['file_name'])
        
        for i, ann_i in enumerate(anns):
            c_i = cat2idx[ann_i['category_id']]
            for j, ann_j in enumerate(anns):
                if i == j: continue
                c_j = cat2idx[ann_j['category_id']]
                if (c_i, c_j) not in valid_pairs: continue
                
                vec = get_spatial_vector(ann_i['bbox'], ann_j['bbox'])
                pair_relations[(c_i, c_j)].append(vec)
                pair_crops[(c_i, c_j)].append((img_path, ann_j['bbox'])) # Collect visual features of associated targets
                break 

    # === Stage 3: Build GMM by Pair and extract [P, 256] templates ===
    print(f"\n3. Extract features, build prior template matrix of size [{P}, 256]...")
    enh_templates = torch.zeros(P, D_MODEL).to(device)
    sup_templates = torch.zeros(P, D_MODEL).to(device)

    def extract_features_batched(crop_list):
        if len(crop_list) == 0: return torch.zeros(D_MODEL).to(device)
        dataloader = DataLoader(CropDataset(crop_list, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        all_feats = []
        for batch in dataloader:
            all_feats.append(extractor(batch.to(device)))
        all_feats = torch.cat(all_feats, dim=0)
        valid_mask = all_feats.abs().sum(dim=1) > 0
        return all_feats[valid_mask].mean(dim=0) if valid_mask.sum() > 0 else torch.zeros(D_MODEL).to(device)
        
    with torch.no_grad():
        for p_idx, (c_i, c_j) in enumerate(valid_pairs):
            vecs = np.array(pair_relations[(c_i, c_j)])
            crops_info = pair_crops[(c_i, c_j)]
            if len(vecs) < 10: continue
            
            # Fit GMM for the current specific high-frequency pair
            gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
            gmm.fit(vecs)
            log_probs = gmm.score_samples(vecs)
            threshold = np.percentile(log_probs, 15) 
            
            normal_crops = [crops_info[k] for k, score in enumerate(log_probs) if score >= threshold]
            anomaly_crops = [crops_info[k] for k, score in enumerate(log_probs) if score < threshold]
            
            # Assign directly to the corresponding row in P dimension
            enh_templates[p_idx] = extract_features_batched(normal_crops)
            sup_templates[p_idx] = extract_features_batched(anomaly_crops)

    torch.save({
        'enh_templates': enh_templates.cpu(), # Shape: [P, 256]
        'sup_templates': sup_templates.cpu(), # Shape: [P, 256]
        'valid_pairs': valid_pairs            # Save Pair list for reference
    }, 'offline_context_templates.pt')

if __name__ == '__main__':
    main()
