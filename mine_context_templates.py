import os
import torch
import torch.nn as nn
import numpy as np
import math
import random 
from collections import defaultdict 
from pycocotools.coco import COCO
from sklearn.mixture import GaussianMixture
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


ANN_FILE = '/hy-tmp/coco_visdrone/annotations/aitodv2_trainval.json' 
IMG_DIR = '/hy-tmp/coco_visdrone/images/trainval'
D_MODEL = 256
NUM_CLASSES = 10
BATCH_SIZE = 256 
N_FREQ_THRESHOLD = 50

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

    def __len__(self):
        return len(self.crop_list)

    def __getitem__(self, idx):
        img_path, bbox = self.crop_list[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            x, y, w, h = [int(v) for v in bbox]
            crop = img.crop((max(0, x), max(0, y), min(img.width, x+w), min(img.height, y+h)))
            if crop.size[0] < 2 or crop.size[1] < 2:
                return torch.zeros(3, 64, 64)
            return self.transform(crop)
        except:
            return torch.zeros(3, 64, 64)

def main():
    print("1. 初始化特征提取器 (ResNet-18 -> 256D)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = models.resnet18(pretrained=True)
    
    extractor = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(), nn.Linear(512, D_MODEL))
    extractor = extractor.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"2. 加载 COCO 标注: {ANN_FILE}")
    coco = COCO(ANN_FILE)
    img_ids = coco.getImgIds()
    cat_ids = sorted(coco.getCatIds())
    cat2idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    
    
    print("2.5 [论文对齐] Stage 1: 进行全局高频共现类别对挖掘...")
    pair_counts = defaultdict(int)
    for img_id in tqdm(img_ids, desc="Counting pairs"):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        cats_in_img = [cat2idx[ann['category_id']] for ann in anns]
        
        
        for i, c_i in enumerate(cats_in_img):
            for j, c_j in enumerate(cats_in_img):
                if i != j:
                    pair_counts[(c_i, c_j)] += 1

    
    P_valid = set(pair for pair, count in pair_counts.items() if count > N_FREQ_THRESHOLD)
    print(f"挖掘完成：找到 {len(P_valid)} 对有效高频共现组合 (N_freq > {N_FREQ_THRESHOLD})")
    
    class_relations = {i: [] for i in range(NUM_CLASSES)}
    class_crops = {i: [] for i in range(NUM_CLASSES)}
    
    print(f"3. [论文对齐] Stage 2: 遍历完整数据集，基于 P_valid 和随机策略收集空间向量...")
    for img_id in tqdm(img_ids, desc="Collecting vectors"): 
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if len(anns) < 2: continue
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMG_DIR, img_info['file_name'])
        
        for i, ann_i in enumerate(anns):
            c_i = cat2idx[ann_i['category_id']]
            
            
            valid_associates = []
            for j, ann_j in enumerate(anns):
                if i == j: continue
                c_j = cat2idx[ann_j['category_id']]
                
                if (c_i, c_j) in P_valid:
                    valid_associates.append(ann_j)
            
            if valid_associates:
                
                chosen_ann_j = random.choice(valid_associates)
                vec = get_spatial_vector(ann_i['bbox'], chosen_ann_j['bbox'])
                class_relations[c_i].append(vec)
                class_crops[c_i].append((img_path, ann_i['bbox']))
            

    enh_templates = torch.zeros(NUM_CLASSES, D_MODEL).to(device)
    sup_templates = torch.zeros(NUM_CLASSES, D_MODEL).to(device)

    print("4. [论文对齐] Stage 3: 使用 GMM 拟合提取特征模板 (K=2)...")
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            vecs = np.array(class_relations[c])
            crops_info = class_crops[c]
            if len(vecs) < 10: 
                print(f"类别 {c} 样本过少，跳过...")
                continue
            
            gmm = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
            gmm.fit(vecs)
            log_probs = gmm.score_samples(vecs)
            
            threshold = np.percentile(log_probs, 15) 
            
            normal_crops = []
            anomaly_crops = []
            
            for k, score in enumerate(log_probs):
                img_path, bbox = crops_info[k]
                if score >= threshold:
                    normal_crops.append((img_path, bbox))
                else:
                    anomaly_crops.append((img_path, bbox))
                    
            def extract_features_batched(crop_list):
                if len(crop_list) == 0: return torch.zeros(D_MODEL).to(device)
                dataset = CropDataset(crop_list, transform)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
                
                all_feats = []
                for batch in dataloader:
                    batch = batch.to(device)
                    feats = extractor(batch)
                    all_feats.append(feats)
                    
                all_feats = torch.cat(all_feats, dim=0)
                valid_mask = all_feats.abs().sum(dim=1) > 0
                if valid_mask.sum() == 0: return torch.zeros(D_MODEL).to(device)
                
                valid_feats = all_feats[valid_mask]
                return valid_feats.mean(dim=0)
                
            enh_templates[c] = extract_features_batched(normal_crops)
            sup_templates[c] = extract_features_batched(anomaly_crops)
            print(f"类别 {c} 模板提取完成 | 正常(Enh): {len(normal_crops)}, 异常(Sup): {len(anomaly_crops)}")

    torch.save({
        'enh_templates': enh_templates.cpu(),
        'sup_templates': sup_templates.cpu()
    }, 'offline_context_templates.pt')
    

if __name__ == '__main__':
    main()
