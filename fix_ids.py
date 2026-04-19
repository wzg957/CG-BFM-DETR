import json
import os

files = [
    '/hy-tmp/coco_visdrone/annotations/aitodv2_trainval.json',
    '/hy-tmp/coco_visdrone/annotations/aitodv2_val.json',
    '/hy-tmp/coco_visdrone/annotations/aitodv2_test.json'
]

for f_path in files:
    if not os.path.exists(f_path):
        print(f"⚠️ 跳过不存在的文件: {f_path}")
        continue
        
    print(f"正在修复: {f_path} ...")
    with open(f_path, 'r') as f:
        data = json.load(f)
    
    # 核心修复逻辑：ID 整体减 1，从 1~10 映射到 0~9
    for ann in data['annotations']:
        ann['category_id'] = ann['category_id'] - 1
        
    for cat in data['categories']:
        cat['id'] = cat['id'] - 1
        
    with open(f_path, 'w') as f:
        json.dump(data, f)
    print(f"✅ 修复完成！")

print("\n✨ 所有 JSON 文件的类别 ID 已成功从 1~10 修正为 0~9！")
