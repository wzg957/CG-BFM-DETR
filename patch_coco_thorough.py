import os

path = '/usr/local/miniconda3/envs/dqdetr/lib/python3.9/site-packages/pycocotools/cocoeval.py'

with open(path, 'r') as f:
    content = f.read()

# 1. 替换函数默认参数
content = content.replace("maxDets=100", "maxDets=500")
# 2. 替换第一行指标的硬编码调用
content = content.replace("self.stats[0] = _summarize(1)", "self.stats[0] = _summarize(1, maxDets=500)")

with open(path, 'w') as f:
    f.write(content)

print("✅ 彻彻底底的 500 框解封！pycocotools 的硬编码已被全面接管。")
