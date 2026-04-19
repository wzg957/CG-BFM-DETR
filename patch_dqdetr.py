import sys

file_path = '/hy-tmp/DQ-DETR/models/dqdetr/dqdetr.py'

with open(file_path, 'r') as f:
    content = f.read()

# 替换所有危险的 range 写法
old_str = "torch.range(0, len(targets[i]['labels']) - 1)"
new_str = "torch.arange(len(targets[i]['labels']))"

if old_str in content:
    content = content.replace(old_str, new_str)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ dqdetr.py 修复成功！原作者的空图索引 Bug 已被铲除！")
else:
    print("⚠️ 没找到目标代码，可能已经被修复过，或者存在格式差异。")
