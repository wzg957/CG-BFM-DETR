import re

file_path = '/hy-tmp/DQ-DETR/engine.py'
with open(file_path, 'r') as f:
    content = f.read()

# 用正则匹配 log_every(data_loader, 任何内容, header) 并替换为 2000
new_content = re.sub(r'log_every\(data_loader,\s*[^,]+,\s*header\)', 'log_every(data_loader, 2000, header)', content)

with open(file_path, 'w') as f:
    f.write(new_content)

print("✅ engine.py 打印频率已成功强制硬编码为 2000！")
