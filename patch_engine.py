import sys

file_path = '/hy-tmp/DQ-DETR/engine.py'

# 读取原始文件
with open(file_path, 'r') as f:
    content = f.read()

# 防止重复修改
if '安全检查：过滤掉宽度或高度' in content:
    print("✅ engine.py 已经包含安全检查补丁，无需重复修改！")
    sys.exit(0)

lines = content.split('\n')
new_lines = []
patched = False

for line in lines:
    # 找到计算 Loss 的那一行
    if 'loss_dict = criterion(outputs, targets)' in line and not patched:
        # 获取当前行的缩进空格数
        indent = line[:len(line) - len(line.lstrip())]
        # 插入带正确缩进的安全检查代码
        new_lines.append(indent + "# --- 安全检查：过滤掉宽度或高度<=0的无效框 ---")
        new_lines.append(indent + "for target in targets:")
        new_lines.append(indent + "    keep = (target['boxes'][:, 2:] > 0).all(1)")
        new_lines.append(indent + "    if not keep.all():")
        new_lines.append(indent + "        target['boxes'] = target['boxes'][keep]")
        new_lines.append(indent + "        target['labels'] = target['labels'][keep]")
        new_lines.append(indent + "# ---------------------------------------------")
        patched = True
    new_lines.append(line)

# 写回文件
with open(file_path, 'w') as f:
    f.write('\n'.join(new_lines))

if patched:
    print("✅ engine.py 修改成功！病灶已清除！")
else:
    print("❌ 未找到目标代码行，请确认 engine.py 是否为原版。")
