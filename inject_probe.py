import os

probe_code = """
# ==========================================================
# 🔥 [论文专属：Params & GFLOPs 双轨体检仪]
# ==========================================================
# (已删掉 import torch，完美避开作用域冲突)
model.eval()
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print()
print('🚀'*20)
print(f'✅ 真实总参数量 (Total Params): {total_params:.3f} M')
try:
    from thop import profile
    from util.misc import nested_tensor_from_tensor_list
    dummy_tensor = torch.randn(1, 3, 640, 640).cuda()
    dummy_nested = nested_tensor_from_tensor_list([dummy_tensor[0]])
    macs, _ = profile(model, inputs=(dummy_nested, ), verbose=False)
    gflops = (macs * 2) / 1e9
    print(f'✅ 真实总计算量 (GFLOPs): {gflops:.3f} G')
except Exception as e:
    print(f'⚠️ GFLOPs 计算失败: {e}')
print('🚀'*20)
print()
exit(0)
"""

for base_dir in ['/hy-tmp/DQ-DETR/', '/hy-tmp/DQ-DETR-Patent/']:
    main_file = os.path.join(base_dir, 'main_aitod.py')
    target_file = os.path.join(base_dir, 'count_params.py')
    
    if os.path.exists(main_file):
        with open(main_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        injected = False
        for line in lines:
            new_lines.append(line)
            if 'model.to(device)' in line and not injected:
                indent = line[:len(line) - len(line.lstrip())]
                for probe_line in probe_code.strip().split('\n'):
                    if probe_line.strip():
                        new_lines.append(indent + probe_line + '\n')
                    else:
                        new_lines.append('\n')
                injected = True
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f'✅ 成功修复并重新植入体检代码: {target_file}')
