import torch
import torch.nn as nn

class CABFM(nn.Module):
    
    
    
    def __init__(self, d_model=256, num_classes=10, template_path='/data/zegao/DQ-DETR-Patent/offline_context_templates.pt'):
        super().__init__()
        self.d_model = d_model
        
        try:
            templates = torch.load(template_path, map_location='cpu')
            self.register_buffer('enh_templates', templates['enh_templates']) 
            self.register_buffer('sup_templates', templates['sup_templates']) 
        except Exception as e:
            print(f"警告: 未能加载离线模板。错误信息: {e}")
            self.register_buffer('enh_templates', torch.zeros(num_classes, d_model))
            self.register_buffer('sup_templates', torch.zeros(num_classes, d_model))
        
        # 全局场景特征投影网络
        self.global_scene_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # ==========================================================
        # 调制权重系数 (Model 1 核心：可学习的独立标量，无动态门控)
        # ==========================================================
        self.w_enh = nn.Parameter(torch.tensor(0.5))
        self.w_scene = nn.Parameter(torch.tensor(0.5))
        self.w_sup = nn.Parameter(torch.tensor(0.5))

    def forward(self, srcs, ccm_class_logits):
        """
        srcs: list of Tensor, 多尺度特征图 [B, C, H, W]
        """
        out_srcs = []
        
        # 提取全局场景特征 (使用最深层特征 srcs[-1] 做全局池化)
        global_feat = srcs[-1].mean(dim=(2, 3)) # [B, d_model]
        scene_context = self.global_scene_proj(global_feat) # [B, d_model]

        for src in srcs:
            # 1. 像素级密集匹配 (Dense Matching)
            enh_sim = torch.einsum('kc,bchw->bkhw', self.enh_templates, src)
            sup_sim = torch.einsum('kc,bchw->bkhw', self.sup_templates, src)
            
            # 2. 跨类别聚合
            enh_map = enh_sim.max(dim=1, keepdim=True)[0].sigmoid()
            sup_map = sup_sim.max(dim=1, keepdim=True)[0].sigmoid()
            
            # 全局场景注意力
            scene_map = torch.einsum('bc,bchw->bhw', scene_context, src).unsqueeze(1).sigmoid()
            

            enh_weight = (enh_map * self.w_enh + scene_map * self.w_scene) + 1.0
            sup_weight = 1.0 - (sup_map * self.w_sup)
            
            # 3. 逐元素特征调制
            modulated_src = src * enh_weight * sup_weight
            out_srcs.append(modulated_src)

        return out_srcs
