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
            print(f"Warning: Failed to load offline templates. Error message: {e}")
            self.register_buffer('enh_templates', torch.zeros(num_classes, d_model))
            self.register_buffer('sup_templates', torch.zeros(num_classes, d_model))
        
        # Global scene feature projection network
        self.global_scene_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # ==========================================================
        # Modulation weight coefficients (Model 1 core: learnable independent scalars, no dynamic gating)
        # ==========================================================
        self.w_enh = nn.Parameter(torch.tensor(0.5))
        self.w_scene = nn.Parameter(torch.tensor(0.5))
        self.w_sup = nn.Parameter(torch.tensor(0.5))

    def forward(self, srcs, ccm_class_logits):
        """
        srcs: list of Tensor, multi-scale feature maps [B, C, H, W]
        """
        out_srcs = []
        
        # Extract global scene features (use the deepest feature srcs[-1] for global pooling)
        global_feat = srcs[-1].mean(dim=(2, 3)) # [B, d_model]
        scene_context = self.global_scene_proj(global_feat) # [B, d_model]

        for src in srcs:
            # 1. Pixel-level dense matching
            enh_sim = torch.einsum('kc,bchw->bkhw', self.enh_templates, src)
            sup_sim = torch.einsum('kc,bchw->bkhw', self.sup_templates, src)
            
            # 2. Cross-category aggregation
            enh_map = enh_sim.max(dim=1, keepdim=True)[0].sigmoid()
            sup_map = sup_sim.max(dim=1, keepdim=True)[0].sigmoid()
            
            # Global scene attention
            scene_map = torch.einsum('bc,bchw->bhw', scene_context, src).unsqueeze(1).sigmoid()
            

            enh_weight = (enh_map * self.w_enh + scene_map * self.w_scene) + 1.0
            sup_weight = 1.0 - (sup_map * self.w_sup)
            
            # 3. Element-wise feature modulation
            modulated_src = src * enh_weight * sup_weight
            out_srcs.append(modulated_src)

        return out_srcs
