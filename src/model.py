import torch
import torch.nn as nn
from torchvision import models
from mamba_ssm import Mamba

class MB_Mamba_Block(nn.Module):
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate_proj = nn.Sequential(nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, x):
        fn = self.ln(x)
        fw = self.gate_proj(fn)
        out_fwd = self.mamba_fwd(fn)
        out_bwd = torch.flip(self.mamba_bwd(torch.flip(fn, dims=[1])), dims=[1])
        return (fw * out_fwd) + (fw * out_bwd)

class BeMambaModel(nn.Module):
    def __init__(self, num_classes=64):
        super(BeMambaModel, self).__init__()
        
        # 1. 图像特征提取
        res34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.img_net = nn.Sequential(res34.conv1, res34.bn1, res34.relu, res34.maxpool, res34.layer1, res34.layer2)
        for name, param in self.img_net.named_parameters():
            if "layer2" not in name: param.requires_grad = False
        
        # 2. 雷达特征提取 (双通道输入)
        res18_r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        res18_r.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.radar_net = nn.Sequential(res18_r.conv1, res18_r.bn1, res18_r.relu, res18_r.maxpool, res18_r.layer1, res18_r.layer2)
        
        # 3. LiDAR特征提取 (单通道 BEV)
        res18_l = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        res18_l.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.lidar_net = nn.Sequential(res18_l.conv1, res18_l.bn1, res18_l.relu, res18_l.maxpool, res18_l.layer1, res18_l.layer2)
        
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.gps_mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128))
        
        # 4. Intra-modal 时序融合 (独立分支)
        self.tsm_img = Mamba(d_model=128)
        self.tsm_rad = Mamba(d_model=128)
        self.tsm_lid = Mamba(d_model=128)
        self.ln_img = nn.LayerNorm(128)
        self.ln_rad = nn.LayerNorm(128)
        self.ln_lid = nn.LayerNorm(128)
        
        # 5. Modal Sequence 交叉融合 (Eq 12 & 14，平行送入三种排列)
        self.msm_block_1 = MB_Mamba_Block(d_model=128)
        self.msm_block_2 = MB_Mamba_Block(d_model=128)
        self.msm_block_3 = MB_Mamba_Block(d_model=128)
        
        # 预测头 (带 Dropout)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes))

    def _process_modality(self, x, net, tsm, ln):
        B, S, C, H, W = x.size()
        x_flat = x.view(-1, C, H, W)
        feat = self.pool(net(x_flat)) 
        feat = feat.view(B, S, 128, 36).transpose(2, 3).reshape(B, S * 36, 128)
        feat_out = tsm(ln(feat)).view(B, S, 36, 128)
        return feat_out.sum(dim=1) 

    def forward(self, imgs, radars, lidars, gps):
        img_fused = self._process_modality(imgs, self.img_net, self.tsm_img, self.ln_img)
        rad_fused = self._process_modality(radars, self.radar_net, self.tsm_rad, self.ln_rad)
        lid_fused = self._process_modality(lidars, self.lidar_net, self.tsm_lid, self.ln_lid)
        
        # ==========================================================
        # 🌟 此处为预留的 Modality Masking (创新点) 位置 🌟
        # 后续你可以在这写：if mask_radar: rad_fused = rad_fused * 0
        # ==========================================================

        gps_start = self.gps_mlp(gps[:, 0, :]).unsqueeze(1)
        gps_end = self.gps_mlp(gps[:, 1, :]).unsqueeze(1)
        
        # 三种混合排列 (Eq. 12)
        comb1 = torch.cat([gps_start, img_fused, lid_fused, rad_fused, gps_end], dim=1) 
        comb2 = torch.cat([gps_start, lid_fused, rad_fused, img_fused, gps_end], dim=1) 
        comb3 = torch.cat([gps_start, rad_fused, img_fused, lid_fused, gps_end], dim=1) 
        
        out1 = self.msm_block_1(comb1)
        out2 = self.msm_block_2(comb2)
        out3 = self.msm_block_3(comb3)
        msm_out = (out1 + out2 + out3) / 3.0 
        
        return self.head(msm_out.mean(dim=1))