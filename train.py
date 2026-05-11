import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np
import csv
import datetime
import time  # <--- 新增导入 time 模块用于计算 ETA

# 从 src 导入你已经写好的模块
from src.dataset import MultimodalDataset
from src.model import BeMambaModel
from src.utils import calculate_topk_accuracy, calculate_dba_score, calculate_apl

# ==========================================
# 1. 带动态 Alpha 权重的 FocalLoss
# ==========================================
class AlphaFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(AlphaFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # alpha 为 [64] 维的 tensor

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) 
        
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
        else:
            at = 1.0
            
        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==========================================
# 2. 动态计算当前场景训练集的波束频率
# ==========================================
def calculate_alpha_weights(csv_path, num_classes=64):
    df = pd.read_csv(csv_path)
    beams = df['unit1_beam'].values - 1 
    class_counts = np.bincount(beams, minlength=num_classes)
    
    total_samples = len(beams)
    # 逆频率加权，+1 防止除 0
    alpha = total_samples / (num_classes * (class_counts + 1))
    # 归一化，保持 loss 整体量级不崩
    alpha = alpha / np.mean(alpha)
    
    return torch.tensor(alpha, dtype=torch.float32)

# ==========================================
# 3. 核心训练与评估主循环
# ==========================================
def run_scenario(scenario_name):
    print(f"\n========== 开始训练场景: {scenario_name} ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    train_csv_path = f"Data/splits/{scenario_name}_train.csv"
    alpha_weights = calculate_alpha_weights(train_csv_path, num_classes=64).to(device)
    print("✅ 已生成自适应 Alpha 类别权重。")
    
    train_ds = MultimodalDataset(mode='train', scenario_name=scenario_name)
    val_ds   = MultimodalDataset(mode='val', scenario_name=scenario_name)
    test_ds  = MultimodalDataset(mode='test', scenario_name=scenario_name)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=8)

    model = BeMambaModel().to(device)
    criterion = AlphaFocalLoss(alpha=alpha_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)

    best_val_loss = float('inf') 
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # ==========================================
    # 🌟 早停机制参数配置 🌟
    # ==========================================
    patience = 10  # 10轮验证损失不下降就停止
    early_stop_counter = 0  # 计数器
    early_stop = False  # 早停标志
    
    # 初始化日志文件
    log_file_path = f"logs/{scenario_name}_train_log.csv"
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Acc@3', 'DBA', 'APL_dB'])
    
    # ==========================================
    # 训练轮数修改为 100 轮
    # ==========================================
    epochs = 100
    start_time = time.time()  # <--- 新增：记录开始时间
    
    for epoch in range(epochs):
        # --- 训练 ---
        model.train()
        train_loss_tot = 0.0
        for imgs, radars, lidars, gps, targets, _ in train_loader:
            imgs, radars, lidars, gps, targets = imgs.to(device), radars.to(device), lidars.to(device), gps.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, radars, lidars, gps)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_tot += loss.item() * targets.size(0)
            
        # --- 验证 ---
        model.eval()
        t1_tot, t3_tot, dba_tot, apl_tot, val_loss_tot = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, radars, lidars, gps, targets, power_vec in val_loader:
                imgs, radars, lidars, gps, targets = imgs.to(device), radars.to(device), lidars.to(device), gps.to(device), targets.to(device)
                outputs = model(imgs, radars, lidars, gps)
                
                v_loss = criterion(outputs, targets)
                bs = targets.size(0)
                val_loss_tot += v_loss.item() * bs
                
                acc1, acc3 = calculate_topk_accuracy(outputs, targets, topk=(1, 3))
                t1_tot += acc1 * bs
                t3_tot += acc3 * bs
                dba_tot += calculate_dba_score(outputs, targets) * bs
                apl_tot += calculate_apl(outputs, power_vec) * bs
                
        n_train, n_val = len(train_ds), len(val_ds)
        epoch_train_loss = train_loss_tot / n_train
        epoch_val_loss = val_loss_tot / n_val
        epoch_dba = dba_tot / n_val
        epoch_apl = apl_tot / n_val
        epoch_acc3 = t3_tot / n_val
        
        # ==========================================
        # 🌟 计算并格式化 ETA 🌟
        # ==========================================
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = int(avg_time_per_epoch * remaining_epochs)
        eta_str = str(datetime.timedelta(seconds=eta_seconds))
        
        # <--- 修改：打印语句加入了 ETA
        print(f"Epoch {epoch+1:02d}/{epochs} | Train L: {epoch_train_loss:.4f} | Val L: {epoch_val_loss:.4f} | Acc@3: {epoch_acc3:.2f}% | DBA: {epoch_dba:.4f} | APL: {epoch_apl:.4f} dB | ⏳ ETA: {eta_str}")
        
        # 写入日志
        with open(log_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{epoch_train_loss:.4f}", f"{epoch_val_loss:.4f}", f"{epoch_acc3:.2f}", f"{epoch_dba:.4f}", f"{epoch_apl:.4f}"])
        
        # ==========================================
        # 🌟 早停逻辑 & 模型保存 🌟
        # ==========================================
        if epoch_val_loss < best_val_loss:
            # 验证损失下降：更新最佳值，重置计数器
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"checkpoints/best_{scenario_name}.pth")
            print(f"  --> [Saved Best Model] Val Loss 创新低: {best_val_loss:.4f}")
        else:
            # 验证损失未下降：计数器+1
            early_stop_counter += 1
            print(f"  --> [Early Stop Counter] 连续 {early_stop_counter}/{patience} 轮无提升")
            
            # 达到耐心值，触发早停
            if early_stop_counter >= patience:
                print(f"\n🚨 验证损失连续 {patience} 轮未下降，触发早停！")
                early_stop = True
                break

    # --- 最终 Test 评估阶段 ---
    print(f"\n>>> 载入最佳权重进行 Test 评估...")
    model.load_state_dict(torch.load(f"checkpoints/best_{scenario_name}.pth"))
    model.eval()
    
    t1_tot, t3_tot, dba_tot, apl_tot = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgs, radars, lidars, gps, targets, power_vec in test_loader:
            imgs, radars, lidars, gps, targets = imgs.to(device), radars.to(device), lidars.to(device), gps.to(device), targets.to(device)
            outputs = model(imgs, radars, lidars, gps)
            
            acc1, acc3 = calculate_topk_accuracy(outputs, targets, topk=(1, 3))
            bs = targets.size(0)
            t1_tot += acc1 * bs
            t3_tot += acc3 * bs
            dba_tot += calculate_dba_score(outputs, targets) * bs
            apl_tot += calculate_apl(outputs, power_vec) * bs
            
    n_test = len(test_ds)
    final_res = f"""
[Final Results for {scenario_name} Test Set]
Top-1 Acc: {t1_tot/n_test:.2f}%
Top-3 Acc: {t3_tot/n_test:.2f}%
DBA Score: {dba_tot/n_test:.4f}
APL Loss:  {apl_tot/n_test:.4f} dB
"""
    print(final_res)
    
    # 保存最终测试结果
    with open(f"logs/{scenario_name}_final_test_result.txt", "w") as f:
        f.write(final_res)

if __name__ == "__main__":
    # 连续依次跑三个场景
    run_scenario("scenario32")
    torch.cuda.empty_cache()  # <--- 新增：清空显存，防止连续运行时 OOM
    
    run_scenario("scenario33")
    torch.cuda.empty_cache()  
    
    run_scenario("scenario34")
