import torch
import numpy as np

def calculate_topk_accuracy(output, target, topk=(1, 3)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def calculate_dba_score(output, target, K=3, delta=5):
    with torch.no_grad():
        N = target.size(0)
        _, pred = output.topk(K, 1, True, True)
        target_expanded = target.view(N, 1)
        diff = torch.abs(pred - target_expanded).float() / delta
        
        eta_k_list = []
        for k in range(1, K + 1):
            min_diff_k = torch.min(diff[:, :k], dim=1)[0]
            clamped_diff = torch.clamp(min_diff_k, max=1.0)
            eta_k = 1.0 - (torch.sum(clamped_diff) / N)
            eta_k_list.append(eta_k.item())
        return sum(eta_k_list) / K

def calculate_apl(output, power_vectors):
    """
    计算 APL (Average Power Loss)
    衡量预测波束与最佳波束之间的功率损失 (dB)
    """
    with torch.no_grad():
        _, pred_indices = output.topk(1, 1, True, True)
        
        loss_list = []
        for i in range(len(pred_indices)):
            pred_idx = pred_indices[i].item()
            p_vec = power_vectors[i].cpu().numpy()
            
            # === 【终极清洗】: 只要遇到 NaN, Inf, -Inf 全都强制变成 0.0 ===
            p_vec = np.nan_to_num(p_vec, nan=0.0, posinf=0.0, neginf=0.0)
            
            p_opt = np.max(p_vec)      
            p_pred = p_vec[pred_idx]   
            
            # 如果整条向量的最大值都是 0 (意味着这个样本根本没测到功率，或者是坏数据)
            # 我们直接记这一次的 Loss 为 0 dB，跳过它的计算，防止干扰平均值
            if p_opt <= 1e-12:
                loss_list.append(0.0)
                continue
                
            # 限制极小值防止除 0
            p_pred = np.clip(p_pred, a_min=1e-12, a_max=None)
            
            # 计算功率损失 (dB)
            power_loss = 10 * np.log10(p_opt / p_pred)
            
            # 再次以防万一，如果算出来还是 nan，就记为 0
            if np.isnan(power_loss):
                power_loss = 0.0
                
            loss_list.append(power_loss)
            
        # 如果整个 batch 的数据全是坏的，返回 0.0
        if len(loss_list) == 0:
            return 0.0
            
        return np.mean(loss_list) # 注意：这里改成了 np.mean(loss_list)，之前是 sum