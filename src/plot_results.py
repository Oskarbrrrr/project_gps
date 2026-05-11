import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_curves(scenario_name):
    csv_path = f"logs/{scenario_name}_train_log.csv"
    if not os.path.exists(csv_path):
        print(f"找不到日志文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 创建一个 2x2 的大图
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制 Loss 曲线
    plt.subplot(2, 2, 1)
    plt.plot(df['Epoch'], df['Train_Loss'], label='Train Loss', marker='o', markersize=3)
    plt.plot(df['Epoch'], df['Val_Loss'], label='Val Loss', marker='s', markersize=3)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 绘制 Acc@3 曲线
    plt.subplot(2, 2, 2)
    plt.plot(df['Epoch'], df['Acc@3'], color='green', marker='^', markersize=3)
    plt.title('Top-3 Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # 3. 绘制 DBA 曲线
    plt.subplot(2, 2, 3)
    plt.plot(df['Epoch'], df['DBA'], color='purple', marker='d', markersize=3)
    plt.title('DBA Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('DBA')
    plt.grid(True)
    
    # 4. 绘制 APL 曲线 (注意 APL 是越低越好)
    plt.subplot(2, 2, 4)
    plt.plot(df['Epoch'], df['APL_dB'], color='red', marker='v', markersize=3)
    plt.title('Average Power Loss (APL) Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    # 保存图片
    plt.savefig(f"logs/{scenario_name}_curves.png", dpi=300)
    print(f"✅ 训练曲线已保存为: logs/{scenario_name}_curves.png")

if __name__ == "__main__":
    # 你跑完哪个场景，就可以画哪个场景的图
    plot_training_curves("scenario32")
    plot_training_curves("scenario33")
    plot_training_curves("scenario34")
