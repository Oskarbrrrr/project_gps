import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_csv(scenario_name, csv_path, output_dir):
    print(f"正在处理: {scenario_name} ...")
    if not os.path.exists(csv_path):
        print(f"找不到 {csv_path}，请检查路径！")
        return
        
    df = pd.read_csv(csv_path)
    
    # 60% Train, 40% Temp
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    # 20% Val, 20% Test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, f"{scenario_name}_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"{scenario_name}_val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"{scenario_name}_test.csv"), index=False)
    
    print(f"✅ {scenario_name} 拆分完成: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    # 注意相对路径，假设你在项目根目录下运行: python src/data_split.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Data", "Multi_Modal")
    out_dir = os.path.join(base_dir, "Data", "splits")
    
    split_csv("scenario32", os.path.join(data_dir, "scenario32.csv"), out_dir)
    split_csv("scenario33", os.path.join(data_dir, "scenario33.csv"), out_dir)
    split_csv("scenario34", os.path.join(data_dir, "scenario34.csv"), out_dir)

    print()