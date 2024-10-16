import os
import random
import shutil
import pandas as pd
import numpy as np


def reduce_dataset(csv_path, img_dir, output_csv_path, output_img_dir, num_samples, seed=None):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 确保 image_name 列存在
    if 'image_name' not in df.columns:
        raise ValueError("CSV file doesn't contain 'image_name' column")

    # 随机选择指定数量的样本
    selected_indices = random.sample(range(len(df)), min(num_samples, len(df)))
    selected_samples = df.iloc[selected_indices]

    # 生成随机评分
    random_scores = np.random.uniform(50.0, 100.0, size=len(selected_samples))
    selected_samples['score'] = random_scores.round(1)  # 四舍五入到一位小数

    # 创建输出图片目录
    os.makedirs(output_img_dir, exist_ok=True)

    # 复制选中的图片并更新DataFrame
    for _, row in selected_samples.iterrows():
        img_name = f"{row['image_name']}.jpg"
        src_path = os.path.join(img_dir, img_name)
        dst_path = os.path.join(output_img_dir, img_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Image not found: {src_path}")

    # 保存更新后的CSV文件
    selected_samples.to_csv(output_csv_path, index=False)

    print(f"Reduced dataset saved to {output_csv_path} with {len(selected_samples)} samples")
    print(f"Score range: {selected_samples['score'].min():.1f} - {selected_samples['score'].max():.1f}")


def main():
    base_dir = 'data'  # 更改为您的数据目录路径
    random_seed = 42  # 设置一个随机种子以确保结果可重复

    datasets = [
        ('train', 400),
        ('validation', 50),
        ('test', 50)
    ]

    for dataset, num_samples in datasets:
        csv_path = os.path.join(base_dir, dataset, f'{dataset}.csv')
        img_dir = os.path.join(base_dir, dataset, 'img')
        output_csv_path = os.path.join(base_dir, f'{dataset}_reduced.csv')
        output_img_dir = os.path.join(base_dir, f'{dataset}_reduced_img')

        reduce_dataset(csv_path, img_dir, output_csv_path, output_img_dir, num_samples, seed=random_seed)


if __name__ == "__main__":
    main()