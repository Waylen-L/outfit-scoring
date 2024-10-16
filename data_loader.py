import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class FashionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        required_columns = ['image_name', 'subCategory', 'score']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in the CSV file")

        self.data['image_name'] = self.data['image_name'].astype(str)

        # 创建类别到索引的映射
        self.category_to_index = {cat: idx for idx, cat in enumerate(self.data['subCategory'].unique())}
        print(f"Category to index mapping: {self.category_to_index}")
        print(f"Unique categories: {self.data['subCategory'].unique()}")

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.data.iloc[idx]['image_name']}.jpg")

        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image file not found: {img_name}")

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        score = torch.tensor(float(self.data.iloc[idx]['score']), dtype=torch.float32)
        category = torch.tensor(self.category_to_index[self.data.iloc[idx]['subCategory']], dtype=torch.long)

        return image, score, category

    def __len__(self):
        return len(self.data)


def get_data_loaders(paths, config):
    train_dataset = FashionDataset(paths.train_csv, paths.train_img_dir)
    val_dataset = FashionDataset(paths.val_csv, paths.val_img_dir)
    test_dataset = FashionDataset(paths.test_csv, paths.test_img_dir)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader