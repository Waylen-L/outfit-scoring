import torch
import torch.nn as nn


class KMeans(nn.Module):
    def __init__(self, n_clusters=5, max_iter=100):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def forward(self, X):
        batch_size, n_points, n_dims = X.shape

        centers = X[:, torch.randperm(n_points)[:self.n_clusters]]

        for _ in range(self.max_iter):
            distances = torch.cdist(X, centers)
            labels = torch.argmin(distances, dim=2)
            new_centers = torch.stack(
                [X[i][labels[i] == k].mean(dim=0) for i in range(batch_size) for k in range(self.n_clusters)]).view(
                batch_size, self.n_clusters, n_dims)
            if torch.allclose(centers, new_centers):
                break
            centers = new_centers

        percentages = torch.stack(
            [(labels[i] == k).float().mean() for i in range(batch_size) for k in range(self.n_clusters)]).view(
            batch_size, self.n_clusters)

        return centers, labels, percentages


class ColorFeatureExtractor(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.kmeans = KMeans(n_clusters=k)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        pixels = x.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)

        # 添加小的epsilon值以避免除以零
        epsilon = 1e-8
        pixels = pixels + epsilon

        centers, _, percentages = self.kmeans(pixels)

        # 使用torch.clamp来限制值的范围
        centers = torch.clamp(centers, min=0, max=1)
        percentages = torch.clamp(percentages, min=0, max=1)

        color_info = torch.cat([centers, percentages.unsqueeze(2)], dim=2)

        # 检查并替换 NaN 值
        color_info = torch.nan_to_num(color_info, nan=0.0, posinf=1.0, neginf=0.0)

        # 添加调试信息
        print(
            f"Color info stats: min={color_info.min().item()}, max={color_info.max().item()}, mean={color_info.mean().item()}")

        return color_info[torch.argsort(color_info[:, :, 3], dim=1, descending=True)]


def extract_color_features(image, k=5):
    extractor = ColorFeatureExtractor(k=k).to(image.device)
    with torch.no_grad():
        color_info = extractor(image)
    return color_info