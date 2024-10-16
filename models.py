import torch
import torch.nn as nn
import torchvision.models as models
from color_feature_extractor import ColorFeatureExtractor

def get_base_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = model.fc.in_features
    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        num_features = model.fc.in_features
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        num_features = model.classifier[6].in_features
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, num_features

class FashionModel(nn.Module):
    def __init__(self, base_model_name, num_classes=2):
        super(FashionModel, self).__init__()
        self.base_model, num_features = get_base_model(base_model_name)

        # 移除最后的全连接层
        if base_model_name.startswith('resnet'):
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        elif base_model_name == 'vgg16':
            self.base_model = self.base_model.features
        elif base_model_name == 'efficientnet_b0':
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        self.color_feature_extractor = ColorFeatureExtractor(k=5)
        self.color_feature_dim = 64
        self.color_feature_processor = nn.Linear(5 * 5 * 4, self.color_feature_dim)

        # 使用自适应池化来处理不同大小的特征图
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(num_features + self.color_feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )

        self.score_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.category_head = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.base_model(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)

        color_info = self.color_feature_extractor(x)
        color_features = self.color_feature_processor(color_info.view(x.size(0), -1))

        combined_features = torch.cat([features, color_features], dim=1)
        x = self.fc(combined_features)

        score = self.score_head(x).squeeze() * 100  # Scale to 0-100
        category = self.category_head(x)

        # 添加调试信息
        print(
            f"Features stats: min={features.min().item()}, max={features.max().item()}, mean={features.mean().item()}")
        print(
            f"Color features stats: min={color_features.min().item()}, max={color_features.max().item()}, mean={color_features.mean().item()}")
        print(
            f"Combined features stats: min={combined_features.min().item()}, max={combined_features.max().item()}, mean={combined_features.mean().item()}")
        print(f"FC output stats: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
        print(f"Score stats: min={score.min().item()}, max={score.max().item()}, mean={score.mean().item()}")
        print(
            f"Category logits stats: min={category.min().item()}, max={category.max().item()}, mean={category.mean().item()}")

        return score, category

def get_model(config):
    return FashionModel(config.base_model_name, num_classes=config.num_classes)