import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_loader import get_data_loaders
from models import get_model
from config import Config


class Paths:
    def __init__(self, data_root='data'):
        self.data_root = data_root
        self.train_csv = os.path.join(self.data_root, 'train', 'train.csv')
        self.val_csv = os.path.join(self.data_root, 'validation', 'validation.csv')
        self.test_csv = os.path.join(self.data_root, 'test', 'test.csv')
        self.train_img_dir = os.path.join(self.data_root, 'train', 'img')
        self.val_img_dir = os.path.join(self.data_root, 'validation', 'img')
        self.test_img_dir = os.path.join(self.data_root, 'test', 'img')
        self.best_model_path = 'best_model.pth'
        self.final_model_path = 'final_model.pth'
        self.training_history_plot = 'training_history.png'

    def validate_paths(self):
        for path in [self.train_csv, self.val_csv, self.test_csv, self.train_img_dir, self.val_img_dir, self.test_img_dir]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found: {path}")


def huber_loss(pred, target, delta=1.0):
    return F.smooth_l1_loss(pred, target, reduction='mean', beta=delta)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience, paths):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_scores = []
        train_categories = []
        train_true_scores = []
        train_true_categories = []

        for inputs, scores, categories in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, scores, categories = inputs.to(device), scores.to(device), categories.to(device)

            optimizer.zero_grad()

            pred_scores, pred_categories = model(inputs)

            reg_loss = criterion['regression'](pred_scores, scores)
            cls_loss = criterion['classification'](pred_categories, categories)
            loss = reg_loss + cls_loss

            if torch.isnan(loss):
                print(f"NaN loss detected. Skipping batch.")
                continue

            loss.backward()

            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            # 检查梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN gradient detected in {name}, replacing with zeros")
                        param.grad[torch.isnan(param.grad)] = 0

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_scores.extend(pred_scores.detach().cpu().numpy())
            train_categories.extend(pred_categories.argmax(dim=1).detach().cpu().numpy())
            train_true_scores.extend(scores.cpu().numpy())
            train_true_categories.extend(categories.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_metrics = calculate_metrics(train_true_scores, train_scores, train_true_categories, train_categories)

        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Val Metrics: {val_metrics}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Learning rate scheduler step
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), paths.best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping!")
                break

    plot_training_history(history, paths.training_history_plot)

    print("\nTraining Summary:")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Epoch: {best_epoch}")

    return model


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_categories = []
    all_true_scores = []
    all_true_categories = []

    with torch.no_grad():
        for inputs, scores, categories in data_loader:
            inputs, scores, categories = inputs.to(device), scores.to(device), categories.to(device)

            pred_scores, pred_categories = model(inputs)

            # 添加调试信息
            print(f"Eval - Pred scores: {pred_scores}")
            print(f"Eval - True scores: {scores}")
            print(f"Eval - Pred categories: {pred_categories}")
            print(f"Eval - True categories: {categories}")

            reg_loss = criterion['regression'](pred_scores, scores)
            cls_loss = criterion['classification'](pred_categories, categories)
            loss = reg_loss + cls_loss

            total_loss += loss.item() * inputs.size(0)
            all_scores.extend(pred_scores.cpu().numpy())
            all_categories.extend(pred_categories.argmax(dim=1).cpu().numpy())
            all_true_scores.extend(scores.cpu().numpy())
            all_true_categories.extend(categories.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)

    # 检查 NaN 值
    if np.isnan(all_scores).any():
        print("NaN detected in predicted scores")
    if np.isnan(all_true_scores).any():
        print("NaN detected in true scores")

    metrics = calculate_metrics(all_true_scores, all_scores, all_true_categories, all_categories)

    return avg_loss, metrics


def calculate_metrics(true_scores, pred_scores, true_categories, pred_categories):
    # 检查并移除 NaN 值
    valid_indices = ~np.isnan(pred_scores) & ~np.isnan(true_scores)
    true_scores = np.array(true_scores)[valid_indices]
    pred_scores = np.array(pred_scores)[valid_indices]
    true_categories = np.array(true_categories)[valid_indices]
    pred_categories = np.array(pred_categories)[valid_indices]

    if len(true_scores) == 0:
        print("Warning: All scores are NaN")
        return {
            'mse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'accuracy': float('nan')
        }

    return {
        'mse': mean_squared_error(true_scores, pred_scores),
        'mae': mean_absolute_error(true_scores, pred_scores),
        'r2': r2_score(true_scores, pred_scores),
        'accuracy': accuracy_score(true_categories, pred_categories)
    }


def plot_training_history(history, plot_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main():
    try:
        config = Config()
        paths = Paths()

        # 验证路径
        paths.validate_paths()

        # 设置 num_classes
        train_data = pd.read_csv(paths.train_csv)
        config.num_classes = len(train_data['subCategory'].unique())

        print("Configuration parameters:")
        for key, value in config.get_display_params().items():
            print(f"{key}: {value}")

        print("\nStarting training with the above configuration.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if device.type == 'cuda':
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        model = get_model(config).to(device)

        train_loader, val_loader, _ = get_data_loaders(paths, config)

        # 检查训练数据
        for batch in train_loader:
            images, scores, categories = batch
            print(f"Batch shapes: images {images.shape}, scores {scores.shape}, categories {categories.shape}")
            print(f"Unique categories in batch: {categories.unique()}")
            print(f"Score range: min={scores.min().item()}, max={scores.max().item()}")
            break

        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal number of parameters: {total_params}")

        criterion = {
            'regression': huber_loss,
            'classification': nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).to(device))  # 假设有两个类别，调整权重
        }

        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9,
                                  weight_decay=config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        if config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        elif config.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif config.scheduler == 'reduce_lr_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                             verbose=True)
        else:
            raise ValueError(f"Unsupported scheduler: {config.scheduler}")

        # 检查模型是否在 GPU 上
        print(f"Model is on GPU: {next(model.parameters()).is_cuda}")

        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                                    config.num_epochs, config.patience, paths)
        torch.save(trained_model.state_dict(), paths.final_model_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()