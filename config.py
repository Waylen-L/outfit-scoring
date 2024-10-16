from dataclasses import dataclass

@dataclass
class Config:
    base_model_name: str = 'resnet101'  # 可以是 'resnet50', 'resnet101', 'vgg16', 或 'efficientnet_b0'
    num_classes: int = None
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    num_epochs: int = 20
    patience: int = 10
    optimizer: str = 'adam'  # 添加了 optimizer 属性
    scheduler: str = 'cosine'

    def get_display_params(self):
        return {
            "Base Model": self.base_model_name,
            "Number of Classes": self.num_classes,
            "Batch Size": self.batch_size,
            "Learning Rate": self.learning_rate,
            "Weight Decay": self.weight_decay,
            "Number of Epochs": self.num_epochs,
            "Patience": self.patience,
            "Optimizer": self.optimizer,
            "Scheduler": self.scheduler,
        }