import torch
from torchvision import transforms
from PIL import Image
import argparse

from models import get_model
from config import Config


def load_model(model_path, config):
    model = get_model(config)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # 处理可能的维度不匹配
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # 特别处理 color_feature_processor
    if 'color_feature_processor.weight' in pretrained_dict and pretrained_dict[
        'color_feature_processor.weight'].shape != model_dict['color_feature_processor.weight'].shape:
        print("Adjusting color_feature_processor dimensions")
        pretrained_dict['color_feature_processor.weight'] = nn.Parameter(torch.randn(64, 20))
        pretrained_dict['color_feature_processor.bias'] = nn.Parameter(torch.zeros(64))

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def predict(model, image_tensor):
    with torch.no_grad():
        score, _ = model(image_tensor)
    return score.item() if score.numel() == 1 else score.mean().item()


def main():
    parser = argparse.ArgumentParser(description="Predict fashion score for an image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    args = parser.parse_args()

    config = Config()
    config.num_classes = 2  # 设置为您训练模型时使用的类别数
    config.base_model_name = 'resnet50'  # 确保这与训练时使用的基础模型相同

    try:
        model = load_model(args.model_path, config)
        print("Model loaded successfully")

        # 打印模型结构
        print(model)

        image_tensor = preprocess_image(args.image_path)

        # 添加更多调试信息
        print(f"Image tensor shape: {image_tensor.shape}")

        score = predict(model, image_tensor)

        print(f"Image: {args.image_path}")
        print(f"Predicted score: {score:.2f} (range: 0-100, higher is better)")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()