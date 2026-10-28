import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from PIL import Image


def predict(image_path, model_path):
    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=True
    )
    if checkpoint["architecture"] != "original_cnn":
        raise ValueError("This script expects the exported original CNN.")

    classes = checkpoint["class_names"]

    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 16 * 16, 256),
        nn.ReLU(),
        nn.Dropout(checkpoint["dropout"]),
        nn.Linear(256, len(classes))
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    settings = checkpoint["preprocessing"]
    transform = transforms.Compose([
        transforms.Resize(tuple(settings["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(settings["mean"], settings["std"])
    ])

    with Image.open(image_path) as image:
        inputs = transform(
            image.convert(settings["color_mode"])
        ).unsqueeze(0)

    with torch.inference_mode():
        scores = model(inputs).softmax(dim=1)[0]

    predicted_index = scores.argmax().item()
    return {
        "predicted_class": classes[predicted_index],
        "softmax_scores": {
            name: float(scores[i]) for i, name in enumerate(classes)
        },
        "note": (
            "Scores are not calibrated clinical probabilities. "
            "Educational use only; not for diagnosis."
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent
        / "exported_model" / "brain_tumor_model.pth"
    )
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.model), indent=2))
