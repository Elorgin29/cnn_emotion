"""
Inférence sur une image (ou un dossier d'images) avec un modèle entraîné.

Usage :
    python -m src.predict --model results/models/v4_regularization_best.pth \
                          --image chemin/vers/image.png
    python -m src.predict --model results/models/v4_regularization_best.pth \
                          --image chemin/vers/dossier
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.constants import CLASSES, IMAGE_SIZE, NUM_CLASSES
from src.model import EmotionCNN

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff"}

TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def load_model(path: str, device: torch.device) -> EmotionCNN:
    model = EmotionCNN(num_classes=NUM_CLASSES)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_image(model, image_path: str | Path, device: torch.device):
    img = Image.open(image_path).convert("L")
    x = TF(img).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
    return CLASSES[int(probs.argmax())], probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, help="Fichier image ou dossier")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Nombre de classes à afficher")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    target = Path(args.image)
    if target.is_dir():
        images = sorted(p for p in target.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
    else:
        images = [target]

    if not images:
        raise SystemExit(f"Aucune image trouvée dans {target}")

    for image_path in images:
        label, probs = predict_image(model, image_path, device)
        order = probs.argsort()[::-1][: args.top_k]
        print(f"\nImage      : {image_path}")
        print(f"Prédiction : {label}")
        for idx in order:
            print(f"  {CLASSES[idx]:>10s} : {probs[idx]:.4f}")


if __name__ == "__main__":
    main()
