"""Constantes partagées (sans dépendance à PyTorch)."""

# Ordre alphabétique : c'est l'ordre imposé par torchvision.datasets.ImageFolder
# et celui utilisé par les labels du dataset FER2013 fourni.
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

IMAGE_SIZE = 48
