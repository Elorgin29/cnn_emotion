"""
Vérifie que l'environnement est correctement installé et que les données
sont présentes.

Usage :
    python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("PIL", "Pillow"),
    ("pyarrow", "pyarrow"),
]


def check_packages() -> bool:
    ok = True
    for module_name, package_name in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "?")
            print(f"  [OK]   {package_name:<16s} {version}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  [FAIL] {package_name:<16s} {exc}")
    return ok


def check_torch() -> bool:
    try:
        import torch
        from src.model import EmotionCNN, count_parameters

        model = EmotionCNN()
        out = model(torch.randn(2, 1, 48, 48))
        assert out.shape == (2, 7)
        print(f"\n  PyTorch fonctionne ({torch.get_num_threads()} threads, "
              f"{count_parameters(model):,} paramètres).")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  [FAIL] PyTorch : {exc}")
        print("\n  Sous Windows, cette erreur vient presque toujours du runtime")
        print("  Visual C++ trop ancien. Installez :")
        print("  https://aka.ms/vs/17/release/vc_redist.x64.exe")
        return False


def check_data() -> bool:
    from src.constants import CLASSES

    base = ROOT / "data" / "raw"
    ok = True
    for split in ("train", "test"):
        split_dir = base / split
        if not split_dir.exists():
            print(f"  [FAIL] {split_dir} absent")
            ok = False
            continue
        total = 0
        for name in CLASSES:
            count = len(list((split_dir / name).glob("*.png"))) if (split_dir / name).exists() else 0
            total += count
            if count == 0:
                print(f"  [FAIL] {split}/{name} vide")
                ok = False
        print(f"  [OK]   {split}: {total} images")
    return ok


def main() -> int:
    print("=== Paquets ===")
    packages_ok = check_packages()
    print("\n=== PyTorch ===")
    torch_ok = check_torch()
    print("\n=== Données ===")
    data_ok = check_data()
    print()
    if packages_ok and torch_ok and data_ok:
        print("Environnement prêt.")
        return 0
    print("Environnement incomplet (voir erreurs ci-dessus).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
