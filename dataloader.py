# dataloader.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# ImageNet 표준 정규화
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_dataloaders(data_dir, batch_size=32, seed=42, strong_aug=False, num_workers=2):
    """
    포켓몬 데이터셋용 DataLoader 생성.

    Args:
        data_dir   : ImageFolder 구조의 경로
        batch_size : 배치 크기
        seed       : 재현성
        strong_aug : True  → Full Fine-tuning 등 무거운 학습용 (강한 augmentation)
                     False → Linear Probing 등 가벼운 학습용 (약한 augmentation)
        num_workers: DataLoader worker 수

    Returns:
        train_loader, val_loader, test_loader, class_names, num_classes
    """

    # ── Augmentation 정의 ─────────────────────────────────────────
    # [수정] 기존 ColorJitter/RandomRotation은 너무 강해서 학습을 방해함
    if strong_aug:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # 가장 안전한 augmentation (Linear Probing용)
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # ── 두 개의 독립 dataset 객체 ────────────────────────────────
    # transform 누출 방지 (얕은 복사 버그 회피)
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    eval_dataset  = datasets.ImageFolder(root=data_dir, transform=eval_transform)

    # RGBA 이미지 자동 변환을 위한 loader 패치
    def _safe_loader(path):
        from PIL import Image
        return Image.open(path).convert('RGB')
    train_dataset.loader = _safe_loader
    eval_dataset.loader  = _safe_loader

    class_names = train_dataset.classes
    num_classes = len(class_names)
    total_size  = len(train_dataset)

    # ── 인덱스 분할 (재현성 보장) ────────────────────────────────
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(total_size).tolist()

    n_train = int(0.70 * total_size)
    n_val   = int(0.15 * total_size)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    train_ds = Subset(train_dataset, train_idx)
    val_ds   = Subset(eval_dataset,  val_idx)
    test_ds  = Subset(eval_dataset,  test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[DataLoader] 클래스 수: {num_classes} | augmentation: {'strong' if strong_aug else 'weak'}")
    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names, num_classes
