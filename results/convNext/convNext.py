import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataloader import get_dataloaders
from utils import DEVICE, train_loop, sanity_check

# config
EXP_NAME     = 'convnext'
EPOCHS       = 15
BATCH_SIZE   = 32
LR_BACKBONE  = 1e-4 
LR_HEAD      = 1e-3    

DATA_DIR     = '/content/pokemon_data/PokemonData'
SAVE_ROOT    = '/content/drive/MyDrive/pokemon_classification/results'


def build_model(num_classes: int) -> nn.Module:
    model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    )
    # 전체 파라미터 학습 (Full Fine-tuning)
    # Head만 교체 (기존 1000 → num_classes)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def get_layerwise_optimizer(model, lr_backbone, lr_head):
    backbone_params = list(model.features.parameters())
    head_params     = list(model.classifier.parameters())

    return optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': 1e-2},
        {'params': head_params,     'lr': lr_head,     'weight_decay': 1e-2},
    ])


def main():
    print(f"\n{'='*60}")
    print(f"🚀 Exp 5: ConvNeXt-Tiny Full Fine-tuning")
    print(f"   Batch={BATCH_SIZE} | LR_backbone={LR_BACKBONE} LR_head={LR_HEAD} | Epochs={EPOCHS}")
    print(f"   Device: {DEVICE}")
    print(f"{'='*60}")

    # 강한 augmentation (Full Fine-tuning)
    train_loader, val_loader, test_loader, class_names, num_classes = \
        get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, strong_aug=True)

    model     = build_model(num_classes)
    optimizer = get_layerwise_optimizer(model, LR_BACKBONE, LR_HEAD)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # label_smoothing

    sanity_check(model, train_loader, criterion)

    # LayerNorm
    return train_loop(
        model, train_loader, val_loader, test_loader, class_names,
        optimizer, criterion, EPOCHS, EXP_NAME, SAVE_ROOT,
        freeze_bn=False
    )


if __name__ == '__main__':
    main()