import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataloader import get_dataloaders
from utils import DEVICE, train_loop, sanity_check

# config
EXP_NAME    = 'resnet50_ft'
EPOCHS      = 15
BATCH_SIZE  = 32
LR_BACKBONE = 1e-4   
LR_HEAD     = 1e-3    

DATA_DIR    = '/content/pokemon_data/PokemonData'
SAVE_ROOT   = '/content/drive/MyDrive/pokemon_classification/results'


def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # 동결하지 않음 → 전체 학습
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def get_layerwise_optimizer(model, lr_backbone, lr_head):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        (head_params if 'fc' in name else backbone_params).append(p)
    return optim.Adam([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params,     'lr': lr_head},
    ])


def main():
    print(f"\n{'='*60}")
    print(f"🚀 Exp 2: ResNet50 Full Fine-tuning")
    print(f"   Batch={BATCH_SIZE} | LR_backbone={LR_BACKBONE} LR_head={LR_HEAD} | Epochs={EPOCHS}")
    print(f"   Device: {DEVICE}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, class_names, num_classes = \
        get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, strong_aug=True)

    model     = build_model(num_classes)
    optimizer = get_layerwise_optimizer(model, LR_BACKBONE, LR_HEAD)
    criterion = nn.CrossEntropyLoss()

    sanity_check(model, train_loader, criterion)

    # Full Fine-tune은 BN도 학습되어야 하므로 freeze_bn=False
    return train_loop(
        model, train_loader, val_loader, test_loader, class_names,
        optimizer, criterion, EPOCHS, EXP_NAME, SAVE_ROOT,
        freeze_bn=False
    )


if __name__ == '__main__':
    main()
