import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataloader import get_dataloaders
from utils import DEVICE, train_loop, sanity_check

# config
EXP_NAME    = 'vit'
EPOCHS      = 15
BATCH_SIZE  = 32
LR          = 1e-3

DATA_DIR    = '/content/pokemon_data/PokemonData'
SAVE_ROOT   = '/content/drive/MyDrive/pokemon_classification/results'


def build_model(num_classes):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_f = model.heads.head.in_features
    model.heads.head = nn.Linear(in_f, num_classes)
    return model.to(DEVICE)


def main():
    print(f"\n{'='*60}")
    print(f"🚀 Exp 3: ViT-B/16 Linear Probing")
    print(f"   Batch={BATCH_SIZE} | LR={LR} | Epochs={EPOCHS}")
    print(f"   Device: {DEVICE}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, class_names, num_classes = \
        get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, strong_aug=False)

    model     = build_model(num_classes)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR)
    criterion = nn.CrossEntropyLoss()

    sanity_check(model, train_loader, criterion)

    # ViT는 BN이 없으므로 freeze_bn=False
    return train_loop(
        model, train_loader, val_loader, test_loader, class_names,
        optimizer, criterion, EPOCHS, EXP_NAME, SAVE_ROOT,
        freeze_bn=False
    )


if __name__ == '__main__':
    main()
