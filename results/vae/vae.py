import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoencoderKL

from dataloader import get_dataloaders
from utils import DEVICE, train_loop, sanity_check

# config
EXP_NAME    = 'vae'
EPOCHS      = 15
BATCH_SIZE  = 16   
LR          = 1e-3

DATA_DIR    = '/content/pokemon_data/PokemonData'
SAVE_ROOT   = '/content/drive/MyDrive/pokemon_classification/results'


class VAEClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.encoder    = vae.encoder
        self.quant_conv = vae.quant_conv

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.quant_conv.parameters():
            p.requires_grad = False

        # Classification Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.BatchNorm1d(8 * 7 * 7),
            nn.Linear(8 * 7 * 7, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # ImageNet 정규화
        self.register_buffer('imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x * self.imagenet_std + self.imagenet_mean
        x = x * 2.0 - 1.0

        with torch.no_grad():
            h = self.encoder(x)
            z = self.quant_conv(h)

        return self.head(z)


def build_model(num_classes):
    return VAEClassifier(num_classes).to(DEVICE)


def main():
    print(f"\n{'='*60}")
    print(f"🚀 Exp 4: VAE Encoder + Classification Head")
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

    return train_loop(
        model, train_loader, val_loader, test_loader, class_names,
        optimizer, criterion, EPOCHS, EXP_NAME, SAVE_ROOT,
        freeze_bn=False 
    )


if __name__ == '__main__':
    main()
