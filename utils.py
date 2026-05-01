# utils.py
# 모든 실험에서 공통으로 쓰는 함수들
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────
# BatchNorm 고정 헬퍼 (Linear Probing 시 매우 중요)
# ──────────────────────────────────────────────────────────────────
def freeze_batchnorm(model: nn.Module):
    """
    모델 내부의 모든 BatchNorm 레이어를 eval 모드로 강제 고정합니다.

    [왜 필요한가?]
    Linear Probing에서 backbone을 freeze해도 model.train()을 호출하면
    BN이 train 모드로 들어가서 running_mean/running_var를 갱신합니다.
    이건 frozen backbone의 사전학습된 표현(representation)을 망가뜨려서
    학습이 진전되지 않는 원인이 됩니다.

    이 함수는 model.train()을 호출한 직후에 BN만 다시 eval 모드로 되돌립니다.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ──────────────────────────────────────────────────────────────────
# 단일 Epoch 실행
# ──────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, phase: str,
              freeze_bn: bool = False):
    """
    phase: 'train' | 'val'
    freeze_bn: True 이면 BN을 eval 모드로 유지 (Linear Probing 권장)
    """
    if phase == 'train':
        model.train()
        if freeze_bn:
            freeze_batchnorm(model)
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss     += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total            += inputs.size(0)

    return running_loss / total, running_corrects / total


# ──────────────────────────────────────────────────────────────────
# Test Set 최종 평가
# ──────────────────────────────────────────────────────────────────
def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'accuracy':  (all_preds == all_labels).mean(),
        'precision': precision_score(all_labels, all_preds,
                                     average='macro', zero_division=0),
        'recall':    recall_score(all_labels, all_preds,
                                  average='macro', zero_division=0),
    }


# ──────────────────────────────────────────────────────────────────
# Learning Curve 저장
# ──────────────────────────────────────────────────────────────────
def save_learning_curve(exp_name, train_losses, val_losses,
                        train_accs, val_accs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{exp_name} Learning Curve', fontsize=14, fontweight='bold')

    axes[0].plot(train_losses, label='Train', marker='o', markersize=4)
    axes[0].plot(val_losses,   label='Val',   marker='s', markersize=4)
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot([a*100 for a in train_accs], label='Train', marker='o', markersize=4)
    axes[1].plot([a*100 for a in val_accs],   label='Val',   marker='s', markersize=4)
    axes[1].set_title('Accuracy (%)'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────────────────────────────
# 통합 학습 루프 (모든 실험 파일에서 호출)
# ──────────────────────────────────────────────────────────────────
def train_loop(model, train_loader, val_loader, test_loader, class_names,
               optimizer, criterion, epochs, exp_name, save_root,
               freeze_bn: bool = False):
    """
    학습 → 검증 → 테스트 → 저장까지 한 번에 처리합니다.

    Returns:
        metrics: {'accuracy': ..., 'precision': ..., 'recall': ...}
    """
    save_dir = os.path.join(save_root, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'best_{exp_name}.pth')
    curve_path = os.path.join(save_dir, f'{exp_name}_curve.png')

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer,
                                  'train', freeze_bn=freeze_bn)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, optimizer,
                                  'val', freeze_bn=False)

        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);    val_accs.append(v_acc)

        is_best = v_acc > best_acc
        if is_best:
            best_acc = v_acc
            best_wts = copy.deepcopy(model.state_dict())

        print(f"  Epoch [{epoch+1:02d}/{epochs}] "
              f"Train: loss={t_loss:.4f} acc={t_acc*100:.2f}% | "
              f"Val: loss={v_loss:.4f} acc={v_acc*100:.2f}%"
              + (" ← Best" if is_best else ""))

    # 최고 모델 복원
    model.load_state_dict(best_wts)

    # Test 평가
    metrics = evaluate_on_test(model, test_loader)

    print(f"\n  ✨ Test 결과:")
    print(f"     Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"     Precision: {metrics['precision']:.4f} (macro)")
    print(f"     Recall   : {metrics['recall']:.4f} (macro)")

    # 저장
    save_learning_curve(exp_name, train_losses, val_losses,
                        train_accs, val_accs, curve_path)
    torch.save({
        'model_name':  exp_name,
        'state_dict':  model.state_dict(),
        'class_names': class_names,
        'metrics':     metrics,
    }, model_path)

    print(f"  💾 모델 저장: {model_path}")
    print(f"  📈 곡선 저장: {curve_path}")
    return metrics


# ──────────────────────────────────────────────────────────────────
# Sanity Check (학습 전 데이터 확인용)
# ──────────────────────────────────────────────────────────────────
def sanity_check(model, train_loader, criterion):
    """
    첫 배치만 모델에 흘려서 정상 작동 여부를 확인합니다.
    학습 전에 호출하면 빠르게 디버깅 가능합니다.
    """
    print("\n[Sanity Check]")
    model.train()
    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

    print(f"  Input shape: {inputs.shape}")
    print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"  Label sample: {labels[:5].cpu().numpy()}")

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    print(f"  Output shape: {outputs.shape}")
    print(f"  Initial loss: {loss.item():.4f}")
    print(f"  Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Total params:     "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print()
