import argparse
import csv
import json
import random
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation：全局池化后生成各通道的权重。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.pool(x).flatten(1)
        weight = self.fc(weight).view(x.size(0), x.size(1), 1, 1)
        return x * weight


class BasicBlockWithOptionalSE(models.resnet.BasicBlock):
    """保持ResNet18基本残差块不变，只在残差相加前选择是否加入SE。"""

    def __init__(self, *args, use_se=True, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        channels = self.conv2.out_channels
        self.se = SEBlock(channels, reduction) if use_se else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


def build_model(use_se, reduction=16):
    block = partial(BasicBlockWithOptionalSE, use_se=use_se, reduction=reduction)
    block.expansion = models.resnet.BasicBlock.expansion
    model = models.ResNet(block, [2, 2, 2, 2], num_classes=10)
    # CIFAR-10为32×32小图，使用3×3卷积并去掉初始最大池化。
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def model_from_checkpoint(path, device):
    checkpoint = load_checkpoint(path)
    config = checkpoint["config"]
    model = build_model(config["use_se"], config["reduction"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, config


def cifar_transforms():
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, test


def get_datasets():
    train_transform, test_transform = cifar_transforms()
    train_aug = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
    train_plain = datasets.CIFAR10(DATA_DIR, train=True, download=False, transform=test_transform)
    test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)

    # 固定划分：45 000张训练、5 000张验证；验证集不使用随机增强。
    indices = torch.randperm(50_000, generator=torch.Generator().manual_seed(42)).tolist()
    train = Subset(train_aug, indices[5_000:])
    val = Subset(train_plain, indices[:5_000])
    return train, val, test


def make_loader(dataset, batch_size, workers, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def accuracy_topk(logits, targets, k):
    predictions = logits.topk(k, dim=1).indices
    return (predictions == targets.unsqueeze(1)).any(dim=1).sum().item()


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        use_amp = device.type == "cuda"

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch = targets.size(0)
        total_loss += loss.item() * batch
        total += batch
        correct1 += accuracy_topk(logits, targets, 1)
        correct5 += accuracy_topk(logits, targets, 5)

    return {
        "loss": total_loss / total,
        "top1": 100.0 * correct1 / total,
        "top5": 100.0 * correct5 / total,
    }


def save_history(path, history):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def read_history(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def train_model(args):
    set_seed(42)
    device = get_device()
    train_set, val_set, _ = get_datasets()
    train_loader = make_loader(train_set, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_set, args.batch_size, args.num_workers)

    use_se = args.model == "se"
    model = build_model(use_se).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_dir = RUNS_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    history = []
    start_epoch = 0
    best_val = float("inf")

    config = {
        "model": args.model,
        "use_se": use_se,
        "reduction": 16,
        "dataset": "CIFAR-10",
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": 10_000,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "SGD",
        "lr": args.lr,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "scheduler": "CosineAnnealingLR",
        "label_smoothing": 0.1,
        "seed": 42,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]
        if history_path.is_file():
            history = read_history(history_path)
        print(f"从第{start_epoch + 1}轮继续训练。")

    print(f"设备：{device}；模型：{args.model}；参数量：{count_parameters(model):,}")
    print(f"训练集：{len(train_set)}；验证集：{len(val_set)}")

    for epoch in range(start_epoch, args.epochs):
        begin = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        minutes = (time.perf_counter() - begin) / 60

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["top1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "minutes": minutes,
        }
        history.append(row)
        save_history(history_path, history)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "model": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": best_val,
                "val_acc": val_metrics["top1"],
            }, output_dir / "best.pth")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "epoch": epoch,
            "best_val": best_val,
        }, output_dir / "last.pth")

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['top1']:.2f}% | "
            f"val loss={val_metrics['loss']:.4f}, acc={val_metrics['top1']:.2f}% | "
            f"{minutes:.2f} min"
        )


def predict_all(model, loader, device):
    model.eval()
    truth = []
    predictions = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            truth.append(targets.numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(truth), np.concatenate(predictions)


def inference_time(model, device, repeats=100):
    sample = torch.randn(1, 3, 32, 32, device=device)
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        sync(device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(sample)
            sync(device)
            times.append((time.perf_counter() - start) * 1000)
    return {"mean": float(np.mean(times)), "std": float(np.std(times))}


def save_confusion_matrix(confusion, model_name):
    row_sum = confusion.sum(axis=1, keepdims=True)
    normalized = confusion / np.maximum(row_sum, 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} normalized confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{model_name}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def evaluate_models(args):
    device = get_device()
    _, _, test_set = get_datasets()
    loader = make_loader(test_set, args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss()
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {"device": str(device), "test_count": len(test_set)}

    for model_name in ("baseline", "se"):
        checkpoint_path = RUNS_DIR / model_name / "best.pth"
        checkpoint = load_checkpoint(checkpoint_path)
        model, config = model_from_checkpoint(checkpoint_path, device)
        metrics = run_epoch(model, loader, device, criterion)
        truth, predictions = predict_all(model, loader, device)
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (truth, predictions), 1)
        class_counts = confusion.sum(axis=1)
        per_class = np.divide(
            confusion.diagonal(), class_counts,
            out=np.zeros(10, dtype=float), where=class_counts > 0
        )
        timing = inference_time(model, device)

        summary[model_name] = {
            "checkpoint": str(checkpoint_path),
            "best_epoch": checkpoint["epoch"] + 1,
            "test_loss": metrics["loss"],
            "top1_accuracy": metrics["top1"],
            "top5_accuracy": metrics["top5"],
            "parameters": count_parameters(model),
            "inference_ms": timing,
            "per_class_accuracy": {name: float(acc * 100) for name, acc in zip(CLASSES, per_class)},
        }
        save_confusion_matrix(confusion, model_name)
        with (RESULTS_DIR / f"{model_name}_per_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "accuracy_percent"])
            writer.writerows((name, acc * 100) for name, acc in zip(CLASSES, per_class))
        print(model_name, json.dumps(summary[model_name], ensure_ascii=False, indent=2))

    summary["improvement"] = {
        "delta_top1_accuracy": summary["se"]["top1_accuracy"] - summary["baseline"]["top1_accuracy"],
        "delta_top5_accuracy": summary["se"]["top5_accuracy"] - summary["baseline"]["top5_accuracy"],
        "delta_parameters": summary["se"]["parameters"] - summary["baseline"]["parameters"],
        "delta_inference_ms": summary["se"]["inference_ms"]["mean"] - summary["baseline"]["inference_ms"]["mean"],
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"评价完成，结果保存到：{RESULTS_DIR}")


def plot_histories():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    histories = {
        "baseline": read_history(RUNS_DIR / "baseline" / "history.csv"),
        "se": read_history(RUNS_DIR / "se" / "history.csv"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for model_name, color in [("baseline", "#355C67"), ("se", "#A67536")]:
        rows = histories[model_name]
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], color=color, label=model_name)
        axes[1].plot(epochs, [float(row["val_loss"]) for row in rows], color=color, label=model_name)
        axes[2].plot(epochs, [float(row["val_acc"]) for row in rows], color=color, label=model_name)
    axes[0].set_title("训练损失")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("验证损失")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[2].set_title("验证集Top-1准确率")
    axes[2].set_ylabel("Accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=200, bbox_inches="tight")
    print(f"曲线已保存：{RESULTS_DIR / 'training_curves.png'}")
    plt.show()


def infer_scene_images(args):
    device = get_device()
    model, _ = model_from_checkpoint(args.checkpoint, device)
    input_path = Path(args.input)
    paths = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not paths:
        raise FileNotFoundError("输入路径中没有可识别的图片。")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    output_dir = RESULTS_DIR / "scene"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in paths:
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        values, indices = probabilities.topk(5)
        top5 = [(CLASSES[i], float(v) * 100) for v, i in zip(values.cpu(), indices.cpu())]
        print(f"\n{path.name}")
        for name, probability in top5:
            print(f"  {name:12s} {probability:6.2f}%")
        rows.append({"image": str(path), "top5": json.dumps(top5, ensure_ascii=False)})

    with (output_dir / "inference_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "top5"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n场景识别结果已保存到：{output_dir / 'inference_results.csv'}")


def check_data():
    train, val, test = get_datasets()
    images, targets = next(iter(make_loader(train, 2, 0)))
    print(f"训练：{len(train)}；验证：{len(val)}；测试：{len(test)}")
    print(f"批量形状：{tuple(images.shape)}；标签形状：{tuple(targets.shape)}")


def smoke_test():
    device = get_device()
    for use_se in (False, True):
        model = build_model(use_se).to(device)
        sample = torch.randn(2, 3, 32, 32, device=device)
        target = torch.tensor([0, 1], device=device)
        loss = nn.CrossEntropyLoss()(model(sample), target)
        loss.backward()
        print(f"use_se={use_se}，输出={tuple(model(sample).shape)}，参数量={count_parameters(model):,}，反向传播通过")


def main():
    parser = argparse.ArgumentParser(description="ResNet18与SE-ResNet18的CIFAR-10分类实验")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--model", choices=["baseline", "se"], required=True)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=0.1)
    train_parser.add_argument("--resume", type=str, default="")

    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--batch-size", type=int, default=256)
    eval_parser.add_argument("--num-workers", type=int, default=4)

    sub.add_parser("plot")
    sub.add_parser("check-data")
    sub.add_parser("smoke")

    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--input", required=True, help="一张图片或图片文件夹")
    infer_parser.add_argument("--checkpoint", default=str(RUNS_DIR / "se" / "best.pth"))

    args = parser.parse_args()
    if args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_models(args)
    elif args.command == "plot":
        plot_histories()
    elif args.command == "check-data":
        check_data()
    elif args.command == "smoke":
        smoke_test()
    elif args.command == "infer":
        infer_scene_images(args)


if __name__ == "__main__":
    main()
