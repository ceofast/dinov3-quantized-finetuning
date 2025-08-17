#!/usr/bin/env python3
"""
DINOv3 Fine-tuning Training Script

This script fine-tunes Facebook's DINOv3 vision transformer model on image classification tasks.
DINOv3 is a self-supervised learning method for learning visual features without labels.
"""

import argparse
import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup, BitsAndBytesConfig
import trackio

try:
    import bitsandbytes as bnb

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. 4-bit quantization will be disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Fine-tuning Training Script")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="ethz/food101",
                        help="Dataset name from Hugging Face datasets")
    parser.add_argument("--train_split_ratio", type=float, default=0.1,
                        help="Ratio of training data to use (default: 0.1 for quick training)")
    parser.add_argument("--val_split_ratio", type=float, default=0.1,
                        help="Ratio of validation data to use (default: 0.1 for quick training)")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vith16plus-pretrain-lvd1689m",
                        help="DINOv3 model name from Hugging Face")
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze backbone and only train classification head (use --no-freeze-backbone to unfreeze)")

    # Quantization arguments
    parser.add_argument("--use_4bit", action="store_true",
                        help="Enable 4-bit quantization using BitsAndBytes")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Compute dtype for 4-bit quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                        choices=["fp4", "nf4"],
                        help="Quantization type for 4-bit quantization")
    parser.add_argument("--use_nested_quant", action="store_true",
                        help="Use nested quantization for even more memory savings")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of data loading workers (default: min(8, cpu_count))")

    # Evaluation and checkpointing
    parser.add_argument("--eval_every_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_dinov3",
                        help="Directory to save checkpoints")

    # Experiment tracking
    parser.add_argument("--project_name", type=str, default="dinov3",
                        help="Project name for experiment tracking")
    parser.add_argument("--no_tracking", action="store_true",
                        help="Disable experiment tracking")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_prepare_dataset(args):
    """Load and prepare the dataset"""
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)

    # Use smaller subsets for quick training
    train_ds = ds["train"].shuffle()
    if args.train_split_ratio < 1.0:
        train_ds = train_ds.train_test_split(test_size=1.0 - args.train_split_ratio)["train"]

    val_key = "validation" if "validation" in ds else ("test" if "test" in ds else None)
    if val_key is None:
        raise ValueError("Dataset must have a validation or test split.")
    val_ds = ds[val_key].shuffle()
    if args.val_split_ratio < 1.0:
        val_ds = val_ds.train_test_split(test_size=1.0 - args.val_split_ratio)["train"]

    num_classes = train_ds.features["label"].num_classes
    id2label = {i: name for i, name in enumerate(train_ds.features["label"].names)}
    label2id = {v: k for k, v in id2label.items()}

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Number of classes: {num_classes}")

    return train_ds, val_ds, num_classes, id2label, label2id


class DinoV3Linear(nn.Module):
    """DINOv3 model with linear classification head"""

    def __init__(self, backbone: AutoModel, hidden_size: int, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # Prefer pooled features if available; else use CLS token
        feats = getattr(outputs, "pooler_output", None)
        if feats is None:
            feats = outputs.last_hidden_state[:, 0]
        logits = self.head(feats)
        return logits


def create_quantization_config(args):
    """Create BitsAndBytes quantization configuration"""
    if not args.use_4bit or not BITSANDBYTES_AVAILABLE:
        return None

    # Map string to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype_mapping[args.bnb_4bit_compute_dtype],
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    return quantization_config


def create_model(model_name: str, num_classes: int, freeze_backbone: bool = True, quantization_config=None):
    """Create DINOv3 model with classification head"""
    print(f"Loading model: {model_name}")

    if quantization_config is not None:
        print("Using 4-bit quantization with BitsAndBytes")
        print(f"  - Compute dtype: {quantization_config.bnb_4bit_compute_dtype}")
        print(f"  - Quantization type: {quantization_config.bnb_4bit_quant_type}")
        print(f"  - Nested quantization: {quantization_config.bnb_4bit_use_double_quant}")

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    # Prefer bf16 on Ampere+; fallback to fp16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        prefer_dtype = torch.bfloat16
    else:
        prefer_dtype = torch.float16

    backbone = AutoModel.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if quantization_config is not None else None,
        torch_dtype=(prefer_dtype if quantization_config is not None else None)
    )

    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not determine hidden size for model {model_name}")

    model = DinoV3Linear(backbone, hidden_size, num_classes, freeze_backbone)
    return model, image_processor


@dataclass
class Collator:
    """Data collator for image classification"""
    processor: AutoImageProcessor

    def __call__(self, batch):
        raw_images = [x["image"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

        rgb_images = []
        for im in raw_images:
            if isinstance(im, Image.Image):
                rgb_images.append(im.convert("RGB"))
            else:
                # datasets may yield numpy arrays sometimes
                rgb_images.append(Image.fromarray(im).convert("RGB"))

        inputs = self.processor(images=rgb_images, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"], "labels": labels}


def create_dataloaders(train_ds, val_ds, image_processor, args):
    """Create training and validation data loaders"""
    collate_fn = Collator(image_processor)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def evaluate(model, val_loader, criterion, device) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def setup_training(model, train_loader, args):
    """Setup optimizer, scheduler, and other training components"""
    # Only train parameters that require gradients
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    # For quantized models, we might want to use different optimizers
    if args.use_4bit and BITSANDBYTES_AVAILABLE:
        try:
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            print("Using 8-bit AdamW optimizer for quantized model")
        except AttributeError:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
            print("Using regular AdamW optimizer (8-bit AdamW not available)")
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    # AMP new API
    use_amp = torch.cuda.is_available() and not args.use_4bit
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if args.use_4bit:
        print("Mixed precision disabled for 4-bit quantized model")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    return optimizer, scheduler, criterion, scaler


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, scaler, args, device):
    """Main training loop"""
    best_acc = 0.0
    global_step = 0

    # Initialize experiment tracking
    if not args.no_tracking:
        trackio.init(project=args.project_name, config={
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "model_name": args.model_name,
            "freeze_backbone": args.freeze_backbone,
            "use_4bit": args.use_4bit,
            "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype if args.use_4bit else None,
            "bnb_4bit_quant_type": args.bnb_4bit_quant_type if args.use_4bit else None,
            "use_nested_quant": args.use_nested_quant if args.use_4bit else None,
        })

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()  # Keep backbone in eval mode if frozen

        running_loss = 0.0
        for i, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                # AMP new API
                with torch.amp.autocast('cuda'):
                    logits = model(pixel_values)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.eval_every_steps == 0:
                avg_train_loss = running_loss / args.eval_every_steps
                metrics = evaluate(model, val_loader, criterion, device)
                print(
                    f"[epoch {epoch} | step {global_step}] "
                    f"train_loss={avg_train_loss:.4f} "
                    f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc'] * 100:.2f}%"
                )

                if not args.no_tracking:
                    # 'step' reserved; pass as argument instead of metric key
                    trackio.log(
                        {
                            "epoch": epoch,
                            "train_loss": avg_train_loss,
                            "val_loss": metrics['val_loss'],
                            "val_acc": metrics['val_acc'],
                        },
                        step=global_step,
                    )

                running_loss = 0.0

                if metrics["val_acc"] > best_acc:
                    best_acc = metrics["val_acc"]
                    ckpt_path = os.path.join(args.checkpoint_dir, f"best_acc_{best_acc:.4f}.pt")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "config": {
                                "model_name": args.model_name,
                                "num_classes": model.head.out_features,
                                "freeze_backbone": args.freeze_backbone,
                                "use_4bit": args.use_4bit,
                                "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype if args.use_4bit else None,
                                "bnb_4bit_quant_type": args.bnb_4bit_quant_type if args.use_4bit else None,
                                "use_nested_quant": args.use_nested_quant if args.use_4bit else None,
                            },
                            "step": global_step,
                            "epoch": epoch,
                            "best_acc": best_acc,
                        },
                        ckpt_path,
                    )
                    print(f"Saved new best model with accuracy: {best_acc * 100:.2f}%")

        # End of epoch evaluation
        metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"END EPOCH {epoch}: val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc'] * 100:.2f}% "
            f"(best_acc={best_acc * 100:.2f}%)"
        )

    if not args.no_tracking:
        trackio.finish()

    print(f"Training completed! Best validation accuracy: {best_acc * 100:.2f}%")
    return best_acc


def main():
    """Main function"""
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare dataset
    train_ds, val_ds, num_classes, id2label, label2id = load_and_prepare_dataset(args)

    # Create quantization config
    quantization_config = create_quantization_config(args)

    # Create model
    model, image_processor = create_model(args.model_name, num_classes, args.freeze_backbone, quantization_config)

    # Move model to device (only if not using quantization, as quantized models handle device placement automatically)
    if quantization_config is None:
        model = model.to(device)
    else:
        # For quantized models, get where it was placed
        device = next(model.parameters()).device
        print(f"Quantized model placed on device: {device}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, image_processor, args)

    # Setup training components
    optimizer, scheduler, criterion, scaler = setup_training(model, train_loader, args)

    # Print training info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if args.use_4bit:
        print(f"4-bit quantization: Enabled")
        print(f"Memory footprint: ~{total_params * 0.5 / 1e6:.1f}MB (estimated)")
    else:
        print(f"4-bit quantization: Disabled")
        print(f"Memory footprint: ~{total_params * 2 / 1e6:.1f}MB (estimated, fp16)")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Total training steps: {args.epochs * len(train_loader)}")

    # Start training
    best_acc = train_model(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion, scaler,
        args, device
    )

    return best_acc


if __name__ == "__main__":
    main()
