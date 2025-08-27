# DINOv3 Fine-tuning Training Script

A comprehensive training script for fine-tuning Facebook's DINOv3 (Vision Transformer) models on image classification tasks.

## What is DINOv3?

DINOv3 (self-DIstillation with NO labels version 3) is a self-supervised learning method developed by Facebook AI Research (Meta) for learning visual representations without requiring labeled data. It represents a significant advancement in computer vision by learning powerful visual features through self-supervision.

### Key Features of DINOv3:

- **Self-Supervised Learning**: Learns visual representations without requiring manually labeled data
- **Vision Transformer Architecture**: Built on the transformer architecture, originally designed for NLP but adapted for vision tasks
- **Strong Transfer Learning**: Pre-trained features transfer well to downstream tasks like image classification, object detection, and segmentation
- **Multiple Model Sizes**: Available in different sizes (small, base, large, huge) to balance performance and computational requirements
- **Robust Features**: Learns features that are robust to various transformations and work well across different domains

### Why DINOv3 was Developed:

1. **Reduce Dependency on Labeled Data**: Traditional supervised learning requires massive amounts of labeled data, which is expensive and time-consuming to collect
2. **Learn General Visual Representations**: Create models that understand visual concepts without being limited to specific labeled categories
3. **Improve Transfer Learning**: Develop features that transfer better to new tasks and domains
4. **Scale with Unlabeled Data**: Leverage the vast amounts of unlabeled images available on the internet

## Project Structure

```
dinov3-quantized-finetuning/
├── main.py              # Main training script
├── README.md           # This file
└── checkpoints_dinov3/ # Model checkpoints (created during training)
```

## Installation

### Requirements

```bash
pip install torch torchvision transformers datasets pillow numpy trackio

# For 4-bit quantization support (optional but recommended)
pip install bitsandbytes
```

### Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for pre-trained models
- **Datasets**: Hugging Face library for dataset loading
- **PIL (Pillow)**: Image processing
- **NumPy**: Numerical computing
- **trackio**: Experiment tracking
- **bitsandbytes**: 4-bit quantization support (optional)

## Quick Start

### Basic Usage

Run training with default parameters on Food101 dataset:

```bash
python main.py
```

### Custom Training

```bash
python main.py \
    --dataset "ethz/food101" \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --checkpoint_dir "./my_checkpoints"
```

### Train on Full Dataset

```bash
python main.py \
    --train_split_ratio 1.0 \
    --val_split_ratio 1.0 \
    --epochs 20 \
    --batch_size 32
```

### 4-bit Quantization (Memory Efficient)

```bash
# Basic 4-bit quantization
python main.py --use_4bit

# Advanced quantization settings
python main.py \
    --use_4bit \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --use_nested_quant \
    --batch_size 16
```

## Training Arguments

### Dataset Arguments

- `--dataset`: Dataset name from Hugging Face datasets (default: "ethz/food101")
- `--train_split_ratio`: Ratio of training data to use (default: 0.1 for quick training)
- `--val_split_ratio`: Ratio of validation data to use (default: 0.1 for quick training)

### Model Arguments

- `--model_name`: DINOv3 model name from Hugging Face (default: "facebook/dinov3-vith16plus-pretrain-lvd1689m")
- `--freeze_backbone`: Freeze backbone and only train classification head (default: True)
  - Use `--no-freeze-backbone` to unfreeze the entire model

### Quantization Arguments

- `--use_4bit`: Enable 4-bit quantization using BitsAndBytes
- `--bnb_4bit_compute_dtype`: Compute dtype for 4-bit quantization (choices: float16, bfloat16, float32, default: float16)
- `--bnb_4bit_quant_type`: Quantization type (choices: fp4, nf4, default: nf4)
- `--use_nested_quant`: Use nested quantization for additional memory savings

### Training Arguments

- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 5e-4)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--warmup_ratio`: Learning rate warmup ratio (default: 0.05)
- `--num_workers`: Number of data loading workers (default: auto-detected, min(8, cpu_count))

### Evaluation and Checkpointing

- `--eval_every_steps`: Evaluate every N steps (default: 100)
- `--checkpoint_dir`: Directory to save checkpoints (default: "./checkpoints_dinov3")

### Experiment Tracking

- `--project_name`: Project name for experiment tracking (default: "dinov3")
- `--no_tracking`: Disable experiment tracking

### Other

- `--seed`: Random seed for reproducibility (default: 42)

## Available DINOv3 Models

The script supports various DINOv3 model sizes:

| Model | Parameters | Description |
|-------|------------|-------------|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | ~21M | Small ViT-S/16 model |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | ~21M | Small ViT-S/16+ model (enhanced) |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | ~85M | Base ViT-B/16 model |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | ~300M | Large ViT-L/16 model |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` | ~1B | Huge ViT-H/16+ model (default) |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | ~7B | Giant ViT-7B/16 model |
| `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | ~28M | ConvNeXt Tiny model |
| `facebook/dinov3-convnext-small-pretrain-lvd1689m` | ~50M | ConvNeXt Small model |

## Training Process

### Transfer Learning Approach

By default, the script uses a transfer learning approach:

1. **Frozen Backbone**: The pre-trained DINOv3 backbone is frozen to preserve learned features
2. **Trainable Head**: Only the linear classification head is trained
3. **Efficient Training**: This approach is computationally efficient and often achieves good results

### Full Fine-tuning

To train the entire model (not recommended without significant computational resources):

```bash
python main.py --no-freeze-backbone
```

### 4-bit Quantization

This script supports 4-bit quantization using BitsAndBytes library, which can significantly reduce memory usage:

#### Benefits of 4-bit Quantization:
- **Memory Reduction**: ~75% less GPU memory usage compared to full precision
- **Faster Loading**: Models load faster due to smaller size
- **Maintained Performance**: Minimal performance degradation in most cases
- **Larger Models**: Allows training larger models on smaller GPUs

#### Quantization Options:
- **NF4 (Normal Float 4)**: Recommended for most use cases (default)
- **FP4 (Float Point 4)**: Alternative quantization scheme
- **Nested Quantization**: Additional memory savings by quantizing quantization constants
- **Compute Dtype**: Controls precision during computation (float16 default, bfloat16 recommended for newer GPUs)

#### Memory Comparison (Estimated):
| Model Size | Full Precision (fp16) | 4-bit Quantized | Memory Savings |
|------------|----------------------|-----------------|----------------|
| ViT-Huge (1B) | ~2GB | ~500MB | 75% |
| ViT-Large (300M) | ~600MB | ~150MB | 75% |
| ViT-Base (85M) | ~170MB | ~43MB | 75% |
| ViT-Small (21M) | ~42MB | ~11MB | 75% |

#### Example Usage:
```bash
# Basic quantization
python main.py --use_4bit

# Optimized for newer GPUs (RTX 30xx/40xx, A100, etc.)
python main.py --use_4bit --bnb_4bit_compute_dtype bfloat16

# Maximum memory savings
python main.py --use_4bit --use_nested_quant --bnb_4bit_compute_dtype bfloat16
```

### Training Features

- **Mixed Precision Training**: Automatic mixed precision for faster training and reduced memory usage (automatically disabled for quantized models)
- **4-bit Quantization**: Memory-efficient training with BitsAndBytes
- **8-bit AdamW Optimizer**: Optimized optimizer attempted for quantized models (falls back to regular AdamW if not available)
- **Cosine Learning Rate Schedule**: Learning rate scheduling with warmup for stable training
- **Automatic Checkpointing**: Best model is automatically saved based on validation accuracy
- **Progress Tracking**: Real-time training progress with validation metrics
- **Experiment Tracking**: Integration with trackio for experiment logging (can be disabled with `--no_tracking`)

## Output and Checkpoints

### Training Output

The script provides detailed training logs:

```
Using device: cuda
Loading dataset: ethz/food101
Training samples: 10100
Validation samples: 10100
Number of classes: 101
Loading model: facebook/dinov3-vith16plus-pretrain-lvd1689m
Using 4-bit quantization with BitsAndBytes
  - Compute dtype: torch.bfloat16
  - Quantization type: nf4
  - Nested quantization: True
Quantized model placed on device: cuda:0
Using 8-bit AdamW optimizer for quantized model
Mixed precision disabled for 4-bit quantized model
Total parameters: 1,139,646,821
Trainable parameters: 25,701
4-bit quantization: Enabled
Memory footprint: ~570.0MB (estimated)
Total training steps: 6,313

[epoch 1 | step 100] train_loss=2.1234 val_loss=1.8765 val_acc=45.67%
Saved new best model with accuracy: 45.67%
END EPOCH 1: val_loss=1.7234 val_acc=52.34% (best_acc=52.34%)
```

### Checkpoint Structure

Saved checkpoints contain:

```python
{
    "model_state_dict": "...",           # Model weights
    "optimizer_state_dict": "...",       # Optimizer state
    "scheduler_state_dict": "...",       # Learning rate scheduler state
    "config": {
        "model_name": "...",             # Model name used
        "num_classes": 101,              # Number of classes
        "freeze_backbone": True,         # Training configuration
        "use_4bit": True,                # Quantization settings
        "bnb_4bit_compute_dtype": "...", # Quantization dtype
        "bnb_4bit_quant_type": "...",    # Quantization type
        "use_nested_quant": False        # Nested quantization setting
    },
    "step": 1500,                        # Training step
    "epoch": 3,                          # Training epoch
    "best_acc": 0.7234                  # Best validation accuracy
}
```

## Example Datasets

### Food101 (Default)
```bash
python main.py --dataset "ethz/food101"
```

### CIFAR-10
```bash
python main.py --dataset "cifar10"
```

### ImageNet (subset for testing)
```bash
python main.py --dataset "imagenet-1k" --train_split_ratio 0.01 --val_split_ratio 0.01
```

### Custom Dataset
Make sure your dataset has 'image' and 'label' fields compatible with Hugging Face datasets format.

## Performance Tips

### Memory Optimization

- **Use 4-bit quantization**: `--use_4bit` for ~75% memory reduction
- **Enable nested quantization**: `--use_nested_quant` for additional savings
- Use smaller batch sizes if running out of GPU memory: `--batch_size 4`
- Use smaller model variants: `--model_name facebook/dinov3-vitb16-pretrain-lvd1689m`
- Reduce data splits for testing: `--train_split_ratio 0.1 --val_split_ratio 0.1`

### Training Speed

- **Use bfloat16 compute dtype**: `--bnb_4bit_compute_dtype bfloat16` for newer GPUs with Ampere+ architecture
- Increase `--num_workers` for faster data loading (default is auto-detected)
- Use larger batch sizes if memory allows (quantization enables this)
- Consider using multiple GPUs (requires code modification)

### Model Performance

- Try different learning rates: `--learning_rate 1e-4` to `--learning_rate 1e-3`
- Experiment with different warmup ratios: `--warmup_ratio 0.1`
- Consider unfreezing the backbone for better performance: `--no-freeze-backbone` (requires more compute)
- **NF4 quantization** typically performs better than FP4 for most tasks

### GPU-Specific Recommendations

#### For RTX 3090/4090, A100, H100 (Ampere+ architecture):
```bash
python main.py --use_4bit --bnb_4bit_compute_dtype bfloat16 --use_nested_quant
```

#### For older GPUs (RTX 20xx, GTX 16xx, V100):
```bash
python main.py --use_4bit --bnb_4bit_compute_dtype float16
```

#### For very limited memory (8GB or less):
```bash
python main.py \
    --use_4bit --use_nested_quant --batch_size 4 \
    --model_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --train_split_ratio 0.1 --val_split_ratio 0.1
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: 
   ```bash
   # Enable 4-bit quantization
   python main.py --use_4bit
   
   # Reduce batch size
   python main.py --batch_size 4
   
   # Use smaller model
   python main.py --model_name facebook/dinov3-vitb16-pretrain-lvd1689m
   ```
   
2. **BitsAndBytes Installation Issues**:
   ```bash
   # For CUDA 11.8
   pip install bitsandbytes>=0.41.0
   
   # For CUDA 12.x
   pip install bitsandbytes>=0.42.0
   
   # For Windows users
   pip install bitsandbytes-windows
   ```

3. **Quantization Not Working**: 
   - Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
   - Check BitsAndBytes version compatibility
   - Try without nested quantization first: remove `--use_nested_quant`
   - Check GPU architecture compatibility

4. **TrackIO Import Error**:
   ```bash
   pip install trackio
   # Or disable tracking
   python main.py --no_tracking
   ```

5. **Slow Data Loading**: 
   ```bash
   # Increase workers (but not more than CPU cores)
   python main.py --num_workers 8
   ```

6. **Poor Performance**: 
   - Try different learning rates: `--learning_rate 1e-4`
   - Unfreeze backbone: `--no-freeze-backbone`
   - Use more training data: `--train_split_ratio 1.0`

7. **Dataset Loading Issues**: 
   - Ensure internet connection for downloading datasets
   - Check dataset name spelling
   - Some datasets may require authentication

### System Requirements

#### Without Quantization:
- **GPU**: NVIDIA GPU with CUDA support (12GB+ VRAM for ViT-Huge)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 5GB+ free space for models and datasets

#### With 4-bit Quantization:
- **GPU**: NVIDIA GPU with CUDA support (4GB+ VRAM sufficient for ViT-Huge)
- **RAM**: 8GB+ system RAM recommended
- **Storage**: 5GB+ free space for models and datasets

#### General Requirements:
- **Internet**: Required for downloading models and datasets
- **CUDA**: 11.8+ or 12.x for optimal BitsAndBytes support
- **Python**: 3.8+ recommended

## Advanced Usage

### Custom Datasets

To use your own dataset, modify the `load_and_prepare_dataset` function or ensure your dataset follows the Hugging Face datasets format with 'image' and 'label' fields.

### Model Customization

The `DinoV3Linear` class can be extended to add more sophisticated classification heads. Current implementation:

```python
class DinoV3Linear(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.head = nn.Linear(hidden_size, num_classes)
```

### Experiment Tracking

The script uses trackio for experiment tracking. You can:
- Enable tracking (default): Logs training metrics automatically
- Disable tracking: `python main.py --no_tracking`
- Change project name: `python main.py --project_name my_experiment`

Tracked metrics include:
- Training loss
- Validation loss  
- Validation accuracy
- Learning rate
- Training configuration

## License

This project uses models and code from Facebook AI Research. Please refer to the original DINOv3 license and terms of use.

## References

- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [Hugging Face DINOv3 Models](https://huggingface.co/models?search=dinov3)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)

## Citation

If you use this code or DINOv3 models in your research, please cite:

```bibtex
@article{oquab2023dinov3,
  title={DINOv3: A SELF-SUPERVISED VISION TRANSFORMER},
  author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
