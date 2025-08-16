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
smol-vision/
├── main.py              # Main training script
├── README.md           # This file
└── checkpoints_dinov3/ # Model checkpoints (created during training)
```

## Installation

### Requirements

```bash
pip install torch torchvision transformers datasets pillow numpy trackio

# For 4-bit quantization support (optional but recommended)
pip install bitsandbytes accelerate
```

### Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for pre-trained models
- **Datasets**: Hugging Face library for dataset loading
- **PIL (Pillow)**: Image processing
- **NumPy**: Numerical computing
- **trackio**: Experiment tracking (optional)
- **bitsandbytes**: 4-bit quantization support (optional)
- **accelerate**: Accelerated training and inference (optional)

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

- `--model_name`: DINOv3 model name from Hugging Face (default: "facebook/dinov3-vith16plus-pretrain-lvd1689")
- `--freeze_backbone`: Freeze backbone and only train classification head (default: True)

### Quantization Arguments

- `--use_4bit`: Enable 4-bit quantization using BitsAndBytes
- `--bnb_4bit_compute_dtype`: Compute dtype for 4-bit quantization (choices: float16, bfloat16, float32)
- `--bnb_4bit_quant_type`: Quantization type (choices: fp4, nf4)
- `--use_nested_quant`: Use nested quantization for additional memory savings

### Training Arguments

- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 5e-4)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--warmup_ratio`: Learning rate warmup ratio (default: 0.05)
- `--num_workers`: Number of data loading workers (default: auto-detected)

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
| `facebook/dinov3-small-384` | 21M | Small model, 384px input |
| `facebook/dinov3-base-384` | 85M | Base model, 384px input |
| `facebook/dinov3-large-384` | 300M | Large model, 384px input |
| `facebook/dinov3-huge-384` | 1B | Huge model, 384px input |
| `facebook/dinov3-vith16plus-pretrain-lvd1689` | 1B+ | ViT-Huge with additional pretraining |

## Training Process

### Transfer Learning Approach

By default, the script uses a transfer learning approach:

1. **Frozen Backbone**: The pre-trained DINOv3 backbone is frozen to preserve learned features
2. **Trainable Head**: Only the linear classification head is trained
3. **Efficient Training**: This approach is computationally efficient and often achieves good results

### Full Fine-tuning

To train the entire model (not recommended without significant computational resources):

```bash
python main.py --freeze_backbone False
```

### 4-bit Quantization

This script supports 4-bit quantization using BitsAndBytes library, which can significantly reduce memory usage:

#### Benefits of 4-bit Quantization:
- **Memory Reduction**: ~75% less GPU memory usage compared to full precision
- **Faster Loading**: Models load faster due to smaller size
- **Maintained Performance**: Minimal performance degradation in most cases
- **Larger Models**: Allows training larger models on smaller GPUs

#### Quantization Options:
- **NF4 (Normal Float 4)**: Recommended for most use cases
- **FP4 (Float Point 4)**: Alternative quantization scheme
- **Nested Quantization**: Additional memory savings by quantizing quantization constants
- **Compute Dtype**: Controls precision during computation (bfloat16 recommended for newer GPUs)

#### Memory Comparison:
| Model Size | Full Precision | 4-bit Quantized | Memory Savings |
|------------|----------------|-----------------|----------------|
| ViT-Huge (1B) | ~4GB | ~1GB | 75% |
| ViT-Large (300M) | ~1.2GB | ~300MB | 75% |
| ViT-Base (85M) | ~340MB | ~85MB | 75% |

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

- **Mixed Precision Training**: Automatic mixed precision for faster training and reduced memory usage (disabled for quantized models)
- **4-bit Quantization**: Memory-efficient training with BitsAndBytes
- **8-bit AdamW Optimizer**: Optimized optimizer for quantized models
- **Cosine Learning Rate Schedule**: Learning rate scheduling with warmup for stable training
- **Automatic Checkpointing**: Best model is automatically saved based on validation accuracy
- **Progress Tracking**: Real-time training progress with validation metrics

## Output and Checkpoints

### Training Output

The script provides detailed training logs:

```
Using device: cuda
Loading dataset: ethz/food101
Loading model: facebook/dinov3-vith16plus-pretrain-lvd1689
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
Training samples: 10,100
Validation samples: 10,100
Total training steps: 6,313

[epoch 1 | step 100] train_loss=2.1234 val_loss=1.8765 val_acc=45.67%
Saved new best model with accuracy: 45.67%
```

### Checkpoint Structure

Saved checkpoints contain:

```python
{
    "model_state_dict": "...",      # Model weights
    "optimizer_state_dict": "...",   # Optimizer state
    "scheduler_state_dict": "...",   # Learning rate scheduler state
    "config": {
        "model_name": "...",         # Model name used
        "num_classes": 101,          # Number of classes
        "freeze_backbone": True      # Training configuration
    },
    "step": 1500,                   # Training step
    "epoch": 3,                     # Training epoch
    "best_acc": 0.7234             # Best validation accuracy
}
```

## Example Datasets

### Food101
```bash
python main.py --dataset "ethz/food101"
```

### CIFAR-10
```bash
python main.py --dataset "cifar10"
```

### ImageNet (subset)
```bash
python main.py --dataset "imagenet-1k" --train_split_ratio 0.01
```

## Performance Tips

### Memory Optimization

- **Use 4-bit quantization**: `--use_4bit` for ~75% memory reduction
- **Enable nested quantization**: `--use_nested_quant` for additional savings
- Use smaller batch sizes if running out of GPU memory
- Use gradient checkpointing for larger models (requires code modification)
- Use smaller model variants for resource-constrained environments

### Training Speed

- **Use bfloat16 compute dtype**: `--bnb_4bit_compute_dtype bfloat16` for newer GPUs
- Increase `num_workers` for faster data loading
- Use larger batch sizes if memory allows (quantization enables this)
- Consider using multiple GPUs (requires code modification)

### Model Performance

- Try different learning rates (1e-4 to 1e-3)
- Experiment with different warmup ratios
- Consider unfreezing the backbone for better performance (requires more compute)
- **NF4 quantization** typically performs better than FP4 for most tasks

### GPU-Specific Recommendations

#### For RTX 3090/4090, A100, H100:
```bash
python main.py --use_4bit --bnb_4bit_compute_dtype bfloat16 --use_nested_quant
```

#### For older GPUs (RTX 20xx, GTX 16xx):
```bash
python main.py --use_4bit --bnb_4bit_compute_dtype float16
```

#### For very limited memory (8GB or less):
```bash
python main.py --use_4bit --use_nested_quant --batch_size 4 --model_name facebook/dinov3-base-384
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: 
   - Enable 4-bit quantization: `--use_4bit`
   - Reduce batch size: `--batch_size 4`
   - Use smaller model: `--model_name facebook/dinov3-base-384`
   
2. **BitsAndBytes Installation Issues**:
   ```bash
   # For CUDA 11.8
   pip install bitsandbytes>=0.41.0
   
   # For CUDA 12.x
   pip install bitsandbytes>=0.42.0
   ```

3. **Quantization Not Working**: 
   - Ensure CUDA is available: `torch.cuda.is_available()`
   - Check BitsAndBytes version compatibility
   - Try without nested quantization first

4. **Slow Data Loading**: Increase `num_workers` or reduce image preprocessing

5. **Poor Performance**: Try different learning rates or unfreeze the backbone

6. **Dataset Loading Issues**: Ensure internet connection and check dataset name

### System Requirements

#### Without Quantization:
- **GPU**: NVIDIA GPU with CUDA support (16GB+ VRAM for ViT-Huge)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 5GB+ free space for models and datasets

#### With 4-bit Quantization:
- **GPU**: NVIDIA GPU with CUDA support (4GB+ VRAM sufficient for ViT-Huge)
- **RAM**: 8GB+ system RAM recommended
- **Storage**: 5GB+ free space for models and datasets

#### General Requirements:
- **Internet**: Required for downloading models and datasets
- **CUDA**: 11.8+ or 12.x for optimal BitsAndBytes support

## Advanced Usage

### Custom Datasets

To use your own dataset, modify the `load_and_prepare_dataset` function to load your data in the expected format with 'image' and 'label' fields.

### Model Customization

The `DinoV3Linear` class can be extended to add more sophisticated classification heads:

```python
class DinoV3MLP(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
```

## License

This project uses models and code from Facebook AI Research. Please refer to the original DINOv3 license and terms of use.

## References

- [DINOv3 Paper](https://arxiv.org/abs/2304.07193)
- [Hugging Face DINOv3 Models](https://huggingface.co/models?search=dinov3)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

## Citation

If you use this code or DINOv3 models in your research, please cite:

```bibtex
@article{dinov3,
  title={DINOv3: A SELF-SUPERVISED VISION TRANSFORMER},
  author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```
