# FLOPs & Complexity Calculator for PyTorch Deep Learning Model

A lightweight Python utility for estimating the computational complexity of PyTorch models. It hooks into a model's forward pass to count floating point operations (FLOPs), number of activations, memory usage, frames per second (FPS), and trainable parameters.


## Package Overview

* **Name:** `flopsmeter`

* **Language:** Python 3.10+

* **Dependencies:**

  * `torch 2.2.1+` (PyTorch)

This package helps deep learning practitioners quickly gauge the computational cost of their PyTorch models, aiding in model optimization, benchmarking, and resource planning.


## Features

* **FLOPs Estimation** — Supports convolution, normalization, pooling, activation, and more.

* **Activation Count** — Measures total activations produced in a forward pass.

* **Memory Usage** — Estimates memory footprint (in MB) during training.

* **FPS (Frames per Second)** — Benchmarks inference speed.

* **Trainable Parameters** — Calculates total learnable weights.

* **Module Exclusion Alerts** — Warns if unsupported layers are skipped.


## Supported Layers

The following PyTorch layers are currently supported by `flopsmeter`:

### Convolution
- `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`
- `nn.ConvTranspose1d`, `nn.ConvTranspose2d`, `nn.ConvTranspose3d`
- `nn.LazyConv1d`, `nn.LazyConv2d`, `nn.LazyConv3d`
- `nn.LazyConvTranspose1d`, `nn.LazyConvTranspose2d`, `nn.LazyConvTranspose3d`

### Normalization
- `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d`
- `nn.LazyBatchNorm1d`, `nn.LazyBatchNorm2d`, `nn.LazyBatchNorm3d`
- `nn.SyncBatchNorm`
- `nn.InstanceNorm1d`, `nn.InstanceNorm2d`, `nn.InstanceNorm3d`
- `nn.LazyInstanceNorm1d`, `nn.LazyInstanceNorm2d`, `nn.LazyInstanceNorm3d`
- `nn.GroupNorm`, `nn.LayerNorm`, `nn.LocalResponseNorm`

### Activation (approximate FLOPs)
- `nn.ELU`, `nn.ReLU`, `nn.ReLU6`, `nn.LeakyReLU`, `nn.PReLU`, `nn.RReLU`, `nn.GELU`, `nn.SELU`
- `nn.Tanh`, `nn.Tanhshrink`, `nn.Hardtanh`, `nn.Sigmoid`, `nn.LogSigmoid`, `nn.SiLU`, `nn.Mish`, `nn.Hardswish`
- `nn.Softplus`, `nn.Softshrink`, `nn.Softsign`, `nn.Hardsigmoid`, `nn.Hardshrink`, `nn.Threshold`
- `nn.GLU`, `nn.Softmin`, `nn.Softmax`, `nn.Softmax2d`, `nn.LogSoftmax`, `nn.AdaptiveLogSoftmaxWithLoss`

### Pooling
- `nn.MaxPool1d`, `nn.MaxPool2d`, `nn.MaxPool3d`
- `nn.AvgPool1d`, `nn.AvgPool2d`, `nn.AvgPool3d`
- `nn.FractionalMaxPool2d`, `nn.FractionalMaxPool3d`
- `nn.AdaptiveMaxPool1d`, `nn.AdaptiveMaxPool2d`, `nn.AdaptiveMaxPool3d`
- `nn.AdaptiveAvgPool1d`, `nn.AdaptiveAvgPool2d`, `nn.AdaptiveAvgPool3d`
- `nn.LPPool1d`, `nn.LPPool2d`

### Fully Connected
- `nn.Linear`, `nn.LazyLinear`, `nn.Bilinear`

### Dropout
- `nn.Dropout`, `nn.Dropout1d`, `nn.Dropout2d`, `nn.Dropout3d`
- `nn.AlphaDropout`, `nn.FeatureAlphaDropout`

### Upsampling
- `nn.Upsample` with `mode`: `nearest`, `linear`, `bilinear`, `bicubic`, `trilinear`
- `nn.UpsamplingNearest2d`, `nn.UpsamplingBilinear2d`

### Padding and Others
- `nn.Identity`, `nn.Flatten`, `nn.PixelShuffle`, `nn.PixelUnshuffle`
- `nn.ChannelShuffle`, `nn.ZeroPad*`, `nn.ConstantPad*`, `nn.ReflectionPad*`, `nn.ReplicationPad*`, `nn.CircularPad*`

*More layers may be supported in the future.*

Note: Unsupported layers will be ignored during FLOPs calculation.


## Installation

Install via pip:

```bash
pip install flopsmeter
```

*(Alternatively, copy the **`Complexity_Calculator`** class file into your project.)*


## Quick Start

```python
import torch
import torch.nn as nn

from flopsmeter import Complexity_Calculator

# Example: A Simple CNN Model
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size = 3)
        self.bn   = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

# Initialize calculator with dummy input shape (C, H, W)
calculator = Complexity_Calculator(model = SimpleCNN(), dummy = (3, 224, 224), device = torch.device('cuda'))

# Print Complexity Report
calculator.log(order = 'G', num_input = 1, batch_size = 16)
```


## API Reference

### `Complexity_Calculator(model, dummy, device = None)`

* **model** (`torch.nn.Module`): Your PyTorch model.

* **dummy** (`tuple[int]`): Input tensor shape for a *single sample*. For 2D input: `(C, H, W)`; for 3D: `(D, C, H, W)`; for 1D: `(L, D)`.

* **device** (`torch.device`, optional): Computation device (`'cpu'` or `'cuda'`). Defaults to CPU.

### `calculator.log(order = 'G', num_input = 1, batch_size = 16)`

Generate and print a detailed report:

* **order** (`Literal['G','M','k']`): Scale for FLOPs (`G`iga, `M`ega, `k`ilo).

* **num\_input** (`int`): How many inputs to simulate concurrently (for multi-input models).

* **batch\_size** (`int`): Size of the input batch used to estimate memory.

**Result Log**:

```
-----------------------------------------------------------------------------------------------
    G FLOPs    |    G FLOPS    |    M Acts     |      FPS      |  Memory (MB)  |    Params     
-----------------------------------------------------------------------------------------------
     1.397     |    109.197    |     67.19     |    78.176     |     8,201     |  88,591,464 
```

* **FLOPs**: Floating Point Operations — the total number of mathematical operations performed during a single forward pass.

* **FLOPS**: Floating Point Operations Per Second — how many FLOPs the model can process per second (a measure of speed).

* **Acts**: Total number of elements in all intermediate feature maps produced during a forward pass. This roughly indicates how much data the model processes internally and helps estimate memory usage and training cost time.

* **FPS**:  Frames Per Second — how many input samples the model can process per second during inference.

* **Memory (MB)**: Estimated GPU memory usage during training, based on the number of activations.

* **Params**: Total number of trainable parameters in the model.

**Warning Log**:

A warning will be printed if any modules are skipped in FLOPs estimation. For example:

```
***********************************************************************************************
Warning !! Above Estimations Ignore Following Modules !! The FLOPs Would be Underestimated !!
***********************************************************************************************

{'StochasticDepth', 'Permute'}
```

A warning block prints any unsupported modules that were excluded from FLOPs calculation.


## Internals

1. **Hook Registration**: Recursively attaches forward hooks to all submodules.

2. **FLOPs Computation**: Implements formulas for convolutions, normalization, pooling, activations, etc.

3. **Warm-up & Timing**: Runs 100 warm-up passes, then times 100 forward passes for stable metrics.

4. **Memory Estimation**: Based on activation count and tensor element size.


## Notes

* This tool is **currently focused on CNN-based models for computer vision**. Transformer-based models (e.g., Vision Transformers, Swin Transformers) are not yet supported in FLOPs estimation.

* Unsupported modules are recorded in `exclude`—you may need to extend formulas for custom layers.

* Memory estimation is rough and assumes no activation checkpointing or optimizer states.


## License

MIT License. Feel free to modify and distribute.