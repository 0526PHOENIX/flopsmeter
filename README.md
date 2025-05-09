# FLOPs Calculator for PyTorch Deep Learning Model

A lightweight Python utility for estimating the computational complexity of PyTorch models. It hooks into a model's forward pass to count floating point operations (FLOPs), activations, memory usage, frames per second (FPS), and trainable parameters.

---

## Package Overview

* **Name:** `Complexity_Calculator`
* **Language:** Python 3.10+
* **Dependencies:**

  * `torch 2.2.1+` (PyTorch)

This package helps deep learning practitioners quickly gauge the computational cost of their PyTorch models, aiding in model optimization, benchmarking, and resource planning.

---

## Features

* **Automatic FLOPs Counting**: Estimates operations for convolution, normalization, activation, pooling, and more.
* **Activation Tracking**: Records total number of activations per forward pass.
* **Memory Footprint**: Computes approximate GPU/CPU memory usage (MB) during inference.
* **FPS Measurement**: Reports frames per second for batch inference.
* **Parameter Counting**: Displays total number of trainable parameters.
* **Module Exclusion Warning**: Alerts if any layer types are not implemented (may lead to underestimation).

---

## Installation

Install via pip:

```bash
pip install torch         # if not already installed
# Clone or download this repository, then:
cd path/to/repo
torch setup.py install    # or add to your project directly
```

*(Alternatively, copy the **`Complexity_Calculator`** class file into your project.)*

---

## Quick Start

```python
import torch
import torch.nn

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

# Create Calculator With Dummy Input (C, H, W)
calculator = Complexity_Calculator(model = SimpleCNN(), dummy = (3, 224, 224), device = torch.device('cuda'))

# Print Complexity Report
calculator.log(order = 'G', num_input = 1, batch_size = 16)
```

---

## API Reference

### `Complexity_Calculator(model, dummy, device = None)`

* **model** (`torch.nn.Module`): Your PyTorch model.
* **dummy** (`tuple[int, int, int]`): Shape of single input tensor (channels, height, width) or (sequence length, feature dims) for 1D.
* **device** (`torch.device`, optional): Computation device (`'cpu'` or `'cuda'`). Defaults to CPU.

### `calculator.log(order = 'G', num_input = 1, batch_size = 16)`

Generate and print a detailed report:

* **order** (`Literal['G','M','k']`): Scale for FLOPs (`G`iga, `M`ega, `k`ilo).
* **num\_input** (`int`): Number of dummy inputs to simulate concurrent inputs.
* **batch\_size** (`int`): Batch size for memory estimation.

**Output Log**:

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

```
***********************************************************************************************
Warning !! Above Estimations Ignore Following Modules !! The FLOPs Would be Underestimated !!
***********************************************************************************************

{'StochasticDepth', 'Permute'}
```

A warning block prints any unsupported modules that were excluded from FLOPs calculation.

---

## Internals

1. **Hook Registration**: Recursively attaches forward hooks to all submodules.
2. **FLOPs Computation**: Implements formulas for convolutions, normalization, pooling, activations, etc.
3. **Warm-up & Timing**: Runs 100 warm-up passes, then times 100 forward passes for stable metrics.
4. **Memory Estimation**: Based on activation count and tensor element size.

---

## Notes

* Unsupported modules are recorded in `exclude`—you may need to extend formulas for custom layers.
* Assumes activations fit in available GPU memory; external context (optimizer states) not considered.

---

## License

MIT License. Feel free to modify and distribute.
