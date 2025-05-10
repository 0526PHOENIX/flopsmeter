"""
========================================================================================================================
Package
========================================================================================================================
"""
import time
import math

from typing import Literal

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module


"""
========================================================================================================================
Complexity Calculator
========================================================================================================================
"""
class Complexity_Calculator():

    """
    ====================================================================================================================
    Initialization
    ====================================================================================================================
    """
    def __init__(self, model: Module, dummy: tuple[int], device: torch.device = None) -> None:

        # Device
        if device.type == 'cuda':
            if torch.cuda.is_available():
                # GPU
                self.device = device
            else:
                # Warning
                print()
                print('WARNING !!! No GPU Found, Falling Back to CPU !!!')
                print()
                # CPU
                self.device = torch.device('cpu')
        else:
            # CPU
            self.device = torch.device('cpu')
    
        # Model and Dummy Input
        self.model = model.to(self.device)
        self.dummy = dummy

        # Activations
        self.actvs = 0

        # FLOPs
        self.flops = 0

        # Not Implemented Module
        self.exclude = set()

        # Capture Information from Forward Pass
        self.hook_layer()

        return
    
    """
    ====================================================================================================================
    Register Forword Hook
    ====================================================================================================================
    """
    def hook_layer(self) -> None:

        """
        ----------------------------------------------------------------------------------------------------------------
        Calculate FLOPs & Activations
        ----------------------------------------------------------------------------------------------------------------
        """
        def forward_hook(module: Module, feature_in: tuple[Tensor], feature_out: Tensor) -> None:

            # Convert Tuple to Tensor
            feature_in: Tensor = feature_in[0]

            # FLOPs
            flops = None

            # Convolution 1D & Transpose Convolution 1D & Lazy Version
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.LazyConv1d, nn.LazyConvTranspose1d)):
                '''
                2 * (C_in / Groups) * C_out * K * L_out
                '''
                # FLOPs
                flops = 2 * math.prod(module.kernel_size) * module.in_channels * feature_out.numel() / module.groups
                # Bias
                if module.bias is not None:
                    flops += feature_out.numel()

            # Convolution 2D & Transpose Convolution 2D & Lazy Version
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.LazyConv2d, nn.LazyConvTranspose2d)):
                '''
                2 * (C_in / Groups) * C_out * K_h * K_w * H_out * W_out
                '''
                # FLOPs
                flops = 2 * math.prod(module.kernel_size) * module.in_channels * feature_out.numel() / module.groups
                # Bias
                if module.bias is not None:
                    flops += feature_out.numel()

            # Convolution 3D & Transpose Convolution 3D & Lazy Version
            elif isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.LazyConv3d, nn.LazyConvTranspose3d)):
                '''
                2 * (C_in / Groups) * C_out * K_d * K_h * K_w * D_out * H_out * W_out
                '''
                # FLOPs
                flops = 2 * math.prod(module.kernel_size) * module.in_channels * feature_out.numel() / module.groups
                # Bias
                if module.bias is not None:
                    flops += feature_out.numel()

            # Normalization
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LazyBatchNorm1d, nn.LazyBatchNorm2d,
                                     nn.LazyBatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                     nn.LazyInstanceNorm1d, nn.LazyInstanceNorm2d, nn.LazyInstanceNorm3d, nn.GroupNorm)):
                '''
                ((x - mean) / sqrt(variance)) * scale + shift
                    substract mean: 1
                    square root:    1
                    variance:       2
                    scale:          1
                    shift:          1
                '''
                # FLOPs
                flops = (6 if module.affine else 4) * feature_out.numel()

            # Normalization
            elif isinstance(module, nn.LayerNorm):
                '''
                ((x - mean) / sqrt(variance)) * scale + shift
                    substract mean: 1
                    square root:    1
                    variance:       2
                    scale:          1
                    shift:          1
                '''
                # FLOPs
                flops = (6 if module.elementwise_affine else 4) * feature_out.numel()

            # Normalization
            elif isinstance(module, nn.LocalResponseNorm):
                '''
                x / (k + alpha * sum(x ^ 2)) ^ beta
                    x ^ 2:         K
                    sum:           K (add & Divise)
                    add k:         1
                    power:         1
                    divide:        1
                '''
                # FLOPs
                flops = (2 * module.size + 3) * feature_out.numel()

            # Dropout
            elif isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)):
                '''
                (x * binary mask) * scale
                    binary mask: 1
                    scale:       1    
                '''
                # FLOPs
                flops = 2 * feature_out.numel()

            # Fully Connected
            elif isinstance(module, (nn.Linear, nn.LazyLinear)):
                '''
                2 * C_in * C_out
                '''
                # FLOPs
                flops = 2 * module.in_features * module.out_features
                # Bias
                if module.bias is not None:
                    flops += module.out_features

            # Identical Mapping
            elif isinstance(module, nn.Identity):
                '''
                pass-through operation
                '''
                # FLOPs
                flops = 0

            # Identical Mapping
            elif isinstance(module, nn.Bilinear):
                '''
                2 * C_in1 * C_in2 * C_out
                '''
                # FLOPs
                flops = 2 * module.in1_features * module.in2_features * module.out_features
                # Bias
                if module.bias is not None:
                    flops += module.out_features

            # Maximum Pooling 
            elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.FractionalMaxPool2d, nn.FractionalMaxPool3d)):
                '''
                (K - 1) comparisons per output element
                '''
                # Kernel Volume Size
                kernel_volume = get_kernel_volume(module)
                # FLOPs
                flops = (kernel_volume - 1) * feature_out.numel()

            # Maximum Unpooling 
            elif isinstance(module, (nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d)):
                '''
                1 scatter per input element
                '''
                # FLOPs
                flops = 1 * feature_in.numel()

            # Average Pooling 
            elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
                '''
                (K - 1) add + 1 division
                '''
                # Kernel Volume Size
                kernel_volume = get_kernel_volume(module)
                # FLOPs
                flops = kernel_volume * feature_out.numel()

            # Power-Averge Pooling
            elif isinstance(module, (nn.LPPool1d, nn.LPPool2d)):
                '''
                K power + (K - 1) add + 1 root
                '''
                # Kernel Volume Size
                kernel_volume = get_kernel_volume(module)
                # FLOPs
                flops = 2 * math.prod(module.kernel_size) * feature_out.numel()

            # Adaptive Maximum Pooling
            elif isinstance(module, (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
                '''
                (K - 1) comparisons per output element
                '''
                # Check Number of Axis
                assert len(feature_in.shape[2:]) == len(feature_out.shape[2:]), "Mismatch in spatial dimensions"
                # kernel Size
                kernel_size = [max(1, math.ceil(in_s / out_s)) for in_s, out_s in zip(feature_in.shape[2:], feature_out.shape[2:])]
                # FLOPs
                flops = (math.prod(kernel_size) - 1) * feature_out.numel()

            # Adaptive Average Pooling
            elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
                '''
                (K - 1) add + 1 division
                '''
                # Check Number of Axis
                assert len(feature_in.shape[2:]) == len(feature_out.shape[2:]), "Mismatch in spatial dimensions"
                # kernel Size
                kernel_size = [max(1, math.ceil(in_s / out_s)) for in_s, out_s in zip(feature_in.shape[2:], feature_out.shape[2:])]
                # FLOPs
                flops = math.prod(kernel_size) * feature_out.numel()

            # Activation (Only Approximation)
            elif isinstance(module, (nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid,
                                     nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.GELU, nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus,
                                     nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Threshold, nn.GLU, nn.Softmin, nn.Softmax,
                                     nn.Softmax2d, nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss)):
                '''
                4 * C_out * Feature Size
                '''
                # FLOPs
                flops = 4 * feature_out.numel()

            # Upsample
            elif isinstance(module, nn.Upsample):
                # Nearest Interpolation (1D, 2D ,3D)
                if module.mode == 'nearest':
                    '''
                    direct copy of nearest input value
                    '''
                    # FLOPs
                    flops = 0
                # Linear Interpolation (1D)
                elif module.mode == 'linear':
                    '''
                    2-point linear interpolation
                        1 multiply + 1 multiply + 1 add
                    '''
                    # FLOPs
                    flops = 3 * feature_out.numel()
                # Bilinear Interpolation (2D)
                elif module.mode == 'bilinear':
                    '''
                    4-point weighted interpolation
                        4 multiply + 3 add
                    '''
                    # FLOPs
                    flops = 7 * feature_out.numel()
                # Bicubic Interpolation (2D) (Only Approximation)
                elif module.mode == 'bicubic':
                    '''
                    16-point cubic interpolation
                        16 multiply + 15 add
                    '''
                    # FLOPs
                    flops = 31 * feature_out.numel()
                # Trilinear Interpolation (3D)
                elif module.mode == 'trilinear':
                    '''
                    8-point weighted interpolation
                        8 multiply + 7 add
                    '''
                    # FLOPs
                    flops = 15 * feature_out.numel()
                else:
                    raise NotImplementedError(f"FLOPs for Upsample mode '{module.mode}' not implemented")
            
            # Nearest Interpolation
            elif isinstance(module, nn.UpsamplingNearest2d):
                '''
                direct copy of nearest input value
                '''
                # FLOPs
                flops = 0

            # Bilinear Interpolation
            elif isinstance(module, nn.UpsamplingBilinear2d):
                '''
                4-point weighted interpolation
                    4 multiply + 3 add
                '''
                # FLOPs
                flops = 7 * feature_out.numel()

            # Pixel Shuffle & Pixel Unshuffle & Channel Shuffle & Flatten
            elif isinstance(module, (nn.PixelShuffle, nn.PixelUnshuffle, nn.ChannelShuffle, nn.Flatten)):
                '''
                only pixel rearrangement
                '''
                # FLOPs
                flops = 0

            # Padding
            elif isinstance(module, (nn.ZeroPad1d, nn.ZeroPad2d, nn.ZeroPad3d, nn.CircularPad1d, nn.CircularPad2d, nn.CircularPad3d,
                                     nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d, nn.ReflectionPad1d, nn.ReflectionPad2d, nn.ReflectionPad3d,
                                     nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d)):
                '''
                no arithmetic computation
                '''
                # FLOPs
                flops = 0

            # Others
            else:
                # Check Layer Name
                self.exclude.add(module._get_name())
                # FLOPs
                flops = 0

            # Total FLOPs
            self.actvs += feature_out.numel()
            self.flops += flops or 0

            return

        """
        ----------------------------------------------------------------------------------------------------------------
        Get Kernel Volume Size of Pooling Layer
        ----------------------------------------------------------------------------------------------------------------
        """   
        def get_kernel_volume(module: Module) -> int:
            # 
            if isinstance(module.kernel_size, int):
                # One Dimension Data
                if isinstance(module, (nn.MaxPool1d, nn.AvgPool1d, nn.LPPool1d)):
                    return module.kernel_size ** 1
                # Two Dimension Data (Square Structure)
                elif isinstance(module, (nn.MaxPool2d, nn.FractionalMaxPool2d, nn.AvgPool2d, nn.LPPool2d)):
                    return module.kernel_size ** 2
                # Three Dimension Data (Cubic Structure)
                elif isinstance(module, (nn.MaxPool3d, nn.FractionalMaxPool3d, nn.AvgPool3d)):
                    return module.kernel_size ** 3
            else:
                return math.prod(module.kernel_size)

        """
        ----------------------------------------------------------------------------------------------------------------
        Break Down Model into Layers
        ----------------------------------------------------------------------------------------------------------------
        """
        def break_block(block: Module) -> None:
            
            # Block is Indeed a Block
            if list(block.children()):
                # Iterate Through Layer in Block
                for child in block.children():
                    # Recursive Break Down
                    break_block(child)

            # Block is a Layer
            else:
                # Register Hook
                block.register_forward_hook(forward_hook)

                return
                
        return break_block(self.model)

    """
    ====================================================================================================================
    Output Log
    ====================================================================================================================
    """
    def log(self, order: str | Literal['G', 'M', 'k'] = 'G', num_input: int = 1, batch_size: int = 16) -> None:

        """
        ----------------------------------------------------------------------------------------------------------------
        FLOPs and FLOPS
        ----------------------------------------------------------------------------------------------------------------
        """
        # Dummy Input
        dummy = torch.rand((1, *self.dummy)).to(self.device)
        dummy = [dummy for _ in range(num_input)]

        # Model Warm Up
        for _ in range(100):
            self.model(*dummy)

        # Synchronize GPU Operation if CUDA is Available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Start Time
        time_start = time.time()

        # Forward Pass
        for _ in range(100):
            self.actvs = 0
            self.flops = 0
            self.model(*dummy)

        # End Time
        time_end = time.time()

        # FLOPs and FLOPs per Second
        flops = self.flops
        flops_per_sec = self.flops / ((time_end - time_start) / 100)

        # Activations
        actvs = self.actvs

        # Memory Usage
        data_size = torch.tensor([], dtype = next(self.model.parameters()).dtype).element_size()
        memory = int((actvs * 2 * data_size * batch_size) / (1024 ** 2))

        # Frame Per Second
        fps = 1 / ((time_end - time_start) / 100)
        fps = round(fps, 3)

        # Magnitude
        actvs = round(actvs * 1e-6, 3)
        if order == 'G':
            flops = round(flops * 1e-9, 3)
            flops_per_sec = round(flops_per_sec * 1e-9, 3)
        elif order == 'M':
            flops = round(flops * 1e-6, 3)
            flops_per_sec = round(flops_per_sec * 1e-6, 3)
        elif order == 'k':
            flops = round(flops * 1e-3, 3)
            flops_per_sec = round(flops_per_sec * 1e-3, 3)
        else:
            raise ValueError('Invalid Order')

        """
        ----------------------------------------------------------------------------------------------------------------
        Parameter
        ----------------------------------------------------------------------------------------------------------------
        """
        # Trainable Parameter
        num_param = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        """
        ----------------------------------------------------------------------------------------------------------------
        Output Log
        ----------------------------------------------------------------------------------------------------------------
        """
        # Output Format
        title = "{:^20}|{:^20}|{:^20}|{:^20}|{:^20}|{:^20}"
        space = "{:^20}|{:^20}|{:^20}|{:^20,}|{:^20,}|{:^20,}"

        # Title
        print('-' * 125)
        print(title.format(order + ' FLOPs', order + ' FLOPS', 'M Acts', 'FPS', 'Memory (MB)', 'Params'))
        print('-' * 125)

        # Output Log
        print(space.format(flops, flops_per_sec, actvs, fps, memory, num_param))
        print()

        """
        ----------------------------------------------------------------------------------------------------------------
        Warning
        ----------------------------------------------------------------------------------------------------------------
        """
        # Check Not Implemented Module Set
        if self.exclude:
            
            # Title
            print('*' * 125)
            print('Warning !! Above Estimations Ignore Following Modules !! The FLOPs Would be Underestimated !!')
            print('*' * 125)

            # Output Log
            print()
            print(self.exclude)
            print()

        return