
"""
========================================================================================================================
Package
========================================================================================================================
"""
import torch
from torchvision import models

from flopsmeter import Complexity_Calculator


"""
========================================================================================================================
Main Function
========================================================================================================================
"""
if __name__ == '__main__':

    # Model
    all_model = {
                    'vgg11':                models.vgg11(),
                    'vgg13':                models.vgg13(),
                    'vgg16':                models.vgg16(),
                    'vgg19':                models.vgg19(),
                    'resnet18':             models.resnet18(),
                    'resnet34':             models.resnet34(),
                    'resnet50':             models.resnet50(),
                    'resnet101':            models.resnet101(),
                    'resnet152':            models.resnet152(),
                    'densenet121':          models.densenet121(),
                    'densenet161':          models.densenet161(),
                    'densenet169':          models.densenet169(),
                    'densenet201':          models.densenet201(),
                    'convnext_base':        models.convnext_base(),
                    'convnext_large':       models.convnext_large(),
                    'convnext small':       models.convnext_small(),
                    'convnext tiny':        models.convnext_tiny(),
                    'efficientnet b0':      models.efficientnet_b0(),
                    'efficientnet b1':      models.efficientnet_b1(),
                    'efficientnet b2':      models.efficientnet_b2(),
                    'efficientnet b3':      models.efficientnet_b3(),
                    'efficientnet b4':      models.efficientnet_b4(),
                    'efficientnet b5':      models.efficientnet_b5(),
                    'efficientnet b6':      models.efficientnet_b6(),
                    'efficientnet b7':      models.efficientnet_b7(),
                    'efficientnet v2 l':    models.efficientnet_v2_l(),
                    'efficientnet v2 m':    models.efficientnet_v2_m(),
                    'efficientnet v2 s':    models.efficientnet_v2_s(),
                }
    
    for name, model in all_model.items():

        print()
        print(name)

        # Initialize calculator with dummy input shape (C, H, W)
        calculator = Complexity_Calculator(model, dummy = (3, 224, 224), device = torch.device('cuda'))

        # Print Complexity Report
        calculator.log(order = 'M', num_input = 1, batch_size = 16)