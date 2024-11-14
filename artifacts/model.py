

import torch
import timm

# create model and load state dict
# the model input is an image with dims [3, 384, 384],
# so you must resize and make sure dims are ordered [batch,C,H,W]
# this is a classification model with 7 classes, so the output size is [batch, 7]
model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=7)
model.load_state_dict(torch.load('vit-7class.pth'))


# if anywhere you need class names, you can use these
classes = [
    'house',
    'tree',
    'bunny',
    'turtle',
    'storm',
    'record-player',
    'ron-howard'
]
