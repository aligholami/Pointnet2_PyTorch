import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl


class ResNet101Encoder(pl.LightningModule):
    def __init__(self, hparams):
        self.image_modules = list(models.resnet101(pretrained=hparams['pretrained'], progress=hparams['show_progress']).children())[:-1]
        self.core = nn.Sequential(*self.image_modules)
        
    def forward(self, image):
        f = self.core(image)
        return f
