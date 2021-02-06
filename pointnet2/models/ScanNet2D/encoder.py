from pointnet2.models.common.resnet_101 import ResNet101Encoder
from torch.utils.data import DataLoader
import torch

class ScanNet2DResNetEncoder(ResNet101Encoder):
    def __init__(self, hparams):
        super().__init__(hparams)

    def forward(self, image):
        return super().forward(image)

    def validation_step(self, batch, batch_idx):
        images = batch
        logits = self.forward(pc)
        loss = F.cross_entropy(logits, labels)
        
        acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        return dict(val_loss=loss, val_acc=acc)

    def validation_end(self, outputs):
        reduced_outputs = {}
        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs
    def get_scene_list(self):
        return []

    def prepare_data(self):
        self.train_dset = []
        self.val_dset = []

    def _build_dataloader(self, dset, mode):

        return DataLoader(
            dset,
            batch_size=self.hparams["batch_size"],
            shuffle=mode == "train",
            num_workers=4,
            pin_memory=True,
            drop_last=mode == "train",
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
