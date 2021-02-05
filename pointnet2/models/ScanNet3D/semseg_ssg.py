import pytorch_lightning as pl
import torch
import json
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader
from pointnet2.data.ScanNet3DLoader import ScanNet3DDataset
from pointnet2.utils.common import compute_acc
from pointnet2.models.common.pointnet2_ssg_sem import PointNet2SemSegSSG
from pointnet2.models.common.pointnet2_ssg_cls import *

class ScanNet3DPointNet2SemSegSSG(PointNet2SemSegSSG):
    def __init__(self, hparams):
        super().__init__(hparams)

    def training_step(self, batch, batch_idx):
        pc, labels, _, _ = batch

        logits = self.forward(pc)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            acc = (torch.argmax(logits, dim=1) == labels).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):
        pc, labels, pc_weights, _ = batch
        logits = self.forward(pc)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, labels)
        acc = (preds == labels).float().mean()

        coords = pc.view(-1, 9).cpu().numpy()            # (B * N, 3)   -> 3 should be dependent on the features
        preds = logits.max(1)[1].view(-1).cpu().numpy()       # (B * N)
        targets = labels.view(-1).cpu().numpy()             # (B * N)
        weights = pc_weights.view(-1).cpu().numpy()             # (B * N)

        pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, mask = compute_acc(
            coords=coords,
            preds=preds,
            targets=targets,
            weights=weights
        )

        return dict(
            val_loss=loss, 
            val_acc=acc,
            point_acc=torch.tensor(pointacc.astype('float32')),
            voxel_accuracy=torch.tensor(voxacc.astype('float32'))
        )

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

    def get_scene_list(self, ds_path):
        all_samples = json.load(open(ds_path))
        scene_ids = []
        for sample in all_samples:
            scene_ids.append(sample['scene_id'])

        return list(set(scene_ids))

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["optimizer.lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["optimizer.decay_step"]
                )
            ),
            lr_clip / self.hparams["optimizer.lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["optimizer.bn_momentum"]
            * self.hparams["optimizer.bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["optimizer.decay_step"]
                )
            ),
            bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["optimizer.lr"],
            weight_decay=self.hparams["optimizer.weight_decay"],
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def prepare_data(self):
        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudScale(),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudRotatePerturbation(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter(),
                d_utils.PointcloudRandomInputDropout(),
            ]
        )

        self.train_dset = ScanNet3DDataset(
                hparams=self.hparams, 
                phase='train',
                scene_list=self.get_scene_list(self.hparams['paths.train_split_json']),
                transforms=train_transforms, 
                num_classes=21, 
                is_weighting=True,
                npoints=8192,
                use_multiview=False,
                use_color=True,
                use_normal=True
        )

        self.val_dset = ScanNet3DDataset(
                hparams=self.hparams, 
                phase='val',
                scene_list=self.get_scene_list(self.hparams['paths.val_split_json']),
                transforms=None, 
                num_classes=21,
                is_weighting=True,
                npoints=8192,
                use_multiview=False,
                use_color=True,
                use_normal=True
        )

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
