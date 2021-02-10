import pytorch_lightning as pl
from pointnet2.models.common.bert import BERTEncoder
from torch.utils.data import DataLoader
from pointnet2.data.ScanRefer3DLoader import ScanRefer3DDataset
import json

class ScanRefer3DRetrieval(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.bert_encoder = BERTEncoder()


    def forward(self, captions_batch):
        self.bert_encoder(captions_batch)

    def training_step(self, captions_batch, batch_idx):
        self.forward(captions_batch=captions_batch)
        return 0

    def validation_step(self, captions_batch, batch_idx):
        self.forward(captions_batch=captions_batch)
        return 0

    def validation_end(self, outputs):
        return 0

    def get_scene_list(self, ds_path) -> list:
        all_samples = json.load(open(ds_path))
        scene_ids = []
        for sample in all_samples:
            scene_ids.append(sample['scene_id'])

        return list(set(scene_ids))

    def get_target_samples(self, ds_path) -> list:
        target_samples = json.load(open(ds_path))

        return target_samples

    def prepare_data(self) -> None:

        self.train_dset = ScanRefer3DDataset(
                hparams=self.hparams, 
                phase='train',
                target_samples=self.get_target_samples(self.hparams['paths.train_split_json']),
                scene_list=self.get_scene_list(self.hparams['paths.train_split_json']),
                transforms=None, 
                num_classes=21, 
                is_weighting=True,
                npoints=8192,
                use_multiview=False,
                use_color=True,
                use_normal=False
        )

        self.val_dset = ScanRefer3DDataset(
                hparams=self.hparams, 
                phase='val',
                target_samples=self.get_target_samples(self.hparams['paths.val_split_json']),
                scene_list=self.get_scene_list(self.hparams['paths.val_split_json']),
                transforms=None, 
                num_classes=21,
                is_weighting=True,
                npoints=8192,
                use_multiview=False,
                use_color=True,
                use_normal=False
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
