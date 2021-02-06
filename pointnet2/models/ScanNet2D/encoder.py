from pointnet2.models.common.resnet_101 import ResNet101Encoder
from torch.utils.data import DataLoader
from pointnet2.data.ScanNet2DLoader import ScanNetFrameOnlyDataset
import torch
import os

class ScanNet2DResNetEncoder(ResNet101Encoder):
    def __init__(self, hparams):
        super().__init__(hparams)

    def forward(self, image):
        return super().forward(image)

    def test_step(self, batch):
        scene_ids, object_ids, ann_ids, images = batch
        batch_resnet_features = super().forward(images)

        return ({
            'scene_id': scene_ids,
            'object_id': object_ids,
            'ann_id': ann_ids,
            'frame_features': batch_resnet_features
        })

    def test_step_end(self, test_summary):
        print(test_summary)
        exit(0)

    # def write_features(self, batches):
    #     """
    #         writes extracted featues as numpy arrays.
    #         batches is a list of dict. Each dict has a batch of results 
    #         in it.
    #     """

    #     for batch in tqdm(batches):
    #         batch_size = len(batch['scene_id'])
    #         for i in range(batch_size):
    #             write_dir = self.args.features_dir.format(str(batch['scene_id'][i]))
    #             if not os.path.isdir(write_dir):
    #                 os.makedirs(write_dir)
    #             lpath = os.path.join(write_dir, str(batch['scene_id'][i]) + '-' + batch['object_id'][i] + '_' + batch['ann_id'][i]  +'.npy')
    #             np.save(lpath, batch['frame_features'][i])

    def get_input_list(self):
        scene_list = [scene_id for scene_id in os.listdir(self.hparams['scannet_scans_dir']) if 'scene' in scene_id]
        
        input_list = []
        for scene_id in scene_list:
            _ = [input_list.append({'scene_id': scene_id, 'object_id': f.split('-')[1].split('_')[0], 'ann_id': f.split('-')[1].split('_')[1].strip('.png')}) for f in os.listdir(os.path.join(self.scene_list_dir, scene_id)) if '.png' in f and 'thumb' not in f]

        return input_list

    def prepare_data(self):
        self.test_dset = ScanNetFrameOnlyDataset(input_list=self.get_input_list())

    def _build_dataloader(self, dset, mode):
        return DataLoader(
            dset,
            batch_size=self.hparams["batch_size"],
            shuffle=mode == "train",
            num_workers=4,
            pin_memory=True,
            drop_last=mode == "train",
        )

    def test_dataloader(self):
        return self._build_dataloader(self.test_dset, mode="test")