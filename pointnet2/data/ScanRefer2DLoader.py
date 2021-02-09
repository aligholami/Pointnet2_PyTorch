from torch.utils.data import Dataset 
from pointnet2.utils.common import *
from pointnet2.data.ScanNet2DLoader import ScanNet2DDataset

class ScanRefer2DDataset(ScanNet2DDataset):
    
    def __init__(self, hparams, phase, scene_list, transforms, num_classes=21, npoints=8192, is_weighting=True, use_color=False, use_normal=False):
        self.hparams = hparams
        self.sample_list = get_samples_list(self.hparams['target_samples'])

    def __len__(self):
        return len([])

    def __getitem__(self, index):
        scene_id = self.sample_list[index]['scene_id']
        object_id = self.
        

        return {}
    