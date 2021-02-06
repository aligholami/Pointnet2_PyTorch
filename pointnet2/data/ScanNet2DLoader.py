import torch
from torchvision import transforms
import numpy as np
import os
import cv2
import json
import torch.utils.data as data_tools
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

class ScanNetFrameOnlyDataset(Dataset):
    def __init__(self, hparams, input_list, transforms):
        self.hparams = hparams
        self.input_list = input_list
        self.transforms = transforms

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        f = self.input_list[idx]
        lpath = self.hparams['scannet_frames_dir'.format(f['scene_id'], f['scene_id'], str(f['object_id']), str(f['ann_id']))]

        try:
            resized = Image.open(lpath).resize((224, 224))
            rgbed = resized.convert('RGB')
            frame_tensor = torch.from_numpy(np.asarray(rgbed).astype(np.float32)).permute(2, 0, 1)
            
            if self.transforms:
                frame_tensor = self.transforms(frame_tensor / 255.0)
        
            f_ret = {
                'use': True, 
                'frame_tensor': frame_tensor,
                'scene_id': f['scene_id'],
                'object_id': f['object_id'],
                'ann_id': f['ann_id']
            }

        except Exception as exc:
            print(exc)
            f_ret = {
                'use': False
            }
    
        return f_ret

    def collate(self, data):
        data = list(filter(lambda x: x['use'] == True, data))

        return data_tools.dataloader.default_collate(data)