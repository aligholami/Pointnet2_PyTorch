# external
from torch.utils.data import Dataset
import torch
import numpy
import json
import os
import sys


# internal 
sys.path.insert(0, '..')
from kepler.aux.utils import *


class ScanReferViewpointsDataset(Dataset):
    def __init__(self, ds_path, debug=False):
        self.fixed_viewpoints_list = get_dict_from_json(ds_path)
        if debug == True:
            self.debug_list = ['scene0583_00', 'scene0063_00', 'scene0261_00', 'scene0149_00']
            self.fixed_viewpoints_list = [item for item in self.fixed_viewpoints_list if item['scene_id'] in target_scenes]

    def __len__(self):
        return len(self.fixed_viewpoints_list)

    def __getitem__(self, idx):
        scene_id = self.fixed_viewpoints_list[idx]['scene_id']
        transformation = self.fixed_viewpoints_list[idx]['transformation']
        quaternion = self.fixed_viewpoints_list[idx]['quaternion']
        translation = self.fixed_viewpoints_list[idx]['translation']
        caption =  self.fixed_viewpoints_list[idx]['description']
        object_id = self.fixed_viewpoints_list[idx]['object_id']
        quaternion = self.fixed_viewpoints_list[idx]['quaternion']
        translation = self.fixed_viewpoints_list[idx]['translation']
        token = self.fixed_viewpoints_list[idx]['token']
        ann_id = self.fixed_viewpoints_list[idx]['ann_id']
        object_name = self.fixed_viewpoints_list[idx]['object_name']

        data_dict = {
            'scene_id': scene_id,
            'transformation': numpy.array(transformation, dtype=numpy.float).reshape(-1, 4, 4),
            'quaternion': numpy.array(quaternion, dtype=numpy.float).reshape(-1, 4),
            'translation': numpy.array(translation).reshape(-1, 3),
            'description': caption,
            'object_id': object_id,
            'object_name': object_name,
            'ann_id': ann_id,
            'token': token
        }
        
        return data_dict

    def collate_fn(self, batch):
        data_dict = {
            'transformation': [],
            'quaternion': [],
            'translation': [],
            'description': [],
            'scene_id': [],
            'object_id': [],
            'object_name': [],
            'ann_id': [],
            'token': []
        }

        for item in batch:
            data_dict['transformation'].append(item['transformation'])
            data_dict['quaternion'].append(item['quaternion'])
            data_dict['translation'].append(item['translation'])
            data_dict['description'].append(item['description'])
            data_dict['scene_id'].append(item['scene_id'])
            data_dict['object_id'].append(item['object_id'])
            data_dict['object_name'].append(item['object_name'])
            data_dict['ann_id'].append(item['ann_id'])
            data_dict['token'].append(item['token'])
        
        data_dict['transformation'] = numpy.vstack(data_dict['transformation'])
        data_dict['quaternion'] = numpy.vstack(data_dict['quaternion'])
        data_dict['translation'] = numpy.vstack(data_dict['translation'])

        return data_dict
        
    def get_subset_scene_ids(self):
        """
            Returns the subset scene ids as a list.
        """
        return list(set([viewpoint['scene_id'] for viewpoint in self.fixed_viewpoints_list]))



