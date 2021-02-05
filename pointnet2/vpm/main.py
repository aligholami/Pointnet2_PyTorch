# external
from config import QUATERNIONS_TRAIN
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import sys

# internal 
sys.path.insert(0, '..')
from kepler.vpm.dataset import ScanReferViewpointsDataset
from kepler.vpm.models import NaiveViewpointMatching, QuaternionViewpointMatching
from kepler.aux.utils import *
from kepler.vpm.config import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('scannet_gallery_dir', nargs="?", type=str, default='/local-scratch/code/scan2cap_extracted/common/galleries')
    parser.add_argument('scannet_all_frames_dir', nargs="?", type=str, default='/local-scratch/code/scannet_extracted/color_and_pose')
    parser.add_argument('best_matches_dir', nargs="?", type=str, default='/local-scratch/code/scan2cap_extracted/match-based/vp_matching')
    parser.add_argument('temp_data', nargs="?", type=str, default='/local-scratch/temp-data')
    parser.add_argument('sampling_freq', nargs="?", type=int, default=1)
    parser.add_argument('--model', type=str, default='naive', help='quaternion')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    
    return parser.parse_args()

def prepare_datasets_and_loaders(batch_size):
    srd_conf = {
        'batch_size': batch_size,
        'shuffle': False
    }
    
    sr_datasets = {
        'train': ScanReferViewpointsDataset(QUATERNIONS_TRAIN),
        'val': ScanReferViewpointsDataset(QUATERNIONS_VAL)
    }

    sr_loaders = {
        'train': DataLoader(sr_datasets['train'], **srd_conf, collate_fn=sr_datasets['train'].collate_fn),
        'val': DataLoader(sr_datasets['val'], **srd_conf, collate_fn=sr_datasets['val'].collate_fn)
    }

    return sr_datasets, sr_loaders

if __name__ == "__main__":
    args = parse_arguments()
    
    full_ds = ScanReferViewpointsDataset(QUATERNIONS)
    target_scenes = full_ds.get_subset_scene_ids()
    scene_translations, scene_rotations, scene_transformations = get_scene_poses_rotations(args, target_scenes)
    scene_quaternions = get_scene_quaternions(args, scene_rotations, target_scenes)
    scene_object_ids = get_scene_object_ids(args, target_scenes)

    reference_data = {
        'scene_translations': scene_translations,
        'scene_rotations': scene_rotations,
        'scene_quaternions': scene_quaternions,
        'scene_object_ids': scene_object_ids
    } 

    # Matching model
    radius = 0.2
    if args.model == 'naive':
        print("Using naive frame matching.")
        get_best_match = NaiveViewpointMatching(reference_data=reference_data, radius=radius)
    elif args.model == 'quaternion':
        print("Using quaternion relevance frame matching.")
        get_best_match = QuaternionViewpointMatching(reference_data=reference_data, radius=radius)
    else:
        raise NotImplementedError;

    # dataset and dataloader
    sr_datasets, sr_loaders = prepare_datasets_and_loaders(args.batch_size)

    for split, loader in sr_loaders.items():
        batch_out = []
        tl = tqdm(loader)
        tl.set_description("{} Frame Matching on {} Split".format(args.model, split))
        for ix, batch_in in enumerate(tl):
            batch_out.append(get_best_match(batch_in))
            
        write_batch_dicts_to_path(batch_out, os.path.join(args.best_matches_dir, 'matched_{}.json'.format(split)))
