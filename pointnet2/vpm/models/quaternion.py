# external
import torch.nn as nn
import torch
import sys

# internal
sys.path.insert(0, '../../')
from kepler.aux.utils import *


class QuaternionViewpointMatching(nn.Module):
    def __init__(self, reference_data, num_debug_samples=20, radius=0.8 , debug=False):
        super().__init__()
        self.all_rotations = reference_data['scene_rotations']
        self.all_object_ids = reference_data['scene_object_ids']
        self.all_translations = reference_data['scene_translations']
        self.all_quaternions = reference_data['scene_quaternions']
        self.num_debug_samples = num_debug_samples
        self.radius = radius
        self.debug = debug

    def forward(self, in_data):
        fixed_viewpoints_object_id = in_data['object_id']
        fixed_viewpoints_scene_id = in_data['scene_id']
        fixed_viewpoints_pose = torch.tensor(in_data['transformation'], dtype=torch.float)
        fixed_viewpoints_quats = torch.tensor(in_data['quaternion'], dtype=torch.float)
        batch_size = len(fixed_viewpoints_object_id)

        batch_selected_frame_main_idx = []
        batch_angle_distance = []
        batch_center_distance = []
        batch_num_in_radius = []
        batch_rotations = []
        for ix in range(batch_size):
            valid_rotations = []
            valid_translations = []
            scene_id = fixed_viewpoints_scene_id[ix]
            object_id = fixed_viewpoints_object_id[ix]
            this_rotations = torch.from_numpy(numpy.vstack(self.all_rotations.get(scene_id)).astype(numpy.float32)).reshape(-1, 3, 3)
            this_translations = torch.from_numpy(numpy.vstack(self.all_translations.get(scene_id)).astype(numpy.float32)).reshape(-1, 3)
            this_quaternions = torch.from_numpy(numpy.vstack(self.all_quaternions.get(scene_id)).astype(numpy.float32)).reshape(-1, 4)
            object_in_frames = self.all_object_ids[str(scene_id)].tolist()[object_id]
            if len(object_in_frames) > 0:
                valid_rotations = this_rotations[object_in_frames]
                valid_translations = this_translations[object_in_frames]
                valid_quaternions = this_quaternions[object_in_frames]
            else:
                print("impossible match {} - {}".format(scene_id, object_id))
                valid_rotations = this_rotations
                valid_translations = this_translations
                valid_quaternions = this_quaternions

            target_quaternion = fixed_viewpoints_quats[ix, :]
            target_origin = fixed_viewpoints_pose[ix, :-1, -1]
            target_rotation = fixed_viewpoints_pose[ix, :3, :3]
    

            valid_origins = valid_translations

            # compute quaternion distance
            # based on https://arxiv.org/pdf/1505.07427.pdf
            beta = 5
            q = ViewpointMatchingTools.quat_to(valid_quaternions, target_quaternion)
            d = ViewpointMatchingTools.l2_to(valid_origins, target_origin)
            l = d + beta * q

            indicies_in_radius = torch.where(d <= self.radius)[0].tolist()

            # assert len(indicies_in_radius) != 0
            for ix in range(q.shape[0]):
                # if (ix not in object_in_frames) # or (ix not in indicies_in_radius):
                if ix not in indicies_in_radius and len(indicies_in_radius) > 0:
                    d[ix] = 10000.0
                    q[ix] = 10000.0
                    l[ix] = 10000.0

            c = l
            # create a mapping from valid rotations indexes to all rotations indexes
            main_indexes = object_in_frames
            selected_frame_idx = torch.argmin(c)
            batch_selected_frame_main_idx.append(main_indexes[selected_frame_idx])
            batch_angle_distance.append(q[selected_frame_idx])
            batch_center_distance.append(d[selected_frame_idx])
            batch_num_in_radius.append(len(indicies_in_radius))
            batch_rotations.append(target_rotation)

        return {
            'scene_id': in_data['scene_id'],
            'object_id': in_data['object_id'],
            'ann_id': in_data['ann_id'],
            'token': in_data['token'],
            'object_name': in_data['object_name'],
            'description': in_data['description'],
            'transformation': in_data['transformation'],
            'quaternion':in_data['quaternion'],
            'translation': in_data['translation'],
            'rotation': batch_rotations,
            'frame_idx': batch_selected_frame_main_idx,
            'angle_distance': batch_angle_distance,
            'center_distance': batch_center_distance,
            'num_in_radius': batch_num_in_radius,
        }