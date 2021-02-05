from pointnet2.utils.common import *
from pointnet2.utils.ptc import read_ply_xyzrgbnormal
from tqdm import tqdm
import numpy as np
import hydra
import json
import os

def get_raw2scannet_label_map(hparams):
    lines = [line.rstrip() for line in open(hparams['paths.scannet_v2_labels_tsv'])]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(get_nyu_40_class_list())
        elements = lines[i].split('\t')
        # raw_name = elements[0]
        # nyu40_name = elements[6]
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'otherprop'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet

def collect_one_scene_data_label(scene_name, scene_folder):
    # Over-segmented segments: maps from segment to vertex/point IDs
    mesh_seg_filename = os.path.join(scene_folder, '%s_vh_clean_2.0.010000.segs.json'%(scene_name))
    #print mesh_seg_filename
    with open(mesh_seg_filename) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
        #print len(seg)
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)
    
    # Raw points in XYZRGBA
    ply_filename = os.path.join(scene_folder, '%s_vh_clean_2.ply' % (scene_name))
    points = read_ply_xyzrgbnormal(ply_filename)
    
    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    # annotation_filename = os.path.join(data_folder, '%s.aggregation.json'%(scene_name))
    annotation_filename = os.path.join(scene_folder, '%s_vh_clean.aggregation.json'%(scene_name))
    #print annotation_filename
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])
    
    #print len(instance_segids)
    #print labels
    
    # Each instance's points
    instance_points_list = []
    instance_labels_list = []
    semantic_labels_list = []
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_points = points[np.array(pointids),:]
        instance_points_list.append(instance_points)
        instance_labels_list.append(np.ones((instance_points.shape[0], 1))*i)   
        label = RAW2SCANNET[labels[i]]
        label = CLASS_NAMES.index(label)
        semantic_labels_list.append(np.ones((instance_points.shape[0], 1))*label)
       
    # Refactor data format
    scene_points = np.concatenate(instance_points_list, 0)
    scene_points = scene_points[:,0:9] # XYZ+RGB+NORMAL
    instance_labels = np.concatenate(instance_labels_list, 0) 
    semantic_labels = np.concatenate(semantic_labels_list, 0)
    data = np.concatenate((scene_points, instance_labels, semantic_labels), 1)

    if data.shape[0] > NUM_MAX_PTS:
        choices = np.random.choice(data.shape[0], NUM_MAX_PTS, replace=False)
        data = data[choices]

    return data

@hydra.main('../config/config.yaml')
def main(cfg):
    hparams = hydra_params_to_dotdict(cfg)
    os.makedirs(hparams['paths.scannet_preped'], exist_ok=True)
    global CLASS_NAMES 
    global RAW2SCANNET
    global NUM_MAX_PTS 
    CLASS_NAMES = get_nyu_40_class_list()
    RAW2SCANNET = get_raw2scannet_label_map(hparams)
    NUM_MAX_PTS = 100000

    scene_list = get_scene_list(hparams['paths.scannet_scans_dir'])
    for scene_id in tqdm(scene_list):
        try:
            out_filename = scene_id + '.npy' # scene0000_00.npy
            out_filename = os.path.join(hparams['paths.scannet_preped'], out_filename)
            data = collect_one_scene_data_label(scene_id, os.path.join(hparams['paths.scannet_scans_dir'], scene_id))
            # print("shape of subsampled scene data: {}".format(data.shape))
            np.save(out_filename, data)

        except Exception as e:
            print(scene_id+'ERROR!!')

    print("done!")

if __name__=='__main__':
    main()
