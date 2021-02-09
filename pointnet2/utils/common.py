from pointnet2.utils.ptc import point_cloud_label_to_surface_voxel_label_fast
import numpy as np
import omegaconf
import json
import os

def filter_points(coords, preds, targets, weights):
    assert coords.shape[0] == preds.shape[0] == targets.shape[0] == weights.shape[0]
    coord_hash = [hash(str(coords[point_idx][0]) + str(coords[point_idx][1]) + str(coords[point_idx][2])) for point_idx in range(coords.shape[0])]
    _, coord_ids = np.unique(np.array(coord_hash), return_index=True)
    coord_filtered, pred_filtered, target_filtered, weight_filtered = coords[coord_ids], preds[coord_ids], targets[coord_ids], weights[coord_ids]

    return coord_filtered, pred_filtered, target_filtered, weight_filtered

def compute_acc(coords, preds, targets, weights):
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
    seen_classes = np.unique(targets)
    NUM_CLASSES = len(get_nyu_40_class_list())
    mask = np.zeros(NUM_CLASSES)
    mask[seen_classes] = 1

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    labelweights = np.zeros(NUM_CLASSES)
    labelweights_vox = np.zeros(NUM_CLASSES)

    correct = np.sum(preds == targets) # evaluate only on 20 categories but not unknown
    total_correct += correct
    total_seen += targets.shape[0]
    tmp,_ = np.histogram(targets,range(NUM_CLASSES+1))
    labelweights += tmp
    for l in seen_classes:
        total_seen_class[l] += np.sum(targets==l)
        total_correct_class[l] += np.sum((preds==l) & (targets==l))

    _, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate((np.expand_dims(targets,1), np.expand_dims(preds,1)), axis=1), res=0.02)
    total_correct_vox += np.sum(uvlabel[:,0]==uvlabel[:,1])
    total_seen_vox += uvlabel[:,0].shape[0]
    tmp,_ = np.histogram(uvlabel[:,0],range(NUM_CLASSES+1))
    labelweights_vox += tmp
    for l in seen_classes:
        total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
        total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    pointacc = total_correct / float(total_seen)
    voxacc = total_correct_vox / float(total_seen_vox)

    labelweights = labelweights.astype(np.float32)/np.sum(labelweights.astype(np.float32))
    labelweights_vox = labelweights_vox.astype(np.float32)/np.sum(labelweights_vox.astype(np.float32))
    caliweights = labelweights_vox
    voxcaliacc = np.average(np.array(total_correct_class_vox)/(np.array(total_seen_class_vox,dtype=np.float)+1e-8),weights=caliweights)

    pointacc_per_class = np.zeros(NUM_CLASSES)
    voxacc_per_class = np.zeros(NUM_CLASSES)
    for l in seen_classes:
        pointacc_per_class[l] = total_correct_class[l]/(total_seen_class[l] + 1e-8)
        voxacc_per_class[l] = total_correct_class_vox[l]/(total_seen_class_vox[l] + 1e-8)

    return pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, mask

def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)

def get_scene_list(scans_dir):
    return [item for item in os.listdir(scans_dir) if 'scene' in item]

def get_samples_list(target_json):
    return json.load(open(target_json))
    
def get_nyu_40_class_list():
    nyu40 = [
        'floor', 
        'wall', 
        'cabinet', 
        'bed', 
        'chair', 
        'sofa', 
        'table', 
        'door', 
        'window', 
        'bookshelf', 
        'picture', 
        'counter', 
        'desk', 
        'curtain', 
        'refrigerator', 
        'bathtub', 
        'shower curtain', 
        'toilet', 
        'sink', 
        'otherprop'
    ]
    return nyu40