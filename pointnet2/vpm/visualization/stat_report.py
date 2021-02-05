import json
import pickle
from tqdm import tqdm
from PIL import Image
import numpy as np
import os


mode = 'match' # or 'render'

if mode == 'match':
    SCANNET_ROOT = "/datasets/released/scannet/public/v2/scans/"
    AGGR_JSON = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean.aggregation.json")
    SCANNET_V2_TSV = "/datasets/released/scannet/public/v2/scannet-labels.combined.tsv"
    TRAIN_MATCHES = '/local-scratch/code/scan2cap_extracted/match-based/vp_matching/matches_train.json'
    VAL_MATCHES = '/local-scratch/code/scan2cap_extracted/match-based/vp_matching/matches_val.json'
    # has object ids, raw name, semantic label and bounding box; extracted using tools/export_scannet_image_bbox.py
    GT_PATH = '/local-scratch/code/scan2cap_extracted/match-based/bbox_pickle/oracle_quaternion/{}/{}.p'
    LABEL_IMG = '/local-scratch/code/scan2cap_extracted/match-based/instance-masks-quaternion/{}/{}.png' # scene_id, frame_id
    tm = json.load(open(TRAIN_MATCHES, 'r'))
    vm = json.load(open(VAL_MATCHES, 'r')) 
    am = tm + vm
    SCENE_LIST = list(set([item['scene_id'] for item in am]))
    exp_name = 'quaternion_annotated'
    ignored_frames = []
    stats_directory = '/local-scratch/code/scan2cap_extracted/match-based/matching_stats'
    ignored_frames_path = '/local-scratch/code/scan2cap_extracted/common/scanrefer/ScanRefer/ignored_frames.json'

if mode == 'render':
    SCANNET_ROOT = "/datasets/released/scannet/public/v2/scans/"
    AGGR_JSON = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean.aggregation.json")
    SCANNET_V2_TSV = "/datasets/released/scannet/public/v2/scannet-labels.combined.tsv"
    FILTERED = '/local-scratch/scan2cap_extracted/common/scanrefer/ScanRefer_fixed_viewpoints/ScanRefer_filtered_fixed_viewpoint.json'
    # has object ids, raw name, semantic label and bounding box; extracted using tools/export_scannet_image_bbox.py
    GT_PATH = '/local-scratch/scan2cap_extracted/render-based/bbox_pickle/oracle/{}/{}-{}_{}.p'
    LABEL_IMG = "/local-scratch/scan2cap_extracted/render-based/instance-masks/{}/{}-{}_{}.objectId.encoded.png" # scene_id, frame_id
    tm = json.load(open(FILTERED, 'r'))
    am = tm
    ignored_renders = []
    exp_name = 'rendered'
    SCENE_LIST = list(set([item['scene_id'] for item in am]))
    stats_directory = '/local-scratch/scan2cap_extracted/match-based/matching_stats'

    ignored_renders_path = '/local-scratch/scan2cap_extracted/common/scanrefer/ScanRefer/ignored_renders.json'

if mode == 'topdown':
    SCANNET_ROOT = "/datasets/released/scannet/public/v2/scans/"
    AGGR_JSON = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean.aggregation.json")
    SCANNET_V2_TSV = "/datasets/released/scannet/public/v2/scannet-labels.combined.tsv"
    FILTERED = '/local-scratch/scan2cap_extracted/common/scanrefer/ScanRefer_fixed_viewpoints/ScanRefer_filtered_fixed_viewpoint.json'
    # has object ids, raw name, semantic label and bounding box; extracted using tools/export_scannet_image_bbox.py
    GT_PATH = '/local-scratch/scan2cap_extracted/topdown-based/bbox_pickle/oracle/{}/{}.p'
    LABEL_IMG = "/local-scratch/scan2cap_extracted/topdown-based/instance_renders/{}/{}.vertexAttribute.encoded.png" # scene_id, frame_id
    tm = json.load(open(FILTERED, 'r'))
    am = tm
    ignored_renders = []
    exp_name = 'topdown'
    SCENE_LIST = list(set([item['scene_id'] for item in am]))
    stats_directory = '/local-scratch/scan2cap_extracted/match-based/matching_stats'

    ignored_renders_path = '/local-scratch/scan2cap_extracted/common/scanrefer/ScanRefer/ignored_renders.json'


if mode == 'estimated':
    SCANNET_ROOT = "/datasets/released/scannet/public/v2/scans/"
    AGGR_JSON = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean.aggregation.json")
    SCANNET_V2_TSV = "/datasets/released/scannet/public/v2/scannet-labels.combined.tsv"
    FILTERED = '/local-scratch/scan2cap_extracted/common/scanrefer/ScanRefer_fixed_viewpoints/ScanRefer_filtered_fixed_viewpoint.json'
    # has object ids, raw name, semantic label and bounding box; extracted using tools/export_scannet_image_bbox.py
    GT_PATH = '/local-scratch/scan2cap_extracted/projected-based/bbox_pickle/votenet_no_synth/{}/{}-{}.p'
    LABEL_IMG = "/local-scratch/scan2cap_extracted/projected-based/predicted_viewpoints/votenet_estimated_viewpoints_20201102/instance_renders/{}/{}-{}.objectId.encoded.png" # scene_id, frame_id
    tm = json.load(open(FILTERED, 'r'))
    am = tm
    ignored_renders = []
    exp_name = 'estimated'
    SCENE_LIST = list(set([item['scene_id'] for item in am]))
    stats_directory = '/local-scratch/code/scan2cap_extracted/match-based/matching_stats'

    ignored_renders_path = '/local-scratch/code/scan2cap_extracted/common/scanrefer/ScanRefer/ignored_renders.json'


########################
########################
#### FIX THE OBJECT ID PROBLEM
def get_id2name_file():
    print("getting id2name...")
    id2name = {}
    item_ids = []
    all_scenes = SCENE_LIST
    print("Number of scenes: ", len(all_scenes))
    for scene_id in tqdm(all_scenes):
        id2name[scene_id] = {}
        aggr_file = json.load(open(AGGR_JSON.format(scene_id, scene_id)))
        for item in aggr_file["segGroups"]:
            item_ids.append(int(item["id"]))
            id2name[scene_id][int(item["id"])] = item["label"]
    
    return id2name

def get_label_info():
    label2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
        'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
        'refridgerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}
    # mapping
    scannet_labels = label2class.keys() 
    scannet2label = {label: i for i, label in enumerate(scannet_labels)}
    class2label = {i: label for i, label in enumerate(scannet_labels)}

    lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
    lines = lines[1:]
    raw2label = {}
    for i in range(len(lines)):
        label_classes_set = set(scannet_labels)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2label[raw_name] = scannet2label['others']
        else:
            raw2label[raw_name] = scannet2label[nyu40_name]

    return raw2label, label2class, class2label # raw class name to nyu40 class idx, nyu40 class idx to nyu40 class name

all_matches_count = len(am)
correct_match_count = 0
num_pixels_in_view_per_object = {}
pixel_ratios_per_object = {}
all_ratios = []
stats_info = []

id2name = get_id2name_file()
raw2label, label2class, class2label = get_label_info()    # raw class name to nyu40 class labels, nyu40 class label to nyu40 class idx

sampling_frequency = 1

num_skipped = 0

num_incorrect = 0

for ix, matched_item in tqdm(enumerate(am)):
    # if ix == 100:
    #     break;
    # matched information
    ann_id = matched_item['ann_id']
    scene_id = matched_item['scene_id']
    if mode == 'match':
        frame_id = int(matched_item['frame_idx']) // sampling_frequency
    object_id = int(matched_item['object_id'])
    object_name = matched_item['object_name']
    if mode == 'match':
        angle_difference = matched_item['angle_distance']
        center_difference = matched_item['center_distance']

    # gt information
    try:
        if mode == 'match':
            gt_info = pickle.load(open(GT_PATH.format(scene_id, frame_id), 'rb'))
        if mode == 'render':
            gt_info = pickle.load(open(GT_PATH.format(scene_id, scene_id, object_id, ann_id), 'rb'))
        if mode == 'topdown':
            gt_info = pickle.load(open(GT_PATH.format(scene_id, scene_id), 'rb'))
        if mode == 'estimated':
            gt_info = pickle.load(open(GT_PATH.format(scene_id, scene_id, object_id), 'rb'))

    except FileNotFoundError:
        print("Handling exception for scene {} and frame {}")
        num_skipped += 1
        continue

    gt_object_id_list = [int(item['object_id']) for item in gt_info]
    
    # compare
    if object_id in gt_object_id_list:
        correct_match_count += 1
    else:
        if mode == 'match':
            ignored_frames.append('{}-{}_{}'.format(scene_id, frame_id, object_id))
        if mode == 'render':
            ignored_renders.append('{}-{}_{}'.format(scene_id, object_id, ann_id))

        num_incorrect += 1
    
    # for pixel distribution
    try:
        if mode == 'match':
            ll = Image.open(LABEL_IMG.format(scene_id, frame_id))
        if mode == 'render':
            ll = Image.open(LABEL_IMG.format(scene_id, scene_id, object_id, ann_id))
        if mode == 'topdown':
            ll = Image.open(LABEL_IMG.format(scene_id, scene_id))
        if mode == 'estimated':
            ll = Image.open(LABEL_IMG.format(scene_id, scene_id, object_id))

        wid, hei = ll.size
        label_img = np.array(ll)
    except FileNotFoundError as fe:
        print("Missing label. ", fe)
        exit(0)
        # num_skipped += 1
        continue

    unique, counts = np.unique(label_img, return_counts=True)

    # object id -> count
    label_counts = dict(zip(unique, counts))

    raw_name = id2name[scene_id][object_id]     # object id to raw class name
    if raw_name in ["floor", "wall", "ceiling"]: continue
    sem_class = raw2label[raw_name] # used for saving the stat results, because we want the stat based on ScanRefer # semantic idx
    sem_label = class2label[sem_class] # semantic name

    # label counts has key values where keys are the object ids
    try:
        num_pixels_with_object_id = label_counts[object_id+1]
    except KeyError as ke:
        # print("K erro 1")
        # num_skipped += 1
        continue
    try:
        num_pixels_in_view_per_object[sem_label].append(num_pixels_with_object_id)
        pixel_ratios_per_object[sem_label].append(num_pixels_with_object_id / (wid * hei))
        all_ratios.append(round(num_pixels_with_object_id / (wid * hei) * 100, 2))
        if mode == 'match':
            stats_info.append({
                'ann_id': ann_id,
                'scene_id': scene_id, 
                'frame_id': frame_id, 
                'object_id': object_id, 
                'object_name': object_name, 
                'pixel_ratio': round(num_pixels_with_object_id / (wid * hei) * 100, 2), 
                'angle_difference': angle_difference, 
                'center_difference': center_difference
                })
        
        if mode == 'render':
            stats_info.append({
                'ann_id': ann_id,
                'scene_id': scene_id, 
                'object_id': object_id, 
                'object_name': object_name, 
                'pixel_ratio': round(num_pixels_with_object_id / (wid * hei) * 100, 2), 
                })

    except KeyError:
        # print("K erro 2")
        num_pixels_in_view_per_object[sem_label] = []
        pixel_ratios_per_object[sem_label] = []

all_matches_count = all_matches_count - num_skipped

# with open(ignored_frames_path, 'w') as f:
#     json.dump(ignored_frames, f)

# print("Dropped ignored.")
# print("Num incorrect: ", num_incorrect)

# exit(0)
print("num_skipped: ", num_skipped)
accuracy = correct_match_count / all_matches_count
print("Matching accuracy: ", accuracy)

avg_pixel_count_per_object = {object_id: round(sum(arr)/(len(arr) + 1), 2) for object_id, arr in num_pixels_in_view_per_object.items()}
print("average pixel count per semantic label: ", avg_pixel_count_per_object)

avg_pixel_ratio_per_object = {object_id: round(sum(arr)/(len(arr) + 1), 2) * 100 for object_id, arr in pixel_ratios_per_object.items()}
print("average pixel ratio per semantic label: ", avg_pixel_ratio_per_object)

# sort based on object category
from collections import OrderedDict

avg_pixel_ratio_per_object = OrderedDict(sorted(avg_pixel_ratio_per_object.items()))

stats_file = os.path.join(stats_directory, exp_name + '.json')
stats_avg_file = os.path.join(stats_directory, exp_name + '_avgs.json')

import matplotlib.pyplot as plt
import numpy as np

N = 1
pos = np.arange(len(avg_pixel_ratio_per_object.keys()))
width = 0.55     # the width of the bars

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111)
rects1 = ax.bar(avg_pixel_ratio_per_object.keys(), avg_pixel_ratio_per_object.values(), width, color='#E91E63')

ax.set_ylabel('Average Object Pixel Ratio', fontsize=12.0, fontweight='bold')
ax.set_xlabel('Object Category', fontsize=12.0, fontweight='bold')
ax.set_xticks(pos)
ax.set_xticklabels(avg_pixel_ratio_per_object.keys())
plt.xticks(rotation = 90)
plt.grid(axis='y', alpha=0.35)

# ax.legend( (rects1[0]), ('pixel_ratio'))

os.makedirs('/project/3dlg-hcvc/scan2cap/code/Scan2Cap/utils/vp_matching/plots/{}'.format(exp_name), exist_ok=True)
plt.savefig('/project/3dlg-hcvc/scan2cap/code/Scan2Cap/utils/vp_matching/plots/{}/ratio_per_object.pdf'.format(exp_name))

plt.clf()

bins=[i for i in range(0, 100, 5)]
la, bins, _ = plt.hist([i for i in all_ratios], density=False, bins=bins, color='#0504aa', rwidth=0.85, alpha=0.7)  # `density=False` would make counts
plt.xticks(bins)
plt.ylabel('Number of Frames', fontsize=12.0, fontweight='bold')
plt.xlabel('Object Pixel Ratio', fontsize=12.0, fontweight='bold')
plt.grid(axis='y', alpha=0.35)
plt.show()
plt.savefig('/project/3dlg-hcvc/scan2cap/code/Scan2Cap/utils/vp_matching/plots/{}/frame_density.pdf'.format(exp_name))

import numpy as np
indices = np.digitize([i for i in all_ratios], bins)
scored_objects = {}
for b_ix, bbb in enumerate(bins):
    target_indices = [s_ix for s_ix, ix in enumerate(indices) if (ix - 1) == b_ix]
    scored_objects[str(int(bbb))] = [stats_info[ix] for ix in target_indices]

# avg scores
avgs = {
    'pixel_ratio': round(sum([score for k, score in avg_pixel_ratio_per_object.items()]) / len(scored_objects.keys()), 2),
    'matching_accuracy': round(accuracy, 2)
}

# save stats

if not os.path.exists(stats_directory):
    os.makedirs(stats_directory)

with open(stats_file, 'w') as f:
    json.dump(scored_objects, f) 

with open(stats_avg_file, 'w') as f:
    json.dump(avgs, f) 


def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '{0:.2f}'.format(h),
                ha='center', va='bottom')

autolabel(rects1)

# plt.show()

# pos = np.arange(len(avg_pixel_count_per_object.keys()))
# width = 0.3  # gives histogram aspect to the bar diagram

# fig = plt.figure()
# ax = fig.add_subplot(111)


# yvals = [4, 9, 2]
# rects1 = ax.bar(ind, yvals, width, color='r')
# ax.set_xticks(pos + width)
# ax.set_xticklabels(avg_pixel_count_per_object.keys())

# # plt.bar(avg_pixel_count_per_object.keys()-0.3, avg_pixel_count_per_object.values(), width, color='g')
# plt.bar(avg_pixel_ratio_per_object.keys(), avg_pixel_ratio_per_object.values(), width, color='b')
# plt.show()

print("completed!")

# mj = json.load(open(MATCHING_FILE, 'r'))

# frames = list([o['frame_idx'] for o in mj])
# unq_scenes = list(set([o['scene_id'] for o in mj]))
# unq_objects = list(set([o['object_name'] for o in mj]))

# print("Number of frames: ", len(frames))
# print("Number of unique scenes: ", len(unq_scenes))
# print("Number of unique objects: ", len(unq_objects))


# print("Average number of unique objects per scene: ", float(len(unq_objects)) / float(len(unq_scenes)))
# print("Average number of unique objects per frame: ", float(len(unq_objects)) / float(len(frames)))
# print("Average number of frames per scene: ", len(frames) / len(unq_scenes) )

