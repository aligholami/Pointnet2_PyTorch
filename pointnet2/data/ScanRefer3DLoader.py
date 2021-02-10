from torch.utils.data import Dataset 
from pointnet2.utils.common import *
from pointnet2.data.ScanNet3DLoader import ScanNet3DDataset

class ScanRefer3DDataset(ScanNet3DDataset):
    
    def __init__(self, hparams, phase, target_samples, scene_list, transforms, num_classes=21, npoints=8192, is_weighting=True, use_color=False, use_normal=False, use_multiview=False):
        super().__init__(
            hparams=hparams,
            phase=phase,
            scene_list=scene_list,
            transforms=transforms,
            num_classes=num_classes,
            npoints=npoints,
            is_weighting=is_weighting,
            use_multiview=use_multiview,
            use_color=use_color,
            use_normal=use_normal
        )
        self.hparams = hparams
        self.sample_list = target_samples
        # self.load_scene_data()
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        scene_id = self.sample_list[index]['scene_id']
        object_id = self.sample_list[index]['object_id']
        object_name = " ".join(self.sample_list[index]["object_name"].split("_"))
        ann_id = self.sample_list[index]["ann_id"]
        description = self.sample_list[index]["description"]
        
        return description

    # def load_scene_data(self):
    #     print("Loading scene data.")
    #     # load scene data
    #     self.scene_data = {}
    #     for scene_id in self.scene_list:
    #         self.scene_data[scene_id] = {}
    #         # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
    #         self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
    #         self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
    #         self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
    #         # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
    #         self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

    #     # prepare class mapping
    #     lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
    #     lines = lines[1:]
    #     raw2nyuid = {}
    #     for i in range(len(lines)):
    #         elements = lines[i].split('\t')
    #         raw_name = elements[1]
    #         nyu40_name = int(elements[4])
    #         raw2nyuid[raw_name] = nyu40_name

    #     # store
    #     self.raw2nyuid = raw2nyuid
    #     self.raw2label = self._get_raw2label()
    #     self.unique_multiple_lookup = self._get_unique_multiple_lookup()
    #     return {}
    