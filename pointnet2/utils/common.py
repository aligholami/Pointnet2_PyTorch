import omegaconf
import os

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