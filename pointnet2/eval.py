import os

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pointnet2.utils.common import hydra_params_to_dotdict
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


@hydra.main("config/config.yaml")
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_data_model, hydra_params_to_dotdict(cfg))
    trainer = pl.Trainer()
    trainer.test(model)

if __name__ == "__main__":
    main()
