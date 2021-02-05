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
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        strict=True,
        verbose=False,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filepath=os.path.join(
            cfg.task_data_model.name, "{epoch}-{val_loss:.2f}-{val_acc:.3f}-{point_acc:.2f}-{voxel_acc:.2f}"
        ),
        verbose=True,
    )
    tb_logger = pl_loggers.TensorBoardLogger('logs/', name=cfg['exp_name'])
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=3,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.distrib_backend,
        logger=tb_logger
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
