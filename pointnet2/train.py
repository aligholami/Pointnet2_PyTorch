import os

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pointnet2.utils.common import hydra_params_to_dotdict
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


@hydra.main("config/config.yaml")
def main(cfg):
    print(cfg)
    exit(0)
    model = hydra.utils.instantiate(cfg.task_data_model, hydra_params_to_dotdict(cfg))
    early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        filepath=os.path.join(
            cfg.task_data_model.name, "{epoch}-{val_loss:.2f}-{val_acc:.3f}"
        ),
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.distrib_backend,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
