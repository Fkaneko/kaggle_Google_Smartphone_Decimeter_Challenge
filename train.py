import sys
from typing import Any, List

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset.datamodule import GsdcDatamodule
from src.modeling.pl_model import LitModel
from src.utils.util import set_random_seed

try:
    LOGGER = "neptune"
    from neptune.new.integrations.pytorch_lightning import NeptuneLogger
except Exception:
    print("use TensorBoardLogger")
    LOGGER = "tensorboard"


@hydra.main(config_path="./src/config", config_name="config")
def main(conf: DictConfig) -> None:

    set_random_seed(conf.seed)

    datamodule = GsdcDatamodule(
        conf=conf,
        val_fold=conf.val_fold,
        batch_size=conf.batch_size,
        aug_mode=conf.aug_mode,
        num_workers=conf.num_workers,
        is_debug=conf.is_debug,
    )

    datamodule.prepare_data()
    print("\t\t ==== TRAIN MODE ====")
    datamodule.setup(stage="fit")
    print(
        "training samples: {}, valid samples: {}".format(
            len(datamodule.train_dataset), len(datamodule.val_dataset)
        )
    )

    if conf.ckpt_path is not None:
        model = LitModel.load_from_checkpoint(
            conf.ckpt_path,
            conf=conf,
            dataset_len=len(datamodule.train_dataset),
            logger_name=LOGGER,
        )
    else:
        model = LitModel(
            conf=conf, dataset_len=len(datamodule.train_dataset), logger_name=LOGGER,
        )

    pl.trainer.seed_everything(seed=conf.seed)
    if LOGGER == "tensorboard":
        logger = TensorBoardLogger("tb_logs", name="my_model")
    elif LOGGER == "neptune":
        logger = NeptuneLogger(
            project="your_project_name",
            name="lightning-run",  # Optional
            mode="debug" if conf.is_debug else "async",
        )
        logger.experiment["params/conf"] = conf
        if conf.nept_tags[0] is not None:
            logger.experiment["sys/tags"].add(list(conf.nept_tags))

    trainer_params = OmegaConf.to_container(conf.trainer)
    trainer_params["callbacks"] = get_callbacks(monitor=conf.monitor)
    trainer_params["logger"] = logger
    trainer = pl.Trainer(**trainer_params)

    # Run lr finder
    if conf.find_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
        lr_finder.plot(suggest=True)
        plt.savefig("./lr_finder.png")
        plt.show()
        sys.exit()

    # Run Training
    trainer.fit(model, datamodule=datamodule)


def get_callbacks(
    ema_decay: float = 0.9, monitor: str = "val_loss", mode: str = "min"
) -> list:
    callbacks: List[Any] = []
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor, save_last=True, mode=mode, verbose=True,
    )
    callbacks.append(checkpoint_callback)
    return callbacks


if __name__ == "__main__":
    main()
