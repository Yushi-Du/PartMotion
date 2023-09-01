from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils
from ipdb import set_trace

log = utils.get_logger(__name__)


def test(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Evaluate model on test set after training
    if not config.trainer.get("fast_dev_run"):

        # 原版就被注释掉了
        # log.info(f"Loading from {config.trainer.resume_from_checkpoint}")
        # model.load_from_checkpoint(config.trainer.resume_from_checkpoint, **config.model)
        # log.info("Starting testing!")
        # trainer.test(model=model, datamodule=datamodule)

        log.info(f"Loading from {config.trainer.resume_from_checkpoint}")
        ckpt = torch.load(config.trainer.resume_from_checkpoint)
        model.load_state_dict(ckpt["state_dict"])
        model = model.eval().cuda()
        log.info("Starting testing!")

        trainer.test(model=model, datamodule=datamodule)
        # 原版就被注释掉了
        # datamodule.setup()
        # results = []
        # for data in datamodule.test_dataloader():
        #     for k, v in data.items():
        #         if isinstance(v, torch.Tensor):
        #             data[k] = v.cuda()
        #     results.append(model.test_step(data, None))
        #     import pdb; pdb.set_trace()

        # model.test_epoch_end(results)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]