import logging
import pathlib
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import albumentations
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from pydantic import BaseSettings, Field
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
import neptune
from lightning.pytorch.loggers import NeptuneLogger
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN

from pytorch_faster_rcnn_tutorial.backbone_resnet import ResNetBackbones
from pytorch_faster_rcnn_tutorial.datasets import ObjectDetectionDataSet
from pytorch_faster_rcnn_tutorial.faster_RCNN import (
    FasterRCNNLightning,
    get_faster_rcnn_resnet,
)
from pytorch_faster_rcnn_tutorial.transformations import (
    AlbumentationWrapper,
    Clip,
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
)
from pytorch_faster_rcnn_tutorial.utils import (
    collate_double,
    get_filenames_of_path,
    log_model_neptune,
)

logger = logging.getLogger(__name__)

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("execution_logs.log"),
    ]
)

# root directory (project directory)
ROOT_PATH: pathlib.Path = pathlib.Path(__file__).parent.absolute()


class NeptuneSettings(BaseSettings):
    """
    Reads the variables from the environment.
    Errors will be raised if the required variables are not set.
    """

    api_key: str = Field(default=..., env="NEPTUNE")
    OWNER: str = "mikmikthemimic"  # set your name here, e.g. mikmikthemimic
    PROJECT: str = "GM-Thesis3"  # set your project name here, e.g. GM-Thesis3
    EXPERIMENT: str = "GM-Thesis3"  # set your experiment name here, e.g. GM-Thesis3

    class Config:
        # this tells pydantic to read the variables from the .env file
        env_file = ".env"


@dataclass
class Parameters:
    """
    Dataclass for the parameters.
    """

    BATCH_SIZE: int = 6
    CACHE: bool = False
    SAVE_DIR: Optional[
        str
    ] = None  # checkpoints will be saved to cwd (current working directory) if None
    LOG_MODEL: bool = True  # whether to log the model to neptune after training
    ACCELERATOR: Optional[str] = "auto"  # set to "gpu" if you want to use GPU
    LR: float = 0.001
    PRECISION: int = 32
    CLASSES: int = 10
    SEED: int = 42
    MAXEPOCHS: int = 25  # 4
    PATIENCE: int = 50
    BACKBONE: ResNetBackbones = ResNetBackbones.RESNET34
    FPN: bool = False
    ANCHOR_SIZE: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),)
    ASPECT_RATIOS: Tuple[Tuple[float, ...]] = ((0.5, 1.0, 2.0),)
    MIN_SIZE: int = 1024
    MAX_SIZE: int = 1025
    IMG_MEAN: List = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMG_STD: List = field(default_factory=lambda: [0.229, 0.224, 0.225])
    IOU_THRESHOLD: float = 0.5
    FAST_DEV_RUN: bool = False

    def __post_init__(self):
        if self.SAVE_DIR is None:
            self.SAVE_DIR: str = str(pathlib.Path.cwd())

def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        val_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, val_indices))
    return folds

# BAWAL TO MAGING 1 ABIIII PLS ANTAGAL MONG NASTUCK DAHIL DITO OMG WAG MO NA ICHANGE EVER -past abi :> labyu
k = 2
reload_dataloaders_every_n_epochs = 5
def train():
    # environment variables (pydantic BaseSettings class)
    neptune_settings = NeptuneSettings()

    # parameters (dataclass)
    parameters = Parameters()

    # data path relative to this file (pathlib)
    data_path: pathlib.Path = (
        ROOT_PATH / "pytorch_faster_rcnn_tutorial" / "data" / "heads"
    )

    # input and target files
    inputs: List[pathlib.Path] = get_filenames_of_path(data_path / "input")
    targets: List[pathlib.Path] = get_filenames_of_path(data_path / "target")
    
    # test files
    tests: List[pathlib.Path] = get_filenames_of_path(data_path/"test")
    tests_targets: List[pathlib.Path] = get_filenames_of_path(data_path/"test_targets")

    # sort inputs and targets
    inputs.sort()
    targets.sort()

    # Get the fold indices
    fold_indices = kfold_indices(inputs, k)

    # mapping
    mapping: Dict[str, int] = {
        "Vehicle": 1,
        "Pedestrian": 2,
        "Enforcer": 3,
        "Vehicle-Violator":4,
        "Pedestrian-Violator": 5
    }

    # training transformations and augmentations
    transforms_training = ComposeDouble(
        [
            Clip(),
            AlbumentationWrapper(albumentations.HorizontalFlip(p=0.5)),
            AlbumentationWrapper(
                albumentations.RandomScale(p=0.5, scale_limit=0.5)
            ),
            # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(function=np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(function=normalize_01),
        ]
    )

    # validation transformations
    transforms_validation = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(function=np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(function=normalize_01),
        ]
    )

    # test transformations
    transforms_test = ComposeDouble(
        [
            Clip(),
            FunctionWrapperDouble(function=np.moveaxis, source=-1, destination=0),
            FunctionWrapperDouble(function=normalize_01),
        ]
    )

    # random seed (function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random)
    seed_everything(parameters.SEED)

    #training validation test split (manually)
    #inputs_train, inputs_valid, inputs_test = inputs[:12], inputs[12:16], inputs[16:]
    #targets_train, targets_valid, targets_test = (
    #    targets[:12],
    #    targets[12:16],
    #    targets[16:],
    #)

    # dataset test
    dataset_test = ObjectDetectionDataSet(
        inputs=tests,
        targets=tests_targets,
        transform=transforms_test,
        use_cache=parameters.CACHE,
        convert_to_format=None,
        mapping=mapping,
    )

    # dataloader test
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_double,
    )

    # neptune logger (neptune-client)
    neptune_logger = NeptuneLogger(
        api_key=neptune_settings.api_key,
        project=f"{neptune_settings.OWNER}/{neptune_settings.PROJECT}",  # use your neptune name here
        name=neptune_settings.PROJECT,
        log_model_checkpoints=False,
    )

    # log hyperparameters
    neptune_logger.log_hyperparams(asdict(parameters))

    # model init
    model: FasterRCNN = get_faster_rcnn_resnet(
        num_classes=parameters.CLASSES,
        backbone_name=parameters.BACKBONE,
        anchor_size=parameters.ANCHOR_SIZE,
        aspect_ratios=parameters.ASPECT_RATIOS,
        fpn=parameters.FPN,
        min_size=parameters.MIN_SIZE,
        max_size=parameters.MAX_SIZE,
    )

    # lightning model
    model: FasterRCNNLightning = FasterRCNNLightning(
        model=model, lr=parameters.LR, iou_threshold=parameters.IOU_THRESHOLD
    )

    # callbacks
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        monitor="Validation_mAP", mode="max"
    )
    learning_rate_callback: LearningRateMonitor = LearningRateMonitor(
        logging_interval="step", log_momentum=False
    )
    early_stopping_callback: EarlyStopping = EarlyStopping(
        monitor="Validation_mAP", patience=parameters.PATIENCE, mode="max"
    )

    # trainer init
    trainer: Trainer = Trainer(
        accelerator=parameters.ACCELERATOR,
        logger=neptune_logger,
        callbacks=[
            checkpoint_callback,
            learning_rate_callback,
            early_stopping_callback,
        ],
        default_root_dir=parameters.SAVE_DIR,  # where checkpoints are saved to
        log_every_n_steps=1,  # increase to reduce the amount of log flushes (lowers the overhead)
        num_sanity_val_steps=0,  # set to 0 to skip sanity check
        max_epochs=parameters.MAXEPOCHS,
        fast_dev_run=parameters.FAST_DEV_RUN,  # set to True to test the pipeline with one batch and without validation, testing and logging
    )
    for epoch in range(parameters.MAXEPOCHS):
        for train_indices, val_indices in fold_indices:
            input_train, targets_train = np.array(inputs)[train_indices.astype(int)], np.array(targets)[train_indices.astype(int)]
            input_val, targets_val = np.array(inputs)[val_indices.astype(int)], np.array(targets)[val_indices.astype(int)]

            # dataset train
            dataset_train = ObjectDetectionDataSet(
                inputs=input_train,
                targets=targets_train,
                transform=transforms_training,
                use_cache=parameters.CACHE,
                convert_to_format=None,
                mapping=mapping,
            )

            # dataset validation
            dataset_valid = ObjectDetectionDataSet(
                inputs=input_val,
                targets=targets_val,
                transform=transforms_validation,
                use_cache=parameters.CACHE,
                convert_to_format=None,
                mapping=mapping,
            )
            # dataloader train
            dataloader_train = DataLoader(
                dataset=dataset_train,
                batch_size=parameters.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_double,
            )

            # dataloader validation
            dataloader_valid = DataLoader(
                dataset=dataset_valid,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_double,
            )
            if not epoch % reload_dataloaders_every_n_epochs:
                train_loader = datamodule.dataloader_train()
            for batch in train_loader:
                # start training
                trainer.fit(
                    model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=dataloader_valid,
                )
        
    # find a way to set FAST_DEV_RUN to false to start the validation
    if not parameters.FAST_DEV_RUN:
        # start testing
        trainer.test(ckpt_path="best", dataloaders=dataloader_test)

        # log model
        if parameters.LOG_MODEL:
            checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
            save_directory = pathlib.Path(ROOT_PATH / "model")
            log_model_neptune(
                checkpoint_path=checkpoint_path,
                save_directory=save_directory,
                name="best_model.pt",
                neptune_logger=neptune_logger,
            )

    # stop logger
    neptune_logger.experiment.stop()
    logger.info("Training finished")
    
if __name__ == "__main__":
    train()
