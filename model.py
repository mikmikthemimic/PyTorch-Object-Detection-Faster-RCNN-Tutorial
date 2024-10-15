import ast
import os
import pathlib

import neptune
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from pytorch_faster_rcnn_tutorial.backbone_resnet import ResNetBackbones
from pytorch_faster_rcnn_tutorial.datasets import (
    ObjectDetectionDataSet,
    ObjectDetectionDatasetSingle,
)
from pytorch_faster_rcnn_tutorial.faster_RCNN import get_faster_rcnn_resnet
from pytorch_faster_rcnn_tutorial.transformations import (
    ComposeDouble,
    ComposeSingle,
    FunctionWrapperDouble,
    FunctionWrapperSingle,
    apply_nms,
    apply_score_threshold,
    normalize_01,
)
from pytorch_faster_rcnn_tutorial.utils import (
    collate_single,
    get_filenames_of_path,
    save_json,
)

from bnd import (
    check_overlap,
    get_light,
)

params = {
    "EXPERIMENT": "GMT3-488",
    "OWNER": "mikmikthemimic",
    #"INPUT_DIR": "pytorch_faster_rcnn_tutorial/data/heads/test",
    "PREDICTIONS_PATH": "predictions",
    "MODEL_DIR": "pytorch_faster_rcnn_tutorial/model/model_statedict_GMT3-488_2024_08_06.pt",
    "DOWNLOAD": False,
    "DOWNLOAD_PATH": "pytorch_faster_rcnn_tutorial/model",
    "PROJECT": "GM-Thesis3",

    "ENSEMBLE": False,
}

# color mapping
color_mapping = {
    1: "blue",
    2: "green",
    3: "white",
    4: "yellow",
    5: "red"
}

class my_model():
    #model = None
    def __init__(self):
        device = torch.device("cpu")
        project_name = f'{params["OWNER"]}/{params["PROJECT"]}'
        print(f"Project: {project_name}")

        parameters = {
            'ACCELERATOR': 'cuda',
            'ANCHOR_SIZE': '((32,), (64,), (128,), (256,))',
            'ASPECT_RATIOS': '((0.5, 1.0, 2.0),)',
            'BACKBONE': 'ResNetBackbones.RESNET34',
            'BATCH_SIZE': 20,
            'CACHE': False,
            'CLASSES': 6,
            'FAST_DEV_RUN': False,
            'FOLDS': 5,
            'FPN': True,
            'IMG_MEAN': '[0.485, 0.456, 0.406]',
            'IMG_STD': '[0.229, 0.224, 0.225]',
            'IOU_THRESHOLD': 0.6,
            'LOG_MODEL': True,
            'LR': 0.0025,
            'MAXEPOCHS': 12,
            'MAX_SIZE': 1025,
            'MIN_SIZE': 1024,
            'PATIENCE': 50,
            'PRECISION': 32,
            'SAVE_DIR': '/content/PyTorch-Object-Detection-Faster-RCNN-Tutorial',
            'SEED': 42,
            'iou_threshold': 0.6,
            'lr': 0.0025,
            'model': None
        }

        # rcnn transform
        transform = GeneralizedRCNNTransform(
            min_size=int(parameters["MIN_SIZE"]),
            max_size=int(parameters["MAX_SIZE"]),
            image_mean=ast.literal_eval(parameters["IMG_MEAN"]),
            image_std=ast.literal_eval(parameters["IMG_STD"]),
            box_nms_thresh = 0.6
        )

        download_path = pathlib.Path(os.getcwd()) / params["DOWNLOAD_PATH"]
        model_name = "model_statedict_GMT3-488_2024_08_06.pt"
        checkpoint = torch.load(
            download_path / model_name, map_location=device
        )

        model = get_faster_rcnn_resnet(
            num_classes=int(parameters["CLASSES"]),
            backbone_name=ResNetBackbones(parameters["BACKBONE"].split(".")[1].lower()),  # reverse look-up enum
            anchor_size=ast.literal_eval(parameters["ANCHOR_SIZE"]),
            aspect_ratios=ast.literal_eval(parameters["ASPECT_RATIOS"]),
            fpn=ast.literal_eval(str(parameters['FPN'])),
            min_size=int(parameters["MIN_SIZE"]),
            max_size=int(parameters["MAX_SIZE"]),
        )
        
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        model.load_state_dict(checkpoint)
        model.eval()

        print("Model loaded")
        self.model = model
    
    def give_model(self):
        return self.model
    #return model

def get_data(input_path, model):
    # transformations
    transforms = ComposeSingle(
        [
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
            FunctionWrapperSingle(normalize_01),
        ]
    )
    dataset = ObjectDetectionDatasetSingle(
        inputs=[pathlib.Path(input_path)],
        transform=transforms,
        use_cache=False,
    )

    # create dataloader
    dataloader_prediction = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_single,
    )

    return predict(dataloader_prediction, input_path, model)
# 
    
def predict(data_input, input_path, model):
    model = model.give_model()
    for sample in data_input:
        x, x_name = sample
        print(x_name)
        with torch.no_grad():
            pred = model(x)
            # Move tensors to CPU before converting to NumPy
            pred = {key: value.cpu().numpy() for key, value in pred[0].items()}
            name = pathlib.Path(x_name[0])
            save_dir = pathlib.Path(os.getcwd()) / params["PREDICTIONS_PATH"]
            save_dir.mkdir(parents=True, exist_ok=True)
            pred_list = {
                key: value.tolist() for key, value in pred.items()
            }  # numpy arrays are not serializable -> .tolist()
            #print(sample)
            light_status = get_light(input_path)

            for p in range(len(pred_list['labels'])):
                pred_list['labels'][p] = check_overlap(pred_list['boxes'][p], pred_list['labels'][p], light_status)

            filename = name.with_suffix(".json")
            save_json(pred_list, path=save_dir / filename)

        output = (save_dir/filename)
        return output
            