# import ast
# import os
# import pathlib
# 
# import neptune
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
# 
# from pytorch_faster_rcnn_tutorial.backbone_resnet import ResNetBackbones
# from pytorch_faster_rcnn_tutorial.datasets import (
#     ObjectDetectionDataSet,
#     ObjectDetectionDatasetSingle,
# )
# from pytorch_faster_rcnn_tutorial.faster_RCNN import get_faster_rcnn_resnet
# from pytorch_faster_rcnn_tutorial.transformations import (
#     ComposeDouble,
#     ComposeSingle,
#     FunctionWrapperDouble,
#     FunctionWrapperSingle,
#     apply_nms,
#     apply_score_threshold,
#     normalize_01,
# )
from pytorch_faster_rcnn_tutorial.utils import (
    collate_single,
    get_filenames_of_path,
    save_json,
)

# params = {
#     "EXPERIMENT": "GMT3-488",
#     "OWNER": "mikmikthemimic",
#     #"INPUT_DIR": "pytorch_faster_rcnn_tutorial/data/heads/test",
#     "PREDICTIONS_PATH": "predictions",
#     "MODEL_DIR": "/content/PyTorch-Object-Detection-Faster-RCNN-Tutorial/pytorch_faster_rcnn_tutorial/model/kfolds/fold_4.ckpt",
#     "DOWNLOAD": False,
#     "DOWNLOAD_PATH": "pytorch_faster_rcnn_tutorial/model/kfolds",
#     "PROJECT": "GM-Thesis3",
# 
#     "ENSEMBLE": False,
# }
# 
# # color mapping
# color_mapping = {
#     1: "blue",
#     2: "green",
#     3: "white",
#     4: "yellow",
#     5: "red"
# }

class Model():
    # THIS IS FOR VIDEO INPUT BUT WE ARE DOING IMAGE INPUT NALANG
    def get_input(image_path : str = ""):
        inputs = get_filenames_of_path(pathlib.Path(image_path))
        inputs.sort()
        print(f"Found {len(inputs)} files in {image_path}")

        return inputs

    # # transformations
    # transforms = ComposeSingle(
    #     [
    #         FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
    #         FunctionWrapperSingle(normalize_01),
    #     ]
    # )
# 
    # dataset = ObjectDetectionDatasetSingle(
    #     inputs=inputs,
    #     transform=transforms,
    #     use_cache=False,
    # )
# 
    # # import experiment from neptune
    # project_name = f'{params["OWNER"]}/{params["PROJECT"]}'
    # print(f"Project: {project_name}")
    # project = neptune.init_run(
    #     project=project_name,
    #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNmE3MzZjYS01MDUxLTQ2YjYtOGU3ZC04YzQwNjg4NmJiZDcifQ==",
    #     with_id=params["EXPERIMENT"],
    #     mode="read-only",
    # )  # get project
    # experiment_id = params["EXPERIMENT"]  # experiment id
    # parameters = project['training/hyperparams'].fetch()
# 
    # # rcnn transform
    # transform = GeneralizedRCNNTransform(
    #     min_size=int(parameters["MIN_SIZE"]),
    #     max_size=int(parameters["MAX_SIZE"]),
    #     image_mean=ast.literal_eval(parameters["IMG_MEAN"]),
    #     image_std=ast.literal_eval(parameters["IMG_STD"]),
    #     box_nms_thresh = 0.6
    # )
# 
    # # download model from neptune or load from checkpoint
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 
    # if params["DOWNLOAD"]:
    #     download_path = pathlib.Path(os.getcwd()) / params["DOWNLOAD_PATH"]
    #     download_path.mkdir(parents=True, exist_ok=True)
    #     if params["ENSEMBLE"]:
    #         model_name = "ensemble_model.pt"  # that's how I called the best model
    #         # model_name = properties['checkpoint_name']  # logged when called log_model_neptune()
    #         if not (download_path / model_name).is_file():
    #             project['artifacts/ensemble_model'].download(
    #                 destination=download_path.as_posix()
    #             )  # download model
# 
    #         model_state_dict = torch.load(
    #             download_path / model_name, map_location=device
    #         )
    #     else:
    #         #new change   08/06/2024      updated to load checkpoint
    #         model_name = "fold_4.ckpt"  # that's how I called the best model                      best-model.pt
    #         # model_name = properties['checkpoint_name']  # logged when called log_model_neptune()
    #         if not (download_path / model_name).is_file():
    #             project['artifacts/fold_4.ckpt'].download(
    #                 destination=download_path.as_posix()
    #             )  # download model
# 
    #         model_state_dict = torch.load(
    #             download_path / model_name, map_location=device
    #         )
    # else:
    #     checkpoint = torch.load(params["MODEL_DIR"], map_location=device)
    #     print(checkpoint.keys())
# 
    # model = get_faster_rcnn_resnet(
    # num_classes=int(parameters["CLASSES"]),
    # backbone_name=ResNetBackbones(parameters["BACKBONE"].split(".")[1].lower()),  # reverse look-up enum
    # anchor_size=ast.literal_eval(parameters["ANCHOR_SIZE"]),
    # aspect_ratios=ast.literal_eval(parameters["ASPECT_RATIOS"]),
    # fpn=ast.literal_eval(str(parameters['FPN'])),
    # min_size=int(parameters["MIN_SIZE"]),
    # max_size=int(parameters["MAX_SIZE"]),
    # )
    #  
    # if 'state_dict' in checkpoint.keys():
    #     checkpoint = checkpoint['state_dict']
# 
    # model.load_state_dict(model_state_dict)
# 
    # # inference (cpu)
    # model.eval()
    # model.to(device)
    # for sample in dataloader_prediction:
    #     x, x_name = sample
    #     with torch.no_grad():
    #         pred = model(x)
    #         # Move tensors to CPU before converting to NumPy
    #         pred = {key: value.cpu().numpy() for key, value in pred[0].items()}
    #         name = pathlib.Path(x_name[0])
    #         save_dir = pathlib.Path(os.getcwd()) / params["PREDICTIONS_PATH"]
    #         save_dir.mkdir(parents=True, exist_ok=True)
    #         pred_list = {
    #             key: value.tolist() for key, value in pred.items()
    #         }  # numpy arrays are not serializable -> .tolist()
    #         save_json(pred_list, path=save_dir / name.with_suffix(".json"))
