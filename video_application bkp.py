import glob
import pathlib
import ast
import os
import shutil
import dearpygui.dearpygui as dpg
import os
import cv2

import numpy as np
import torch
import json
import math

from torchvision.ops import nms
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from pytorch_faster_rcnn_tutorial.backbone_resnet import ResNetBackbones
from pytorch_faster_rcnn_tutorial.datasets import (
    ObjectDetectionDatasetSingle,
)
from pytorch_faster_rcnn_tutorial.faster_RCNN import get_faster_rcnn_resnet
from pytorch_faster_rcnn_tutorial.transformations import (
    ComposeSingle,
    FunctionWrapperSingle,
    normalize_01,
)
from pytorch_faster_rcnn_tutorial.utils import (
    collate_single,
    get_filenames_of_path,
    save_json,
)

from bnd import (
    new_check_overlap,
    get_light_from_image,
)

# color mapping
color_mapping = {
    1: "blue",
    2: "green",
    3: "white",
    4: "yellow",
    5: "red"
}

dpg.create_context()
global model

class FinalModel():
    def __init__(self):
        # device = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        parameters = {
            "DOWNLOAD_PATH": "pytorch_faster_rcnn_tutorial/model",
            'ANCHOR_SIZE': '((32,), (64,), (128,), (256,))',
            'ASPECT_RATIOS': '((0.5, 1.0, 2.0),)',
            'BACKBONE': 'ResNetBackbones.RESNET34',
            'CLASSES': 6,
            'FPN': True,
            'MAX_SIZE': 1025,
            'MIN_SIZE': 1024,
        }

        download_path = pathlib.Path(os.getcwd()) / parameters["DOWNLOAD_PATH"]
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
            device=device,
            map_location=device,
        ).to(device)
        
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        model.load_state_dict(checkpoint)
        model.eval()
        # model.to(device)

        print("Model loaded")
        self.model = model

        # transformations
        self.transforms = ComposeSingle(
            [
                FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
                FunctionWrapperSingle(normalize_01),
            ]
        )
    
    def give_model(self):
        return self.model
    #return model

    #def get_data(self, image_data: np.ndarray):
    #    # Check if input is a directory
    #    if os.path.isdir(input_path):
    #        input_path = [file for file in get_filenames_of_path(pathlib.Path(input_path))]
    #        print(input_path)
    #        print(len(input_path))
    #    else:
    #        input_path = [pathlib.Path(input_path)]
#
#
    #    dataset = ObjectDetectionDatasetSingle(
    #        inputs=input_path,
    #        transform=transforms,
    #        use_cache=False,
    #    )
#
    #    # create dataloader
    #    dataloader_prediction = DataLoader(
    #        dataset=dataset,
    #        batch_size=1,
    #        shuffle=False,
    #        num_workers=0,
    #        collate_fn=collate_single,
    #    )
#
    #    return predict(dataloader_prediction, input_path, model)
# 
    
    def predict(self, image_data: np.ndarray):
        output = []
        image = self.transforms(image_data)
        image = torch.from_numpy(image).to(dtype=torch.float32)
        image = image.unsqueeze(0)

        light_status = get_light_from_image(image_data, (1132, 203, 1140, 224))

        #if not os.path.exists('temp/predictions'):
        #    os.makedirs('temp/predictions')
#
        #for file in os.listdir('temp/predictions'):
        #    os.remove(f'temp/predictions/{file}')
        with torch.no_grad():
            prediction = self.model(image)

        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']

        labels = new_check_overlap(boxes, labels, light_status)

        return { 'boxes': boxes, 'labels': labels, 'scores': scores }

        #for i, sample in enumerate(data_input):
        #    x, x_name = sample
        #    with torch.no_grad():
        #        pred = model(x)
        #        # Move tensors to CPU before converting to NumPy
        #        pred = {key: value.cpu().numpy() for key, value in pred[0].items()}
        #        name = pathlib.Path(x_name[0])
        #        save_dir = pathlib.Path(os.getcwd()) / 'temp' / 'predictions'
        #        pred_list = {
        #            key: value.tolist() for key, value in pred.items()
        #        }  # numpy arrays are not serializable -> .tolist()
        #        #print(sample)
        #        light_status = get_light(input_path[i].as_posix())
#
        #        for p in range(len(pred_list['labels'])):
        #            pred_list['labels'][p] = check_overlap(pred_list['boxes'][p], pred_list['labels'][p], light_status)
#
        #        filename = name.with_suffix(".json")
        #        save_json(pred_list, path=save_dir / filename)
#
        #        output.append(save_dir/filename)
        #        video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (1280, 720))
        #        img = load_predicted_images([save_dir/filename], [pathlib.Path(os.getcwd()) / 'temp' / 'frames' / name.with_suffix(".jpg")])
        #        video.write(img)
        #return output

labels = {
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Bicycle',
    4: 'Vehicle-Violator',
    5: 'Pedestrian-Violator',
    6: 'Bicycle-Violator',
    99: 'Unknown',
}

if not os.path.exists('output'):
    os.makedirs('output')

def load_predicted_images(
        predictions,
        image_data,
    ):
    # Constants
    IOU_THRESHOLD = 0.65
    SCORE_THRESHOLD = 0

    USE_NMS = True
    #

    image = image_data

    if USE_NMS:
        indices = nms(predictions['boxes'], predictions['scores'], iou_threshold=IOU_THRESHOLD)
        indices = (np.array(indices.cpu().numpy()),)

        predictions['boxes'] = np.asarray(predictions['boxes'].cpu().numpy())[indices]
        predictions['labels'] = np.asarray(predictions['labels'].cpu().numpy())[indices]
        predictions['scores'] = np.asarray(predictions['scores'].cpu().numpy())[indices]

    for i in range(len(predictions['boxes'])):
        if (predictions['labels'][i] == 1 or predictions['labels'][i] == 4):
            SCORE_THRESHOLD = 0.5

        elif (predictions['labels'][i] == 2 or predictions['labels'][i] == 5):
            SCORE_THRESHOLD = 0.2

        elif (predictions['labels'][i] == 3 or predictions['labels'][i] == 6):
            SCORE_THRESHOLD = 0.2
        else:
            print(f"Label: {predictions['labels'][i]}, iteration: {i}")

        if predictions['scores'][i] < SCORE_THRESHOLD:
            continue

        label = labels[predictions['labels'][i].item()]
        box = [int(v) for v in predictions['boxes'][i]]

        match label:
            case 'Vehicle':
                bnd_color = (0, 255, 0)
                text_color = (1, 250, 32)
            case 'Pedestrian':
                bnd_color = (0, 255, 0)
                text_color = (1, 250, 32)
            case 'Bicycle':
                bnd_color = (0, 255, 0)
                text_color = (1, 250, 32)
            case 'Vehicle-Violator':
                bnd_color = (0, 0, 139)
                text_color = (0, 0, 255)
            case 'Pedestrian-Violator':
                bnd_color = (0, 0, 139)
                text_color = (0, 0, 255)
            case 'Bicycle-Violator':
                bnd_color = (0, 0, 139)
                text_color = (0, 0, 255)
            case 'Outside':
                bnd_color = (0, 0, 0)
                text_color = (36, 255, 12)
            case _:
                bnd_color = (255, 0, 0)
                text_color = (36, 255, 12)

        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), bnd_color, 1)
        cv2.putText(image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    cv2.imshow('image', image)
    cv2.waitKey(1)
    #return image

    if dpg.does_item_exist("texture_tag"):
        print("Deleting existing image")
        dpg.delete_item("image_tag")
        dpg.delete_item("texture_tag")
    #image = cv2.imwrite(f'temp/output/{filename}.jpg', image_data)
    #add_and_load_image(f'temp/output/{filename}.jpg', parent="Primary Window")
    add_and_load_image(image, parent="Primary Window")
    return image

# Callbacks
def select_file(sender, app_data, user_data):
    selections = app_data.get("selections")
    file_path = next(iter(selections.values()))
    print(f"App Data: {app_data}")
    print(f"Selected: {file_path}")

    file_name = next(iter(selections.keys()))

    cap = cv2.VideoCapture(file_path)
    video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (1280, 720))
    while True:
        ret, frame = cap.read()
        frame_count = 0
        print("Reading video")
        if not ret:
            break
        predictions = model.predict(image_data=frame)
        image = load_predicted_images(predictions, frame)
        video.write(image)
        frame_count += 1
    print("Finished reading video, total frames: ", frame_count)
    cap.release()
    


    # check if the file is a video
    #if file_name.endswith('.mp4'):
    #    # Create a temp folder to store the frames
    #    if not os.path.exists('temp/frames'):
    #        os.makedirs('temp/frames')
    #    
    #    # Ensures the temp folder is empty
    #    for file in os.listdir('temp/frames'):
    #        os.remove(f'temp/frames/{file}')
    #    
    #    cap = cv2.VideoCapture(file_path)
    #    frame_count = 0
    #    print("Reading video")
    #    while True:
    #        ret, frame = cap.read()
    #        if not ret:
    #            break
    #        cv2.imwrite(f'temp/frames/frame_{frame_count:06d}.jpg', frame)
    #        frame_count += 1
    #    print("Finished reading video, total frames: ", frame_count)
#
    #    cap.release()
    #    output = get_data(os.getcwd() + '/temp/frames', model)
    #    # load_predicted_images(output, glob.glob(os.getcwd() + '/temp/frames/*.jpg'))
    #else:
    #    # Copy item to temp/frames folder
    #    if not os.path.exists('temp/frames'):
    #        os.makedirs('temp/frames')
#
    #    shutil.copy(file_path, f'temp/frames/{os.path.basename(file_path)}')
    #    output = get_data(file_path, model)
        #load_predicted_images(output, [file_path])

# Lambdas
def add_and_load_image(image_path, parent=None):
    width, height, channels, data = dpg.load_image(image_path)

    with dpg.texture_registry() as reg_id:
        texture_id = dpg.add_raw_texture(width, height, data, parent=reg_id, tag="texture_tag")
        dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag2", format=dpg.mvFormat_Float_rgb)

    if parent is None:
        return dpg.add_image(texture_id, tag="image_tag")
    else:
        return dpg.add_image(texture_id, parent=parent, tag="image_tag")

with dpg.file_dialog(
    tag="file_dialog_id",
    directory_selector=False,
    show=False,
    file_count=1,
    height=300,
    callback=select_file
):
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
    dpg.add_file_extension(".png", color=(0, 255, 0, 255))
    dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))

    dpg.add_file_extension(".mp4", color=(127, 255, 255, 255))

# Main Window
with dpg.window(
    label="Main Window",
    tag="Primary Window",
    width=1280,
    height=840,
    no_resize=True,
    no_collapse=True,
    no_move=True,
    no_scroll_with_mouse=False,
    no_scrollbar=False,
    no_title_bar=True,
    no_background=False,
):
    model = FinalModel()
    dpg.add_button(label="Select Video", callback=lambda: dpg.show_item("file_dialog_id"))

# Setup
dpg.create_viewport(
    title="Test", width=1280, height=840, x_pos=0, y_pos=0, resizable=False
)

dpg.setup_dearpygui()
dpg.show_viewport()
#dpg.start_dearpygui()
while dpg.is_dearpygui_running():
    #ret, frame = cap.read()
    #data = np.asfarray(frame, dtype='f')
    #texture = dpg.create_texture((1280, 720), data)
    #dpg.set_value("texture_tag", texture)
    dpg.render_dearpygui_frame()
dpg.destroy_context()
