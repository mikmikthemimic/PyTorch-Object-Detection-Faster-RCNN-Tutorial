import array
import json
import pathlib
import ast
import dearpygui.dearpygui as dpg
import os
import cv2

import numpy as np
import torch

from torchvision.ops import nms

from pytorch_faster_rcnn_tutorial.backbone_resnet import ResNetBackbones
from pytorch_faster_rcnn_tutorial.faster_RCNN import get_faster_rcnn_resnet
from pytorch_faster_rcnn_tutorial.transformations import (
    ComposeSingle,
    FunctionWrapperSingle,
    normalize_01,
)

from bnd import (
    new_check_overlap,
    get_light_from_image,
)

labels = {
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Bicycle',
    4: 'Vehicle-Violator',
    5: 'Pedestrian-Violator',
    6: 'Bicycle-Violator',
    99: 'Unknown',
}

dpg.create_context()
dpg.create_viewport(
    title="Test", width=1280, height=840, x_pos=0, y_pos=0, resizable=False
)
dpg.setup_dearpygui()

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

        self.transforms = ComposeSingle(
            [
                FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
                FunctionWrapperSingle(normalize_01),
            ]
        )
    
    def predict(self, image_data: np.ndarray):
        # image_path = "C:/Users/Angelo/Documents/GitHub/PyTorch-Object-Detection-Faster-RCNN-Tutorial/temp/frames/frame_000000.jpg"
        image = self.transforms(image_data)
        image = torch.from_numpy(image).to(dtype=torch.float32)
        image = image.unsqueeze(0)

        light_status = get_light_from_image(image_data, (1132, 203, 1140, 224))

        with torch.no_grad():
            prediction = self.model(image)

        boxes = prediction[0]['boxes']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']

        labels = new_check_overlap(boxes, labels, light_status)

        return { 'boxes': boxes, 'labels': labels, 'scores': scores }

def create_prediction_image(predictions, image_data):
    # Constants
    IOU_THRESHOLD = 0.4
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

        #roi_pedlane = [(46.80, 366.90), (17.30, 463.30), (199.67, 443.63), (342.79, 426.02), (720.40, 383.08), (1228.00, 333.00), (1194.90, 326.94), (1155.27, 320.33), (1092.51, 306.02), (1052.88, 296.11), (1031.96, 289.50), (680.70, 313.70), (687.38, 321.43), (692.88, 329.14), (692.90, 338.70), (681.50, 345.20), (664.80, 346.80), (647.70, 346.70), (625.90, 344.50), (607.30, 340.90), (590.20, 333.60), (570.30, 320.70), (326.27, 342.35)]
        roi_pedlane = [(45.41,364.32), (1015.1,281.1), (1157.8,329.7), (19.46,450.81)]
        roi_street = [(106.54,162.43), (1.54,506.21), (0.0,603.48), (0.0,719.59), (1278.44,719.59), (1278.44,321.61), (1207.31,328.39), (1163.28,321.61), (678.94,194.6), (560.4,153.96)]

        roi_pedlane = np.array(roi_pedlane, np.int32)
        roi_street = np.array(roi_street, np.int32)

        roi_pedlane = roi_pedlane.reshape(-1, 1, 2)
        roi_street = roi_street.reshape(-1, 1, 2)

        x1, y1, x2, y2 = box
        # Set the new y2 to be make the bounding box 1/5th of the original height
        # But at the bottom of the bounding box
        new_y1 = y1 + ((y2-y1) // 5) * 3
        image = cv2.rectangle(image, (x1, y1), (x2, y2), bnd_color, 1)
        # color_mask = cv2.rectangle(image, (x1, new_y1), (x2, y2), bnd_color, cv2.FILLED)
        # alpha = 0.4
        # image = cv2.addWeighted(color_mask, alpha, image, 1-alpha, 0)
        image = cv2.polylines(image, [roi_pedlane], True, (255, 255, 0), 1)
        image = cv2.polylines(image, [roi_street], True, (255, 0, 255), 1)
        cv2.rectangle(image, (1132, 203), (1140, 224), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # cv2.imshow('image', image)
    # cv2.waitKey(1)
    return image

# Callbacks
def select_file(sender, app_data, user_data):
    selections = app_data.get("selections")

    file_path = next(iter(selections.values())) # Path to the selected file
    file_name = next(iter(selections.keys()))   # Name of the selected file

    # check if the file is a video
    if file_name.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))

        frame_count = 0
        print("Reading video")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            predictions = model.predict(frame)
            image = create_prediction_image(predictions, frame)
            video.write(image)

            for key in predictions:
                predictions[key] = predictions[key].tolist()

            json.dump(predictions, open(f'app/{frame_count:06d}.json', 'w'))

            # Update the texture with the new image
            frame = np.flip(image, 2)
            frame = frame.ravel()
            frame = np.asfarray(frame, dtype='f')
            texture_data = np.true_divide(frame, 255.0)
            dpg.set_value("frame_texture", texture_data)

            frame_count += 1
        print("Finished reading video, total frames: ", frame_count)
        cap.release()
    else: # Selected an image
        image_data = cv2.imread(file_path)
        predictions = model.predict(image_data)
        image = create_prediction_image(predictions, image_data)

        cv2.imwrite('output.jpg', image)

        # Update the texture with the new image
        frame = np.flip(image, 2)
        frame = frame.ravel()
        frame = np.asfarray(frame, dtype='f')
        texture_data = np.true_divide(frame, 255.0)
        dpg.set_value("frame_texture", texture_data)

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

texture_data = []
for i in range(0, 1280 * 720):
    texture_data.append(0)
    texture_data.append(0)
    texture_data.append(0)

raw_data = array.array('f', texture_data)

with dpg.texture_registry():
    # Create a raw texture to display frames
    dpg.add_raw_texture(
        width=1280,
        height=720,
        default_value=raw_data,
        tag="frame_texture",
        format=dpg.mvFormat_Float_rgb
    )

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
    dpg.add_button(label="Select Image", callback=lambda: dpg.show_item("file_dialog_id"))
    dpg.add_image(texture_tag="frame_texture")

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()