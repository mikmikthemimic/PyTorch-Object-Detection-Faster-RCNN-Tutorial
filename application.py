import dearpygui.dearpygui as dpg
import os
import cv2

import numpy as np
import torch
import json
import math

from torchvision.ops import nms

from model import (
    predict
)

dpg.create_context()

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

def load_predicted_images(json_file_name, path_to_image):
    filename = os.path.splitext(os.path.basename(path_to_image))[0]

    IOU_THRESHOLD = 0.90
    SCORE_THRESHOLD = 0

    USE_NMS = False

    with open(json_file_name) as f:
        data = json.load(f)
        
        image = cv2.imread(path_to_image)
        print(path_to_image)

        nms_boxes = torch.tensor(data['boxes']) 
        if nms_boxes.size()[0] > 0 and USE_NMS:
            nms_labels = torch.tensor(data['labels'])
            nms_scores = torch.tensor(data['scores'])

            mask = nms(nms_boxes, nms_scores, iou_threshold=IOU_THRESHOLD)
            mask = (np.array(mask),)

            data['boxes'] = np.asarray(nms_boxes)[mask]
            data['labels'] = np.asarray(nms_labels)[mask]
            data['scores'] = np.asarray(nms_scores)[mask]

        for j in range(len(data['labels'])):
            #label = data['labels'][j]
            confidence = data['scores'][j]

            if (data['labels'][j] == 1 or data['labels'][j] == 4):
                SCORE_THRESHOLD = 0.5

            elif (data['labels'][j] == 2 or data['labels'][j] == 5):
                SCORE_THRESHOLD = 0.2

            elif (data['labels'][j] == 3 or data['labels'][j] == 6):
                SCORE_THRESHOLD = 0.2
                
            else:
                print(f"Label: {data['labels'][j]}, iteration: {j}")

            if confidence < SCORE_THRESHOLD:
                continue

            label = labels[data['labels'][j]]
            box = [math.trunc(float(i)) for i in data['boxes'][j]]

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
            #cv2.putText(image, f'{confidence:.2f}', (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(image, filename, (1150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.putText(image, ("NMS" if USE_NMS else "No NMS"), (1150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 255, 0) if USE_NMS else (0, 0, 255)), 2)
        
    cv2.imwrite(f'output/{filename}.jpg', image)
    add_and_load_image(f'output/{filename}.jpg', parent="Primary Window")

# Callbacks
def select_image(sender, app_data, user_data):
    selections = app_data.get("selections")
    image = next(iter(selections.values()))
    print(f"App Data: {app_data}")
    print(f"Image: {image}")

    image_name = next(iter(selections.keys()))

    if dpg.does_item_exist("image_tag"):
        print("Deleting existing image")
        dpg.delete_item("image_tag")
        dpg.delete_item("texture_tag")
    
    #TODO:
    output = predict(image)
    print(output)
    load_predicted_images(output, image)

# Lambdas
def add_and_load_image(image_path, parent=None):
    width, height, channels, data = dpg.load_image(image_path)

    with dpg.texture_registry() as reg_id:
        texture_id = dpg.add_static_texture(width, height, data, parent=reg_id, tag="texture_tag")

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
    callback=select_image
):
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
    dpg.add_file_extension(".png", color=(0, 255, 0, 255))
    dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))

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
    dpg.add_button(label="Select Image", callback=lambda: dpg.show_item("file_dialog_id"))

# Setup
dpg.create_viewport(
    title="Test", width=1280, height=840, x_pos=0, y_pos=0, resizable=False
)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()