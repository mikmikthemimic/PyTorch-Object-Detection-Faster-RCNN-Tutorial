import pathlib
import ast
import os
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
    return image

def main(input_file):
    model = FinalModel()

    cap = cv2.VideoCapture(input_file)
    frame_count = 0
    print("Reading video")
    video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (1280, 720))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imwrite(f'temp/frames/frame_{frame_count:06d}.jpg', frame)
        predictions = model.predict(frame)
        image = create_prediction_image(predictions, frame)

        video.write(image)

        frame_count += 1
    print("Finished reading video, total frames: ", frame_count)
    cap.release()

if __name__ == "__main__":
    input_file = "C:/Users/Angelo/Documents/GitHub/PyTorch-Object-Detection-Faster-RCNN-Tutorial/Video_1-00.00.00.000-00.01.40.686.mp4"

    main(input_file=input_file)