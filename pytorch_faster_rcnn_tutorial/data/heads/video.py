import cv2
import json
import math
import glob
import os

import numpy as np
import torch
from torchvision.ops import nms

predictions = glob.glob('predictions/*.json')
images = glob.glob('test/*.jpg')

labels = {
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Bicycle',
    4: 'Motorcycle',
    5: 'Tricycle',
    99: 'Unknown'
}

IOU_THRESHOLD = 0.7
SCORE_THRESHOLD = 0.55

USE_NMS = True

video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (1280, 720))

if not os.path.exists('output'):
    os.makedirs('output')

for i in range(len(predictions)):
    filename = os.path.basename(images[i])
    filename = filename.split('.')[0]

    with open(predictions[i]) as f:
        data = json.load(f)
        image = cv2.imread(images[i])

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
            confidence = data['scores'][j]
            if confidence < SCORE_THRESHOLD:
                continue

            label = data['labels'][j]
            box = [math.trunc(float(i)) for i in data['boxes'][j]]

            if label not in labels:
                print(f'Found unknown label: {label} from {predictions[i]}')
                label = 99

            match labels[label]:
                case 'Vehicle':
                    bnd_color = (0, 0, 139)
                    text_color = (0, 0, 255)
                case 'Pedestrian':
                    bnd_color = (0, 255, 0)
                    text_color = (1, 250, 32)
                case _:
                    bnd_color = (255, 0, 0)
                    text_color = (36, 255, 12)
            
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), bnd_color, 1)
            cv2.putText(image, labels[label], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(image, f'{confidence:.2f}', (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(image, filename, (1150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, ("NMS" if USE_NMS else "No NMS"), (1150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 255, 0) if USE_NMS else "No NMS"), 2)
        
    video.write(image)
    cv2.imwrite(f'output/{filename}.jpg', image)

video.release()