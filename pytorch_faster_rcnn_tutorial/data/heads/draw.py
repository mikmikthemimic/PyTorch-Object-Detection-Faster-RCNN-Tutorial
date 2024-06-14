import cv2
import json
import math

filename = '016001'
score_threshold = 0.6
image = cv2.imread('test/0' + str(int(filename)) + '.jpg') #str(int(filename) + 121 if hindi test
annotation = open('predictions/' + filename + '.json', 'r')

data = json.load(annotation)

for i in range(0, len(data['labels'])):
    label = data['labels'][i]
    if label == 1:
        label = 'Vehicle'
    if label == 2:
        label = 'Pedestrian'

    box = [math.trunc(float(i)) for i in data['boxes'][i]]
    
    min_x, min_y, max_x, max_y = box[0], box[1], box[2], box[3]

    # switch case for different labels
    match label:
        #BGR
        case 'Vehicle':
            bnd_color = (0, 0, 139)
            text_color = (0, 0, 255)
        case 'Pedestrian':  
            bnd_color = (0, 255, 0)
            text_color = (1, 250, 32)
        case _:
            bnd_color = (255, 0, 0)
            text_color = (36, 255, 12)

    image = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), bnd_color, 1)
    cv2.putText(image, label, (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
cv2.imshow('image', image)
cv2.waitKey(0)