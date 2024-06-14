import glob
import json
import cv2
import math

images = glob.glob("input/*.jpg")

annotations = glob.glob("target/*.json")

video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280, 720))

for i in range(0, len(images)):
    annotation = open(annotations[i], 'r')
    image = cv2.imread(images[i])
    data = json.load(annotation)

    image_name = images[i].split('\\')[1]
    image_name = image_name.split('.')[0]
    image_name = int(image_name)
    print(image_name)

    for k in range(0, len(data['labels'])):
        label = data['labels'][k]
        box = [math.trunc(float(i)) for i in data['boxes'][k]]
        
        min_x, min_y, max_x, max_y = box[0], box[1], box[2], box[3]

        # switch case for different labels
        match label:
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

        # put the frame number on the upper right corner
        cv2.putText(image, str(image_name), (1150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    video.write(image)

video.release()
