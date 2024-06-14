import cv2
import json
import math
import glob
import os

predictions = glob.glob('predictions/*.json')
images = glob.glob('test/*.jpg')

labels = {
    1: 'Vehicle',
    2: 'Pedestrian',
}

video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 18, (1280, 720))

if not os.path.exists('output'):
    os.makedirs('output')

for i in range(len(predictions)):
    with open(predictions[i]) as f:
        data = json.load(f)
        image = cv2.imread(images[i])

        for j in range(len(data['labels'])):
            confidence = data['scores'][j]
            if confidence < 0.4:
                continue

            label = data['labels'][j]
            box = [math.trunc(float(i)) for i in data['boxes'][j]]

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
            cv2.putText(image, str(i), (1150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    video.write(image)
    cv2.imwrite(f'output/{i}.jpg', image)

video.release()