import glob
import json
import cv2

images = glob.glob("input/*.jpg")
images.extend(glob.glob("test/*.jpg"))

annotations = glob.glob("target/*.json")
annotations.extend(glob.glob("test_targets/*.json"))

# for image in images:
#     img = cv2.imread(image)
#     if img.shape != (720, 1280, 3):
#         print(f"{image} is not 1280x720")

for annotation in annotations:
    with open(annotation) as f:
        data = json.load(f)
        for bbox in data['boxes']:
            x_min, y_min, x_max, y_max = bbox
            if x_min < 0 or y_min < 0:
                print(f"Negative coordinates in {annotation}")
                print(bbox)
            if x_max > 1280 or y_max > 720:
                print(f"Coordinates out of bounds in {annotation}")
                print(bbox)

                if x_max > 1280:
                    x_max = 1280.0
                
                if y_max > 720:
                    y_max = 720.0

                index = data['boxes'].index(bbox)
                print(f'Changing [{index}] {data["labels"][index]}: {bbox} to [{x_min}, {y_min}, {x_max}, {y_max}]')
                data['boxes'][index] = [x_min, y_min, x_max, y_max]

                with open(annotation, 'w') as f:
                    json.dump(data, f, indent=4)
            if x_min > x_max or y_min > y_max:
                print(f"Invalid bounding box in {annotation}")
                print(bbox)
            if (x_max - x_min) * (y_max - y_min) < 100:
                print(f"Bounding box too small in {annotation}")
                print(bbox)
                
                index = data['boxes'].index(bbox)
                print(f'Removing [{index}] {data["labels"][index]}: {bbox}')
                del data['boxes'][index]
                del data['labels'][index]

                with open(annotation, 'w') as f:
                    json.dump(data, f, indent=4)
                
                