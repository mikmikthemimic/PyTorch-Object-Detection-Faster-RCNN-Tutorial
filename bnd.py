from shapely.geometry import Polygon
import cv2
import numpy as np
import json
from glob import glob

def check_overlap(coords, label, ped_status):
    """
    coords: array of coords of the object e.g. [x1, y1, x2, y2]
    label: provided label of the object
    img: filepath to image
    """
    # Taken from CVAT.ai annotation tool
    #roi_pedlane = [(46.80, 366.90), (17.30, 463.30), (199.67, 443.63), (342.79, 426.02), (720.40, 383.08), (1228.00, 333.00), (1194.90, 326.94), (1155.27, 320.33), (1092.51, 306.02), (1052.88, 296.11), (1031.96, 289.50), (680.70, 313.70), (687.38, 321.43), (692.88, 329.14), (692.90, 338.70), (681.50, 345.20), (664.80, 346.80), (647.70, 346.70), (625.90, 344.50), (607.30, 340.90), (590.20, 333.60), (570.30, 320.70), (326.27, 342.35)]
    roi_pedlane = [(45.41,364.32), (1015.1,281.1), (1157.8,329.7), (19.46,450.81)]
    roi_pedlane = Polygon(roi_pedlane)

    roi_street = [(106.54,162.43), (1.54,506.21), (0.0,603.48), (0.0,719.59), (1278.44,719.59), (1278.44,321.61), (1207.31,328.39), (1163.28,321.61), (678.94,194.6), (560.4,153.96)]
    roi_street = Polygon(roi_street)
    #print(f'roi_pedlane: {roi_pedlane}')
    #print(f'coords: {coords}')
    x1, y1, x2, y2 = coords
    # y2 = y1+((y2-y1)//2)
    y1 = y1 + ((y2-y1) // 5) * 4
    object_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    
    #if roi_street.intersection(object_polygon).area > 0.2 * object_polygon.area:
    #    return "Outside"
    
    if roi_pedlane.intersection(object_polygon).area > 0.25 * object_polygon.area:
        if label == 2 or label == 5:     # Pedestrian and Pedestrian Violator
            # Pedestrian is inside pedestrian lane ROI and light is red
            if ped_status == 'Red light':
                print("Pedestrian out of lane")
                return 5                            # Pedestrian Violator
            # Pedestrian is inside pedestrian lane ROI and light is green
            else:
                return 2                                     # Pedestrian
            
        elif label == 1 or label == 4:         # Vehicle and Vehicle Violator
            # Vehicle is inside pedestrian lane ROI and light is green
            if ped_status == 'Green light':
                return 4                               # Vehicle Violator
            # Vehicle is inside pedestrian lane ROI and light is red
            else:
                return 1                                        # Vehicle
            
        elif label == 3 or label == 6:         # Bicycle and Bicycle Violator
            # Bicycle is inside pedestrian lane ROI and light is green
            if ped_status == 'Green light':
                return 6                               # Bicycle Violator
            # Bicycle is inside pedestrian lane ROI and light is red
            else:
                return 3                                        # Bicycle
    # Kapag wala sa pedestrian lane
    else:
        if label == 2 or label == 5:     # Pedestrian and Pedestrian Violator
            if roi_street.intersection(object_polygon).area > 0.2 * object_polygon.area:
                # Pedestrian is outside the pedestrian lane, but inside the street
                # They shouldn't be there whether the light is red or green
                return 5                            # Pedestrian Violator
            else:
                # Pedestrian is outside the street,
                # they're not violating any rules whether the light is red or green
                return 2                                     # Pedestrian
        
        elif label == 1 or label == 4:         # Vehicle and Vehicle Violator
            # Vehicle is outside pedestrian lane ROI and light is green
            if ped_status == 'Green light':
                return 1                                        # Vehicle
            # Vehicle is outside pedestrian lane ROI and light is red
            else:
                return label # Return lang kung ano yung original label
            
        elif label == 3 or label == 6:         # Bicycle and Bicycle Violator
            # Bicycle is outside pedestrian lane ROI and light is green
            if ped_status == 'Green light':
                return 3                                        # Bicycle
            # Bicycle is outside pedestrian lane ROI and light is red
            else:
                return label
            
def new_check_overlap(coords: list, labels: list, ped_status):
    """
    coords: list of array of coords of the object e.g. [x1, y1, x2, y2]
    label: list of labels of the object
    ped_status: filepath to image
    """
    # Taken from CVAT.ai annotation tool
    roi_pedlane = [(45.41,364.32), (1015.1,281.1), (1157.8,329.7), (19.46,450.81)]
    roi_pedlane = Polygon(roi_pedlane)

    roi_street = [(106.54,162.43), (1.54,506.21), (0.0,603.48), (0.0,719.59), (1278.44,719.59), (1278.44,321.61), (1207.31,328.39), (1163.28,321.61), (678.94,194.6), (560.4,153.96)]
    roi_street = Polygon(roi_street)
    #print(f'roi_pedlane: {roi_pedlane}')
    #print(f'coords: {coords}')
    
    #print(object_polygon)
    
    #if roi_street.intersection(object_polygon).area > 0.2 * object_polygon.area:
    #    return "Outside"
    
    for i, label in enumerate(labels):

        x1, y1, x2, y2 = coords[i]
        # y2 = y1+((y2-y1)//2)
        y1 = y1 + ((y2-y1) // 5) * 4
        object_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

        if roi_pedlane.intersection(object_polygon).area > 0.25 * object_polygon.area:
            if label == 2 or label == 5:     # Pedestrian and Pedestrian Violator
                # Pedestrian is inside pedestrian lane ROI and light is red
                if ped_status == 'Red light':
                    labels[i] = 5                            # Pedestrian Violator
                # Pedestrian is inside pedestrian lane ROI and light is green
                else:
                    labels[i] = 2                                     # Pedestrian
                
            elif label == 1 or label == 4:         # Vehicle and Vehicle Violator
                # Vehicle is inside pedestrian lane ROI and light is green
                if ped_status == 'Green light':
                    labels[i] = 4                               # Vehicle Violator
                # Vehicle is inside pedestrian lane ROI and light is red
                else:
                    labels[i] = 1                                        # Vehicle
                
            elif label == 3 or label == 6:         # Bicycle and Bicycle Violator
                # Bicycle is inside pedestrian lane ROI and light is green
                if ped_status == 'Green light':
                    labels[i] = 6                               # Bicycle Violator
                # Bicycle is inside pedestrian lane ROI and light is red
                else:
                    labels[i] = 3                                        # Bicycle
        # Kapag wala sa pedestrian lane
        else:
            if label == 2 or label == 5:     # Pedestrian and Pedestrian Violator
                if roi_street.intersection(object_polygon).area > 0.2 * object_polygon.area:
                    # Pedestrian is outside the pedestrian lane, but inside the street
                    # They shouldn't be there whether the light is red or green
                    labels[i] = 5                            # Pedestrian Violator
                else:
                    # Pedestrian is outside the street,
                    # they're not violating any rules whether the light is red or green
                    labels[i] = 2                                     # Pedestrian
            
            elif label == 1 or label == 4:         # Vehicle and Vehicle Violator
                # Vehicle is outside pedestrian lane ROI and light is green
                if ped_status == 'Green light':
                    labels[i] = 1                                        # Vehicle
                # Vehicle is outside pedestrian lane ROI and light is red
                # else:
                    # labels[i] = label # Return lang kung ano yung original label
                
            elif label == 3 or label == 6:         # Bicycle and Bicycle Violator
                # Bicycle is outside pedestrian lane ROI and light is green
                if ped_status == 'Green light':
                    labels[i] = 3                                        # Bicycle
                # Bicycle is outside pedestrian lane ROI and light is red
                # else:
                    # return label
    
    return labels
            
def get_light(image_path):
    """
    image: filepath to image
    """
    img = cv2.imread(image_path)
    # cv2.rectangle(img, (1132, 203), (1140, 224), (0, 255, 0), 2)
    # Get the average color of the rectangle
    avg_color = np.array(cv2.mean(img[203:224, 1132:1140])).astype(np.uint8)
    # Convert BGR to RGB
    avg_color = avg_color[[2, 1, 0]]

    # Check if there is more red than green in the average color
    if avg_color[0] > avg_color[2]:
        return 'Red light'
    # Check if there is more green than red in the average color
    elif avg_color[0] < avg_color[2]:
        return 'Green light'
    return 'Unknown light'

def get_light_from_image(image_data, coordinates):
    """
    image: numpy.ndarray (cv2.imread)
    coordinates: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = coordinates
    # cv2.rectangle(img, (1132, 203), (1140, 224), (0, 255, 0), 2)
    # Get the average color of the rectangle
    avg_color = np.array(cv2.mean(image_data[y1:y2, x1:x2])).astype(np.uint8)
    # Convert BGR to RGB
    avg_color = avg_color[[2, 1, 0]]

    # Check if there is more red than green in the average color
    if avg_color[0] > avg_color[2]:
        return 'Red light'
    # Check if there is more green than red in the average color
    elif avg_color[0] < avg_color[2]:
        return 'Green light'
    return 'Unknown light'

def test():
    prediction_path = glob('pytorch_faster_rcnn_tutorial/data/heads/test_targets/*.json')
    input_path = glob('pytorch_faster_rcnn_tutorial/data/heads/test/*.jpg')

    for i in range(len(prediction_path)):
        ped_status = get_light(input_path[i])
        print(f'Processing {prediction_path[i]} | Light status: {ped_status}')
        with open(prediction_path[i], 'r') as f:
            data = json.load(f)
            for j in range(len(data['labels'])):
                result = check_overlap(data['boxes'][j], data['labels'][j], ped_status)
                if result == None:
                    print(f"{data['labels'][j]} is not a pedestrian or vehicle")

                data['labels'][j] = result
                # print(data['boxes'][i])

        with open(prediction_path[i], 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    test()

