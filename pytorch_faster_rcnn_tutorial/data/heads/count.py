import glob
import json

training_json = glob.glob('target/*.json')
test_json = glob.glob('test_targets/*.json')

training_count = {
    "Vehicle": 0,
    "Pedestrian": 0,
    "Bicycle": 0,
    "Bicycle-Violator": 0,
}

test_count = {
    "Vehicle": 0,
    "Pedestrian": 0,
    "Bicycle": 0,
    "Bicycle-Violator": 0,
}

for file in training_json:
    with open(file) as f:
        data = json.load(f)
        for label in data['labels']:
            if label == 'Vehicle':
                training_count['Vehicle'] += 1
            elif label == 'Pedestrian':
                training_count['Pedestrian'] += 1
            elif label == 'Bicycle':
                training_count['Bicycle'] += 1

for file in test_json:
    with open(file) as f:
        data = json.load(f)
        for label in data['labels']:
            if label == 'Vehicle':
                test_count['Vehicle'] += 1
            elif label == 'Pedestrian':
                test_count['Pedestrian'] += 1
            elif label == 'Bicycle':
                test_count['Bicycle'] += 1
            elif label == 'Bicycle-Violator':
                test_count['Bicycle-Violator'] += 1

print(training_count)
print(test_count)