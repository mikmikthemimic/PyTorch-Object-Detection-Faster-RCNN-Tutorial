# ONLY FOR GETTING THE GROUND TRUTH
import json
import math

from torchvision.ops import nms
from glob import glob

files = glob('predictions/*.json')
files.extend(glob('test_targets/*.json'))

labels = {
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Bicycle',
    4: 'Vehicle-Violator',
    5: 'Pedestrian-Violator',
    6: 'Bicycle-Violator',
    99: 'Unknown',
}
reverse_labels = {
    'Vehicle': 1,
    'Pedestrian': 2,
    'Bicycle': 3,
    'Vehicle-Violator': 4,
    'Pedestrian-Violator': 5,
    'Bicycle-Violator': 6,
    'Unknown' : 99
}

for file in files:
    print(file)
    with open(file, 'r') as f:
        predictions = json.load(f)
        for i in range(len(predictions['labels'])):
            predictions['labels'][i] = reverse_labels[predictions['labels'][i]]
    
    with open(file, 'w') as f:
        json.dump(predictions, f, indent=4)