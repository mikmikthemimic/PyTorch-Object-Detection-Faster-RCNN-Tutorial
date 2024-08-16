# ONLY FOR GETTING THE GROUND TRUTH
import json
import math

from torchvision.ops import nms
from glob import glob

files = glob('predictions/*.json')

labels = {
    'Vehicle':1,
    'Pedestrian':2,
    'Bicycle':3,
    'Unknown':99,
}

for file in files:
    print(file)
    with open(file, 'r') as f:
        predictions = json.load(f)
        for i in range(len(predictions['labels'])):
            predictions['labels'][i] = labels[predictions['labels'][i]]
    
    with open(file, 'w') as f:
        json.dump(predictions, f, indent=4)