# ONLY FOR GETTING THE GROUND TRUTH
import json
import math

from torchvision.ops import nms
from glob import glob

files = glob('predictions/New folder/predictions/*.json')

labels = {
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Bicycle',
    99: 'Unknown'
}

for file in files:
    print(file)
    with open(file, 'r') as f:
        predictions = json.load(f)
        for i in range(len(predictions['labels'])):
            predictions['labels'][i] = labels[predictions['labels'][i]]
    
    with open(file, 'w') as f:
        json.dump(predictions, f, indent=4)