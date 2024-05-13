from glob import glob
import json
import os

files = glob('target/*.json')
files.extend(glob('test_targets/*.json'))

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        for i in range(len(data['labels'])):
            if data['labels'][0].startswith('\n'):
                print(f'Sanitizing {file}...')
            data['labels'][i] = data['labels'][i].strip()

    with open(file, 'w') as f:
        json.dump(data, f, indent=4)