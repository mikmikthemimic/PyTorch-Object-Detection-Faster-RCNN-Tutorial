import os
import json

folders = os.listdir()
folders = folders[1:]

for folder in folders:
    annotations = os.listdir(f'{folder}/Annotations')

    for annotation in annotations:
        with open(f'{folder}/Annotations/{annotation}', 'r') as file:
            json_data = json.load(file)
            
            for i in range(len(json_data['boxes'])):
                for j in range(len(json_data['boxes'][i])):
                    json_data['boxes'][i][j] = float(json_data['boxes'][i][j])
                
            file.close()

        with open(f'{folder}/Annotations/{annotation}', 'w') as file:
            json.dump(json_data, file, indent=4)
            file.close()