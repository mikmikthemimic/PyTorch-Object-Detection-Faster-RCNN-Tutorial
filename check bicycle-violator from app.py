import glob
import json

app_predictions = glob.glob('app/*.json')


count = 0
for prediction in app_predictions:
    with open(prediction, 'r') as f:
        data = json.load(f)

        for i, label in enumerate(data['labels']):
            if label == 6 and data['scores'][i] > 0.2:
                count += 1
                print(f'{prediction} has label 6 at index {i} with confidence {data["scores"][i]}')

print(f'Total count: {count}')