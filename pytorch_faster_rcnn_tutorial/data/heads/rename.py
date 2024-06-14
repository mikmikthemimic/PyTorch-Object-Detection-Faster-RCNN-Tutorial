import glob
import os

images = glob.glob("test/*.jpg")

for image in images:
    image_name = image.split('\\')[1]
    image_name = image_name.split('.')[0]
    image_name = int(image_name) - 121
    image_name = str(image_name).zfill(6)

    print(f'{image} -> {image_name}.jpg')
    os.rename(image, f"test/{image_name}.jpg")