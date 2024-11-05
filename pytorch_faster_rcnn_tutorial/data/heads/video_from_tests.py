import glob
import cv2

images = glob.glob('./test/*.jpg')

video = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (1280, 720))
for i, image in enumerate(images):
    print(f'{i}/{len(images)}')
    image_data = cv2.imread(image)
    video.write(image_data)