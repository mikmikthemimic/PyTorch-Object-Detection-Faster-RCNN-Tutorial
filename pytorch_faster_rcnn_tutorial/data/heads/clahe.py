import cv2
import glob
import numpy as np

images = glob.glob('input/023026.jpg')

def main():
    # Random choose an image
    choice = np.random.choice(images)
    print('Image:', choice)
    orig_image = cv2.imread(choice)

    # Convert to grayscale
    gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(gray_image)

    clahe = cv2.createCLAHE(clipLimit=0.9, tileGridSize=(35, 35))
    clahe_images = [clahe.apply(plane) for plane in lab_planes]

    updated_lab_img2 = cv2.merge(clahe_images)

    # Convert back to BGR
    final_image = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

    # Display the images
    cv2.imshow('Original Image', orig_image)
    cv2.waitKey(0)
    cv2.destroyWindow('Original Image')
    cv2.imshow('Final Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the final image
    cv2.imwrite('original.jpg', orig_image)
    cv2.imwrite('clahe.jpg', final_image)

if __name__ == '__main__':
    main()