import cv2
import sys
import numpy as np
import time
from frameimage import FrameImage


def get_center(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # save x and y in array
    center = []

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 3)
    kernel = np.ones((4, 4), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 120, param1=25,
                               param2=50, minRadius=10, maxRadius=200)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # draw the circle outline
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)

            # get center of circle
            cx = x - 5
            cy = y - 5
            center.append(cx)
            center.append(cy)

    # if center is empty (meaning circle WAS NOT detected), store 0 for x and y
    if len(center) == 0:
        center.append(0)
        center.append(0)

    return center


def take_photo():
    video = cv2.VideoCapture(0)
    ret, img = video.read()
    video.release()

    return img


def get_dimension(file_name):
    photo = cv2.imread(file_name)

    # Get dimensions of the image
    height, width, channels = photo.shape  # for color images

    return height, width


def photo_3_sec():
    images = []
    img_counter = 0

    # capture max of 100 images and store in images array
    for i in range(100):
        img_name = f"images/img_{img_counter}.png"
        img_counter += 1
        new_image = take_photo()

        # save image in images folder if circle is detected
        cv2.imwrite(img_name, new_image)
        c_pixel = get_center(img_name)  # c_pixel = [cx, cy]
        img_dim = get_dimension(img_name)  # img_dim = [height, width]
        store_image = FrameImage(new_image, img_name, c_pixel[0], c_pixel[1], img_dim[0], img_dim[1])
        images.append(store_image)

        # show image that JUST GOT CAPTURED
        cv2.namedWindow('Image')
        cv2.imshow('Image', new_image)
        cv2.resizeWindow('Image', 200, 200)
        # print('Center x: ' + str(images[i].cx))
        # print('Center y: ' + str(images[i].cy))

        print('Image height: ' + str(images[i].dim_height))
        print('Image width: ' + str(images[i].dim_width))
        print()

        time.sleep(1)  # Wait for 1 second before capturing the next image

        if cv2.waitKey(2000) & 0xFF == ord('q'):  # Wait 2 seconds to quit showing frames
            cv2.destroyWindow('Image')
            break

    return images


if __name__ == "__main__":
    captured_images = photo_3_sec()
    cv2.destroyAllWindows()

    print('Gathered the images...')
    index_num = int(input('Input the image number you desire: '))

    print(captured_images[index_num].img_name)
   
