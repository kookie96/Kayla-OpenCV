import sys
import numpy as np
import cv2
import time


def take_photo(counter):
    video = cv2.VideoCapture(0)
    ret, img = video.read()
    img_counter = counter

    if not ret:
        print('Error: Could not read frame')
        return

    img_name = 'img{}.png'.format(img_counter)
    cv2.imwrite(img_name, img)
    video.release()


def color_circles(target_color):
    video = cv2.VideoCapture(0)  # Set Capture Device

    # image counter
    img_counter = 1

    # max number of pictures to take when target circle is detected
    max_num = 3
    num_count = 0

    while True:
        # Capture frame-by-frame
        ret, img = video.read()

        # load the image, clone it for output, and then convert it to grayscale
        output = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        gray = cv2.medianBlur(gray, 5)

        # Adaptive Guassian Threshold is to detect sharp edges in the image. Yields better results
        # with varying illumination for different regions
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 3)

        # Erode function for "shrinking" or "eroding" boundaries of regions in binary image.
        # Also used to remove small white regions orq separate objects close to each other.
        kernel = np.ones((4, 4), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=1)
        # # gray = erosion
        #
        # gray = cv2.dilate(gray, kernel, iterations=1)
        # gray = dilation
        #
        # # get the size of the final image
        # # img_size = gray.shape
        # # print img_size

        # detect circles in the image
        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 190, param1=25, param2=45, minRadius=10, maxRadius=100)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 120, param1=25,
                                   param2=50, minRadius=10, maxRadius=200)

        # ensure at least some circles were found
        color = 'UNDEFINED'
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle in the image
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                # print color of the circle from the pixel center of circle (the rectangle
                # in center of circle)
                height, width, _ = img.shape
                cx = x - 5
                cy = y - 5
                pixel_center = hsv_frame[cy, cx]
                hue_value = pixel_center[0]  # only take first value in the
                # tuple which is the hue value
                sat_value = pixel_center[1]  # saturation values
                val_value = pixel_center[2]  # value (V in HSV)

                # cv2.putText(output, 'CENTER',(cx, cy), 0, 1, (255, 0, 0), 2)

                if hue_value < 5:
                    color = 'RED'
                elif hue_value < 22:
                    color = 'ORANGE'
                elif hue_value < 33:
                    color = 'YELLOW'
                elif hue_value < 78:
                    color = 'GREEN'
                elif hue_value < 131:
                    color = 'BLUE'
                elif hue_value < 170:
                    color = 'VIOLENT'
                elif sat_value < 21:
                    color = 'WHITE'
                else:
                    color = 'RED'

                # if color matches the color of target, take a photo of frame
                if target_color == color and num_count < max_num:
                    img_name = f"images/img_{img_counter}.png"
                    cv2.imwrite(img_name, output)
                    img_counter += 1
                    num_count = img_counter - 1

                    # get pixels for GPS
                    target_x = x
                    target_y = y

                    print('Target is detected')

                # time.sleep(0.5)
                # print("Column Number: ")
                # print(x)
                # print("Row Number: ")
                # print(y)
                # print("Radius is: ")
                # print(r)
                # print(hue_value, end=' ')
                # print(color)

        # Display the resulting frame and the color detected
        cv2.putText(output, color, (10, 50), 0, 1, (255, 0, 0), 2)
        # cv2.imshow('gray', gray)
        cv2.imshow('frame', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if cv2.waitKey(1) & 0xFF == ord('s'):

        # reset counter for pictures
        num_count = 0

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    color_input = sys.argv[1]
    color_circles(color_input)
    # color_circles('RED')
