
### Import the libraries
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from model.yolo_model import YOLO


# input the file name of the video to be analyzed
videoname = "sample.mp4"



def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """
    Get class names from a file.

    # Argument:
        file: str, path to the file containing class names.

    # Returns:
        List[str]: Class names read from the file.
    """
    with open(file) as f:
        class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
    return class_names



def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def detect_video(video, yolo, all_classes):
    """Use yolo v3 to detect video.

    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    video_path = os.path.join("videos", "test", video)
    print()
    print(video_path)
    camera = cv2.VideoCapture(video_path)
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')

    
    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

    while True:
        res, frame = camera.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", image)

        # Save the video frame by frame
        vout.write(image)

        if cv2.waitKey(110) & 0xff == 27:
                break

    vout.release()
    camera.release()


yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'
all_classes = get_classes(file)


### Detecting Images
# f = 'spoon_knife.jpg'
# f = 'carrot_apple.jpg'
# f = 'bus_traffic_light.jpg'
# f = 'central_market.jpg'
# f = 'friend_groups.jpg'
# f = 'fruit_basket.jpg'
# f = 'house_kitchen.jpg'
# f = 'spoon_knife.jpg'
# f = 'tv_laptop.jpg'
# f = 'vegetable.jpg'
# f = 'woman_on_track.jpg'
# f = 'WIN_20241215_22_31_09_Pro.mp4'
# path = 'images/test/'+f

detect_video(videoname, yolo, all_classes)
# cv2.imwrite('images/res/' + f, image)


# video = cv2.VideoCapture(path)
# while True:
#     ret, frame = video.read()  # Read a frame
#     if not ret:  # Break the loop if no more frames
#         print("End of video.")
#         break

#     # Display the frame
#     cv2.imshow("Video Playback", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break



# path = 'images/test/'+f
# image = cv2.imread(path)
# image = detect_image(image, yolo, all_classes)
# cv2.imwrite('images/res/' + f, image)