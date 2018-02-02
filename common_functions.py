import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from yad2k.models.keras_yolo import preprocess_true_boxes
import os

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


def load_images():
    images = [
        #Image.open("train_images/1.jpg"),
        Image.open("train_images/2.jpg"),
        Image.open("train_images/3.jpg"),
        Image.open("train_images/4.jpg")
    ]
    orig_size_saved = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size_saved, axis=0)
    orig_size = np.expand_dims(orig_size, axis=0) #TODO fix it should be one statement

    # Images preprocessing.
    image_data = []
    for i in range(len(images)):
        resized = images[i].resize((416, 416), PIL.Image.BICUBIC)
        resized = np.array(resized, dtype=np.float)
        image_data.append(resized / 255)

    return np.array(image_data), orig_size


def get_boxes(orig_size):
    # Box preprocessing.
    # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
    boxes = np.array(
        [
            #[[0, 226, 128, 343, 243], [1, 23, 355, 135, 467]],
            [[1, 56, 117, 157, 215], [0, 225, 127, 342, 241], [2, 190, 320, 313, 440]],
            [[0, 36, 35, 158, 153], [2, 196, 169, 318, 290], [1, 372, 47, 474, 145]],
            [[0, 286, 137, 409, 254], [1, 140, 295, 241, 394], [2, 410, 367, 540, 493]],
        ])
    # Get extents as y_min, x_min, y_max, x_max, class for comparision with
    # model output.
    orig_boxes = boxes #just for printing

    # Get box parameters as x_center, y_center, box_width, box_height, class.
    boxes_xy = 0.5 * (boxes[:, :, 3:5] + boxes[:, :, 1:3])
    boxes_wh = boxes[:, :, 3:5] - boxes[:, :, 1:3]
    boxes_xy = boxes_xy / orig_size
    boxes_wh = boxes_wh / orig_size
    boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, :, 0:1]), axis=2)

    return boxes, orig_boxes


def get_classes():
    classes_path = os.path.expanduser("model_data/figure_classes.txt")
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)
