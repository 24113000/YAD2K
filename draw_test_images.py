import argparse
import io
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes
from common_functions import load_images, get_boxes, get_detector_mask, get_classes, YOLO_ANCHORS

OUT_PATH = "output_images"
IMAGE_INDEX = 1
WEIGHTS_NAME = "overfit_weights.h5"


def _main():
    class_names = get_classes()
    image_data, orig_size = load_images()

    image_input = Input(shape=(416, 416, 3))
    model_body = yolo_body(image_input, len(YOLO_ANCHORS), len(class_names))
    model_body = Model(image_input, model_body.output)
    model_body.load_weights(WEIGHTS_NAME)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, YOLO_ANCHORS, len(class_names))
    input_image_shape = K.placeholder(shape=(2,))
    boxes, scores, classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=.3, iou_threshold=.9)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    curr_image = image_data[IMAGE_INDEX]
    curr_image = np.expand_dims(curr_image, axis=0)
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            model_body.input: curr_image,
            input_image_shape: [416, 416],
            K.learning_phase(): 0
        })
    print('Found {} boxes for image.'.format(len(out_boxes)))
    print(out_boxes)

    # Save images
    image_with_boxes = draw_boxes(curr_image[0], out_boxes, out_classes, class_names, out_scores)
    result_image = PIL.Image.fromarray(image_with_boxes)
    result_image.save(os.path.join(OUT_PATH, str(IMAGE_INDEX) + 'c.png'))


if __name__ == '__main__':
    _main()
