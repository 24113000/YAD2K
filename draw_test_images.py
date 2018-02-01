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


# Create output variables for prediction.
yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2,))
boxes, scores, classes = yolo_eval(
    yolo_outputs, input_image_shape, score_threshold=.3, iou_threshold=.9)

# Run prediction on overfit image.
sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
out_boxes, out_scores, out_classes = sess.run(
    [boxes, scores, classes],
    feed_dict={
        model_body.input: image_data,
        input_image_shape: [416, 416],
        K.learning_phase(): 0
    })
print('Found {} boxes for image.'.format(len(out_boxes)))
print(out_boxes)

# Plot image with predicted boxes.
image_with_boxes = draw_boxes(image_data[0], out_boxes, out_classes,
                              class_names, out_scores)
plt.imshow(image_with_boxes, interpolation='nearest')
plt.show()
