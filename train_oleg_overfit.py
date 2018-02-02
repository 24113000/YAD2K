#! /usr/bin/env python
"""Overfit a YOLO_v2 model to a single image from the Pascal VOC dataset.

This is a sample training script used to test the implementation of the
YOLO localization loss function.
"""
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

EPOH = 3000

def _main():

    anchors = YOLO_ANCHORS
    class_names = get_classes()
    image_data, orig_size = load_images()
    boxes, orig_boxes = get_boxes(orig_size)

    # Precompute detectors_mask and matching_true_boxes for training.
    # Detectors mask is 1 for each spatial position in the final conv layer and
    # anchor that should be active for the given boxes and 0 otherwise.
    # Matching true boxes gives the regression targets for the ground truth box
    # that caused a detector to be active or 0 otherwise.
    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    print('Boxes:')
    print(boxes)
    print('Box corners:')
    print(orig_boxes)
    print('Active detectors:')
    print(np.where(detectors_mask == 1)[:-1])
    print('Matching boxes for active detectors:')
    print(matching_true_boxes[np.where(detectors_mask == 1)[:-1]])

    # Create model body.
    model_body = yolo_body(image_input, len(anchors), len(class_names))
    model_body = Model(image_input, model_body.output)
    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])
    model = Model(
        [image_input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    num_steps = EPOH
    # TODO: For full training, put preprocessing inside training loop.
    # for i in range(num_steps):
    #     loss = model.train_on_batch(
    #         [image_data, boxes, detectors_mask, matching_true_boxes],
    #         np.zeros(len(image_data)))
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              batch_size=1,
              epochs=num_steps)
    model.save_weights('overfit_weights.h5')
    
    print("=== FINISHED ===")        


if __name__ == '__main__':
    _main()
