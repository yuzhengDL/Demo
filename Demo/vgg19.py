import numpy as np
import os
import tensorflow as tf

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

import matplotlib.pyplot as plt
from skimage import io
import json

from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing

from tensorflow.contrib import slim

checkpoints_dir = 'checkpoints/'

image_size = vgg.vgg_19.default_image_size

class vgg_exactor():
    def __init__(self):

        self.image_string = tf.placeholder(tf.string, name="image_string")
        #image_string = urllib.urlopen(self.url).read()
        image = tf.image.decode_jpeg(self.image_string, channels=3)
        processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_images  = tf.expand_dims(processed_image, 0)

        with slim.arg_scope(vgg.vgg_arg_scope()):
            self.logits, _ = vgg.vgg_19(processed_images, num_classes=None, is_training=False)

        self.init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'vgg_19.ckpt'),
            slim.get_model_variables('vgg_19'))

