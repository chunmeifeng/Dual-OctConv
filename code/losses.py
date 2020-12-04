import tensorflow as tf
import tensorflow.keras as K
import numpy as np
from data import real2complex


def mse(y_true, y_pred):
    y_pred_complex = real2complex(y_pred)
    y_true_complex = real2complex(y_true)
    diff = tf.square(tf.abs(y_pred_complex - y_true_complex))
    loss = tf.reduce_mean(tf.reduce_mean(diff,axis=[1,2,3]))

    return loss

def mae(y_true, y_pred):
    y_pred_complex = real2complex(y_pred)
    y_true_complex = real2complex(y_true)
    diff = tf.abs(y_pred_complex - y_true_complex)
    loss = tf.reduce_mean(tf.reduce_mean(diff,axis=[1,2,3]))

    return loss

# This is our perceptual loss function
def perceptual_loss(y_true,y_pred):
    vgg_inp=K.Input(shape=y_true.shape[1:])
    vgg= K.applications.VGG16(include_top=False, input_tensor=vgg_inp)
    for l in vgg.layers:
        l.trainable=False
    vgg_out_layer = vgg.get_layer(index=5).output

    # making model Model(inputs, outputs)
    vgg_content = K.Model(vgg_inp, vgg_out_layer)
    y_t = vgg_content(y_true)
    y_p = vgg_content(y_pred)
    loss = mse(y_t,y_p)
    return loss
