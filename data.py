import h5py
import numpy    as np
import scipy.io as sio
import tensorflow as tf
import os
import random
from option import args
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def augment(img):
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    if random.random()<0.5:
        img = tf.image.rot90(img)
    return img

def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:,:,:channel], x[:,:,channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])

def complex2real(x):
    x_real = tf.math.real(x)
    x_imag = tf.math.imag(x)
    return tf.concat([x_real,x_imag], axis=-1)

def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]

def load_data(data_path, batch_size):
    if args.mask_name == '1Duniform2.98_ac29' or args.mask_name == '1Un2.98' or 'new1D-Cartesian-0.3_320' or 'new1D-Cartesian-0.3_256':
        mask = sio.loadmat(os.path.join(args.mask_path, '{}.mat'.format(args.mask_name)))
        mask = mask['mask']
    else:
        with h5py.File(os.path.join(args.mask_path, '{}.mat'.format(args.mask_name)), 'r') as f:
            mask = f['mask'][:]
    mask = np.fft.ifftshift(mask)
    train_data = read_and_decode(data_path+'/train.tfrecords', batch_size)
    validate_data = read_and_decode(data_path+'/val.tfrecords', batch_size)
    with h5py.File(os.path.join(data_path,'test.h5'),'r') as f:
        test_data = f['test'][:]
    test_data = np.transpose(test_data, (0, 2, 3, 1))
    channel = test_data.shape[-1] // 2
    test_data_real = test_data[:, :, :, :channel]
    test_data_imag = test_data[:, :, :, channel:]
    test_data = test_data_real + 1j * test_data_imag
    print('Dataset {} Loading Done.'.format(args.data_dst))
    return train_data, validate_data, test_data, mask

def load_data_test(data_path, batch_size):
    if args.mask_name == '1Duniform2.98_ac29' or args.mask_name == '1D-Uniform-0.3_320' or args.mask_name == '1D-Uniform-0.2_320'\
         or args.mask_name == '1D-Cartesian-0.2_320' or args.mask_name == '1D-Cartesian-0.3_320' or args.mask_name == '1D-Cartesian-0.4_320'\
         or args.mask_name == '2D-Random-0.2_320' or args.mask_name == '2D-Random-0.3_320' or args.mask_name == '2D-Random-0.4_320' or args.mask_name == '2D-Random-0.5_320'\
         or args.mask_name == '1D-Cartesian-0.2_320' or args.mask_name == '1D-Cartesian-0.3_320' or args.mask_name == '1D-Cartesian-0.4_320'\
         or args.mask_name == 'radial_mask_4x' or args.mask_name == 'radial_mask_6x' or args.mask_name == 'radial_mask_9x':
        mask = sio.loadmat(os.path.join(args.mask_path, '{}.mat'.format(args.mask_name)))
        mask = mask['mask']
    else:
        with h5py.File(os.path.join(args.mask_path, '{}.mat'.format(args.mask_name)), 'r') as f:
            mask = f['mask'][:]  # uint8  in fd(K-space)
    mask = np.fft.ifftshift(mask)
    with h5py.File(os.path.join(data_path,'test.h5'),'r') as f:
        test_data = f['test'][:]
    test_data = np.transpose(test_data, (0, 2, 3, 1))
    channel = test_data.shape[-1] // 2
    test_data_real = test_data[:, :, :, :channel]
    test_data_imag = test_data[:, :, :, channel:]
    test_data = test_data_real + 1j * test_data_imag
    print('Dataset {} Loading Done.'.format(args.data_dst))

    return test_data, mask

def read_and_decode(filename, batch_size):
    filename_queue = tf.compat.v1.train.string_input_producer([filename])
    reader = tf.compat.v1.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    feature = {'train/label':tf.io.FixedLenFeature([],tf.string)}
    features = tf.io.parse_single_example(serialized_example, features=feature)
    img = tf.decode_raw(features['train/label'], tf.float64)
    img = tf.reshape(img, shape=[args.data_size, args.data_size, args.in_channels])

    if args.augment is True:
        img = real2complex(img)
        img = augment(img)
        img = complex2real(img)

    return img

def setup_inputs(x, mask, batch_size):
    """setup_inputs
    input:  x,              (256, 256, 30) sd float64   | mask,   (256, 256)     fd uint8
    output: features,    (?, 256, 256, 30) sd float32   | labels, (?, 256, 256, 30) sd float32
            kx_mask,     (?, 256, 256, 15) fd complex64 | masks,  (?, 256, 256, 15) fd complex64
    """
    channel = x.shape[-1].value // 2  # 15
    mask_tf_c = tf.tile(tf.expand_dims(mask, 0), (channel, 1, 1))
    mask_tf_c = tf.cast(mask_tf_c, tf.complex64)
    
    x_complex = real2complex(x)
    x_complex = tf.cast(x_complex, tf.complex64)
    x_complex = tf.transpose(x_complex, [2, 0, 1])
    kx = tf.signal.fft2d(x_complex)
    kx_u = kx * mask_tf_c
    x_u = tf.signal.ifft2d(kx_u)
    x_u = tf.transpose(x_u, [1, 2, 0])
    kx_u = tf.transpose(kx_u, [1, 2, 0])

    x_u_cat = complex2real(x_u)
    x_cat = tf.cast(x, tf.float32)
    mask_tf_c = tf.transpose(mask_tf_c, [1, 2, 0])

    features, labels, kx_u, masks = tf.compat.v1.train.shuffle_batch([x_u_cat, x_cat, kx_u, mask_tf_c],
                                                     batch_size=batch_size,
                                                     num_threads=64,
                                                     capacity=50,
                                                     min_after_dequeue=10)
    return features, labels, kx_u, masks

def setup_inputs_test(x, mask, norm=None):
    batch = x.shape[0]
    channel = x.shape[-1]
    mask = np.tile(mask, (batch, channel, 1, 1))
    mask = np.transpose(mask, (0, 2, 3, 1))
    kx = np.fft.fft2(x, axes=(1,2), norm=norm)
    kx_mask = kx * mask
    x_u = np.fft.ifft2(kx_mask, axes=(1,2), norm=norm)

    x_u_cat = np.concatenate((np.real(x_u), np.imag(x_u)), axis=-1)
    x_cat = np.concatenate((np.real(x), np.imag(x)), axis=-1)
    mask_c = mask.astype(np.complex64)
    return x_u_cat, x_cat, kx_mask, mask_c