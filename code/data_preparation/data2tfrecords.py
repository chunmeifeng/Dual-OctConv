"""
Original version
"""

import os
import sys
import time
import glob
import h5py
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

def normalization(data, dtype='max'):
    if dtype == 'max':
        norm = np.max(np.abs(data))
    elif dtype == 'no':
        norm = 1.0
    else:
        raise ValueError("Normalization has to be in ['max', 'no']")

    data /= norm
    return data

def fft2c(img):
    """ Centered fft2 """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))) / np.sqrt(img.shape[-2]*img.shape[-1])

def ifft2c(img):
    """ Centered ifft2 """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img))) * np.sqrt(img.shape[-2]*img.shape[-1])

def complex_center_crop(data, shape):
    """ Apply a center crop to the input image or batch of complex images. """
    if data.shape[-2] < shape[1]:
        zeros_pad = np.zeros((shape[0], shape[1]-data.shape[-2], shape[-1]))
        left_pad, right_pad = np.array_split(zeros_pad, 2, axis=1)
        data = np.concatenate((left_pad, data, right_pad), axis=1)
    if data.shape[-3] < shape[0]:
        zeros_pad = np.zeros((shape[0]-data.shape[-3], shape[1], shape[-1]))
        left_pad, right_pad = np.array_split(zeros_pad, 2, axis=0)
        data = np.concatenate((left_pad, data, right_pad), axis=0)

    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]

def create_all(dataset_dir, save_path, custom_train=20, shuffle=False):
    if not os.path.exists(dataset_dir):
        print('Error in creating TFRECORD. File directory or file does not exist, please check the path.\n')
        exit()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    patient_idx_list = os.listdir(dataset_dir)
    patient_idx_list = sorted(patient_idx_list, key=lambda x: int(x))
    print('Total {} patients\n'.format(len(patient_idx_list)))
    if shuffle:
        random.shuffle(patient_idx_list)
    if custom_train is not None:
        patient_train_list = patient_idx_list[:custom_train]
        patient_val_list = patient_idx_list[14:17]
        patient_test_list = patient_idx_list[17:]
    else:
        patient_train_list = patient_idx_list[:14]
        patient_val_list = patient_idx_list[14:17]
        patient_test_list = patient_idx_list[17:]

    with open(os.path.join(save_path, r'.\dataset_info.txt'), 'w') as f:
        f.write('train_set_idx: {}\nval_set_idx: {}\ntest_set_idx: {}'.format(patient_train_list,patient_val_list,patient_test_list))
    print('train_set_idx: {}\nval_set_idx: {}\ntest_set_idx: {}'.format(patient_train_list,patient_val_list,patient_test_list))
    
    create_tfr(dataset_dir,save_path,patient_train_list,"train.tfrecords")
    create_tfr(dataset_dir,save_path,patient_val_list,"val.tfrecords")
    create_h5(dataset_dir,save_path,patient_test_list,"test.h5")
    # create_mat(dataset_dir,save_path,patient_train_list)
    
def create_h5(dataset_dir,tfrecord_path,idx_list,name="test.h5"):
    writer = h5py.File(os.path.join(tfrecord_path,name), 'w')
    slice_count = 0
    dataset = []
    for index,idx in enumerate(idx_list):
        print('\nProcessing slices of No.{} patient.'.format(idx))
        time0 = time.time()  # 显示当前时间
        filepath = os.path.join(dataset_dir,idx)
        filesNames = os.listdir(filepath)
        rawdata_list = glob.glob(os.path.join(filepath, 'rawdata*.mat'))
        rawdata_list = sorted(rawdata_list, key=lambda x: int(x.split('rawdata')[-1].split('.mat')[0]))
        start_slice = 11
        end_slice = 30
        for i in range(len(rawdata_list)):
            if (i+1)<start_slice or (i+1)>end_slice:
                rawdata_list.remove(os.path.join(filepath, 'rawdata{}.mat'.format(i+1)))
        for i,rawdata_dir in enumerate(rawdata_list):
            slice_count += len(rawdata_list)
            data = loadmat(rawdata_dir)
            rawdata = np.ascontiguousarray(np.transpose(data['rawdata'], (2, 0, 1)).astype(np.complex64))
            
            rawdata = ifft2c(rawdata)
            rawdata = np.transpose(rawdata, (1, 2, 0))
            rawdata = complex_center_crop(rawdata, shape=[320, 320, 15])
            rawdata = normalization(rawdata, dtype='max')
            rd_real = rawdata.real.astype(np.float64)
            rd_imag = rawdata.imag.astype(np.float64)
            rawdata_r = np.concatenate((rd_real, rd_imag), axis=-1)
            rawdata_r = np.transpose(rawdata_r, (2, 0, 1))
            dataset.append(rawdata_r)

            dataset_arr = np.asarray(dataset)
            print(dataset_arr.shape, dataset_arr.dtype)

    dataset_arr = np.asarray(dataset)
    writer.create_dataset('test', data=dataset_arr)
    writer.close()
            
    print('\nFinished writing data to h5 files.')

def create_tfr(dataset_dir,tfrecord_path,idx_list,tfrecord_name="val.tfrecords"):
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path,tfrecord_name))
    slice_count = 0
    for index,idx in enumerate(idx_list):
        print('\nProcessing slices of No.{} patient.'.format(idx))
        time0 = time.time()
        filepath = os.path.join(dataset_dir,idx)
        filesNames = os.listdir(filepath)
        rawdata_list = glob.glob(os.path.join(filepath, 'rawdata*.mat'))
        rawdata_list = sorted(rawdata_list, key=lambda x: int(x.split('rawdata')[-1].split('.mat')[0]))
        start_slice = 11
        end_slice = 30
        for i in range(len(rawdata_list)):
            if (i+1)<start_slice or (i+1)>end_slice:
                rawdata_list.remove(os.path.join(filepath, 'rawdata{}.mat'.format(i+1)))
        for i,rawdata_dir in enumerate(rawdata_list):
            slice_count += len(rawdata_list)
            data = loadmat(rawdata_dir)
            rawdata = np.ascontiguousarray(np.transpose(data['rawdata'], (2, 0, 1)).astype(np.complex64))
            
            rawdata = ifft2c(rawdata)
            rawdata = np.transpose(rawdata, (1, 2, 0))
            rawdata = complex_center_crop(rawdata, shape=[320, 320, 15])
            rawdata = normalization(rawdata, dtype='max')
            rd_real = rawdata.real.astype(np.float64)
            rd_imag = rawdata.imag.astype(np.float64)
            rawdata_r = np.concatenate((rd_real, rd_imag), axis=-1)

            sp_slice = rawdata_r.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'train/label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sp_slice]))
            }))
            writer.write(example.SerializeToString())
            sys.stdout.write('\r>> Converting image {}/{}, total:{}slices, consume:{}s'.format(i+1, len(rawdata_list), slice_count, time.time() - time0))
    writer.close()
    print('\nFinished writing data to tfrecord files.')

def read_direct(filename):
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        img = example.features.feature['img'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        return img,label

if __name__ == "__main__":
    dataset_dir = r'E:\SRdatasets\VN\coronal_pd'  # objective path
    save_path = r'E:\test_dataset'  # save path

    create_all(dataset_dir, save_path, shuffle=False)