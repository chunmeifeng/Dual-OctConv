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


def create_mat(dataset_dir,tfrecord_path,idx_list,shapesize=320):
    """
    Creating .mat for training in traditional methods
    """
    slice_count = 0
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
        # print(rawdata_list)
        for i,rawdata_dir in enumerate(rawdata_list):
            slice_count += 1
            data = loadmat(rawdata_dir)
            rawdata = np.ascontiguousarray(np.transpose(data['rawdata'], (2, 0, 1)).astype(np.complex64))
            rawdata = ifft2c(rawdata)
            rawdata = np.transpose(rawdata, (1, 2, 0))
            rawdata = complex_center_crop(rawdata, shape=[shapesize, shapesize, 15])
            rawdata = normalization(rawdata, dtype='max')
            savemat(os.path.join(tfrecord_path, '{:0>2d}.mat'.format(slice_count)), {'Img': rawdata})

    print('\nFinished writing data to tfrecord files.')