"""
Common utilities for training process visualization
"""

import os, sys, glob
import time,datetime
import numpy as np
from scipy import signal
from scipy.signal import correlate2d
import scipy.io as sio
import imageio
# from skimage.measure import compare_ssim, compare_psnr ## removed in v0.18
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt
import tensorflow as tf
from option import args

def save2mat(inputs, pictype, count, epoch):
    if not os.path.exists('../experiment/{}/mats'.format(args.name)):
        os.makedirs('../experiment/{}/mats'.format(args.name))
    pred_c = real2complex_array(inputs)
    pred = np.squeeze(pred_c, axis=0) if inputs.ndim == 4 else pred_c
    mat_name = os.path.join('../experiment/{}/mats'.format(args.name),
                            '{}_{}_{}.mat'.format(count, pictype, epoch+1))
    sio.savemat(mat_name, {'rawdata':pred})

def save2img(inputs, pictype, count, epoch):
    if not os.path.exists('../experiment/{}/imgs'.format(args.name)):
        os.makedirs('../experiment/{}/imgs'.format(args.name))
    pred_c = real2complex_array(inputs)
    pred = np.squeeze(np.sqrt(np.sum(np.square(np.abs(pred_c)), axis=-1)))
    fig_name = os.path.join('../experiment/{}/imgs'.format(args.name),
                            '{}_{}_{}.png'.format(count, pictype, epoch+1))
    imageio.imsave(fig_name, pred)

def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)
    n_idx = np.arange(n)
    n_batch = (n+batch_size-1) // batch_size
    if shuffle:
        np.random.shuffle(n_idx)
    for i in range(n_batch):
        batch_idx = n_idx[i*batch_size : (i+1)*batch_size]
        yield data[batch_idx], n_batch

def real2complex_array(x):
    x = np.asarray(x)
    channel = x.shape[-1] // 2
    if len(x.shape) == 3:
        x_real = x[:,:,:channel]
        x_imag = x[:,:,channel:]
    elif len(x.shape) == 4:
        x_real = x[:,:,:,:channel]
        x_imag = x[:,:,:,channel:]
    return x_real + x_imag * 1j

def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:,:,:channel], x[:,:,channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])

def complex2real(x):
    x_real = tf.real(x)
    x_imag = tf.imag(x)
    return tf.concat([x_real,x_imag], axis=-1)

def imshow(img, title=""):
    """ Show image as grayscale. """
    if img.dtype == np.complex64 or img.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.show()

def loss_plot(loss, val_loss, loss_type, save_dir):
    iters = np.linspace(1, len(loss[loss_type]), len(loss[loss_type]))
    idx_min_train = np.argmin(loss[loss_type])
    fig = plt.figure()
    plt.title('Restoration on {}'.format('ComplexMRI'))
    
    # train loss
    plt.plot(iters, loss[loss_type], 'g', label='train loss')
    plt.plot(idx_min_train+1, loss[loss_type][idx_min_train], marker='v', color='k')
    plt.annotate('min:{:.6f}'.format(loss[loss_type][idx_min_train]),
                                xytext=(idx_min_train-5,loss[loss_type][idx_min_train]),
                                xy=(idx_min_train,loss[loss_type][idx_min_train]),
                                textcoords='data'
                                )
    # val loss
    plt.plot(iters, val_loss, 'r', label='val loss')
    idx_min_val = np.argmin(val_loss)
    plt.plot(idx_min_val+1, val_loss[idx_min_val], marker='v', color='k')
    plt.annotate('min:{:.6f}'.format(val_loss[idx_min_val]),
                                xytext=(idx_min_val-5,val_loss[idx_min_val]),
                                xy=(idx_min_val,val_loss[idx_min_val]),
                                textcoords='data'
                                )
    plt.grid(True)
    plt.xlabel(loss_type)
    plt.ylabel('Loss')
    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,2))
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir,'Loss_{}.png'.format(args.name)))
    plt.close(fig)


def lr_plot(lr_, save_dir):
    iters = np.linspace(1, len(lr_), len(lr_))
    min_idx = np.argmin(lr_)
    
    fig = plt.figure()
    plt.title('Restoration on {}'.format('ComplexMRI'))
    plt.plot(iters, lr_, 'g', label='epoch_psnr')
    plt.plot(min_idx+1, lr_[min_idx], marker='.', color='k')
    plt.annotate('min:{:.6f}'.format(lr_[min_idx]),
                                xytext=(min_idx-5,lr_[min_idx]),
                                xy=(min_idx,lr_[min_idx]),
                                textcoords='data'
                                )
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.legend(loc="best")
    a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    plt.savefig(os.path.join(save_dir,'lr{}.png'.format(a)))
    plt.close(fig)

def any_plot(dst_dict, dst_type, save_dir):
    assert type(dst_type) is str
    if dst_type.lower() == 'loss':
        # train loss
        input_list_train = dst_dict['loss_train_count']
        iters = np.linspace(1, 50, len(input_list_train))
        idx_min_train = np.argmin(input_list_train)
        fig = plt.figure()
        plt.title('Restoration on {}'.format('ComplexMRI'))
        
        
        plt.plot(iters, input_list_train, 'g', label='train loss')

        # val loss
        input_list_val = dst_dict['loss_val_count']
        idx_min_val = np.argmin(input_list_val)
        plt.plot(iters, input_list_val, 'r', label='val loss')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel(dst_type)
        ax = plt.gca()
        ax.yaxis.get_major_formatter().set_powerlimits((0,2))
        plt.legend(loc="best")
        a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        plt.savefig(os.path.join(save_dir,'Loss_{}.png'.format(a)))
        plt.close(fig)
    
    else:
        print(dst_type.lower())
        if dst_type.lower() == 'psnr':
            input_list = dst_dict['psnr_epoch']
        elif dst_type.lower() == 'ssim':
            input_list = dst_dict['ssim_epoch']
        elif dst_type.lower() == 'lr' or dst_type.lower().find('learning rate')>=0:
            input_list = dst_dict['lr_epoch']
        else: raise ValueError('Invalid Criterion: {}'.format(dst_type))

        iters = np.linspace(1, len(input_list), len(input_list))
        max_idx = np.argmax(input_list)
        
        fig = plt.figure()
        plt.title('Restoration on {}'.format(args.name))
        plt.plot(iters, input_list, 'g', label='{}_epoch'.format(dst_type))
        plt.plot(max_idx+1, input_list[max_idx], marker='v', color='k')
        plt.annotate('max:{:.3f}'.format(input_list[max_idx]),
                                    xytext=(max_idx,input_list[max_idx]),
                                    xy=(max_idx,input_list[max_idx]),
                                    textcoords='data'
                                    )
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel(dst_type)
        plt.legend(loc="best")
        a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        plt.savefig(os.path.join(save_dir,'{}_{}.png'.format(dst_type,a)))
        plt.close(fig)


def psnr_plot(psnr_, psnr_type, save_dir):
    iters = np.linspace(1, len(psnr_[psnr_type]), len(psnr_[psnr_type]))
    max_idx = np.argmax(psnr_[psnr_type])
    
    fig = plt.figure()
    plt.title('Restoration on {}'.format('ComplexMRI'))
    plt.plot(iters, psnr_[psnr_type], 'g', label='epoch_psnr')
    plt.plot(max_idx+1, psnr_[psnr_type][max_idx], marker='v', color='k')
    plt.annotate('max:{:.3f}'.format(psnr_[psnr_type][max_idx]),
                                xytext=(max_idx,psnr_[psnr_type][max_idx]),
                                xy=(max_idx,psnr_[psnr_type][max_idx]),
                                textcoords='data'
                                )
    plt.grid(True)
    plt.xlabel(psnr_type)
    plt.ylabel('PSNR')
    plt.legend(loc="best")
    a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
    plt.savefig(os.path.join(save_dir,'PSNR{}.png'.format(a)))
    plt.close(fig)

def norm(data, dtype='max'):
    """ Normalization """
    if dtype == 'max':
        norm = np.max(np.abs(data))
    elif dtype == 'no':
        norm = 1.0
    else:
        raise ValueError("Normalization has to be in ['max', 'no']")

    data /= norm
    return data

def norm_tf(data, dtype='max'):
    """ Normalization """
    if dtype == 'max':
        norm = tf.reduce_max(tf.abs(data))
    elif dtype == 'no':
        norm = 1.0
    else:
        raise ValueError("Normalization has to be in ['max', 'no']")

    data /= norm
    return data

def sos(im, axes=-1):
    '''Root sum of squares combination along given axes.

    Parameters
    ----------
    im : array_like
        Input image.
    axes : tuple
        Dimensions to sum across.

    Returns
    -------
    array_like
        SOS combination of image.
    '''
    return np.sqrt(np.sum(np.abs(im)**2, axis=axes))

def AdaptiveCoilCombine(img, Rn=None):  # (256,256,12)
    """
    Coil Combine according to DO Walsh et al. Adaptive Reconstruction of
    Phased Array MR Imagery, MRM, 2000.
    Img: 2D images of individual coil (row * column * TEs * coils)
    Rn: noise correlation matrix
    """
    if img.ndim == 4:
        H, W, TEs, coils = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    elif img.ndim == 3:
        TEs = 1
        H, W, coils = img.shape[0], img.shape[1], img.shape[2]
        img = img[:,:,np.newaxis,:]
    else: raise ValueError('Invalid image data!')

    if Rn is None:
        Rn = np.eye(coils)
    kernel_size = 7
    iRn = np.linalg.inv(Rn)  # 求逆

    Rs = np.zeros((H, W, coils, coils)).astype(np.complex128)  # 数据类型不匹配 allclose也会报错
    for i in range(coils):
        for j in range(coils):
            for n in range(TEs):
                Rs[:,:,i,j] = Rs[:,:,i,j] + correlate2d(img[:,:,n,i]*img[:,:,n,j].conj(), np.ones((kernel_size, kernel_size)), 'same')
                
    v = np.ones((H, W, coils)).astype(np.complex128)
    N = 2
    for i in range(N):
        v = np.squeeze(np.sum(Rs * np.transpose(np.tile(v,(coils,1,1,1)), (1,2,3,0)), axis=2))  # axis相较MATLABsum()的一定要减1
        d = np.sqrt(np.sum(v*v.conj(), axis=2))
        d[d<=np.spacing(1)] = np.spacing(1)
        v = v / np.transpose(np.tile(d, (coils,1,1)), (1,2,0))    
    v = np.transpose(np.tile(v, (TEs, 1,1,1)), (1,2,0,3))

    C_img = np.squeeze(np.sum(img * v, axis=3))
    
    return C_img

def ACC_tf(img, Rn=None):  # (256,256,12)
    """
    Coil Combine according to DO Walsh et al. Adaptive Reconstruction of
    Phased Array MR Imagery, MRM, 2000.
    Img: 2D images of individual coil (row * column * TEs * coils)
    Rn: noise correlation matrix
    """
    if img.shape.ndims == 4:
        [H, W, TEs, coils] = img.shape[0].value, img.shape[1].value, img.shape[2].value, img.shape[3].value, 
    elif img.shape.ndims == 3:
        TEs = 1
        [H, W, coils] = img.shape[0].value, img.shape[1].value, img.shape[2].value, 
        img = tf.expand_dims(img, axis=2)
    else: raise ValueError('Invalid image data!')

    if Rn is None:
        Rn = tf.eye(coils)
    kernel_size = 7
    iRn = tf.matrix_inverse(Rn)

    Rs = tf.zeros([H, W, coils, coils], dtype=tf.complex64)
    i_list = []
    for i in range(coils):
        j_list = []
        for j in range(coils):
            e = Rs[:,:,i,j]
            for n in range(TEs):
                a1 = img[:,:,n,i]
                a2 = tf.math.conj(img[:,:,n,j])
                a = a1 * a2
                a = tf.expand_dims(tf.expand_dims(a, 0), -1)
                a_complex = tf.concat([tf.real(a), tf.imag(a)], -1)
                b = tf.ones([kernel_size, kernel_size], dtype=tf.complex64)
                b = tf.expand_dims(tf.expand_dims(b, -1), -1)
                b_real = tf.concat([tf.real(b), -1*tf.imag(b)], -2)
                b_imag = tf.concat([tf.imag(b), tf.real(b)], -2)
                b_complex = tf.concat([b_real, b_imag], -1)
                c = tf.nn.convolution(a_complex, b_complex, padding='SAME')
                c = tf.complex(c[:,:,:,:1], -c[:,:,:,1:])
                d = tf.squeeze(c)
                e = e + d
            j_list.append(e)
        Rs_j = tf.stack(j_list, -1)
        i_list.append(Rs_j)
    Rs_ij = tf.stack(i_list, -2)
    Rs = tf.transpose(Rs_ij, [0,1,3,2])

    v = tf.ones([H, W, coils], dtype=tf.complex64)
    N = 2
    for i in range(N):
        a = tf.tile(tf.expand_dims(v, -1), [1,1,1,coils])
        b = tf.reduce_sum(Rs*a, axis=2)
        v = tf.squeeze(tf.reduce_sum(Rs * tf.tile(tf.expand_dims(v, -1), [1,1,1,coils]), axis=2))
        d = tf.cast(tf.sqrt(tf.reduce_sum(v*tf.conj(v), axis=2)), tf.float32)
        e = tf.ones_like(d)*np.spacing(1)
        d = tf.cast(tf.where(tf.less_equal(d, e), e, d), tf.complex64)
        v = tf.divide(v, tf.tile(tf.expand_dims(d, -1), [1,1,coils]))
    
    v = tf.expand_dims(v, 2)

    C_img = tf.squeeze(tf.reduce_sum(img*v, axis=3))
    
    return C_img

def psnr_tf(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return tf.image.psnr(gt, pred, max_val=tf.reduce_max(gt))

def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

def psnr1(org,recon):
    """ This function calculates PSNR between the original and
    the reconstructed images"""
    mse=np.sum(np.square( np.abs(org-recon)))/org.size
    psnr=20*np.log10(org.max()/(np.sqrt(mse)+1e-10 ))
    return psnr

def psnr2(org,recon):
    """ Designed like MATLAB. 
    This function calculates PSNR between the original and
    the reconstructed images"""
    mse=np.sum(np.square( np.abs(org-recon)))/org.size
    psnr=20*np.log10(org.max()/(np.sqrt(mse)))
    return psnr

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
  """
  2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

def calc_ssim(X, Y, sigma=1.5, K1=0.01, K2=0.03, R=255):
    '''
    X : y channel (i.e., luminance) of transformed YCbCr space of X
    Y : y channel (i.e., luminance) of transformed YCbCr space of Y
    Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
    Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
    The authors of EDSR use MATLAB's ssim as the evaluation tool, 
    thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2. 
    '''
    gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    window = gaussian_filter

    ux = signal.convolve2d(X, window, mode='same', boundary='symm')
    uy = signal.convolve2d(Y, window, mode='same', boundary='symm')

    uxx = signal.convolve2d(X*X, window, mode='same', boundary='symm')
    uyy = signal.convolve2d(Y*Y, window, mode='same', boundary='symm')
    uxy = signal.convolve2d(X*Y, window, mode='same', boundary='symm')

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    mssim = S.mean()

    return mssim

def ssim(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    # structural_similarity(gt.squeeze(), pred.squeeze(), multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
    return structural_similarity(gt.squeeze(), pred.squeeze(), multichannel=True, data_range=gt.max())

def ssim1(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return structural_similarity(np.abs(gt.squeeze()), np.abs(pred.squeeze()), multichannel=True, data_range=gt.max())

def ssim2(img, ref, dynamic_range=None):
    """ Compute SSIM. If inputs are 3D, average over axis=0.
        If dynamic_range != None, the same given dynamic range will be used for all slices in the volume. """
    assert img.ndim == ref.ndim
    assert img.ndim in [2, 3]
    if img.ndim == 2:
        img = img[np.newaxis]
        ref = ref[np.newaxis]

    # ssim averaged over slices
    ssim_slices = []
    ref_abs = np.transpose(np.abs(ref), (2, 0, 1))
    img_abs = np.transpose(np.abs(img), (2, 0, 1))  # (30,256,256)

    for i in range(ref_abs.shape[0]):
        if dynamic_range == None:
            drange = np.max(ref_abs[i]) - np.min(ref_abs[i])
        else:
            drange = dynamic_range
        _, ssim_i = structural_similarity(img_abs[i], ref_abs[i],
                                 data_range=drange,
                                 gaussian_weights=True,
                                 use_sample_covariance=False,
                                 full=True)
        ssim_slices.append(np.mean(ssim_i))

    return np.mean(ssim_slices)


def rmse(img, ref):
    """ Compute RMSE. If inputs are 3D, average over axis=0 """
    assert img.ndim == ref.ndim
    assert img.ndim in [2,3]
    if img.ndim == 2:
        axis = (0,1)
    elif img.ndim == 3:
        axis = (1,2)
    # else not possible

    denominator = np.sum(np.real(ref * np.conj(ref)), axis=axis)
    nominator = np.sum(np.real((img - ref) * np.conj(img - ref)), axis=axis)
    rmse = np.mean(np.sqrt(nominator / denominator))
    return rmse


def count_para1():
    print("Total number of trainable parameters: %d" % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

def count_para2():
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Total number of trainable parameters: %d" % total_parameters)

def count_para3():
    from functools import reduce
    from operator import mul
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    print(num_params)

def flops():
    pass

def plot_txt(txt_dir, xlabel='epoch'):
    # os.path.join(path, data_test)
    with open(txt_dir, 'r') as f:
        data_list = f.readlines()
        # axis = range(len(data_list))
        axis = np.linspace(1, len(data_list), len(data_list))
        data_y = np.array(data_list, dtype=np.float64)
        max_idx = np.argmax(data_y)
        min_idx = np.argmin(data_y)

        # plot this
        fig = plt.figure()
        plt.title('Restoration on {}'.format('ComplexMRI'))
        plt.plot(axis, data_y, label='baseline')
        plt.plot(max_idx+1, data_y[max_idx], marker='.', color='k')
        plt.plot(min_idx+1, data_y[min_idx], marker='.', color='k')
        plt.annotate('max:{:.6f}'.format(data_y[max_idx]),xytext=(max_idx+1,data_y[max_idx]),xy=(max_idx+1,data_y[max_idx]),textcoords='offset points',weight='heavy')
        plt.annotate('min:{:.6f}'.format(data_y[min_idx]),xytext=(min_idx-5,data_y[min_idx]),xy=(min_idx-5,data_y[min_idx]),textcoords='offset points',weight='heavy')
        plt.legend(loc="best")
        plt.xlabel(xlabel)
        ax = plt.gca()
        ax.yaxis.get_major_formatter().set_powerlimits((0,2))  # y轴使用科学计数法且显示数位为3
        plt.ylabel('PSNR')
        plt.grid(True)
        a = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
        plt.savefig('PSNR{}.png'.format(a))
        plt.close(fig)
        
        f.close()
    print('Plotting finished')


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0