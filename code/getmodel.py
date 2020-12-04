import tensorflow as tf
from model.common import *
# from model.KernelInit import ComplexInit
# from model.oct_conv2d import OctConv2D
from option import args

def getModel(X, X_k, mask, alpha):
    if args.name.find('DeepComplex')>=0:
        with open('../experiment/{}/DeepComplex indeed.txt'.format(args.name), 'w') as f:
            f.write('Modle is DeepComplex indeed.')
        temp = X
        for i in range(args.n_blocks):
            conv1 = normal_complex_conv2d(temp, 'conv' + str(i + 1) + '1', kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
            conv2 = normal_complex_conv2d(conv1, 'conv' + str(i + 1) + '2', kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
            conv3 = normal_complex_conv2d(conv2, 'conv' + str(i + 1) + '3', kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
            conv4 = normal_complex_conv2d(conv3, 'conv' + str(i + 1) + '4', kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
            conv5 = normal_complex_conv2d(conv4, 'conv' + str(i + 1) + '5', kw=3, kh=3, n_out=args.in_channels//2, sw=1, sh=1, activation=False)

            block = conv5 + temp
            temp = data_consistency(block, X_k, mask)

    elif args.name.find('OctComplex')>=0:
        with open('../experiment/{}/OctComplex indeed.txt'.format(args.name), 'w') as f:
            f.write('Modle is OctComplex indeed.')
        C = args.n_feats // 2
        temp = X
        for i in range(args.n_blocks):
            conv1_h, conv1_l = firstOct_complex_conv2d(temp, 'conv' + str(i + 1) + '1', kw=3, kh=3, n_out=C, sw=1, sh=1, activation=True, alpha=alpha)
            conv2_h, conv2_l = oct_complex_conv2d([conv1_h,conv1_l], 'conv' + str(i + 1) + '2', kw=3, kh=3, n_out=C, sw=1, sh=1, activation=True, alpha=alpha)
            conv3_h, conv3_l = oct_complex_conv2d([conv2_h,conv2_l], 'conv' + str(i + 1) + '3', kw=3, kh=3, n_out=C, sw=1, sh=1, activation=True, alpha=alpha)
            conv4_h, conv4_l = oct_complex_conv2d([conv3_h,conv3_l], 'conv' + str(i + 1) + '4', kw=3, kh=3, n_out=C, sw=1, sh=1, activation=True, alpha=alpha)
            conv4_l = tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(conv4_l)
            conv4 = tf.concat([conv4_h, conv4_l], axis=-1)
            conv5 = lastOct_complex_conv2d(conv4, 'conv' + str(i + 1) + '5', kw=3, kh=3, n_out=args.in_channels//2, sw=1, sh=1, activation=False, alpha=alpha)

            block = conv5 + temp
            temp = data_consistency(block, X_k, mask)

    return temp

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args.name = 'OctComplex_B10_a0.125_cpd320_1Un3'

    X = tf.Variable(tf.random_uniform([1, 320, 320, 30]), name="X", trainable=False)
    
    real = tf.Variable(tf.random_uniform([1, 320, 320, 15]), trainable=False)
    imag = tf.Variable(tf.random_uniform([1, 320, 320, 15]), trainable=False)
    X_k = tf.complex(real, imag, name='X_k')
    
    m_real = tf.Variable(tf.ones([1, 320, 320, 15]), trainable=False)
    m_imag = tf.Variable(tf.ones([1, 320, 320, 15]), trainable=False)
    mask = tf.complex(m_real, m_imag, name="mask")
    
    y_ = tf.Variable(tf.random_uniform([1, 320, 320, 30]), name="y_label", trainable=False)

    y = getModel(X,X_k,mask,alpha=0.5)
    
    from losses import mae
    loss = mae(y_, y)
    
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    flops = tf.compat.v1.profiler.profile(options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('Original flops :',flops.total_float_ops)
    print('Original params:',params.total_parameters)


    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out_loss = sess.run(loss)
        print(out_loss)