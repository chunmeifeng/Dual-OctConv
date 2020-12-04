import tensorflow as tf
from model.KernelInit import ComplexInit
from model.oct_conv2d import OctConv2D
from option import args

def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:,:,:channel], x[:,:,channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])

def data_consistency(generated, X_k, mask):
    gene_complex = real2complex(generated)
    gene_complex = tf.transpose(gene_complex,[0, 3, 1, 2])
    mask = tf.transpose(mask,[0, 3, 1, 2])
    X_k = tf.transpose(X_k,[0, 3, 1, 2])
    gene_fft = tf.fft2d(gene_complex)
    out_fft = X_k + gene_fft * (1.0 - mask)
    output_complex = tf.ifft2d(out_fft)
    output_complex = tf.transpose(output_complex, [0, 2, 3, 1])
    output_real = tf.cast(tf.real(output_complex), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(output_complex), dtype=tf.float32)
    output = tf.concat([output_real,output_imag], axis=-1)

    return output

def normal_complex_conv2d(input_data,layer_name,kw=3,kh=3,n_out=32,sw=1,sh=1,activation=True):
    n_in = input_data.get_shape()[-1].value // 2
    with tf.variable_scope(layer_name):
        kernel_init = ComplexInit(kernel_size=(kh,kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=n_out,
                                  criterion='he')
        """kernel
        name: convi1/weights:0
        shape: (3, 3, 12, 64)
        """
        kernel = tf.get_variable('weights',
                                 shape=[kh,kw,n_in,n_out*2],
                                 dtype=tf.float32,
                                 initializer=kernel_init)
        bias_init = tf.constant(0.0001,dtype=tf.float32,shape=[n_out*2])
        biases = tf.get_variable('biases', dtype=tf.float32, initializer=bias_init)
        kernel_real = kernel[:,:,:,:n_out]
        kernel_imag = kernel[:,:,:,n_out:]
        
        cat_kernel_real = tf.concat([kernel_real, -kernel_imag], axis=-2)
        cat_kernel_imag = tf.concat([kernel_imag, kernel_real], axis=-2)
        cat_kernel_complex = tf.concat([cat_kernel_real,cat_kernel_imag], axis=-1)
        # conv = tf.keras.layers.conv2d(input, 2*n_out, (kh, kw), padding='SAME')  # w/o complex kernel initialization
        conv = tf.nn.conv2d(input_data,cat_kernel_complex,strides=[1,sh,sw,1],padding='SAME')
        conv_bias = tf.nn.bias_add(conv,biases)
        if activation:
            act = tf.nn.relu(conv_bias)
            output = act

        else:
            output = conv_bias
        return output

def firstOct_complex_conv2d(inp_data,layer_name,kw=3,kh=3,n_out=32,sw=1,sh=1,activation=True, alpha=0.5):
    n_in = inp_data.get_shape()[-1].value // 2
    with tf.variable_scope(layer_name):
        kernel_init = ComplexInit(kernel_size=(kh,kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=n_out,
                                  criterion='he')
        kernel = tf.compat.v1.get_variable('weights',
                                 shape=[kh,kw,n_in,n_out*2],
                                 dtype=tf.float32,
                                 initializer=kernel_init)
        kernel_real = kernel[:,:,:,:n_out]
        kernel_imag = kernel[:,:,:,n_out:]
        
        bias_init = tf.constant(0.0001,dtype=tf.float32,shape=[n_out*2])
        bias = tf.compat.v1.get_variable('bias', dtype=tf.float32, initializer=bias_init)
        bias_h = bias[:int(n_out*2*(1-alpha))]
        bias_l = bias[int(n_out*2*(1-alpha)):]
        
        input_r = inp_data[:,:,:,:n_in]
        input_i = inp_data[:,:,:,n_in:]
        
        kernel_1 = kernel_real
        kernel_2 = -kernel_imag
        kernel_3 = kernel_imag
        kernel_4 = kernel_real

        high1, low1 = OctConv2D(filters_out=n_out, kernel=kernel_1, alpha_out=alpha)(input_r)
        high2, low2 = OctConv2D(filters_out=n_out, kernel=kernel_2, alpha_out=alpha)(input_i)
        high3, low3 = OctConv2D(filters_out=n_out, kernel=kernel_3, alpha_out=alpha)(input_r)
        high4, low4 = OctConv2D(filters_out=n_out, kernel=kernel_4, alpha_out=alpha)(input_i)

        conv_m1_h = high1+high2
        conv_m1_l = low1 +low2
        conv_m2_h = high3+high4
        conv_m2_l = low3 +low4
        conv_o_h = tf.concat([conv_m1_h, conv_m2_h], axis=-1)
        conv_o_l = tf.concat([conv_m1_l, conv_m2_l], axis=-1)
        
        conv_bias_h = tf.nn.bias_add(conv_o_h,bias_h)
        conv_bias_l = tf.nn.bias_add(conv_o_l,bias_l)
        if activation:
            act_h = tf.nn.relu(conv_bias_h)
            act_l = tf.nn.relu(conv_bias_l)
            output_h = act_h
            output_l = act_l
        else:
            output_h = conv_bias_h
            output_l = conv_bias_l
        return output_h, output_l

def lastOct_complex_conv2d(input_data,layer_name,kw=3,kh=3,n_out=32,sw=1,sh=1,activation=True, alpha=0.5):
    n_in = input_data.get_shape()[-1].value // 2
    with tf.variable_scope(layer_name):
        kernel_init = ComplexInit(kernel_size=(kh,kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=n_out,
                                  criterion='he')
        kernel = tf.get_variable('weights',
                                 shape=[kh,kw,n_in,n_out*2],
                                 dtype=tf.float32,
                                 initializer=kernel_init)
        kernel_real = kernel[:,:,:,:n_out]
        kernel_imag = kernel[:,:,:,n_out:]
        
        bias_init = tf.constant(0.0001,dtype=tf.float32,shape=[n_out*2])
        biases = tf.get_variable('biases', dtype=tf.float32, initializer=bias_init)

        # separate kernel
        input_r = input_data[:,:,:,:n_in]
        input_i = input_data[:,:,:,n_in:]
        kernel_1 = kernel_real
        kernel_2 = -kernel_imag
        kernel_3 = kernel_imag
        kernel_4 = kernel_real

        conv1 = tf.nn.conv2d(input_r, kernel_1,strides=[1,sh,sw,1],padding='SAME')
        conv2 = tf.nn.conv2d(input_i, kernel_2,strides=[1,sh,sw,1],padding='SAME')
        conv3 = tf.nn.conv2d(input_r, kernel_3,strides=[1,sh,sw,1],padding='SAME')
        conv4 = tf.nn.conv2d(input_i, kernel_4,strides=[1,sh,sw,1],padding='SAME')

        conv_m1 = conv1+conv2
        conv_m2 = conv3+conv4
        conv_o = tf.concat([conv_m1, conv_m2], axis=-1)
        
        conv_bias = tf.nn.bias_add(conv_o,biases)
        if activation:
            act = tf.nn.relu(conv_bias)
            output = act

        else:
            output = conv_bias
        return output

def oct_complex_conv2d(input_data,layer_name,kw=3,kh=3,n_out=32,sw=1,sh=1,activation=True, alpha=0.5):
    assert len(input_data) == 2
    n_in_h = input_data[0].get_shape()[-1].value
    n_in_l = input_data[1].get_shape()[-1].value
    n_in = (n_in_h+n_in_l) // 2  # 32
    with tf.variable_scope(layer_name):
        kernel_init = ComplexInit(kernel_size=(kh,kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=n_out,
                                  criterion='he')
        kernel = tf.get_variable('weights',
                                 shape=[kh,kw,n_in,n_out*2],
                                 dtype=tf.float32,
                                 initializer=kernel_init)
        kernel_real = kernel[:,:,:,:n_out]
        kernel_imag = kernel[:,:,:,n_out:]
        
        bias_init = tf.constant(0.0001,dtype=tf.float32,shape=[n_out*2])
        bias = tf.compat.v1.get_variable('bias', dtype=tf.float32, initializer=bias_init)
        bias_h = bias[:int(n_out*2*(1-alpha))]
        bias_l = bias[int(n_out*2*(1-alpha)):]

        input_h_r = input_data[0][:,:,:,:n_in_h//2]
        input_h_i = input_data[0][:,:,:,n_in_h//2:]
        input_l_r = input_data[1][:,:,:,:n_in_l//2]
        input_l_i = input_data[1][:,:,:,n_in_l//2:]
        
        kernel_1 = kernel_real
        kernel_2 = -kernel_imag
        kernel_3 = kernel_imag
        kernel_4 = kernel_real

        high1, low1 = OctConv2D(filters_out=n_out, kernel=kernel_1, alpha_out=alpha)([input_h_r, input_l_r])
        high2, low2 = OctConv2D(filters_out=n_out, kernel=kernel_2, alpha_out=alpha)([input_h_i, input_l_i])
        high3, low3 = OctConv2D(filters_out=n_out, kernel=kernel_3, alpha_out=alpha)([input_h_r, input_l_r])
        high4, low4 = OctConv2D(filters_out=n_out, kernel=kernel_4, alpha_out=alpha)([input_h_i, input_l_i])

        conv_m1_h = high1+high2
        conv_m1_l = low1 +low2
        conv_m2_h = high3+high4
        conv_m2_l = low3 +low4
        conv_o_h = tf.concat([conv_m1_h, conv_m2_h], axis=-1)
        conv_o_l = tf.concat([conv_m1_l, conv_m2_l], axis=-1)
        
        conv_bias_h = tf.nn.bias_add(conv_o_h,bias_h)
        conv_bias_l = tf.nn.bias_add(conv_o_l,bias_l)
        if activation:
            act_h = tf.nn.relu(conv_bias_h)
            act_l = tf.nn.relu(conv_bias_l)
            output_h = act_h
            output_l = act_l
        else:
            output_h = conv_bias_h
            output_l = conv_bias_l
        return output_h, output_l