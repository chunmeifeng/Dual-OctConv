"""
1.Add eval_every
"""

import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from getmodel import getModel
from losses import mae, mse, perceptual_loss
from data   import setup_inputs, load_data, setup_inputs_test
from utils import *
from option import args
from tensorflow.python.framework import graph_util


def tower_loss(label, pred, scope, reuse_variables=None):
    if args.loss is 'L1':
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            loss = mae(label, pred)
    elif args.loss is 'L2':
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            loss = mse(label, pred)
    else: raise ValueError('Invalid Criterion: {}'.format(args.loss))

    return loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y_label, batch_kx_u, batch_mask_c):
    for i in range(len(models)):
        x, y_label, kx_u, mask_k, _, _, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i + 1) * payload_per_gpu
        inp_dict[x] = batch_x[int(start_pos):int(stop_pos)]
        inp_dict[y_label] = batch_y_label[int(start_pos):int(stop_pos)]
        inp_dict[kx_u] = batch_kx_u[int(start_pos):int(stop_pos)]
        inp_dict[mask_k] = batch_mask_c[int(start_pos):int(stop_pos)]

    return inp_dict


def train(train_batch, validate_batch, test_data, mask):
    """setup_inputs
    input:  train_batch, (?, 256, 256, 24) sd float64   | mask,   (?, 256, 256)     fd uint8
    output: features,    (?, 256, 256, 24) sd float32   | labels, (?, 256, 256, 24) sd float32
            kx_mask,     (?, 256, 256, 12) fd complex64 | mask_c,  (?, 256, 256, 12) fd complex64
    """

    experiment_dir = os.path.join('../experiment', args.name)
    model_dir = os.path.join(experiment_dir, 'model')
    result_dir = os.path.join(experiment_dir,'result')
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    assessment_dict={'loss_train_count':[], 'loss_val_count':[], 'lr_epoch':[], 'psnr_epoch':[], 'ssim_epoch':[]}

    # Loading data.
    features, labels, kx_undersamp, mask_c = setup_inputs(train_batch, mask, args.batch_size)
    f_val, l_val, kx_uds_val, m_val = setup_inputs(validate_batch, mask, args.batch_size)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    lr = tf.compat.v1.train.exponential_decay(args.lr,
                                    global_step=global_step,
                                    decay_steps=(args.nb_train*8 + args.batch_size-1) // args.batch_size,  # ???!!!
                                    decay_rate=args.lr_decay_rate,
                                    staircase=False)

    AdamOpt = tf.compat.v1.train.AdamOptimizer(lr)

    models = []
    reuse_variables = False
    for gpu_id in range(args.n_GPU):
        with tf.device('/gpu:{}'.format(gpu_id)):
            print('GPU No.{}...'.format(gpu_id))
            with tf.name_scope('gpu_{}'.format(gpu_id)) as scope:
                with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                    x = tf.compat.v1.placeholder(tf.float32,shape=(None,args.data_size,args.data_size,args.in_channels),name='x_input')
                    y_label = tf.compat.v1.placeholder(tf.float32,shape=(None,args.data_size,args.data_size,args.in_channels),name='y_label')
                    kx_u = tf.compat.v1.placeholder(tf.complex64,shape=(None,args.data_size,args.data_size,args.in_channels//2),name='kx_undersamp')
                    masks = tf.compat.v1.placeholder(tf.complex64,shape=(None,args.data_size,args.data_size,args.in_channels//2),name='mask_coils')
                    pred = getModel(x, kx_u, masks, alpha=args.alpha)
                    loss = tower_loss(y_label, pred, scope, reuse_variables)
                    reuse_variables = True
                    grads = AdamOpt.compute_gradients(loss)
                    models.append((x, y_label, kx_u, masks, pred, loss, grads))

    tower_x, tower_y_label, tower_x_k, tower_mask_c, tower_preds, tower_losses, tower_grads = zip(*models)
    aver_loss_op = tf.reduce_mean(tower_losses)
    apply_gradient_op = AdamOpt.apply_gradients(average_gradients(tower_grads), global_step=global_step)

    all_y_label = tf.reshape(tf.stack(tower_y_label, 0), [-1,args.data_size,args.data_size,args.in_channels])
    all_preds = tf.reshape(tf.stack(tower_preds, 0), [-1,args.data_size,args.data_size,args.in_channels])

    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Print params and flops
    flops = tf.compat.v1.profiler.profile(options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('Original flops :',flops.total_float_ops)
    print('Original params:',params.total_parameters)

    with open('../experiment/{}/config.txt'.format(args.name), 'w') as f:
        f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M') + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            print('{}: {}'.format(arg, getattr(args, arg)))
        f.write('\n')
        f.write('\nOriginal flops:{}\nOriginal params:{}'.format(flops.total_float_ops, params.total_parameters))

    with tf.compat.v1.Session(config=config) as sess:
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        start_epoch = 0
        if args.rsname is not None:
            from_exp_dir = os.path.join('../experiment', args.rsname)
            from_model_dir = os.path.join(from_exp_dir, 'model')
            saver.restore(sess, tf.train.latest_checkpoint(from_model_dir))

            # Change based learning rate
            if args.reset_lr is not None:
                sess.run(global_step.initializer)
                lr = tf.train.exponential_decay(args.reset_lr,
                                    global_step=global_step,
                                    decay_steps=(args.nb_train*8 + args.batch_size-1) // args.batch_size,
                                    decay_rate=args.lr_decay_rate,
                                    staircase=False)
                with tf.name_scope("train"):
                    train_step = tf.train.AdamOptimizer(lr).minimize(total_loss,global_step=global_step)

                var_list = sess.run(tf.report_uninitialized_variables())
                var_r_list = [var_list[i].decode('utf-8') for i in range(var_list.shape[0])]
                variables_to_restore = tf.contrib.slim.get_variables_to_restore(include=var_r_list)
                sess.run(tf.variables_initializer(var_list=variables_to_restore))

            # Change optimizer
            if args.reset_op is not None:
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                print('Trainable variables', len(var_list))
                a_list = tf.get_collection(tf.GraphKeys.VARIABLES)
                b_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                print('All variables :', a_list)
                print('Tra variables :', b_list)

                opt = tf.train.GradientDescentOptimizer(lr)
                train_step = opt.minimize(total_loss, var_list=var_list, global_step=global_step)
                opt_var_list = [opt.get_slot(var, name) for name in opt.get_slot_names() for var in var_list]
                reset_opt_op = tf.variables_initializer(opt_var_list)
                
                sess.run(reset_opt_op)
            
            print("Model restored from file: {}".format(tf.train.latest_checkpoint(from_model_dir)))
            start_epoch = int(tf.train.latest_checkpoint(from_model_dir).split('-')[-1])//int(np.ceil(args.nb_train*8//args.batch_size))
            with open('../experiment/{}/result/lr_{}.txt'.format(args.rsname,args.rsname), 'r') as f:
                assessment_dict['lr_epoch'] = [float(i) for i in f.readlines()[:start_epoch]]
            with open('../experiment/{}/result/train_loss_{}.txt'.format(args.rsname,args.rsname), 'r') as f:
                assessment_dict['loss_train_count'] = [float(i) for i in f.readlines()[:start_epoch]]
            with open('../experiment/{}/result/val_loss_{}.txt'.format(args.rsname,args.rsname), 'r') as f:
                assessment_dict['loss_val_count'] = [float(i) for i in f.readlines()[:start_epoch]]
            with open('../experiment/{}/result/train_psnr_{}.txt'.format(args.rsname,args.rsname), 'r') as f:
                assessment_dict['psnr_epoch'] = [float(i) for i in f.readlines()[:start_epoch]]
            with open('../experiment/{}/result/train_ssim_{}.txt'.format(args.rsname,args.rsname), 'r') as f:
                assessment_dict['ssim_epoch'] = [float(i) for i in f.readlines()[:start_epoch]]

        imgs_epoch_list = np.linspace(1, args.epochs, 5, dtype=int)
        payload_per_gpu = args.batch_size / args.n_GPU

        for epoch in range(start_epoch, args.epochs):
            # Training Step
            loss_sum = 0.
            count_batch = 0
            ave_loss = 0.
            total_batches = int(np.ceil(args.nb_train*8/ args.batch_size))

            # show and record learning rate
            lr_now = sess.run(lr)
            print('\n====================',' Epoch:{:>3d} | Learning rate:{:>8e}'.format(epoch+1, lr_now), 
                    '====================')
            assessment_dict['lr_epoch'].append(lr_now)
            tbar = tqdm(range(total_batches))
            for n_batch in tbar:
                inp_dict = {}
                features_trian, labels_train, kx_u_train, mask_c_train = sess.run([features, labels, kx_undersamp, mask_c])  # output np.array
                inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, features_trian, labels_train, kx_u_train, mask_c_train)
                _, loss_value, step = sess.run([apply_gradient_op, aver_loss_op, global_step], feed_dict=inp_dict)
                                    
                loss_sum += loss_value
                count_batch += 1
                ave_loss = loss_sum / count_batch
                tbar.set_description('[Epoch{:>3d}]  batch: {:>3d}/{:>3d} |   training loss: {:.8f} |'.format(
                                                                    epoch+1, count_batch, total_batches, ave_loss))
            
                # Validation Step
                if count_batch % args.eval_every == 0:
                    count_batch_val = 0
                    loss_sum_val = 0.
                    ave_loss_val = 0.
                    total_batches_val = int(np.ceil(args.nb_val // args.batch_size))
                    inp_dict = {}
                    tbar = tqdm(range(total_batches_val))
                    for n_batch_val in tbar:
                        features_val, labels_val, kx_u_val, mask_c_val = sess.run([f_val, l_val, kx_uds_val, m_val])
                        inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, features_val, labels_val, kx_u_val, mask_c_val)
                        loss_value_val = sess.run(aver_loss_op, feed_dict=inp_dict)

                        loss_sum_val += loss_value_val
                        count_batch_val += 1
                        ave_loss_val = loss_sum_val / count_batch_val
                        tbar.set_description('[Epoch{:>3d}]  batch: {:>3d}/{:>3d} | validation loss: {:.8f} |'.format(
                                                            epoch+1, count_batch_val, total_batches_val, ave_loss_val))
                        
                    assessment_dict['loss_val_count'].append(ave_loss_val)
                    assessment_dict['loss_train_count'].append(ave_loss)

            # Testing Step
            if (epoch+1) % args.test_every == 0:
                count = 0
                test_PSNR = 0
                ave_test_PSNR = 0
                test_SSIM = 0
                ave_test_SSIM = 0

                for y_test, n_batch in iterate_minibatch(test_data, batch_size=4, shuffle=False):
                    features_test, labels_test, kx_u_test, mask_c_test = setup_inputs_test(y_test, mask, norm=None)
                    inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, features_test, labels_test, kx_u_test, mask_c_test)
                    batch_prediction = sess.run(all_preds, inp_dict)
                    for i in range(batch_prediction.shape[0]):
                        count += 1
                        label = labels_test[i]
                        prediction = batch_prediction[i]
                        
                        if args.save_gt is True and epoch==0 and epoch+1 in imgs_epoch_list:
                            save2img(label, 'gt', count, -1)
                        if args.save_results is True and epoch+1 in imgs_epoch_list:
                            save2img(prediction, 'ee', count, epoch)
                        
                        # PSNR = psnr(norm(abs(label_acc)), norm(abs(prediction_acc)))  # way in MATLAB
                        PSNR = psnr(label, prediction)  # way in Python
                        # SSIM = ssim(abs(label_acc), abs(prediction_acc))  # way in MATLAB
                        SSIM = ssim(prediction, label)  # way in Python
                        test_PSNR += PSNR
                        test_SSIM += SSIM

                ave_test_PSNR = test_PSNR / count
                ave_test_SSIM = test_SSIM / count

                assessment_dict['psnr_epoch'].append(ave_test_PSNR)
                assessment_dict['ssim_epoch'].append(ave_test_SSIM)
                idx_psnr_max = np.argmax(assessment_dict['psnr_epoch'])
                idx_ssim_max = np.argmax(assessment_dict['ssim_epoch'])

                print('[Epoch{:3d}]  PSNR: {:.3f} (Best:{:.3f} @epoch {})  SSIM: {:.5f} (Best:{:.5f} @epoch {})'.format(
                        epoch+1, ave_test_PSNR, assessment_dict['psnr_epoch'][idx_psnr_max], idx_psnr_max+1,
                                 ave_test_SSIM, assessment_dict['ssim_epoch'][idx_ssim_max], idx_ssim_max+1))
                                 
                if ave_test_PSNR >= assessment_dict['psnr_epoch'][idx_psnr_max]:
                    saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=global_step)

            np.savetxt(os.path.join(result_dir, 'train_psnr_{}.txt'.format(args.name)), np.asarray(assessment_dict['psnr_epoch']))
            np.savetxt(os.path.join(result_dir, 'train_ssim_{}.txt'.format(args.name)), np.asarray(assessment_dict['ssim_epoch']))
            np.savetxt(os.path.join(result_dir, 'train_loss_{}.txt'.format(args.name)), np.asarray(assessment_dict['loss_train_count']))
            np.savetxt(os.path.join(result_dir, 'val_loss_{}.txt'.format(args.name)), np.asarray(assessment_dict['loss_val_count']))
            np.savetxt(os.path.join(result_dir, 'lr_{}.txt'.format(args.name)), np.asarray(assessment_dict['lr_epoch']))

        with open('../experiment/{}/config.txt'.format(args.name), 'a') as f:
            f.write('\nPSNR Best:{:.3f} @epoch {} | SSIM Best:{:.5f} @epoch {}'.format(
                    assessment_dict['psnr_epoch'][idx_psnr_max], idx_psnr_max+1,assessment_dict['ssim_epoch'][idx_ssim_max], idx_ssim_max+1))

        coord.request_stop()
        coord.join(threads)

        any_plot(assessment_dict, 'psnr', save_dir=experiment_dir)
        any_plot(assessment_dict, 'ssim', save_dir=experiment_dir)
        any_plot(assessment_dict, 'loss', save_dir=experiment_dir)
        any_plot(assessment_dict, 'lr', save_dir=experiment_dir)

def main(argv=None):
    # if not os.path.exists('../checkpoint'):
    #     os.makedirs('../checkpoint')
    
    train_data, validate_data, test_data, mask = load_data(args.data_dir, args.batch_size)
    train(train_data, validate_data, test_data, mask)

if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
    tf.compat.v1.app.run()