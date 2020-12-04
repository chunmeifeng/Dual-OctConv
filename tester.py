"""
1.Add Parallel Data Training Version0.1
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
from data   import load_data_test, setup_inputs_test
from utils import *
from option import args


def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y_label, batch_kx_u, batch_mask_c):
    for i in range(len(models)):
        x, y_label, kx_u, mask_k, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i + 1) * payload_per_gpu
        inp_dict[x] = batch_x[int(start_pos):int(stop_pos)]
        inp_dict[y_label] = batch_y_label[int(start_pos):int(stop_pos)]
        inp_dict[kx_u] = batch_kx_u[int(start_pos):int(stop_pos)]
        inp_dict[mask_k] = batch_mask_c[int(start_pos):int(stop_pos)]

    return inp_dict

def test(test_data, mask):
    args.name = args.rsname
    experiment_dir = os.path.join('../experiment', args.name)
    model_dir = os.path.join(experiment_dir, 'model')
    result_dir = os.path.join(experiment_dir,'result')
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    assessment_dict={'loss_train_epoch':[], 'loss_val_epoch':[], 'lr_epoch':[], 'psnr_epoch':[], 'ssim_epoch':[], 'psnr_slice':[], 'ssim_slice':[]}

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
                    models.append((x, y_label, kx_u, masks, pred))
    tower_x, tower_y_label, tower_x_k, tower_mask_c, tower_preds = zip(*models)

    all_preds = tf.reshape(tf.stack(tower_preds, 0), [-1,args.data_size,args.data_size,args.in_channels])

    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Print params and flops
    flops = tf.compat.v1.profiler.profile(options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('Original flops :',flops.total_float_ops)
    print('Original params:',params.total_parameters)

    with tf.compat.v1.Session(config=config) as sess:
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        best_epoch = 0
        if args.rsname is not None:
            from_exp_dir = os.path.join('../experiment', args.rsname)
            from_model_dir = os.path.join(from_exp_dir, 'model')
            print()
            saver.restore(sess, tf.train.latest_checkpoint(from_model_dir))
            
            print("Model restored from file: {}".format(tf.train.latest_checkpoint(from_model_dir)))
            best_epoch = int(tf.train.latest_checkpoint(from_model_dir).split('-')[-1])//int(np.ceil(args.nb_train*8//args.batch_size))
                       
        payload_per_gpu = args.batch_size / args.n_GPU

        if args.test_only:
            count = 0
            test_PSNR = 0
            ave_test_PSNR = 0
            test_SSIM = 0
            ave_test_SSIM = 0
            inp_dict = {}

            for y_test, n_batch in iterate_minibatch(test_data, batch_size=4, shuffle=False):
                features_test, labels_test, kx_u_test, mask_c_test = setup_inputs_test(y_test, mask, norm=None)
                inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, features_test, labels_test, kx_u_test, mask_c_test)
                batch_prediction = sess.run(all_preds, inp_dict)
                for i in range(batch_prediction.shape[0]):
                    count += 1
                    label = labels_test[i]
                    prediction = batch_prediction[i]

                    if args.save_gt is True:
                        save2img(label, 'gt', count, -1)
                        save2mat(label, 'gt', count, -1)
                    if args.save_results is True:
                        save2img(prediction, 'ee', count, best_epoch)
                        save2mat(prediction, 'ee', count, best_epoch)
                    
                    label_acc = AdaptiveCoilCombine(real2complex_array(label))  # way in MATLAB
                    prediction_acc = AdaptiveCoilCombine(real2complex_array(prediction))  # way in MATLAB

                    ################## postprocessing #####################
                    # if args.data_dst.find('coronal_pd')>=0 or args.data_dst.find('axial_t2')>=0 or args.data_dst.find('coronal_pd_fs')>=0:
                    #     label_acc = np.flipud(np.fliplr(label_acc))
                    #     prediction_acc = np.flipud(np.fliplr(prediction_acc))
                    # elif args.data_dst.find('sagittal_pd')>=0 or args.data_dst.find('sagittal_t2')>=0:
                    #     label_acc = np.flipud(np.rot90(label_acc))
                    #     prediction_acc = np.flipud(np.rot90(prediction_acc))
                    # else:
                    #     print(Warning("Postprocessing not defined for dataset %s" % dataset))
                    ################################################

                    PSNR = psnr(norm(abs(label_acc)), norm(abs(prediction_acc)))  # way in MATLAB
                    SSIM = calc_ssim(norm(abs(label_acc)), norm(abs(prediction_acc)), R=1.0)  # way in MATLAB

                    if args.everyslices:
                        assessment_dict['psnr_slice'].append(PSNR)
                        assessment_dict['ssim_slice'].append(SSIM)

                    test_PSNR += PSNR
                    test_SSIM += SSIM

            ave_test_PSNR = test_PSNR / count
            ave_test_SSIM = test_SSIM / count

            assessment_dict['psnr_epoch'].append(ave_test_PSNR)
            assessment_dict['ssim_epoch'].append(ave_test_SSIM)
            idx_psnr_max = np.argmax(assessment_dict['psnr_epoch'])
            idx_ssim_max = np.argmax(assessment_dict['ssim_epoch'])
            print('inmatlab PSNR:{:.4f}  inmatlab SSIM:{:.4f}\n'.format(ave_test_PSNR, ave_test_SSIM))
            print('[Epoch{:3d}]  PSNR: {:.4f} (Best:{:.4f} @epoch {})  SSIM: {:.4f} (Best:{:.4f} @epoch {})'.format(
                    best_epoch, ave_test_PSNR, assessment_dict['psnr_epoch'][idx_psnr_max], idx_psnr_max+1,
                                ave_test_SSIM, assessment_dict['ssim_epoch'][idx_ssim_max], idx_ssim_max+1))

        with open('../experiment/tester_list.txt'.format(args.name), 'a') as f:
            f.write('\nModel name:{}\n[Epoch{:4d}]  PSNR: {:.4f} (Best:{:.4f} @epoch {})  SSIM: {:.4f} (Best:{:.4f} @epoch {})'.format(
                    args.name, best_epoch, ave_test_PSNR, assessment_dict['psnr_epoch'][idx_psnr_max], idx_psnr_max+1,
                                ave_test_SSIM, assessment_dict['ssim_epoch'][idx_ssim_max], idx_ssim_max+1))

        if args.everyslices:
            np.savetxt(os.path.join(result_dir, 'slices_psnr_{}.txt'.format(args.name)), np.asarray(assessment_dict['psnr_slice']), fmt='%.4f')
            np.savetxt(os.path.join(result_dir, 'slices_ssim_{}.txt'.format(args.name)), np.asarray(assessment_dict['ssim_slice']), fmt='%.4f')

        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    # if not os.path.exists('../checkpoint'):
    #     os.makedirs('../checkpoint')
    
    test_data, mask = load_data_test(args.data_dir, args.batch_size)
    test(test_data, mask)

if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
    # os.environ["CUDA_VISIBLE_DEVICES"]='5'
    tf.compat.v1.app.run()