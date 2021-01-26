import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from model.UnetGAN_percetual_loss import UnetGAN
from datetime import datetime
import logging
import load as loader
import random
import numpy as np
import lib.metrics as metrics

tf_summary = tf.compat.v1.summary

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')

# IBIS HR
tf.flags.DEFINE_integer('image_size_z', 144, 'image size z dimension')
tf.flags.DEFINE_integer('image_size_y', 192, 'image size y dimension')
tf.flags.DEFINE_integer('image_size_x', 160, 'image size x dimension')


tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_float('lamda_l1', 25.0,
                      'coefficient of l1 norm, default: 100')
tf.flags.DEFINE_float('lamda_p', 25.0,
                      'coefficient of l1 norm, default: 0.0')

# IBIS
tf.flags.DEFINE_string('paired_image_path',
                       '/ASD/Autism/IBIS2/IBIS_DL_Prediction/Data_preprocessed/pickle/paired_6_12_HR_QC',
                       # '/Users/yusenlin/Documents/github/pickledata_test/data_tmp',
                       'where pickcle data saved. format:[x,y,path]')

tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')

tf.flags.DEFINE_string('weight_dir', '/ASD/Autism/IBIS2/IBIS_DL_Prediction/Code/model_weights/unet_3d.pkl',
                       'the dir of weight file for feature extractor')

# IBIS
data_train = loader.Data(os.path.join(FLAGS.paired_image_path, 'train', 't1'), 'train')
data_val = loader.Data(os.path.join(FLAGS.paired_image_path, 'val', 't1'), 'val')
data_test = loader.Data(os.path.join(FLAGS.paired_image_path, 'test', 't1'), 'test')

# shuffle data
indices = np.array(range(data_train.len))
random.seed(10)
random.shuffle(indices)
data_train.set_path_list(indices)

# start fetching data
data_train.start_thread()
data_val.start_thread()
data_test.start_thread()

# calculate epoch
epoch = int(data_train.len / FLAGS.batch_size)
epoch_val = int(data_val.len / FLAGS.batch_size)
epoch_test = int(data_test.len / FLAGS.batch_size)


def get_checkpoint_dir():
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    # IBIS
    checkpoints_dir = "../../checkpoints/IBIS_HR/brain_imputation_UnetGAN_p_loss/{}".format(current_time)

    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass

    return checkpoints_dir


def init_writer(checkpoints_dir, sess):
    train_step_summary_writer = tf.summary.FileWriter(os.path.join(checkpoints_dir, 'train_step'), sess.graph)
    train_epoch_summary_writer = tf.summary.FileWriter(os.path.join(checkpoints_dir, 'train_epoch'), sess.graph)
    val_epoch_summary_writer = tf.summary.FileWriter(os.path.join(checkpoints_dir, 'val_epoch'), sess.graph)
    test_epoch_summary_writer = tf.summary.FileWriter(os.path.join(checkpoints_dir, 'test_epoch'), sess.graph)
    return train_step_summary_writer, train_epoch_summary_writer, val_epoch_summary_writer, test_epoch_summary_writer


def add_summary(sess, summary_op, feed_dict, summary_writer, step):
    summary_str = sess.run(summary_op, feed_dict)
    summary_writer.add_summary(summary_str, step)
    summary_writer.flush()

def train():
    if FLAGS.load_model is not None:
        # IBIS
        checkpoints_dir = os.path.join("../../checkpoints/IBIS_HR/GAN/", FLAGS.load_model)

    else:
        checkpoints_dir = get_checkpoint_dir()

    graph = tf.Graph()
    with graph.as_default():
        inputs_x = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_size_z, FLAGS.image_size_y,
                                                               FLAGS.image_size_x, 1])
        inputs_y = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_size_z, FLAGS.image_size_y,
                                                               FLAGS.image_size_x, 1])
        ph_ssim = tf.placeholder(tf.float32, name='ssim')
        tf_summary.scalar(name='ssim', tensor=ph_ssim)

        unet_gan = UnetGAN(
            inputs_x,
            inputs_y,
            condition=None,
            weight_dir=FLAGS.weight_dir,
            batch_size=FLAGS.batch_size,
            image_size_z=FLAGS.image_size_z,
            image_size_y=FLAGS.image_size_y,
            image_size_x=FLAGS.image_size_x,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lamda_l1=FLAGS.lamda_l1,
            beta_cor=FLAGS.beta_cor,
            lamda_p=FLAGS.lamda_p,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf
        )

        G_loss, D_Y_loss, fake_y = unet_gan.model()
        #optimizers = unet_gan.optimize(G_loss, D_Y_loss)

        optimizer_G = unet_gan.optimize_G(G_loss)
        optimizer_D = unet_gan.optimize_D(D_Y_loss)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            # checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            # meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            # restore = tf.train.import_meta_graph(meta_graph_path)
            # restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            # start_step = int(meta_graph_path.split("-")[2].split(".")[0])
            # checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = os.path.join(checkpoints_dir, 'model.ckpt-25872.meta')
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, os.path.join(checkpoints_dir, 'model.ckpt-25872'))
            start_step = 25872
            checkpoints_dir = get_checkpoint_dir()
            # train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        else:
            sess.run(tf.global_variables_initializer())
            # train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
            start_step = 0

        summary_dir = os.path.join(checkpoints_dir, 'tensorboard')
        if not os.path.isdir(summary_dir):
            os.mkdir(summary_dir)

        train_step_summary_writer, train_epoch_summary_writer, \
        val_epoch_summary_writer, test_epoch_summary_writer = init_writer(summary_dir, sess)

        # initialize measurements
        ssim_per_epoch_train = 0.0
        best_val_ssim = 0.0

        for step in range(start_step, 30000):
            # 6 -> 12 month
            # x_, y_ = data_train.next_batch(FLAGS.batch_size)

            # 12 -> 6 month
            y_, x_ = data_train.next_batch(FLAGS.batch_size)

            x_ = x_ * 2 - 1
            y_ = y_ * 2 - 1

            # train
            feed_dict = {inputs_x: x_, inputs_y: y_, unet_gan.is_training: True, ph_ssim: 0.}

            # _, G_loss_train, D_Y_loss_train, fake_y_train = sess.run([optimizers, G_loss, D_Y_loss, fake_y],
            #                                                          feed_dict=feed_dict)

            _, G_loss_train, D_Y_loss_train, fake_y_train = sess.run([optimizer_G, G_loss, D_Y_loss, fake_y],
                                                                     feed_dict=feed_dict)

            # calculate measurement for each batch
            ssim_per_batch_train = metrics.compute_ssim(y_, fake_y_train)

            # calculate measurement for each epoch
            ssim_per_epoch_train += ssim_per_batch_train

            if step % 2 == 0:
                _ = sess.run(optimizer_D, feed_dict=feed_dict)
                add_summary(sess, summary_op, feed_dict, train_step_summary_writer, step)

            if step % epoch == 0:
                # train
                cur_epoch = int(step / epoch)
                ssim_per_epoch_train /= float(epoch)

                print('ssim_per_epoch_train')
                print(ssim_per_epoch_train)

                feed_dict[ph_ssim] = ssim_per_epoch_train
                add_summary(sess, summary_op, feed_dict, train_epoch_summary_writer, step)

                ssim_per_epoch_train = 0.0

                # val
                # initialize measurements
                val_feed_dict = {}
                ssim_per_epoch_val = 0.0
                for step_val in range(epoch_val):
                    # 6 -> 12 month
                    # x_val, y_val = data_val.next_batch(FLAGS.batch_size)

                    # 12 -> 6 month
                    y_val, x_val = data_val.next_batch(FLAGS.batch_size)

                    x_val = x_val * 2 - 1
                    y_val = y_val * 2 - 1

                    val_feed_dict = {inputs_x: x_val, inputs_y: y_val, unet_gan.is_training: False}
                    fake_y_val = (sess.run(fake_y, feed_dict=val_feed_dict))

                    # calculate measurement for each batch
                    ssim_per_batch_val = metrics.compute_ssim(y_val, fake_y_val)

                    # calculate measurement for each epoch
                    ssim_per_epoch_val += ssim_per_batch_val

                ssim_per_epoch_val /= float(epoch_val)

                val_feed_dict[ph_ssim] = ssim_per_epoch_val
                add_summary(sess, summary_op, val_feed_dict, val_epoch_summary_writer, step)

                if step > 10000 and ssim_per_epoch_val > best_val_ssim:
                    best_val_ssim = ssim_per_epoch_val

                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  Best ssim in valset   : {}'.format(best_val_ssim))

                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                logging.info('-----------Step %d:-------------' % step)
                logging.info('  G_loss   : {}'.format(G_loss_train))
                logging.info('  D_Y_loss : {}'.format(D_Y_loss_train))


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.compat.v1.app.run()
