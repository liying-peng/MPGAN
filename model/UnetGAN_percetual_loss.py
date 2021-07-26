import tensorflow as tf
from tf_models.discriminator import Discriminator
from tf_models.feature_extractor import model_genesis_encoder as feature_extractor
from tf_models.generator_unet3d import unet_model_3d as Generator
from six.moves import cPickle as pickle
from loss.utils import correlation_loss, discriminator_loss, generator_loss

keras = tf.keras
layers = keras.layers
tf_summary = tf.compat.v1.summary

REAL_LABEL = 0.9

class UnetGAN:
    def __init__(self,
                 x,
                 y,
                 condition=None,
                 weight_dir=None,
                 batch_size=32,
                 image_size_z=72,
                 image_size_y=92,
                 image_size_x=80,
                 use_lsgan=True,
                 norm='instance',
                 lamda_l1=0.0,
                 beta_cor=0.0,
                 lamda_p=0.0,
                 learning_rate=1e-4,
                 beta1=0.5,
                 ngf=64
                 ):
        '''

        :param x: input x
        :param y: input y
        :param condition: condition on discriminator
        :param weight_dir: the path of weight of model genesis
        :param batch_size: batch size
        :param image_size_z: image_size_z
        :param image_size_y: image_size_y
        :param image_size_x: image_size_x
        :param use_lsgan:
        :param norm: norm type
        :param lamda_l1: the weight of lvr
        :param beta_cor:
        :param lamda_p: the weight of lp
        :param learning_rate: learning_rate
        :param beta1:
        :param ngf:
        '''

        self.use_lsgan = use_lsgan
        self.batch_size = batch_size
        self.image_size_z = image_size_z
        self.image_size_y = image_size_y
        self.image_size_x = image_size_x
        self.learning_rate = learning_rate
        self.lamda_l1 = lamda_l1
        self.beta_cor = beta_cor
        self.lamda_p = lamda_p
        self.beta1 = beta1
        self.x = x
        self.y = y
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        weights = None
        if weight_dir:
            with open(weight_dir, 'rb') as f:
                weights = pickle.load(f)

        kernel_initializer = tf.compat.v1.keras.initializers.RandomNormal(mean=0, stddev=0.02)

        self.G = Generator((self.image_size_z, self.image_size_y, self.image_size_x, 1),
                           batch_normalization=False,
                           instance_normalization=True,
                           encoder_trainable=True,
                           decoder_trainable=True,
                           encoder_weight_dict=None,
                           decoder_weight_dict=None,
                           final_weight_dict=None,
                           is_training=self.is_training)

        self.D_Y = Discriminator('D_Y', kernel_initializer=kernel_initializer, norm_type=norm)
        self.Extractor = feature_extractor((self.image_size_z, self.image_size_y, self.image_size_x, 1),
                                           batch_normalization=True,
                                           instance_normalization=False,
                                           encoder_trainable=False,
                                           encoder_weight_dict=weights,
                                           is_training=True)

    def model(self):
        # X -> Y
        fake_y = self.G(self.x)
        feature_fake_y_list = self.Extractor(fake_y)
        feature_y_list = self.Extractor(self.y)

        G_gan_loss = generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
        l1_loss = tf.reduce_mean(tf.abs(self.y - fake_y))
        cor_loss = correlation_loss(fake_y, self.y)
        # percetual_loss = (tf.reduce_mean(tf.abs(feature_y_list[0] - feature_fake_y_list[0])) + \
        #                   tf.reduce_mean(tf.abs(feature_y_list[1] - feature_fake_y_list[1])) + \
        #                   tf.reduce_mean(tf.abs(feature_y_list[2] - feature_fake_y_list[2])) + \
        #                   tf.reduce_mean(tf.abs(feature_y_list[3] - feature_fake_y_list[3]))) / 4.0

        percetual_loss = tf.reduce_mean(tf.abs(feature_y_list[0] - feature_fake_y_list[0]))

        G_loss = G_gan_loss + self.lamda_l1 * l1_loss + self.beta_cor * cor_loss + self.lamda_p * percetual_loss
        D_Y_loss = discriminator_loss(self.D_Y, self.y, fake_y, use_lsgan=self.use_lsgan)

        # summary
        tf_summary.scalar('loss/G', G_gan_loss)
        tf_summary.scalar('loss/D_Y', D_Y_loss)

        # select some slices to visualize
        # X
        input_x_40 = self.x[0:1, 120, :, :, :]
        input_x_30 = self.x[0:1, 100, :, :, :]
        input_x_20 = self.x[0:1, 90, :, :, :]
        input_x_10 = self.x[0:1, 80, :, :, :]

        input_y_40 = self.y[0:1, 120, :, :, :]
        input_y_30 = self.y[0:1, 100, :, :, :]
        input_y_20 = self.y[0:1, 90, :, :, :]
        input_y_10 = self.y[0:1, 80, :, :, :]

        x_generated = self.G(self.x)
        x_generated_40 = x_generated[0:1, 120, :, :, :]
        x_generated_30 = x_generated[0:1, 100, :, :, :]
        x_generated_20 = x_generated[0:1, 90, :, :, :]
        x_generated_10 = x_generated[0:1, 80, :, :, :]

        # image visualization X
        tf_summary.image('X_40/input', input_x_40)
        tf_summary.image('X_40/generated', x_generated_40)
        tf_summary.image('X_40/ground_truth', input_y_40)

        tf_summary.image('X_30/input', input_x_30)
        tf_summary.image('X_30/generated', x_generated_30)
        tf_summary.image('X_30/ground_truth', input_y_30)

        tf_summary.image('X_20/input', input_x_20)
        tf_summary.image('X_20/generated', x_generated_20)
        tf_summary.image('X_20/ground_truth', input_y_20)

        tf_summary.image('X_10/input', input_x_10)
        tf_summary.image('X_10/generated', x_generated_10)
        tf_summary.image('X_10/ground_truth', input_y_10)

        return G_loss, D_Y_loss, fake_y

    def make_optimizer(self, loss, variables, start_decay_step, decay_steps, name='Adam', starter_learning_rate=1e-4):
        global_step = tf.Variable(0, trainable=False)
        start_decay_step = start_decay_step
        decay_steps = decay_steps
        beta1 = self.beta1

        # --------------------#
        # decay_steps = 20000
        # learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, decay_steps,
        #                                                      decay_rate=0.01, staircase=False)
        # -------------------#

        # -------------------#
        learning_rate = (
            tf.where(
                tf.greater_equal(global_step, start_decay_step),
                tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, decay_steps,
                                                     decay_rate=0.5, staircase=True),
                starter_learning_rate
            )

        )
        # --------------------#

        # --------------------#
        # decay_steps = 50000
        # learning_rate = (
        #     tf.where(
        #         tf.greater_equal(global_step, start_decay_step),
        #         tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, decay_steps,
        #                                              decay_rate=0.0001, staircase=False),
        #         starter_learning_rate
        #     )
        #
        # )
        # --------------------#
        tf_summary.scalar('learning_rate/{}'.format(name), learning_rate)

        learning_step = (
            tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                .minimize(loss, global_step=global_step, var_list=variables)
        )

        return learning_step

    def optimize_G(self, G_loss):
        # G_optimizer = self.make_optimizer(G_loss, self.G.trainable_variables,
        #                                   start_decay_step=10000,
        #                                   decay_steps=5000,
        #                                   name='Adam_G',
        #                                   starter_learning_rate=self.learning_rate)

        G_optimizer = self.make_optimizer(G_loss, self.G.trainable_variables,
                                          start_decay_step=20000,
                                          decay_steps=10000,
                                          name='Adam_G',
                                          starter_learning_rate=self.learning_rate)

        return G_optimizer

    def optimize_D(self, D_Y_loss):
        # D_Y_optimizer = self.make_optimizer(D_Y_loss, self.D_Y.trainable_variables,
        #                                     start_decay_step=2500,
        #                                     decay_steps=1250,
        #                                     name='Adam_D_Y', starter_learning_rate=self.learning_rate)

        D_Y_optimizer = self.make_optimizer(D_Y_loss, self.D_Y.trainable_variables,
                                            start_decay_step=10000,
                                            decay_steps=5000,
                                            name='Adam_D_Y', starter_learning_rate=self.learning_rate)

        return D_Y_optimizer
