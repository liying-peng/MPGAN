import tensorflow as tf
from tf_models.discriminator import Discriminator
from tf_models.unet3d_multi_decoder_fusion.fusion_skff_concat_conv_debug import unet_model_3d as Generator
from tf_models.feature_extractor import model_genesis_encoder as feature_extractor
from six.moves import cPickle as pickle
from loss.utils import discriminator_loss, generator_loss

keras = tf.keras
layers = keras.layers
tf_summary = tf.compat.v1.summary

REAL_LABEL = 0.9


class UnetGAN:
    NAME = 'skff_conv_fusion_two_branch_fusion'

    def __init__(self,
                 x_t1,
                 x_t2,
                 y_t1,
                 y_t2,
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
                 lamda_p1=0.0,
                 lamda_p2=0.0,
                 learning_rate=1e-4,
                 beta1=0.5,
                 ngf=64
                 ):
        """
        Args:
          x: placehold inputs x 4D tensor [batch_size, image_width, image_height, image_depth]
          y: placehold inputs y 4D tensor [batch_size, image_width, image_height, image_depth]
          batch_size: integer, batch size
          image_size: integer, image size
          lambda1: integer, weight for forward cycle loss (X->Y->X)
          lambda2: integer, weight for backward cycle loss (Y->X->Y)
          use_lsgan: boolean
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
        """
        self.use_lsgan = use_lsgan
        self.batch_size = batch_size
        self.image_size_z = image_size_z
        self.image_size_y = image_size_y
        self.image_size_x = image_size_x
        self.learning_rate = learning_rate
        self.lamda_l1 = lamda_l1
        self.beta_cor = beta_cor
        self.lamda_p1 = lamda_p1
        self.lamda_p2 = lamda_p2
        self.beta1 = beta1
        self.x_t1 = x_t1
        self.x_t2 = x_t2
        self.y_t1 = y_t1
        self.y_t2 = y_t2
        self.x = tf.concat([self.x_t1, self.x_t2], axis=-1)
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        weights = None
        if weight_dir:
            with open(weight_dir, 'rb') as f:
                weights = pickle.load(f)

        kernel_initializer = tf.compat.v1.keras.initializers.RandomNormal(mean=0, stddev=0.02)

        self.G = Generator((self.image_size_z, self.image_size_y, self.image_size_x, 2),
                           batch_normalization=False,
                           instance_normalization=True,
                           encoder_trainable=True,
                           decoder_trainable=True,
                           encoder_weight_dict=None,
                           decoder_weight_dict=None,
                           final_weight_dict=None,
                           is_training=self.is_training)

        self.D_Y_t1 = Discriminator('D_Y_t1', kernel_initializer=kernel_initializer, norm_type=norm)
        self.D_Y_t2 = Discriminator('D_Y_t2', kernel_initializer=kernel_initializer, norm_type=norm)
        self.Extractor = feature_extractor((self.image_size_z, self.image_size_y, self.image_size_x, 1),
                                           batch_normalization=True,
                                           instance_normalization=False,
                                           encoder_trainable=False,
                                           encoder_weight_dict=weights,
                                           is_training=True)

        # print('\n--------------------------------------------------')
        # print('Generator')
        # G_variables = self.G.variables
        # shapes = []
        # for v in G_variables:
        #     shapes.append(v.shape)
        #     print(v)
        #     print(v.shape)
        #     print('')
        #
        # print('\n--------------------------------------------------')
        # print('Generator')
        # D_variables = self.D_Y_t1.variables
        # shapes = []
        # for v in D_variables:
        #     shapes.append(v.shape)
        #     print(v)
        #     print(v.shape)
        #     print('')

    def model(self):
        # X -> Y
        fake_list = self.G(self.x)
        fake_y_t1 = fake_list[0]
        fake_y_t2 = fake_list[1]

        feature_fake_y_list_t1 = self.Extractor(fake_y_t1)
        feature_y_list_t1 = self.Extractor(self.y_t1)

        feature_fake_y_list_t2 = self.Extractor(fake_y_t2)
        feature_y_list_t2 = self.Extractor(self.y_t2)

        # t1
        G_gan_loss_t1 = generator_loss(self.D_Y_t1, fake_y_t1, use_lsgan=self.use_lsgan)
        l1_loss_t1 = tf.reduce_mean(tf.abs(self.y_t1 - fake_y_t1))

        # percetual_loss_t1 = (tf.reduce_mean(tf.abs(feature_y_list_t1[0] - feature_fake_y_list_t1[0])) + \
        #                   tf.reduce_mean(tf.abs(feature_y_list_t1[1] - feature_fake_y_list_t1[1])) + \
        #                   tf.reduce_mean(tf.abs(feature_y_list_t1[2] - feature_fake_y_list_t1[2])) + \
        #                   tf.reduce_mean(tf.abs(feature_y_list_t1[3] - feature_fake_y_list_t1[3]))) / 4.0

        percetual_loss_t1 = tf.reduce_mean(tf.abs(feature_y_list_t1[0] - feature_fake_y_list_t1[0]))

        G_loss_t1 = G_gan_loss_t1 + self.lamda_l1 * l1_loss_t1  + self.lamda_p1 * percetual_loss_t1
        D_Y_loss_t1 = discriminator_loss(self.D_Y_t1, self.y_t1, fake_y_t1, use_lsgan=self.use_lsgan)

        # t2
        G_gan_loss_t2 = generator_loss(self.D_Y_t2, fake_y_t2, use_lsgan=self.use_lsgan)
        l1_loss_t2 = tf.reduce_mean(tf.abs(self.y_t2 - fake_y_t2))

        # percetual_loss_t2 = (tf.reduce_mean(tf.abs(feature_y_list_t2[0] - feature_fake_y_list_t2[0])) + \
        #                      tf.reduce_mean(tf.abs(feature_y_list_t2[1] - feature_fake_y_list_t2[1])) + \
        #                      tf.reduce_mean(tf.abs(feature_y_list_t2[2] - feature_fake_y_list_t2[2])) + \
        #                      tf.reduce_mean(tf.abs(feature_y_list_t2[3] - feature_fake_y_list_t2[3]))) / 4.0

        percetual_loss_t2 = tf.reduce_mean(tf.abs(feature_y_list_t2[0] - feature_fake_y_list_t2[0]))

        G_loss_t2 = G_gan_loss_t2 + self.lamda_l1 * l1_loss_t2  + self.lamda_p2 * percetual_loss_t2
        D_Y_loss_t2 = discriminator_loss(self.D_Y_t2, self.y_t2, fake_y_t2, use_lsgan=self.use_lsgan)

        # summary
        tf_summary.scalar('loss/G', G_gan_loss_t1)
        tf_summary.scalar('loss/D_Y', D_Y_loss_t1)
        tf_summary.scalar('loss/G', G_gan_loss_t2)
        tf_summary.scalar('loss/D_Y', D_Y_loss_t2)

        # select some slices to visualize
        # # t1
        # input_x_40_t1 = self.x_t1[0:1, 120, :, :, :]
        # input_x_30_t1 = self.x_t1[0:1, 100, :, :, :]
        # input_x_20_t1 = self.x_t1[0:1, 80, :, :, :]
        # input_x_10_t1 = self.x_t1[0:1, 60, :, :, :]
        #
        # input_y_40_t1 = self.y_t1[0:1, 120, :, :, :]
        # input_y_30_t1 = self.y_t1[0:1, 100, :, :, :]
        # input_y_20_t1 = self.y_t1[0:1, 80, :, :, :]
        # input_y_10_t1 = self.y_t1[0:1, 60, :, :, :]
        #
        # x_generated_40_t1 = fake_y_t1[0:1, 120, :, :, :]
        # x_generated_30_t1 = fake_y_t1[0:1, 100, :, :, :]
        # x_generated_20_t1 = fake_y_t1[0:1, 80, :, :, :]
        # x_generated_10_t1 = fake_y_t1[0:1, 60, :, :, :]
        #
        # # t2
        # input_x_40_t2 = self.x_t2[0:1, 120, :, :, :]
        # input_x_30_t2 = self.x_t2[0:1, 100, :, :, :]
        # input_x_20_t2 = self.x_t2[0:1, 80, :, :, :]
        # input_x_10_t2 = self.x_t2[0:1, 60, :, :, :]
        #
        # input_y_40_t2 = self.y_t2[0:1, 120, :, :, :]
        # input_y_30_t2 = self.y_t2[0:1, 100, :, :, :]
        # input_y_20_t2 = self.y_t2[0:1, 80, :, :, :]
        # input_y_10_t2 = self.y_t2[0:1, 60, :, :, :]
        #
        # x_generated_40_t2 = fake_y_t2[0:1, 120, :, :, :]
        # x_generated_30_t2 = fake_y_t2[0:1, 100, :, :, :]
        # x_generated_20_t2 = fake_y_t2[0:1, 80, :, :, :]
        # x_generated_10_t2 = fake_y_t2[0:1, 60, :, :, :]

        # t1
        input_x_40_t1 = self.x_t1[0:1, 50, :, :, :]
        input_x_30_t1 = self.x_t1[0:1, 40, :, :, :]
        input_x_20_t1 = self.x_t1[0:1, 30, :, :, :]
        input_x_10_t1 = self.x_t1[0:1, 20, :, :, :]

        input_y_40_t1 = self.y_t1[0:1, 50, :, :, :]
        input_y_30_t1 = self.y_t1[0:1, 40, :, :, :]
        input_y_20_t1 = self.y_t1[0:1, 30, :, :, :]
        input_y_10_t1 = self.y_t1[0:1, 20, :, :, :]

        x_generated_40_t1 = fake_y_t1[0:1, 50, :, :, :]
        x_generated_30_t1 = fake_y_t1[0:1, 40, :, :, :]
        x_generated_20_t1 = fake_y_t1[0:1, 30, :, :, :]
        x_generated_10_t1 = fake_y_t1[0:1, 20, :, :, :]

        # t2
        input_x_40_t2 = self.x_t2[0:1, 50, :, :, :]
        input_x_30_t2 = self.x_t2[0:1, 40, :, :, :]
        input_x_20_t2 = self.x_t2[0:1, 30, :, :, :]
        input_x_10_t2 = self.x_t2[0:1, 20, :, :, :]

        input_y_40_t2 = self.y_t2[0:1, 50, :, :, :]
        input_y_30_t2 = self.y_t2[0:1, 40, :, :, :]
        input_y_20_t2 = self.y_t2[0:1, 30, :, :, :]
        input_y_10_t2 = self.y_t2[0:1, 20, :, :, :]

        x_generated_40_t2 = fake_y_t2[0:1, 50, :, :, :]
        x_generated_30_t2 = fake_y_t2[0:1, 40, :, :, :]
        x_generated_20_t2 = fake_y_t2[0:1, 30, :, :, :]
        x_generated_10_t2 = fake_y_t2[0:1, 20, :, :, :]

        # image visualization t1
        tf_summary.image('X_40/t1/input', input_x_40_t1)
        tf_summary.image('X_40/t1/generated', x_generated_40_t1)
        tf_summary.image('X_40/t1/ground_truth', input_y_40_t1)

        tf_summary.image('X_30/t1/input', input_x_30_t1)
        tf_summary.image('X_30/t1/generated', x_generated_30_t1)
        tf_summary.image('X_30/t1/ground_truth', input_y_30_t1)

        tf_summary.image('X_20/t1/input', input_x_20_t1)
        tf_summary.image('X_20/t1/generated', x_generated_20_t1)
        tf_summary.image('X_20/t1/ground_truth', input_y_20_t1)

        tf_summary.image('X_10/t1/input', input_x_10_t1)
        tf_summary.image('X_10/t1/generated', x_generated_10_t1)
        tf_summary.image('X_10/t1/ground_truth', input_y_10_t1)

        # image visualization t2
        tf_summary.image('X_40/t2/input', input_x_40_t2)
        tf_summary.image('X_40/t2/generated', x_generated_40_t2)
        tf_summary.image('X_40/t2/ground_truth', input_y_40_t2)

        tf_summary.image('X_30/t2/input', input_x_30_t2)
        tf_summary.image('X_30/t2/generated', x_generated_30_t2)
        tf_summary.image('X_30/t2/ground_truth', input_y_30_t2)

        tf_summary.image('X_20/t2/input', input_x_20_t2)
        tf_summary.image('X_20/t2/generated', x_generated_20_t2)
        tf_summary.image('X_20/t2/ground_truth', input_y_20_t2)

        tf_summary.image('X_10/t2/input', input_x_10_t2)
        tf_summary.image('X_10/t2/generated', x_generated_10_t2)
        tf_summary.image('X_10/t2/ground_truth', input_y_10_t2)

        return G_loss_t1, D_Y_loss_t1, G_loss_t2, D_Y_loss_t2, fake_y_t1, fake_y_t2

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
                                          start_decay_step=60000,
                                          decay_steps=5000,
                                          name='Adam_G',
                                          starter_learning_rate=self.learning_rate)

        return G_optimizer

    def optimize_D_t1(self, D_Y_loss_t1):
        # D_Y_optimizer_t1 = self.make_optimizer(D_Y_loss_t1, self.D_Y_t1.trainable_variables,
        #                                        start_decay_step=2500,
        #                                        decay_steps=1250,
        #                                        name='Adam_D_Y_t1', starter_learning_rate=self.learning_rate)

        D_Y_optimizer_t1 = self.make_optimizer(D_Y_loss_t1, self.D_Y_t1.trainable_variables,
                                               start_decay_step=60000,
                                               decay_steps=1250,
                                               name='Adam_D_Y_t1', starter_learning_rate=self.learning_rate)

        return D_Y_optimizer_t1

    def optimize_D_t2(self, D_Y_loss_t2):
        # D_Y_optimizer_t2 = self.make_optimizer(D_Y_loss_t2, self.D_Y_t2.trainable_variables,
        #                                        start_decay_step=2500,
        #                                        decay_steps=1250,
        #                                        name='Adam_D_Y_t2', starter_learning_rate=self.learning_rate)

        D_Y_optimizer_t2 = self.make_optimizer(D_Y_loss_t2, self.D_Y_t2.trainable_variables,
                                               start_decay_step=60000,
                                               decay_steps=1250,
                                               name='Adam_D_Y_t2', starter_learning_rate=self.learning_rate)

        return D_Y_optimizer_t2
