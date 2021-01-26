import numpy as np
import tensorflow as tf
from tf_models.generator import Instance_Norm
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

keras = tf.keras
layers = keras.layers
RANDOM_SEED = 36


def model_genesis_encoder(input_shape, pool_size=(2, 2, 2),
                          depth=4, n_base_filters=32, batch_normalization=False, instance_normalization=False,
                          encoder_trainable=True, encoder_weight_dict=None, is_training=True):
    """
    feature list length:4
    """
    inputs = keras.Input(input_shape)
    current_layer = inputs
    levels = []
    feature_list = []
    num_layer = 0

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters * (2 ** layer_depth),
                                          batch_normalization=batch_normalization,
                                          instance_normalization=instance_normalization, layer_depth=num_layer,
                                          trainable=encoder_trainable, weight_dict=encoder_weight_dict,
                                          is_training=is_training)
        num_layer += 1

        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters * (2 ** layer_depth) * 2,
                                          batch_normalization=batch_normalization,
                                          instance_normalization=instance_normalization, layer_depth=num_layer,
                                          trainable=encoder_trainable, weight_dict=encoder_weight_dict,
                                          is_training=is_training)

        num_layer += 1

        if layer_depth < depth - 1:
            current_layer = layers.MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

        print(current_layer)
        feature_list.append(current_layer)

    model = keras.Model(inputs=inputs, outputs=feature_list)

    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, instance_normalization=False,
                             kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), layer_depth=None,
                             trainable=True, weight_dict=None, is_training=True):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """

    conv_name = "depth_" + str(layer_depth) + "_conv"
    bn_name = "depth_" + str(layer_depth) + "_bn"

    kernel_initializer = keras.initializers.glorot_uniform(seed=RANDOM_SEED) if not weight_dict else \
        keras.initializers.constant(weight_dict[conv_name + '/kernel:0'])
    bias_initializer = 'zeros' if not weight_dict else keras.initializers.constant(
        weight_dict[conv_name + '/bias:0'])

    layer = layers.Conv3D(n_filters, kernel, padding=padding, strides=strides, name=conv_name, trainable=trainable,
                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(input_layer)
    if batch_normalization:
        # weight_dict = None
        beta_initializer = 'zeros' if not weight_dict else keras.initializers.constant(weight_dict[bn_name + '/beta:0'])
        gamma_initializer = 'ones' if not weight_dict else keras.initializers.constant(
            weight_dict[bn_name + '/gamma:0'])
        moving_mean_initializer = 'zeros' if not weight_dict else keras.initializers.constant(
            weight_dict[bn_name + '/moving_mean:0'])
        moving_variance_initializer = 'ones' if not weight_dict else keras.initializers.constant(
            weight_dict[bn_name + '/moving_variance:0'])

        layer = layers.BatchNormalization(axis=-1, name=bn_name, trainable=trainable,
                                          beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                                          moving_mean_initializer=moving_mean_initializer,
                                          moving_variance_initializer=moving_variance_initializer)(layer,
                                                                                                   training=is_training)

        '''
        x_shape = layer.get_shape()
        params_shape = x_shape[-1:]
        _beta = np.zeros(params_shape) if not weight_dict else weight_dict[conv_name + '/bias:0']
        _gamma = np.zeros(params_shape) if not weight_dict else weight_dict[bn_name + '/gamma:0']
        _moving_mean = np.zeros(params_shape) if not weight_dict else weight_dict[bn_name + '/moving_mean:0']
        _moving_variance = np.zeros(params_shape) if not weight_dict else weight_dict[bn_name + '/moving_variance:0']

        beta = tf.Variable(_beta, name='beta', dtype=tf.float32)
        gamma = tf.Variable(_gamma, name='gamma', dtype=tf.float32)
        moving_mean = tf.Variable(_moving_mean, name='moving_mean', trainable=False, dtype=tf.float32)
        moving_variance = tf.Variable(_moving_variance, name='moving_variance', trainable=False, dtype=tf.float32)

        mean, variance = tf.nn.moments(layer, [0, 1, 2, 3])

        BN_DECAY = 0.9997
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)

        update_collection = '%s_%d' % ('update_ops', layer_depth)
        tf.add_to_collection(update_collection, update_moving_mean)
        tf.add_to_collection(update_collection, update_moving_variance)

        mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance),
                                               lambda: (moving_mean, moving_variance))

        layer = tf.nn.batch_normalization(layer, mean, variance, beta, gamma, 0.001)
        '''

    elif instance_normalization:
        layer = Instance_Norm(n_filters, name="depth_" + str(layer_depth) + "_instance_norm", trainable=trainable)(
            layer)
    if activation is None:
        return layers.Activation('relu', name="depth_" + str(layer_depth) + "_relu")(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False, trainable=True):
    if deconvolution:
        return layers.Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                                      strides=strides, trainable=trainable,
                                      kernel_initializer=keras.initializers.glorot_uniform(seed=RANDOM_SEED))
    else:
        return layers.UpSampling3D(size=pool_size)


# from six.moves import cPickle as pickle
#
# with open(r'/Users/yusenlin/Documents/github/pickledata_test/unet_3d/unet_3d.pkl', 'rb') as f:
#     weights = pickle.load(f)

# print(weights)
#

# model = model_genesis_encoder((128, 128, 128, 1),
#                               batch_normalization=True,
#                               instance_normalization=False,
#                               encoder_trainable=False,
#                               encoder_weight_dict=weights,
#                               is_training=True)
# sess = tf.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
#
# print(sess.run(model.weights[0][-1, -1, -1, -1]))
# exit()

# -------------------------- Discriminator -----------------------------------

# num_class = 2
#
# from six.moves import cPickle as pickle
#
# with open(r'/Users/yusenlin/Documents/github/pickledata_test/unet_3d/unet_3d.pkl', 'rb') as f:
#     weights = pickle.load(f)
#
# new_model = unet_model_3d((128, 128, 128, 1), batch_normalization=True, encoder_weight_dict=weights)
# encoder_output = new_model.layers[27].output
#
# final_convolution = layers.Conv3D(num_class, (1, 1, 1))(encoder_output)
# output = layers.Activation('softmax')(final_convolution)
#
# Discriminator = keras.models.Model(inputs=new_model.input, outputs=output)
#
# for i, l in enumerate(new_model.layers):
#     print(i, l)
#
# print('\n----------------------------------------')
# print(new_model.layers[27])
# print(new_model.get_layer('depth_7_relu'))
