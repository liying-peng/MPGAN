import tensorflow as tf
import ops as ops

REAL_LABEL = 0.9


def discriminator_loss(D, y, fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
        # use mean squared error
        error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(D(fake_y)))
    else:
        # use cross entropy
        error_real = -tf.reduce_mean(ops.safe_log(D(y)))
        error_fake = -tf.reduce_mean(ops.safe_log(1 - D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss


def generator_loss(D, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
        # use mean squared error
        loss = tf.reduce_mean(tf.math.squared_difference(D(fake_y), REAL_LABEL))
    else:
        # heuristic, non-saturating loss
        loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    return loss


def correlation_loss(fake_y, y):
    fake_y_m = fake_y - tf.reduce_mean(fake_y)
    y_m = y - tf.reduce_mean(y)
    inner_product = tf.reduce_sum(fake_y_m * y_m)
    fake_y_norm = tf.sqrt(tf.reduce_sum(fake_y_m * fake_y_m))
    y_norm = tf.sqrt(tf.reduce_sum(y_m * y_m))

    return -tf.abs((inner_product / (fake_y_norm * y_norm)))


def conditioned_discriminator_loss(D, condition, y, fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
        # use mean squared error
        error_real = tf.reduce_mean(tf.squared_difference(D(condition, y), REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(D(condition, fake_y)))
    else:
        # use cross entropy
        error_real = -tf.reduce_mean(ops.safe_log(D(condition, y)))
        error_fake = -tf.reduce_mean(ops.safe_log(1 - D(condition, fake_y)))
    loss = (error_real + error_fake) / 2
    return loss

def conditioned_generator_loss(D, condition, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
        # use mean squared error
        loss = tf.reduce_mean(tf.math.squared_difference(D(condition, fake_y), REAL_LABEL))
    else:
        # heuristic, non-saturating loss
        loss = -tf.reduce_mean(ops.safe_log(D(condition, fake_y))) / 2
    return loss

def discriminator_loss_slim(D_y, D_fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
        # use mean squared error
        error_real = tf.reduce_mean(tf.squared_difference(D_y, REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(D_fake_y))
    else:
        # use cross entropy
        error_real = -tf.reduce_mean(ops.safe_log(D_y))
        error_fake = -tf.reduce_mean(ops.safe_log(1 - D_fake_y))
    loss = (error_real + error_fake) / 2
    return loss


def generator_loss_slim(D_fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
        # use mean squared error
        loss = tf.reduce_mean(tf.math.squared_difference(D_fake_y, REAL_LABEL))
    else:
        # heuristic, non-saturating loss
        loss = -tf.reduce_mean(ops.safe_log(D_fake_y)) / 2
    return loss
