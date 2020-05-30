
import numpy as np
from keras import backend
from tensorflow.keras import layers
import tensorflow as tf


FILEHASHKEY = 123

def square(x:int):
    return x*x

# ----------- LossFunction --------- #

def return_loss_func(key: str):
    def binarycrossentropy_loss():
        # This method returns a helper function to compute cross entropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy

    # calculate wasserstein loss
    def wasserstein_loss(y_true, y_pred):
        return backend.mean(y_true * y_pred)

    if key=="binarycrossentropy":
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy

    elif key== "wasserstein":
        return wasserstein_loss
    else:
        raise Exception('Incorrect "Key" argument')

# For non saturating loss
# Add to possible loss functions maybe??
# haven't been able to get it to work

# Change code in Train_Step whether you want normal or non discriminating loss

def ns_discriminator_loss(real_output, generated_output):
    return -tf.reduce_mean(tf.math.log(real_output) + tf.math.log(1-generated_output))

def ns_generator_loss(generated_output):
    return -tf.reduce_mean(tf.math.log(generated_output))


def discriminator_loss(real_output, fake_output, loss_func):
    real_loss = loss_func(tf.ones_like(real_output), real_output)
    fake_loss = loss_func(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output,loss_func):
    #wasserstein_loss(tf.ones_like(fake_output), fake_output)
    return loss_func(tf.ones_like(fake_output), fake_output)



# ----------- Generators --------- #

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense( 7 * 7 *256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))

    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 3)

    return model

def make_generator_model_MNIST():
  model = tf.keras.Sequential()
  model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((7,7,256)))
  assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

  return model


# --------- Discriminators --------- #

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def make_discriminator_model2():
  #https://medium.com/@scrybe.ml/paper-explained-dcgan-using-keras-based-on-chintala-et-als-work-d37043dfe909
    model = tf.keras.Sequential()

    # Model architecture from https://arxiv.org/abs/1511.06434
    model.add(layers.Conv2D(32, kernel_size=3, strides=2,
                     input_shape=[28, 28, 3], padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def make_discriminator_model3():
  # Deeper
#   Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
# convolutions (generator).
# • Use batchnorm in both the generator and the discriminator.
# • Remove fully connected hidden layers for deeper architectures.
# • Use ReLU activation in generator for all layers except for the output, which uses Tanh.
# • Use LeakyReLU activation in the discriminator for all layers.
    model = tf.keras.Sequential()

    # Model architecture from https://arxiv.org/abs/1511.06434
    model.add(layers.Conv2D(32, kernel_size=3, strides=2,
                     input_shape=[28, 28, 3], padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    return model


def make_discriminator_model_MNIST():
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  return model
