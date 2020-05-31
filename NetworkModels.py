
import numpy as np
from keras import backend
from tensorflow.keras import layers
import tensorflow as tf


FILEHASHKEY = 123

def square(x:int):
    return x*x


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
    
def make_generator_model_28(z, out_channel_dim, is_train=True, alpha=0.2, keep_prob=0.5):
    """ From https://github.com/HACKERSHUBH/Face-Genaration-using-Generative-Adversarial-Network/blob/master/face_gen.ipynb
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    with tf.variable_scope('generator', reuse = (not is_train)):
        fc = layers.dense(z, 4*4*1024, use_bias=False)
        fc = tf.reshape(fc, (-1, 4, 4, 1024))
        bn0 = layers.batch_normalization(fc, training=is_train)
        lrelu0 = tf.maximum(alpha*bn0, bn0)
        drop0 = layers.dropout(lrelu0, keep_prob, training=is_train)
        
        # Deconvolution, 7x7x512
        conv1= layers.Conv2DTranspose(drop0, 512, 4, 1, 'valid', use_bias=False)
        bn1 = layers.batch_normalization(conv1, training=is_train)
        lrelu1 = tf.maximum(alpha*bn1, bn1)
        drop1 = layers.dropout(lrelu1, keep_prob, training=is_train)
        
        # Deconvolution 14x14x256
        conv2 = layers.Conv2DTranspose(drop1, 256, 5, 2, 'same', use_bias=False)
        bn2 = layers.batch_normalization(conv2, training=is_train)
        lrelu2 = tf.maximum(alpha*bn2, bn2)
        drop2 = layers.dropout(lrelu2, keep_prob, training=is_train)
        
        # Output layer, 28x28xn
        logits= layers.Conv2DTranspose(drop2, out_channel_dim, 5, 2, 'same')
        
        out = tf.tanh(logits)
        
        return out
    


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
    
def make_discriminator_model_28(images, alpha=0.2, keep_prob=0.5):
    """ Adapted from https://github.com/HACKERSHUBH/Face-Genaration-using-Generative-Adversarial-Network/blob/master/face_gen.ipynb
        Uses 28*28*3 inputs
        Probable original source is
        https://www.floydhub.com/udacity/projects/face-generation/5/output/dlnd_face_generation.ipynb
    """
    # Convolutional layer, 14x14x64
    conv1 = layers.Conv2D(images, 64, 5, 2, padding='same', kernel_initiazire=layers.xavier_initializer())
    lrelu1 = tf.maximum(alpha * conv1, conv1)
    drop1 = layers.dropout(lrelu1, keep_prob)
    
    # Strided convolutional layer, 7x7x128
    conv2 = layers.Conv2D(drop1, 128, 5, 2, 'same', use_bias=False)
    bn2 = layers.batch_normalization(conv2)
    lrelu2 = tf.maximum(alpha*bn2, bn2)
    drop2 = layers.dropout(lrelu2, keep_prob)
    
    # Strided convolutional layer, 4x4x256
    conv3 = layers.Conv2D(drop2, 256, 5, 2, 'same', use_bias=False)
    bn3 = layers.batch_normalization(conv3)
    lrelu3 = tf.maximum(alpha*bn3, bn3)
    drop3 = layers.dropout(lrelu3, keep_prob)
    
    # fully connected
    flat = tf.reshape(drop3, (-1, 4*4*256))
    logits = layers.dense(flat, 1)
    out = tf.sigmoid(logits)
    
    return out, logits
    
    
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
