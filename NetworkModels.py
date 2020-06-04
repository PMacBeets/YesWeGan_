
import numpy as np
from keras import backend
from tensorflow.keras import layers
import tensorflow as tf


FILEHASHKEY = 8

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
    
def make_generator_model2():
    model = tf.keras.Sequential()
    model.add(layers.Dense( 4 * 4 * 1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #dropout layer?

    model.add(layers.Reshape((7, 7, 512)))
    assert model.output_shape == (None, 7, 7, 512) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(512, 4, 1, padding='valid', use_bias=False))

    # assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, 5, 2, padding='same', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, 5, 2, padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 28, 28, 3)

    return model
    
def make_generator_model_28(is_train=True, alpha=0.2, keep_prob=0.5):
    """ From https://github.com/HACKERSHUBH/Face-Genaration-using-Generative-Adversarial-Network/blob/master/face_gen.ipynb
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """
    out_channel_dim = 3
    z = 3
    with tf.compat.v1.variable_scope('generator', reuse = (not is_train)):
        fc = layers.Dense(z, 4*4*1024, use_bias=False)
        fc = tf.reshape(fc, (-1, 4, 4, 1024))
        bn0 = layers.BatchNormalization(fc, training=is_train)
        lrelu0 = tf.maximum(alpha*bn0, bn0)
        drop0 = layers.dropout(lrelu0, keep_prob, training=is_train)
        
        # Deconvolution, 7x7x512
        conv1= layers.Conv2DTranspose(drop0, 512, 4, 1, 'valid', use_bias=False)
        bn1 = layers.BatchNormalization(conv1, training=is_train)
        lrelu1 = tf.maximum(alpha*bn1, bn1)
        drop1 = layers.dropout(lrelu1, keep_prob, training=is_train)
        
        # Deconvolution 14x14x256
        conv2 = layers.Conv2DTranspose(drop1, 256, 5, 2, 'same', use_bias=False)
        bn2 = layers.BatchNormalization(conv2, training=is_train)
        lrelu2 = tf.maximum(alpha*bn2, bn2)
        drop2 = layers.dropout(lrelu2, keep_prob, training=is_train)
        
        # Output layer, 28x28xn
        logits= layers.Conv2DTranspose(drop2, out_channel_dim, 5, 2, 'same')
        
        out = tf.tanh(logits)
        
        return out

    
def make_generator_model_DCGAN():
    model = tf.keras.Sequential(name="DCGAN")
    model.add(layers.Dense(4* 4 * 1024, use_bias=False, input_shape=(100,)))
    model.add(layers.Reshape((4, 4, 1024)))

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False,activation='tanh'))
    model.add(layers.BatchNormalization())

    return model


def make_generator_model_DCGAN_Final():
    data_format = 'channels_last'

    # generator
    latent_size = 128
    latent_shape = (1024, 4, 4) if data_format == 'channels_first' else (4, 4, 1024)
    axis = 1 if data_format == 'channels_first' else -1

    generator = tf.keras.Sequential(name="generator")
    generator.add(layers.Dense(4 * 4 * 1024, use_bias=True, input_shape=(latent_size,), name='dense'))
    generator.add(layers.Reshape(latent_shape, name='reshape'))

    generator.add(
        layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                               name='up1'))
    generator.add(layers.BatchNormalization(axis=axis, name='bn1'))
    generator.add(layers.ReLU(name='relu1'))

    generator.add(
        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                               name='up2'))
    generator.add(layers.BatchNormalization(axis=axis, name='bn2'))
    generator.add(layers.ReLU(name='relu2'))

    generator.add(
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                               name='up3'))
    generator.add(layers.BatchNormalization(axis=axis, name='bn3'))
    generator.add(layers.ReLU(name='relu3'))

    generator.add(
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                               name='up4'))
    generator.add(layers.BatchNormalization(axis=axis, name='bn4'))
    generator.add(layers.ReLU(name='relu4'))

    generator.add(layers.Conv2D(3, (3, 3), padding='same', use_bias=True, activation='tanh', data_format=data_format,
                                name='to_im'))

    return generator




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


def make_generator_model_MNIST_Deep():
  # use input of size 128

  model = tf.keras.Sequential()
  model.add(layers.Dense(7 * 7 * 512, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((7, 7, 512)))
  assert model.output_shape == (None, 7, 7, 512)  # Note: None is the batch size

  model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  #assert model.output_shape == (None, 9, 9, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  #assert model.output_shape == (None, 7, 7, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  #assert model.output_shape == (None, 14, 14, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))


  return model


def make_generator_model_MNIST_Deep3():
  # use input of size 128

  model = tf.keras.Sequential()
  model.add(layers.Dense(7 * 7 * 512, use_bias=False, input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((7, 7, 512)))
  assert model.output_shape == (None, 7, 7, 512)  # Note: None is the batch size

  model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  #assert model.output_shape == (None, 9, 9, 256)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
  #assert model.output_shape == (None, 7, 7, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  #assert model.output_shape == (None, 14, 14, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))


  return model




# --------- Discriminators --------- #

def make_discriminator_model_DCGAN():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[64, 64, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation="sigmoid"))

    return model

def make_discriminator_model_DCGAN_Final(Batch_Norm:bool):
    data_format = 'channels_last'
    axis = 1 if data_format == 'channels_first' else -1
    im_size = (64, 64, 3)

    discriminator = tf.keras.Sequential(name='discriminator')
    discriminator.add(
        layers.Conv2D(64, (3, 3), padding='same', use_bias=True, data_format=data_format, input_shape=im_size,
                      name='from_im'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu0'))

    discriminator.add(
        layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down1'))
    if Batch_Norm:
        discriminator.add(layers.BatchNormalization(axis=axis, name='bn1'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu1'))
    discriminator.add(
        layers.Conv2D(128, (3, 3), padding='same', use_bias=False, data_format=data_format, name='trans1'))
    if Batch_Norm:
        discriminator.add(layers.BatchNormalization(axis=axis, name='bn2'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu2'))

    discriminator.add(
        layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down2'))
    if Batch_Norm:
        discriminator.add(layers.BatchNormalization(axis=axis, name='bn3'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu3'))
    discriminator.add(
        layers.Conv2D(256, (3, 3), padding='same', use_bias=False, data_format=data_format, name='trans2'))
    if Batch_Norm:
        discriminator.add(layers.BatchNormalization(axis=axis, name='bn4'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu4'))

    discriminator.add(
        layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down3'))
    if Batch_Norm:
        discriminator.add(layers.BatchNormalization(axis=axis, name='bn5'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu5'))
    discriminator.add(
        layers.Conv2D(512, (3, 3), padding='same', use_bias=False, data_format=data_format, name='trans3'))
    if Batch_Norm:
        discriminator.add(layers.BatchNormalization(axis=axis, name='bn6'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu6'))

    discriminator.add(
        layers.Conv2D(1024, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down4'))
    if Batch_Norm:
        discriminator.add(layers.BatchNormalization(axis=axis, name='bn7'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu7'))
    discriminator.add(
        layers.Conv2D(1024, (3, 3), padding='same', use_bias=False, data_format=data_format, name='trans4'))
    if Batch_Norm:
        discriminator.add(layers.BatchNormalization(axis=axis, name='bn8'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu8'))

    discriminator.add(layers.Flatten(name='flatten'))
    discriminator.add(layers.Dense(1, activation='linear', name='score'))



    return discriminator


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


def make_discriminator_model_MNIST_Dynamic(dropout: bool,batch_norm: bool,activation):
  axis=-1
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  if batch_norm:
    model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())
  if dropout:
      model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  if batch_norm:
    model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())
  if dropout:
      model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation=activation))

  return model

#try batch1_2 0.29 mil paramters
#try deep!_4 0.4 mill paramaeters
def make_discriminator_model_MNIST_Dynamic_Deeper(dropout: bool,batch_norm: bool,activation,ksize:int,Deeper:int,Double:bool):
  axis=-1
  model = tf.keras.Sequential()

  model.add(layers.Conv2D(32, (4+ksize, 4+ksize), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  # NO Batch Normalisation
  model.add(layers.LeakyReLU())


  model.add(layers.Conv2D(64, (4+ksize, 4+ksize), strides=(2, 2), padding='same'))
  if batch_norm:
    model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())
  if dropout:
      model.add(layers.Dropout(0.3))

  if Double:
      model.add(layers.Conv2D(64, (4 + ksize, 4 + ksize), strides=(2, 2), padding='same'))
      if batch_norm:
          model.add(layers.BatchNormalization(axis=axis))
      model.add(layers.LeakyReLU())
      if dropout:
          model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (3+ksize, 3+ksize), strides=(2, 2), padding='same'))
  if batch_norm:
    model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())
  if dropout:
      model.add(layers.Dropout(0.3))

  if Double:
      model.add(layers.Conv2D(128, (3 + ksize, 3 + ksize), strides=(2, 2), padding='same'))
      if batch_norm:
          model.add(layers.BatchNormalization(axis=axis))
      model.add(layers.LeakyReLU())
      if dropout:
          model.add(layers.Dropout(0.3))

  if Deeper == 1:
      model.add(layers.Conv2D(256, (3 + ksize, 3 + ksize), strides=(2, 2), padding='same'))
      if batch_norm:
          model.add(layers.BatchNormalization(axis=axis))
      model.add(layers.LeakyReLU())
      if dropout:
          model.add(layers.Dropout(0.3))

      if Double:
          model.add(layers.Conv2D(256, (3 + ksize, 3 + ksize), strides=(2, 2), padding='same'))
          if batch_norm:
              model.add(layers.BatchNormalization(axis=axis))
          model.add(layers.LeakyReLU())
          if dropout:
              model.add(layers.Dropout(0.3))

  if Deeper==2:
      model.add(layers.Conv2D(512, (3+ksize, 3+ksize), strides=(2, 2), padding='same'))
      if batch_norm:
        model.add(layers.BatchNormalization(axis=axis))
      model.add(layers.LeakyReLU())
      if dropout:
          model.add(layers.Dropout(0.3))

      if Double:
          model.add(layers.Conv2D(512, (3 + ksize, 3 + ksize), strides=(2, 2), padding='same'))
          if batch_norm:
              model.add(layers.BatchNormalization(axis=axis))
          model.add(layers.LeakyReLU())
          if dropout:
              model.add(layers.Dropout(0.3))


  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation=activation))

  return model




def make_discriminator_model_MNIST_Batch1_2():
  axis=-1
  model = tf.keras.Sequential()

  model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())


  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation='sigmoid'))

  return model



def make_discriminator_model_MNIST_Deep1():
  axis = -1
  model = tf.keras.Sequential()

  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation='sigmoid'))

  return model

def make_discriminator_model_MNIST_Batch1_2():
    axis = -1
    data_format = 'channels_last'
    im_size = (28, 28, 1)

    discriminator = tf.keras.Sequential(name='discriminator')
    discriminator.add(
        layers.Conv2D(32, (5, 5), padding='same', use_bias=True, data_format=data_format, input_shape=im_size,
                      name='from_im'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu0'))

    discriminator.add(
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down1'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn1'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu1'))
    discriminator.add(
        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, data_format=data_format, name='trans1'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn2'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu2'))

    discriminator.add(
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down2'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn3'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu3'))
    discriminator.add(
        layers.Conv2D(128, (3, 3), padding='same', use_bias=False, data_format=data_format, name='trans2'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn4'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu4'))

    return discriminator


def make_discriminator_model_MNIST_Deep1_3():
  axis = -1
  model = tf.keras.Sequential()

  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  # model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
  # model.add(layers.BatchNormalization(axis=axis))
  # model.add(layers.LeakyReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation='sigmoid'))

  return model

def make_discriminator_model_MNIST_Deep1_4():
  axis = -1
  model = tf.keras.Sequential()

  model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization(axis=axis))
  model.add(layers.LeakyReLU())

  # model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
  # model.add(layers.BatchNormalization(axis=axis))
  # model.add(layers.LeakyReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation='sigmoid'))

  return model



def make_discriminator_model_MNIST_Deep():
  axis = -1
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation='sigmoid'))

  return model



def make_discriminator_model_MNIST_Deep3():
    axis = -1
    data_format = 'channels_last'
    im_size = (28, 28, 1)

    discriminator = tf.keras.Sequential(name='discriminator')
    discriminator.add(
        layers.Conv2D(32, (5, 5), padding='same', use_bias=True, data_format=data_format, input_shape=im_size,
                      name='from_im'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu0'))

    discriminator.add(
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down1'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn1'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu1'))
    discriminator.add(
        layers.Conv2D(64, (3, 3), padding='same', use_bias=False, data_format=data_format, name='trans1'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn2'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu2'))

    discriminator.add(
        layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down2'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn3'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu3'))
    discriminator.add(
        layers.Conv2D(128, (5, 5), padding='same', use_bias=False, data_format=data_format, name='trans2'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn4'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu4'))

    discriminator.add(
        layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, data_format=data_format,
                      name='down3'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn5'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu5'))
    discriminator.add(
        layers.Conv2D(256, (3, 3), padding='same', use_bias=False, data_format=data_format, name='trans3'))
    discriminator.add(layers.BatchNormalization(axis=axis, name='bn6'))
    discriminator.add(layers.LeakyReLU(alpha=0.1, name='lrelu6'))

    discriminator.add(layers.Flatten(name='flatten'))
    discriminator.add(layers.Dense(1, activation='sigmoid', name='score'))

    return discriminator


