
import time
from IPython import display
import tensorflow as tf
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
from skimage.transform import resize
import importlib.util

#import NetworkModels
# Files for Generatring Network Models
#spec = importlib.util.spec_from_file_location("NetworkModels","/content/gdrive/My Drive/PythonFiles/NetworkModels.py")
#NetworkModels = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(NetworkModels)


FILEHASHKEY = 4 # Change this to check you are running the most recent version on colab

def print_hashkey():
    print(FILEHASHKEY)

def calc_FID_MC(generator,num_FID_MC,num_latent,modelv3):

    V3input_shape =(75, 75, 3)
    #modelv3 = InceptionV3(include_top=False, pooling='avg', input_shape=V3input_shape)

    seed_FID2 = tf.random.normal([num_FID_MC, num_latent])
    seed_FID1 = tf.random.normal([num_FID_MC, num_latent])
    predictions1 = generator(seed_FID1, training=False)
    predictions2 = generator(seed_FID2, training=False)


    images2 = tf.dtypes.cast(predictions2, tf.float32)
    images1 = tf.dtypes.cast(predictions1, tf.float32)

    # resize images
    images1 = scale_images(images1, V3input_shape)
    images2 = scale_images(images2, V3input_shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # calculate fid
    fid = calculate_fid(modelv3, images1, images2)
    #print('FID: %.3f' % fid)
    return fid

def calc_FID(generator,FID,seed_FID,num_ex_FID,batch_size,V3input_shape,modelv3,start,image_batch):
    predictions = generator(seed_FID, training=False)
    images2 = predictions

    # may return error
    if num_ex_FID > batch_size:
        raise Exception("num_ex_FID>batch_size")

    images1 = image_batch[0][:num_ex_FID]
    images2 = tf.dtypes.cast(images2, tf.float32)
    images1 = tf.dtypes.cast(images1, tf.float32)
    # images2 = tf.image.convert_image_dtype(images2, dtype=tf.float32, saturate=False)
    # print(images1)
    # print(images2)

    # images1 = images1.astype('float32')
    # images2 = images2.astype('float32')
    # resize images
    images1 = scale_images(images1, V3input_shape)
    images2 = scale_images(images2, V3input_shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # calculate fid
    fid = calculate_fid(modelv3, images1, images2)
    FID.append(fid)
    print('FID: %.3f' % fid)



def train2(dataset,var,instr,seeds,model):

    epochs = var["epochs"]
    num_latent = var["num_latent"]
    batch_size = var["batch_size"]
    label_smoothing = var["label_smoothing"]

    FID_bool = instr["FID"]
    FID_batch_interval = instr['FID_batch_interval']
    data_flag = instr["dataflag"]

    seed_GIF = seeds["seed_GIF"]
    seed_FID = seeds["seed_FID"]
    num_ex_GIF = seeds["num_ex_GIF"]
    num_ex_FID = seeds["num_ex_FID"]

    generator = model["generator"]
    discriminator = model["discriminator"]
    loss_func_key = model["loss_func"]
    generator_optimizer = model["generator_optimizer"]
    discriminator_optimizer = model["discriminator_optimizer"]

    if data_flag == "CelebA":
        num_instance = 202599
    elif data_flag == "MNIST":
        num_instance = 60000


    num_steps_per_epoch = num_instance // batch_size




    FID = []
    FID_MC = []
    history = {"D" :[],
               "G" :[],
               "ave_real":[],
               "ave_fake":[],
               "mse_real":[],
               "mse_fake":[],
                }


    # prepare the inception v3 model
    V3input_shape =(75, 75, 3)
    modelv3 = InceptionV3(include_top=False, pooling='avg', input_shape=V3input_shape)
    mse = tf.keras.losses.MeanSquaredError()
    discriminator_loss_func = return_discriminator_loss_func(loss_func_key)
    generator_loss_func = return_generator_loss_func(loss_func_key)
    bar = tf.keras.utils.Progbar(num_steps_per_epoch, stateful_metrics=('loss_gen', 'loss_dis'))


    for epoch in range(epochs):
        start = time.time()

        i = 0
        for image_batch in dataset:

            # Because CelebA batch contains labels that need to be parsed
            if data_flag == "CelebA":
                image_batch = image_batch[0]

            loss = train_step(image_batch,num_latent,generator,discriminator
                              ,generator_optimizer,discriminator_optimizer,mse,label_smoothing,discriminator_loss_func,generator_loss_func)

            history["D"].append(loss[0])
            history["G"].append(loss[1])
            history["ave_real"].append(loss[2])
            history["ave_fake"].append(loss[3])
            history["mse_real"].append(loss[4])
            history["mse_fake"].append(loss[5])
            i+=1

            if i%10 == 0:
                display.clear_output(wait=True)
                print("Epoch:{}  Number of images processed: {}".format(epoch,i*batch_size) )

            if i%5 == 0:
                bar.update(i + 1, [('loss_gen', loss[1]), ('loss_dis', loss[0])])

            if FID_batch_interval!=0 and i%FID_batch_interval==0:
                calc_FID(generator, FID, seed_FID, num_ex_FID, batch_size, V3input_shape, modelv3, start, image_batch)
                num_FID_MC = 64
                fid_mc = calc_FID_MC(generator, num_FID_MC, num_latent, modelv3)
                FID_MC.append(fid_mc)




        time2 = time.time() - start

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        #generate_and_save_images(generator, epoch + 1, seed_GIF)
        test_seed = tf.random.normal([num_ex_GIF, num_latent])
        gen_image = generator(test_seed, training=False)
        generate_and_save_images2(gen_image)


        time3 = time.time() - start
        if FID_bool:

            calc_FID(generator, FID, seed_FID, num_ex_FID, batch_size, V3input_shape, modelv3, start, image_batch)
            num_FID_MC = 64
            fid_mc = calc_FID_MC(generator, num_FID_MC, num_latent, modelv3)
            FID_MC.append(fid_mc)
            time4 = time.time() - start
            # Produce Images to be evaluated by FID

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            pass
            # checkpoint.save(file_prefix=checkpoint_prefix)

        # Save the images every 1 epochs
        if (epoch + 1) % 1 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epochs,
                                     seed_GIF)

        print('Time for epoch {} is {} sec\n'.format(epoch + 1, time2))  #
        print('Time to save gif is {} sec\n'.format(time3 - time2))  #

        if FID_bool:
            print('Time to calc FID is {} sec\n'.format(time4 - time3))
        # Generate after the final epoch
        # What is the purpose of this??
        # display.clear_output(wait=True)
        # generate_and_save_images(generator, epochs, seed)
    return (FID, history, FID_MC)



# `tf.function' causes function to be 'compiled'

@tf.function
def train_step(images,num_latent,generator,discriminator,generator_optimizer,discriminator_optimizer,mse,label_smoothing,discriminator_loss,generator_loss):

    noise = tf.random.normal([images.shape[0], num_latent])


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)


        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)


        gen_loss = generator_loss(fake_output,label_smoothing)
        disc_loss = discriminator_loss(real_output, fake_output, label_smoothing)

        # Or if you want to try discriminating loss
        # Could add parameter if you wanted once we get it to work
        #gen_loss = NetworkModels.ns_generator_loss(fake_output)
        #disc_loss = NetworkModels.ns_discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # real_output & fake_output have size (batch_size,1) of type Tensor
    ave_real_output = tf.reduce_mean(real_output)
    ave_fake_output = tf.reduce_mean(fake_output)

    # WHY IS mse_real full of nonsense results

    mse_fake = tf.reduce_mean(tf.math.square(tf.zeros_like(fake_output) + label_smoothing - fake_output))
    mse_real = tf.reduce_mean(tf.math.square(tf.ones_like(real_output) - label_smoothing - real_output))

    return (gen_loss, disc_loss,ave_real_output,ave_fake_output,mse_real,mse_fake)
    


def calc_accuracy(real_output,key:str):
    if key == "real":
        pass
    elif key =="fake":
        pass
    else:
        raise Exception("Incorrect Key")


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

def generate_and_save_images2(gen_image):

    plt.figure(figsize=(10, 10))

    for ind, im in enumerate(gen_image[:9]):
        im = tf.keras.preprocessing.image.array_to_img(im)
        plt.subplot(3, 3, ind + 1)
        plt.imshow(im)
    plt.tight_layout()
    plt.show()


# example of calculating the frechet inception distance in Keras for cifar10



# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# ----------- LossFunction --------- #

def return_loss_func(key: str):


    # calculate wasserstein loss
    def wasserstein_loss(y_true, y_pred):
        return backend.mean(y_true * y_pred)

    if key == "binarycrossentropy":
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy

    elif key == "wasserstein":
        return wasserstein_loss
    else:
        raise Exception('Incorrect "Key" argument')


def return_discriminator_loss_func(key: str):
    def ns_discriminator_loss(real_output, generated_output,label_smoothing):
        return -tf.reduce_mean(tf.math.log(real_output) + tf.math.log(1 - generated_output))

    def ns_discriminator_loss_Wei(score_real, score_gen, label_smoothing):
        return tf.reduce_mean(tf.nn.softplus(score_gen) + tf.nn.softplus(-score_real) + label_smoothing * score_real)


    if key =="n_s":
        return ns_discriminator_loss

    if key == "non_sat":
        return ns_discriminator_loss_Wei

    else:
        loss_func = return_loss_func(key)

        def discriminator_loss(real_output, fake_output, label_smoothing):
            real_loss = loss_func(tf.ones_like(real_output) - label_smoothing, real_output)
            fake_loss = loss_func(tf.zeros_like(fake_output) + label_smoothing, fake_output)
            total_loss = real_loss + fake_loss
            return total_loss


        return discriminator_loss

def return_generator_loss_func(key:str):
    def ns_generator_loss(generated_output,label_smoothing):
        return -tf.reduce_mean(tf.math.log(generated_output))

    def ns_generator_loss_Wei(score_gen, label_smoothing):
        return tf.reduce_mean(tf.nn.softplus(-score_gen))

    if key =="n_s":
        return ns_generator_loss

    if key =="non_sat":
        return ns_generator_loss_Wei

    else:
        loss_func = return_loss_func(key)

        def generator_loss(fake_output, label_smoothing):
            return loss_func(tf.ones_like(fake_output) - label_smoothing, fake_output)

        return generator_loss



# For non saturating loss
# Add to possible loss functions maybe??
# haven't been able to get it to work

# Change code in Train_Step whether you want normal or non discriminating loss

def ns_discriminator_loss(real_output, generated_output):
    return -tf.reduce_mean(tf.math.log(real_output) + tf.math.log(1 - generated_output))


def ns_generator_loss(generated_output):
    return -tf.reduce_mean(tf.math.log(generated_output))


def discriminator_loss(real_output, fake_output, loss_func,label_smoothing):
    real_loss = loss_func(tf.ones_like(real_output)-label_smoothing, real_output)
    fake_loss = loss_func(tf.zeros_like(fake_output)+label_smoothing, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, loss_func,label_smoothing):
    # wasserstein_loss(tf.ones_like(fake_output), fake_output)
    return loss_func(tf.ones_like(fake_output)-label_smoothing, fake_output)


def generator_loss_28(input_z, out_channel_dim=3, alpha=0.2):
    input_fake = generator(input_z, out_channel_dim, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(input_fake, reuse=True, alpha=alpha)
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
    return g_loss


def discriminator_loss_28(input_real, out_channel=3, alpha=0.2, smooth_factor=0.1):


    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * (1 - smooth_factor)))
    input_fake = generator(input_z, out_channel_dim, alpha=alpha)
    _, d_logits_fake = discriminator(input_fake, reuse=True, alpha=alpha)
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    return d_loss_real + d_loss_fake


def model_loss(input_real, input_z, out_channel_dim, alpha=0.2, smooth_factor=0.1):
    """
    Non-saturating loss
    https://github.com/HACKERSHUBH/Face-Genaration-using-Generative-Adversarial-Network/blob/master/face_gen.ipynb
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # TODO: Implement Function


    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * (1 - smooth_factor)))

    input_fake = generator(input_z, out_channel_dim, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(input_fake, reuse=True, alpha=alpha)

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))



    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    return d_loss_real + d_loss_fake, g_loss

