
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
spec = importlib.util.spec_from_file_location("NetworkModels","/content/gdrive/My Drive/PythonFiles/NetworkModels.py")
NetworkModels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NetworkModels)


FILEHASHKEY = 15 # Change this to check you are running the most recent version on colab

def print_hashkey():
    print(FILEHASHKEY)


def train2(dataset,var,instr,seeds,model):

    epochs = var["epochs"]
    num_latent = var["num_latent"]
    batch_size = var["batch_size"]

    FID_bool = instr["FID"]
    data_flag = instr["dataflag"]

    seed_GIF = seeds["seed_GIF"]
    seed_FID = seeds["seed_FID"]
    num_ex_GIF = seeds["num_ex_GIF"]
    num_ex_FID = seeds["num_ex_FID"]

    generator = model["generator"]
    discriminator = model["discriminator"]
    loss_func = model["loss_func"]
    generator_optimizer = model["generator_optimizer"]
    discriminator_optimizer = model["discriminator_optimizer"]



    FID = []
    loss_history = {"D" :[],
                    "G" :[]
                    }

    # prepare the inception v3 model
    V3input_shape =(75, 75, 3)
    modelv3 = InceptionV3(include_top=False, pooling='avg', input_shape=V3input_shape)

    for epoch in range(epochs):
        start = time.time()

        i = 0
        for image_batch in dataset:

            # Because CelebA batch contains labels that need to be parsed
            if data_flag == "CelebA":
                image_batch = image_batch[0]

            loss = train_step(image_batch,loss_func,num_latent,generator,discriminator
                              ,generator_optimizer,discriminator_optimizer)
            loss_history["D"].append(loss[0])
            loss_history["G"].append(loss[1])
            i+=1

            if i%100 == 0:
                display.clear_output(wait=True)
                print("Number of images processed: {}".format(i*batch_size) )



        time2 = time.time() - start

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed_GIF)

        time3 = time.time() - start
        if FID_bool:
            # Produce Images to be evaluated by FID

            predictions = generator(seed_FID, training=False)
            images2 = predictions

            time4_1 = time.time() - start
            # may return error
            if num_ex_FID>batch_size:
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
            time4_2 = time.time() - start
            # calculate fid
            fid = calculate_fid(modelv3, images1, images2)
            FID.append(fid)
            print('FID: %.3f' % fid)
            print(loss_history["D"])
            print(loss_history["G"])
            time4 = time.time() - start

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Save the images every 10 epochs
        if (epoch + 1) % 1 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epochs,
                                     seed_GIF)

        print('Time for epoch {} is {} sec\n'.format(epoch + 1, time2))  # Why doesn't this display??
        print('Time to save gif is {} sec\n'.format(time3 - time2))  # Why doesn't this display??

        if FID_bool:
            print('Time to calc FID is {} sec\n'.format(time4 - time3))  # Why doesn't this display??
            print('Time to generate images & build model is {} sec\n'.format(
                time4_1 - time3))  # Why doesn't this display??
            print('Time to preprocess images {} sec\n'.format(time4_2 - time4_1))  # Why doesn't this display??
            print('Time to actually calculate FID {} sec\n'.format(time4 - time4_2))  # Why doesn't this display??

        # Generate after the final epoch
        # What is the purpose of this??
        # display.clear_output(wait=True)
        # generate_and_save_images(generator, epochs, seed)
    return (FID, loss_history)

    # `tf.function' causes function to be 'compiled'


@tf.function
def train_step(images,loss_func,num_latent,generator,discriminator,generator_optimizer,discriminator_optimizer):
    noise = tf.random.normal([images.shape[0], num_latent])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)


        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        # print(real_output)
        # print(images.shape)
        # print(real_output.shape)
        # print(fake_output.shape)

        gen_loss = NetworkModels.generator_loss(fake_output, loss_func)
        disc_loss = NetworkModels.discriminator_loss(real_output, fake_output, loss_func)

        # Or if you want to try discriminating loss
        # Could add parameter if you wanted once we get it to work
        #gen_loss = NetworkModels.ns_generator_loss(fake_output)
        #disc_loss = NetworkModels.ns_discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return (gen_loss, disc_loss)



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