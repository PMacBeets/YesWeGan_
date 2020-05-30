import os
import tensorflow as tf
import matplotlib.pyplot as plt

FILEHASHKEY = 123 # Change this to check you are running the most recent version on colab


def ReadData(flag: str, plotdata: bool, batch_size: int,zoom: bool, size: int, channels: int):
    """

    :param flag: string, MNIST or CelebA
    :param folder: string, Path of folder if required
    :param plotdata: Boolean that indicates if data wants to be plotted with plt
    :return:
    """

    dataset= 0
    #batch_size = 0
    buffer_size = 0

    if flag == "MNIST":
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalise the images to [-1, 1]
        buffer_size = 60000
        #batch_size = 256
        # Batch and shuffle the data
        # If MNIST dataset dosnt work its probably due to different application of these shuffling and batching functions
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images)#.shuffle(buffer_size).batch(batch_size)
        dataset = train_dataset
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)


    elif flag == "CelebA":
        if zoom and size == 28 and channels == 3:
            file_folder = '/content/gdrive/My Drive/TfRecords_28_28_Zoom'
            input_size = (28, 28, 3)  # NCHW or channels_first format
            data_format = 'channels_last'
            num_file = 9
            filename = ['data_{}'.format(i) for i in range(num_file)]

            #batch_size = 64
            buffer_size = 10000
            num_labels = 0
            num_threads = 7
            shuffle_file = True

            training_data = ReadTFRecords(
                filename, num_features=28 * 28 * 3, batch_size=batch_size,
                file_folder=file_folder, num_threads=num_threads,
                shuffle_file=shuffle_file)

            dataset = training_data.dataset
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    else:
        raise Exception('Incorrect "flag argument')

    if plotdata:

        print(dataset)
        plt.figure(figsize=(10, 10))

        # Check Images are correct
        for i, data in enumerate(dataset.take(8)):
            if flag == "CelebA":
                data = data[0]

            img = tf.keras.preprocessing.image.array_to_img(data)
            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
        plt.show()


    # Post dataplotting batching of data
    dataset = dataset.batch(batch_size)
    # Not sure if these 2 are nessercary
    # I think the first shuffle, shuffle teh entire dataset while the second shuffle
    # shuffles the batches constantly as its past through the epochs which is more important
    dataset.repeat(None)

    return dataset


class ReadTFRecords(object):
    def __init__(
            self, filename, num_features=None, num_labels=0, dtype=tf.string, batch_size=64,
            skip_count=0, file_repeat=1, num_epoch=None, file_folder=None, num_threads=8,
            buffer_size=10000, shuffle_file=False):
        """ This function creates a dataset object that reads data from files.

        :param filename:
        :param num_features:
        :param num_labels:
        :param dtype: default tf.string, the dtype of features stored in tfrecord file
        :param num_epoch:
        :param buffer_size:
        :param batch_size:
        :param skip_count: if num_instance % batch_size != 0, we could skip some instances
        :param file_repeat: if num_instance % batch_size != 0, we could repeat the files for k times
        :param num_epoch:
        :param file_folder: if not specified, DEFAULT_IN_FILE_DIR is used.
        :param num_threads:
        :param buffer_size:
        :param shuffle_file: bool, whether to shuffle the filename list

        """
        if file_folder is None:
            file_folder = DEFAULT_IN_FILE_DIR
        # check inputs
        if isinstance(filename, str):  # if string, add file location and .tfrecords
            filename = [os.path.join(file_folder, filename + '.tfrecords')]
        else:  # if list, add file location and .tfrecords to each element in list
            filename = [os.path.join(file_folder, file + '.tfrecords') for file in filename]
        for file in filename:
            assert os.path.isfile(file), 'File {} does not exist.'.format(file)
        if file_repeat > 1:
            filename = filename * int(file_repeat)
        if shuffle_file:  # shuffle operates on the original list and returns None / does not return anything
            from random import shuffle
            shuffle(filename)

        # training information
        self.num_features = num_features
        self.num_labels = num_labels
        self.dtype = dtype
        self.batch_size = batch_size
        self.batch_shape = [self.batch_size, self.num_features]
        self.num_epoch = num_epoch
        self.skip_count = skip_count
        # read data,
        dataset = tf.data.TFRecordDataset(filename)  # setting num_parallel_reads=num_threads decreased the performance
        self.dataset = dataset.map(self.__parser__, num_parallel_calls=num_threads)
        self.iterator = None
        self.buffer_size = buffer_size
        self.scheduled = False
        self.num_threads = num_threads

    ###################################################################
    # DO NOT USE THIS PARSER BUT OUR OWN read_tfrecords
    # def __parser__(self, example_proto):
    #     """ This function parses a single datum

    #     :param example_proto:
    #     :return:
    #     """
    #     # configure feature and label length
    #     x_config = tf.io.FixedLenFeature([self.num_features], tf.float32) \
    #         if self.dtype == tf.float32 else tf.io.FixedLenFeature([], tf.string)
    #     if self.num_labels == 0:
    #         proto_config = {'x': x_config}
    #     else:
    #         y_config = tf.io.FixedLenFeature([self.num_labels], tf.int64)
    #         proto_config = {'x': x_config, 'y': y_config}

    #     # decode examples
    #     datum = tf.io.parse_single_example(example_proto, features=proto_config)
    #     if self.dtype == tf.string:  # if input is string / bytes, decode it to float32
    #         _temp = tf.io.decode_raw(datum['x'], tf.uint8)
    #         datum['x'] = tf.cast(_temp, tf.float32)

    #     # return data
    #     if 'y' in datum:
    #         # datum['y'] = tf.cast(datum['y'], tf.int32)
    #         return datum['x'], datum['y']
    #     else:
    #         return datum['x']

    def __parser__(self, serialized_example):
        feature_description = {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.string),
            'height': tf.io.FixedLenFeature((), tf.int64),
            'width': tf.io.FixedLenFeature((), tf.int64),
            'depth': tf.io.FixedLenFeature((), tf.int64)
        }


        example = tf.io.parse_single_example(serialized_example, feature_description)

        image = tf.io.parse_tensor(example['image'], out_type=float)
        label = tf.io.parse_tensor(example['label'], out_type=float)
        image_shape = [example['height'], example['width'], example['depth']]
        image = tf.reshape(image, image_shape)

        return (image, example['label'])

    ###################################################################
    def shape2image(self, channels, height, width, data_format):
        """ This function shapes the input instance to image tensor

        :param channels:
        :param height:
        :param width:
        :return:
        """

        def image_preprocessor(image):
            # scale image to [-1,1]
            image = tf.subtract(tf.divide(image, 127.5), 1)
            # reshape
            image = tf.reshape(image, (channels, height, width)) \
                if data_format == 'channels_first' else tf.reshape(image, (height, width, channels))

            return image

        self.batch_shape = [self.batch_size, height, width, channels] \
            if data_format == 'channels_last' else [self.batch_size, channels, height, width]
        #
        if self.num_labels == 0:
            self.dataset = self.dataset.map(
                lambda image_data: image_preprocessor(image_data),
                num_parallel_calls=self.num_threads)
        else:
            self.dataset = self.dataset.map(
                lambda image_data, label: (image_preprocessor(image_data[0]), label),
                num_parallel_calls=self.num_threads)

    ###################################################################
    def scheduler(
            self, batch_size=None, num_epoch=None, shuffle_data=True, buffer_size=None, skip_count=None,
            sample_same_class=False, sample_class=None):
        """ This function schedules the batching process

        :param batch_size:
        :param num_epoch:
        :param buffer_size:
        :param skip_count:
        :param sample_same_class: if the data must be sampled from the same class at one iteration
        :param sample_class: if provided, the data will be sampled from class of this label, otherwise,
            data of a random class aresampled.
        :param shuffle_data:
        :return:
        """
        if not self.scheduled:
            # update batch information
            if batch_size is not None:
                self.batch_size = batch_size
                self.batch_shape[0] = self.batch_size
            if num_epoch is not None:
                self.num_epoch = num_epoch
            if buffer_size is not None:
                self.buffer_size = buffer_size
            if skip_count is not None:
                self.skip_count = skip_count
            # skip instances
            if self.skip_count > 0:
                print('Number of {} instances skipped.'.format(self.skip_count))
                self.dataset = self.dataset.skip(self.skip_count)
            # shuffle
            if shuffle_data:
                self.dataset = self.dataset.shuffle(self.buffer_size)
            # set batching process
            if sample_same_class:
                if sample_class is None:
                    print('Caution: samples from the same class at each call.')
                    group_fun = tf.contrib.data.group_by_window(
                        key_func=lambda data_x, data_y: data_y,
                        reduce_func=lambda key, d: d.batch(self.batch_size),
                        window_size=self.batch_size)
                    self.dataset = self.dataset.apply(group_fun)
                else:
                    print('Caution: samples from class {}. This should not be used in training'.format(sample_class))
                    self.dataset = self.dataset.filter(lambda x, y: tf.equal(y[0], sample_class))
                    self.dataset = self.dataset.batch(self.batch_size)
            else:
                self.dataset = self.dataset.batch(self.batch_size)
            # self.dataset = self.dataset.padded_batch(batch_size)
            if self.num_epoch is None:
                self.dataset = self.dataset.repeat()
            else:
                FLAGS.print('Num_epoch set: {} epochs.'.format(num_epoch))
                self.dataset = self.dataset.repeat(self.num_epoch)

# -------------- END CLASS ----------------------#


def cube(x: int):
    return x*x*x

