import numpy as np
import math
import struct

class Data:
    def __init__(self, training_img_file="data/train-images-idx3-ubyte", training_lbl_file="data/train-labels-idx1-ubyte", test_img_file="data/t10k-images-idx3-ubyte", test_lbl_file="data/t10k-labels-idx1-ubyte", batch_size=100):
        self.default_batch_size = batch_size
        self.training_img_file = training_img_file
        self.training_lbl_file = training_lbl_file
        self.test_img_file = test_img_file
        self.test_lbl_file = test_lbl_file
        self.training_images, self.training_labels = self.load_training_data()
        self.test_images, self.test_labels = self.load_test_data()

    """take binary image file, and load the images into an ndarray"""
    def format_images(self, data_file):
        try:
            magic_num = struct.unpack(">L", data_file.read(4))[0] # magic number isn't used, but has some info about the file
            num_images = struct.unpack(">L", data_file.read(4))[0] # number of total images
            rows = struct.unpack(">L", data_file.read(4))[0] # per image
            cols = struct.unpack(">L", data_file.read(4))[0] # per image

            img_buffer = data_file.read(num_images * rows * cols) # reads all the data for all the images
            dt = np.dtype(np.uint8).newbyteorder('>') # big endian byte order
            img_array = np.frombuffer(img_buffer, dtype=dt, count=-1, offset=0) # make a one dimensional array of all the data
            img_array = np.reshape(img_array, (num_images, rows * cols)).transpose() # reshape array so that each column is an image
            img_array = img_array.astype(dtype=np.float32, casting='safe') # change data type to float32
        finally:
            return img_array

    """take binary label file, and load the labels into an ndarray"""
    def format_labels(self, data_file):
        magic_num = struct.unpack(">L", data_file.read(4))[0] # not used(see above)
        num_labels = struct.unpack(">L", data_file.read(4))[0] # total number of labels(same as number of images)
        try:
            lbl_buffer = data_file.read(num_labels) # reads all the data for all the images
            dt = np.dtype(np.uint8).newbyteorder('>') # big endian byte order
            lbl_array = np.frombuffer(lbl_buffer, dtype=dt, count=-1, offset=0) # one d array with all images
            lbl_array = lbl_array.astype(dtype=np.float32, casting='safe') # change data type to float32
        finally:
            return lbl_array

    """perform zero mean, and unit variance normalization on images"""
    def normalize_images(self, imgs):
        mean = np.mean(imgs) # calculates the mean of all the pixels of all the images
        std = np.std(imgs) # calculates the standard deviation of all the pixels
        return (imgs - mean) / std # centers the values around zero, and devides by the deviation

    """randomize input images and labels(they will still line up)"""
    def randomize_data(self, imgs, lbls):
        permutation = np.random.permutation(imgs.shape[1]) # make a permutation of the indices of the images/labels
        shuffled_imgs = np.take(imgs, permutation, axis=-1) # apply the permutation to the images
        shuffled_lbls = np.take(lbls, permutation, axis=-1) # apply the permutation to the labels
        return shuffled_imgs, shuffled_lbls

    """display image for visualization purposes"""
    def display_image(self, img):
        disp = ['.', ',', ';', 'x'] # index of symbols
        image = img.reshape(784) # select one image to use
        for y in range(28):
            for x in range(28):
                symbol = disp[min(math.floor(image[y*28 + x]*(len(disp))), 3)] # determine the symbol to use for the current pixel
                print(symbol + symbol, end="") # display two of the symbols, to make the image square
            print("", end="\n") # start new line

    """load all the training images and labels"""
    def load_training_data(self, randomize=True):
        # open the raw data files
        training_images_raw = open(self.training_img_file, "rb")
        training_labels_raw = open(self.training_lbl_file, "rb")

        # create numpy arrays with the raw data
        training_images = self.format_images(training_images_raw)
        training_labels = self.format_labels(training_labels_raw)

        # close input streams
        training_images_raw.close()
        training_labels_raw.close()

        # process the data, and prepare it for training
        training_images = self.normalize_images(training_images)
        training_images = np.append(training_images, np.ones((1, training_images.shape[1])), axis=0)

        #randomize
        if randomize:
            training_images, training_labels = self.randomize_data(training_images, training_labels)

        # return the processed data
        return training_images, training_labels

    """load all the test images and labels"""
    def load_test_data(self, randomize=True):
        # open the raw data files
        test_images_raw = open(self.test_img_file, "rb")
        test_labels_raw = open(self.test_lbl_file, "rb")

        # create numpy arrays with the correct (raw) data
        test_images = self.format_images(test_images_raw)
        test_labels = self.format_labels(test_labels_raw)

        # close input streams
        test_images_raw.close()
        test_labels_raw.close()

        # process the data, and prepare it for training
        test_images = self.normalize_images(test_images)
        test_images = np.append(test_images, np.ones((1, test_images.shape[1])), axis=0)

        #randomize
        if randomize:
            test_images, test_labels = self.randomize_data(test_images, test_labels)

        # return the processed data
        return test_images, test_labels

    """create a random batch of images with specified size"""
    def next_batch(self, size=-2):
        if size == -1:
            return self.randomize_data(self.training_images, self.training_labels)
        elif size == -2:
            size = self.default_batch_size
        indices = np.arange(self.training_images.shape[1])
        np.random.shuffle(indices)
        return np.take(self.training_images, indices[:size], axis=-1), np.take(self.training_labels, indices[:size], axis=-1) # use the permutation as indices to select random image-label pairs