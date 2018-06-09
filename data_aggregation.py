import numpy as np
import math
import struct

"""take binary image file, and load the images into an ndarray"""
def format_images(data_file):
    images = np.array([]) # initialize return variable
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
        images = img_array
    finally:
        return images

"""take binary label file, and load the labels into an ndarray"""
def format_labels(data_file):
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
def normalize_images(imgs):
    mean = np.mean(imgs) # calculates the mean of all the pixels of all the images
    std = np.std(imgs) # calculates the standard deviation of all the pixels
    return (imgs - mean) / std # centers the values around zero, and devides by the deviation

"""randomize input images and labels(they will still line up)"""
def randomize_data(imgs, lbls):
    permutation = np.random.permutation(imgs.shape[1]) # make a permutation of the indices of the images/labels
    shuffled_imgs = np.take(imgs, permutation, axis=-1) # apply the permutation to the images
    shuffled_lbls = np.take(lbls, permutation, axis=-1) # apply the permutation to the labels
    return shuffled_imgs, shuffled_lbls

"""select all the images of ones and zeros, and also handles the intercept term"""
def process_data(imgs, lbls):
    index = np.sort(np.append(np.where(lbls==0)[0], np.where(lbls==1)[0])) # sort the indices of the imgs/lbls that are 1 or 0
    labels = np.take(lbls, index) # take the labels that correspond to 1 or 0
    images = np.take(imgs, index, axis=-1) # take the images that correspond to 1 or 0
    images = normalize_images(images) # apply zero mean and unit variance normalization to the images
    images = np.append(images, np.ones((1, images.shape[1])), axis=0) # append ones to image data for intercept term
    return images, labels

"""display image for visualization purposes"""
def display_image(imgs, index):
    disp = ['.', ',', ';', 'x'] # index of symbols
    image = imgs[:784, index:index+1].reshape(784) # select one image to use
    for y in range(28):
        for x in range(28):
            symbol = disp[min(math.floor(image[y*28 + x]*(len(disp))), 3)] # determine the symbol to use for the current pixel
            print(symbol+symbol, end="") # display two of the symbols, to make the image square
        print("", end="\n") # start new line

"""load all the training images and labels"""
def load_training_data(randomize=True):
    # open the raw data files
    training_images_raw = open("data/train-images-idx3-ubyte", "rb")
    training_labels_raw = open("data/train-labels-idx1-ubyte", "rb")

    # create numpy arrays with the raw data
    training_images = format_images(training_images_raw)
    training_labels = format_labels(training_labels_raw)

    # close input streams
    training_images_raw.close()
    training_labels_raw.close()

    # process the data, and prepare it for training
    training_images, training_labels = process_data(training_images, training_labels)

    #randomize
    if randomize:
        training_images, training_labels = randomize_data(training_images, training_labels)

    # return the processed data
    return training_images, training_labels

"""load all the test images and labels"""
def load_test_data(randomize=True):
    # open the raw data files
    test_images_raw = open("data/t10k-images-idx3-ubyte", "rb")
    test_labels_raw = open("data/t10k-labels-idx1-ubyte", "rb")

    # create numpy arrays with the correct (raw) data
    test_images = format_images(test_images_raw)
    test_labels = format_labels(test_labels_raw)

    # close input streams
    test_images_raw.close()
    test_labels_raw.close()

    # process the data, and prepare it for training
    test_images, test_labels = process_data(test_images, test_labels)

    #randomize
    if randomize:
        test_images, test_labels = randomize_data(test_images, test_labels)

    # return the processed data
    return test_images, test_labels

"""create a random batch of images with specified size"""
def batch(imgs, lbls, size):
    if size == -1:
        return randomize_data(imgs, lbls)
    shuff_imgs, shuff_lbls = randomize_data(imgs, lbls) # randomize the images before selecting images
    perm = np.random.permutation(size) # create a random permutation of the specified size
    return np.take(shuff_imgs, perm, axis=-1), np.take(shuff_lbls, perm, axis=-1) # use the permutation as indices to select random image-label pairs