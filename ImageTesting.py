import tensorflow as tf
import matplotlib.image as img
import numpy as np
saver=tf.train.Saver()

def read():
    my_file=img.read("C:\\Users\\Ayse\\Documents\\RBC ML\\Test_Files\\KAAN_50.jpg")
    return my_file

def reshape(image_array):
    # --Reshaping images to 55*55, setting up DataSet tensor--#
    image_array_2d = []
    for image in image_array:
        image_array_2d.append(
            image)  # can reshape here if you want, such as image_array_2d.append(image.reshape(55,55))
    dataset = np.array(image_array_2d, dtype='float32')
    dataset_with_channel = np.expand_dims(dataset, axis=3)
    dataset_with_channel[:, :, 0, :] = 1
    number_of_images, height, width, channels = dataset_with_channel.shape  # defining dimensions
    return number_of_images, height, width, channels, dataset_with_channel

test_image=read()
print(test_image.shape)
with tf.Session() as sess:
    saver.restore(sess,"./model_three_image.ckpt")
