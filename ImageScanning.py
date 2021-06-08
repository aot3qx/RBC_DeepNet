import tensorflow as tf
import re
import numpy as np
import os as os
import matplotlib.pyplot as plt
import matplotlib.image as img
from datetime import datetime

def read_multiple(mypath,label_vec):
    #--Reading in images--#
    image_array=[]
    label_vec=label_vec.split(',')
    for image_path in os.listdir(mypath):
        if image_path.endswith(".jpg"):
            for individual in label_vec:
                image = open(os.path.join(mypath, image_path), mode='rb')
                image_array.append((np.fromfile(image, dtype='float32')))
                image_label=re.search(individual,image_path)
                if image_label:
                    image_array.append(image_label.groups(1))
    return image_array

def reshape(image_array):
    #--Reshaping images to 55*55, setting up DataSet tensor--#
    image_array_2d=[]
    for image in image_array:
        image_array_2d.append(image) #can reshape here if you want, such as image_array_2d.append(image.reshape(55,55))
    dataset=np.array(image_array_2d,dtype='float32')
    print(dataset.shape)
    dataset_with_channel=dataset.reshape((-1,60,60,1))
    number_of_images, height, width, channels=dataset_with_channel.shape #defining dimensions
    return number_of_images,height,width,channels,dataset_with_channel

def read_test(path):
    my_file = img.imread("C:\\Users\\Ayse\\Documents\\RBC ML\\Experimental_test\\"+path)
    my_file = my_file/255
    return my_file

def image_average(image1,image2):
    image_avg=(image1+image2)/2
    return image_avg

def read():
    #rigidty files
    """my_path=('C:\\Users\\Ayse\\Documents\\RBC ML\\Train_Files')
    kaan_path=my_path+('\\KAAN_50.jpg')
    ozge_path=my_path+('\\OZGE_30.jpg')
    ozlem_path_ten=my_path+('\\OZLEM_10.jpg')
    ozlem_path_control=my_path+('\\OZLEM_0.jpg')
    image_labels=np.array([0,1,2,3])

    image_kaan = img.imread(fname=os.path.join(my_path,kaan_path))
    image_kaan=image_kaan/255

    image_ozge = img.imread(fname=os.path.join(my_path, ozge_path))
    image_ozge=image_ozge/255

    image_ozlem_ten=img.imread(fname=os.path.join(my_path,ozlem_path_ten))
    image_ozlem_ten=image_ozlem_ten/255

    image_ozlem_control = img.imread(fname=os.path.join(my_path, ozlem_path_control))
    image_ozlem_control=image_ozlem_control/255"""
    #concentration files
    my_path=('C:\\Users\\Ayse\\Documents\\RBC ML\\Train_concentration')
    image_one_path=my_path+('\\one.jpg')
    image_two_path = my_path + ('\\two.jpg')
    image_three_path = my_path + ('\\three.jpg')
    image_four_path = my_path + ('\\four.jpg')
    image_five_path = my_path + ('\\five.jpg')
    image_six_path = my_path + ('\\six.jpg')

    image_array=[img.imread(fname=image_one_path),img.imread(fname=image_two_path),img.imread(fname=image_three_path),
                 img.imread(fname=image_four_path),img.imread(fname=image_five_path),
                 img.imread(fname=image_six_path)]
    image_array_scaled=[]
    for image in image_array:
        image_array_scaled.append(image/255)
    image_labels=np.array([0,1,2,3,4,5])

    return image_labels,image_array_scaled[0],image_array_scaled[1],image_array_scaled[2],image_array_scaled[3],image_array_scaled[4],image_array_scaled[5]

n_outputs=4
def convolutional_layer(X,filters,kernel_size):
    #--Defining random kernels for each convolutional layer--#
    conv=tf.layers.conv2d(X,filters=filters
                          ,kernel_size=kernel_size,strides=[1,1],padding="SAME")
    return tf.nn.elu(conv)
def avg_layer(X):
    avg_pool=tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    return avg_pool

# Reading in data and formatting into 4d tensor of [mini-batch size, height, width, channels] <-dataset_with_channel
batch_size=1
image_labels,image_one,image_two,image_three,image_four,image_five,image_six=read()
print(image_one)
image_array=[image_one,image_two,image_three,image_four,image_five,image_six]
number_of_images,height,width,channels,dataset_with_channel=reshape(image_array=image_array)

test_image=read_test(path="OZLEM_control_50.jpg")
#test_ozlem_image=read_test(path="five.jpg")
#test_ozlem_ten_image=read_test(path="six.jpg")
#test_img_avg=image_average(image1=test_ozlem_image,image2=test_ozlem_ten_image)

test_image_array=[test_image]
test_number_of_images,test_height,test_width,test_channels,test_dataset_with_channel=reshape(image_array=test_image_array)

# Defining placeholder nodes, convolutional/pooling nodes, etc.
X=tf.compat.v1.placeholder(dtype='float32',shape=(None,height,width,channels))
y=tf.compat.v1.placeholder(dtype='int32',shape=(None))


with tf.name_scope("convolutional_nn"):
    conv_layer_1=convolutional_layer(X,filters=32,kernel_size=5)
    pool_layer_1=avg_layer(conv_layer_1)
    conv_layer_2=convolutional_layer(pool_layer_1,filters=64,kernel_size=3)
    pool_layer_2=avg_layer(conv_layer_2)
    conv_layer_3=convolutional_layer(pool_layer_2,filters=128,kernel_size=2)
    pool_layer_3=avg_layer(conv_layer_3)
    hidden_1_input=tf.layers.flatten(pool_layer_3)

with tf.name_scope("fully_connected_layer"):
    hidden_1=tf.layers.dense(hidden_1_input,200,activation=tf.nn.elu,name="hidden1")
    hidden_2=tf.layers.dense(hidden_1,100,activation=tf.nn.elu,name="hidden2")
    logits=tf.layers.dense(hidden_2,units=n_outputs,name="logits")
    softmax=tf.nn.softmax(logits=logits)

with tf.name_scope("loss"):
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(cross_entropy,name="loss")

learning_rate=.1
with tf.name_scope("training"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_operation=optimizer.minimize(loss=loss)

with tf.name_scope("evaluation"):
    number_correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(number_correct,tf.float32))
    acc_summary=tf.summary.scalar('Accuracy',accuracy)
    softmax_summary=tf.summary.tensor_summary('Softmax',softmax)
    loss_summary=tf.summary.scalar('Loss',loss)

now=datetime.utcnow().strftime("Year_%Y_Month_%m_Day_%d_Time_%H%M%S")
root_logdir="tf_logs"
logdirectory="{}/run-{}/".format(root_logdir,now)
filewriter=tf.summary.FileWriter(logdir=logdirectory,graph=tf.get_default_graph())

print_logits=tf.print(logits)
print_labels=tf.print(y)
number_epochs=600
init = tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    """
    #train block
    init.run()
    accuracy_vec=[]
    epoch_vec=[]
    conv_layer_output=sess.run(conv_layer_1,feed_dict={X:dataset_with_channel})
    pool_layer_output=sess.run(pool_layer_1,feed_dict={conv_layer_1:conv_layer_output})
    for epoch in range(number_epochs):
        print(epoch)
        sess.run(training_operation,feed_dict={X:dataset_with_channel,y:image_labels})
        loss_train=loss.eval(feed_dict={X:dataset_with_channel,y:image_labels})
        loss_string=loss_summary.eval(feed_dict={X:dataset_with_channel,y:image_labels})
        filewriter.add_summary(loss_string,epoch)
        accuracy_vec.append(loss_train)
        epoch_vec.append(epoch)
    saver.save(sess,"./concentration_image_train_fig.ckpt")
    """
    #test block
    saver.restore(sess,"./model_four_image.ckpt")
    softmax_eval=softmax.eval(feed_dict={X:test_dataset_with_channel})
    print(softmax_eval)

"""
plt.figure(figsize=(9,3))
plt.suptitle("Training CNN")
plt.subplot(131)
plt.imshow(conv_layer_output[0,:,:,2])
plt.title('Convolutional Layer 1 Example')
plt.subplot(132)
plt.imshow(pool_layer_output[0,:,:,2])
plt.title('Pool Layer 1 Example')
plt.subplot(133)
plt.plot(accuracy_vec[0:200])
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.show()
"""
filewriter.close()