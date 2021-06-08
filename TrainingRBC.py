import tensorflow as tf
import re
import numpy as np
import os as os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from datetime import datetime
import sys
def train_test_split(image_array,label_array,number_images_per_class,k_fold,seed,first_run,number_of_classes):
    number_images_per_class=int(number_images_per_class)
    k_fold=float(k_fold)
    first_run=int(first_run)
    seed=int(seed)
    number_of_classes=int(number_of_classes)
    holdout_number_of_images = int(np.ceil(k_fold * number_images_per_class))
    if first_run==1:
        seed_int=np.random.random_integers(0,10000000)
        seed_construct=np.random.seed(seed_int)
        test_images=[]
        test_labels=[]
        print("For test runs, your seed is: "+ str(seed_int))
        for i in range(0, number_of_classes):
            indices = np.where(label_array == i)
            random_array=np.random.choice(a=len(indices[0]),size=int(holdout_number_of_images),replace=False)
            random_holdout_indices=[]
            for num in random_array:
                random_holdout_indices.append(indices[0][num])
            for index in random_holdout_indices:
                test_images.append(np.array(image_array[index],dtype='float32'))
                test_labels.append(np.array(label_array[index]))
            image_array=np.delete(image_array,random_holdout_indices,axis=0)
            label_array=np.delete(label_array,random_holdout_indices,axis=0)
        return image_array,label_array,np.array(test_images),np.array(test_labels)

    else:
        seed_construct = np.random.seed(seed)
        test_images = []
        test_labels = []
        print("Using seed integer of: "+ str(seed))
        for i in range(0, number_of_classes):
            indices = np.where(label_array == i)
            random_array=np.random.choice(a=len(indices[0]),size=int(holdout_number_of_images),replace=False)
            random_holdout_indices=[]
            for num in random_array:
                random_holdout_indices.append(indices[0][num])
            for index in random_holdout_indices:
                test_images.append(np.array(image_array[index],dtype='float32'))
                test_labels.append(np.array(label_array[index]))
            image_array=np.delete(image_array,random_holdout_indices,axis=0)
            label_array=np.delete(label_array,random_holdout_indices,axis=0)
        return image_array,label_array,np.array(test_images),np.array(test_labels)


def read_multiple(mypath,label_vec):
    #--Reading in images--#
    image_array=[]
    label_name_array=[]
    label_vec=label_vec.split(',')
    for image_path in os.listdir(mypath):
        if image_path.endswith(".jpg"):
            image_array.append(np.array(img.imread(fname=mypath+"\\"+image_path)))
            for individual in label_vec:
                image_label=re.search(individual,image_path)
                if image_label:
                    label_name_array.append(image_label.group())
    image_array_scaled=[]
    for image in image_array:
        image_scaled=image/255
        image_array_scaled.append(image_scaled)

    label_array_duplicates_removed=dict.fromkeys(label_name_array)
    i=0
    for key in label_array_duplicates_removed.keys():
        label_array_duplicates_removed[key]=i
        i=i+1
    i=0
    for i in range(0,len(label_name_array)):
        label_name_array[i]=label_array_duplicates_removed[label_name_array[i]]
    image_array_scaled=np.array(image_array_scaled)
    label_array=np.array(label_name_array)
    print("Image array dimensions: "+ str(np.shape(image_array_scaled)))
    print("Label array dimensions: "+ str(np.shape(label_array)))
    print("These dimensions should match. If not, the parser did not find equivalent amount of images + labels")
    return image_array_scaled,label_array

def reshape(image_array,shape):
    #--Reshaping images to 55*55, setting up DataSet tensor--#
    dataset_with_channel=image_array.reshape((-1,shape,shape,1))
    number_of_images, height, width, channels=dataset_with_channel.shape #defining dimensions
    return number_of_images,height,width,channels,dataset_with_channel

def convolutional_layer(X,filters,kernel_size):
    #--Defining random kernels for each convolutional layer--#
    conv=tf.layers.conv2d(X,filters=filters
                          ,kernel_size=kernel_size,strides=[1,1],padding="SAME")
    return tf.nn.elu(conv)
def avg_layer(X):
    avg_pool=tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    return avg_pool

test_override=input("Are you working solely with test images? 1 for yes, 0 for no: ")
if test_override=="1":
    directory = input("Please enter a path: ")
    label_vec = input("Enter subject names as comma-delimited list: ")
    number_of_classes = input("Enter number of classes: ")
    first_run = input("Is this the first run (i.e. training run)?"
                      "Enter 1 for yes, 0 for no. A seed will be generated that should be used for subsequent"
                      "test runs. Remember this seed: ")
else:
    directory = input("Please enter a path: ")
    label_vec = input("Enter subject names as comma-delimited list: ")
    number_of_classes = input("Enter number of classes: ")
    number_of_images_per_class=input("Enter number of images per class: ")
    k_fold = input("Enter ratio of holdout set to train set (if odd, will take ceil): ")
    first_run = input("Is this the first run (i.e. training run)?"
                      "Enter 1 for yes, 0 for no/test run. A seed will be generated that should be used for subsequent"
                      "test runs. Remember this seed: ")

if test_override=="1":
    trained_model_name = input("Enter trained model name (.ckpt file): ")
elif (first_run!="1") and (test_override!="1"):
    seed=input("Enter seed: ")
    trained_model_name=input("Enter trained model name (.ckpt file): ")
else:
    seed=0
    model_name=input("Enter model name to be saved to folder (will be saved as .ckpt file): ")

if test_override=="1":
    image_array,label_array=read_multiple(mypath=directory,label_vec=label_vec)
    print("X_test shape is: " + str(image_array.shape))
    print("Y_test shape is: "+ str(label_array.shape))
else:
    image_array, label_array = read_multiple(mypath=directory, label_vec=label_vec)
    X_train, y_train, X_test, y_test = train_test_split(image_array=image_array, label_array=label_array,
                                                        number_images_per_class=number_of_images_per_class,
                                                        k_fold=k_fold, seed=seed, first_run=first_run,
                                                        number_of_classes=number_of_classes)
    print("X_train shape is: " + str(X_train.shape))
    print("Y_train shape is: " + str(y_train.shape))
    print("X_test shape is: " + str(X_test.shape))
    print("Y_test shape is: " + str(y_test.shape))

    if (X_train.shape[0] != y_train.shape[0]) or (X_test.shape[0] != y_test.shape[0]):
        print(
            "First dimension of X train does not match with y train, or first dimension of X test does not match y test."
            "Exiting...make sure that file names in folder are unique for each class to avoid duplicates.")
        sys.exit("Error. See above message.")

if first_run=="1" and test_override!="1":
    number_of_images,height,width,channels,dataset_with_channel=reshape(X_train,shape=60)
    print("Dataset shape is: " + str(dataset_with_channel.shape))
elif first_run!="1" and test_override!="1":
    number_of_images, height, width, channels, dataset_with_channel = reshape(X_test, shape=60)
    print("Dataset shape is: " + str(dataset_with_channel.shape))
elif test_override=="1":
    number_of_images, height, width, channels, dataset_with_channel = reshape(image_array, shape=60)
    print("Dataset shape is: " + str(dataset_with_channel.shape))
else:
    print("Unclear input")
    sys.exit("Clean up input")
n_outputs=number_of_classes

print("Dataset shape:"+str(dataset_with_channel.shape))

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
if first_run=="1":
    number_epochs=int(input("Enter epoch number: "))
init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    if first_run=="1":
        #train block
        init.run()
        accuracy_vec=[]
        epoch_vec=[]
        conv_layer_output=sess.run(conv_layer_1,feed_dict={X:dataset_with_channel})
        pool_layer_output=sess.run(pool_layer_1,feed_dict={conv_layer_1:conv_layer_output})
        for epoch in range(number_epochs):
            print(epoch)
            sess.run(training_operation,feed_dict={X:dataset_with_channel,y:y_train})
            loss_train=loss.eval(feed_dict={X:dataset_with_channel,y:y_train})
            loss_string=loss_summary.eval(feed_dict={X:dataset_with_channel,y:y_train})
            filewriter.add_summary(loss_string,epoch)
            accuracy_vec.append(loss_train)
            epoch_vec.append(epoch)
            print(loss_train)
        saver.save(sess,"./"+model_name+".ckpt")
    else:
        saver.restore(sess,"./"+trained_model_name+".ckpt")
        softmax_eval = softmax.eval(feed_dict={X: dataset_with_channel})
        print(softmax_eval)

if first_run=="1":
    print(accuracy_vec)