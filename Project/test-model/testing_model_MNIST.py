import os
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def MNIST_prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (28, 28))
    return new_array.reshape(-1, 28, 28, 1)

def plot_image(i, predictions_array, img):
  prediction = predictions_array[i]
  img = mpimg.imread(img[i])
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_index = np.argmax(prediction[0])

  class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  plt.xlabel("prediction : {}".format(class_names[predicted_index]), color='blue')

def plot_value_array(i, predictions_array):
  prediction = predictions_array[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), prediction[0], color="#777777")
  plt.ylim([-2, 2])
  predicted_label = np.argmax(prediction[0])
  thisplot[predicted_label].set_color('red')



if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    data_directory = "./data/GAN_generated_MNIST"
    test_datas = os.listdir(data_directory)
    test_datas.sort()

    test_data_dir_array = []
    for test_data in test_datas:
        full_directory = os.path.join(data_directory, test_data)
        test_data_dir_array.append(full_directory)

    MNIST_model = keras.models.load_model('../AITest-model/MNIST/random1_mnist.h5')
    iter = len(test_data_dir_array)
    predictions = []
    for i in range(0, iter):
        img_reshaped = MNIST_prepare(test_data_dir_array[i])
        img_reshaped = tf.cast(img_reshaped, tf.float32)
        prediction = MNIST_model.predict(img_reshaped,steps=1)
        print(prediction)
        predicted_index = np.argmax(prediction[0])
        print("prediction : " + str(predicted_index), '\n')
        predictions.append(prediction)

    num_rows = 6
    num_cols = 3
    num_images = len(predictions)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for i in range(0, 10):
        result_fileName = './random1_mnist_prediction_'+str(i)+'_200dpi.png'
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        start = 0
        #print(str(i)+"_start = "+str(start))
        for prediction_index in range(0, len(predictions)):
            prediction = predictions.__getitem__(prediction_index)
            predicted_result = np.argmax(prediction[0])
            if predicted_result == i:
                if start == 30:
                    break;
                plt.subplot(num_rows, 2 * num_cols, start + 1)
                #print(str(i) + "_image_subplot = "+str(start + 1))
                plot_image(prediction_index, predictions, test_data_dir_array)
                plt.subplot(num_rows, 2 * num_cols, start + 2)
                #print(str(i) + "_value_subplot = " + str(start + 2))
                plot_value_array(prediction_index, predictions)
                _ = plt.xticks(range(10), class_names, rotation=45)
                start = start + 2
        plt.savefig(result_fileName, dpi=200)








