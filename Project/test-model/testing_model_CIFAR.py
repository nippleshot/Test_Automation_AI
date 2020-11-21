import os
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def CIFAR_prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    try:
        new_array = cv2.resize(img_array, (32, 32))
    except:
        print(filepath)
    return new_array.reshape(-1, 32, 32, 3)

def plot_image(i, predictions_array, img):
  prediction = predictions_array[i]
  img = mpimg.imread(img[i])
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_index = np.argmax(prediction[0])

  plt.xlabel("{}".format(class_names[predicted_index]), color='blue')

def plot_value_array(i, predictions_array):
  prediction = predictions_array[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), prediction[0], color="#777777")
  plt.ylim([-2, 2])
  predicted_label = np.argmax(prediction[0])
  thisplot[predicted_label].set_color('red')


name = 'lenet5_without_dropout.h5'

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    data_directory = "../AITest-data/CIFAR100_JPG_test"
    #data_directory = "./data/GAN_generated_CIFAR100"

    test_datas = os.listdir(data_directory)
    test_datas.sort()

    test_data_dir_array = []
    for test_data in test_datas:
        if test_data.find(".jpg") == -1:
            continue
        # if test_data.find(".png") == -1:
        #     continue
        full_directory = os.path.join(data_directory, test_data)
        test_data_dir_array.append(full_directory)

    CIFAR_model = keras.models.load_model('../AITest-model/cifar100/'+name)
    iter = len(test_data_dir_array)
    predictions = []
    for i in range(0, iter):
        img_reshaped = CIFAR_prepare(test_data_dir_array[i])
        img_reshaped = tf.cast(img_reshaped, tf.float32)
        prediction = CIFAR_model.predict(img_reshaped,steps=1)
        #print(prediction)
        predicted_index = np.argmax(prediction[0])
        #print("prediction : " + str(predicted_index), '\n')
        predictions.append(prediction)


    num_rows = 3
    num_cols = 3
    num_images = 9
    result_fileName = './'+name+'_prediction_oriPic_100dpi.png'
    # result_fileName = './'+name+'_prediction_fakePic_100dpi.png'
    plt.figure(figsize=(num_cols, num_rows))
    for i in range(1,10):
        plt.subplot(num_rows, num_cols, i)
        plot_image(i, predictions, test_data_dir_array)
    plt.savefig(result_fileName, dpi=100)



