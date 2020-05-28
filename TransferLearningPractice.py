import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers

'''
    Transfer learning is the process whereby one uses neural network models trained in a 
    related domain to accelerate the development of accurate models in your more specific 
    domain of interest.

    In this example,
    We will be using an advanced pre trained image classification model - ResNet50 - 
    to improve a more specific image classification task - the cats vs dogs classification

    Benefits:
    1) Speeds up learning
    2) Works with less data
    3) You can rake advantage of expert state of art of models (get rid of tuning of models and rely upon efforts of expert)

    What is ResNet50?
    Residual Neural Network.
    50 indicates the number of layers in network
    As we know in very deep NN there is often a problem of vanishing gradients due to number of
    calculations performed during back propagation.
    The main innovation of ResNet is the skip connection.
    In the NN it skips the layers which it finds are not very useful or are less relevant in training.

    Why are we using Resnet50 Model ?
    Resnet50 is a pre-trained Deep learning model.
    A pre-trained model is trained on a different task than the task at hand and provides 
    a good starting point since the features learned on the old task are useful for the new task.

    https://i1.wp.com/adventuresinmachinelearning.com/wp-content/uploads/2019/06/ResNet50-transfer-learning.png?resize=617%2C140&ssl=1

    The full ResNet50 model shown in the image above, in addition to a Global Average Pooling 
    (GAP) layer, contains a 1000 node dense / fully connected layer which acts as a “classifier”
    of the 2048 (4 x 4) feature maps output from the ResNet CNN layers. In this transfer learning 
    task, we’ll be removing these last two layers (GAP and Dense layer) and replacing these with 
    our own GAP and dense layer (in this example, we have a binary classification task – 
    hence the output size is only 1).

'''
def visualize_data(raw_train, metadata):
    get_label_name = metadata.features['label'].int2str
    for image, label in raw_train.take(2):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))
        plt.show()

def pre_process_image(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image, label

def standard_cnn_model():
    head = tf.keras.Sequential()
    head.add(layers.Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    head.add(layers.Conv2D(32, (3, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    head.add(layers.Conv2D(64, (3, 3)))
    head.add(layers.BatchNormalization())
    head.add(layers.Activation('relu'))
    head.add(layers.MaxPooling2D(pool_size=(2, 2)))

    average_pool = tf.keras.Sequential()
    average_pool.add(layers.AveragePooling2D())
    average_pool.add(layers.Flatten())
    average_pool.add(layers.Dense(1, activation='sigmoid'))

    standard_model = tf.keras.Sequential([
        head, 
        average_pool
    ])

    return standard_model

def train_standard_cnn_model(standard_model, train, validation, TRAIN_BATCH_SIZE):
    standard_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    # callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./log/standard_model', update_freq='batch')]
    standard_model.fit(train, steps_per_epoch = 23262//TRAIN_BATCH_SIZE, epochs=5, 
                validation_data=validation, validation_steps=10)

def tf_cnn_model(IMG_SHAPE):
    res_net = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    res_net.trainable = False

    global_average_layer = layers.GlobalAveragePooling2D()
    output_layer = layers.Dense(1, activation='sigmoid')

    tl_model = tf.keras.Sequential([
        res_net,
        global_average_layer,
        output_layer
    ])
    return tl_model

def train_tl_cnn_model(tl_model, train, validation, TRAIN_BATCH_SIZE):
    tl_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    # callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./log/transer_learning_model', update_freq='batch')]
    tl_model.fit(train, steps_per_epoch = 23262//TRAIN_BATCH_SIZE, epochs=5, 
                validation_data=validation, validation_steps=10)

if __name__ == "__main__":
    
    # Train - 80% | Validation - 10% | Test - 10%

    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )   
    print(raw_train)
    print(raw_validation)
    print(raw_test)
    visualize_data(raw_train, metadata)
    
    IMAGE_SIZE = 100

    TRAIN_BATCH_SIZE = 64

    train = raw_train.map(pre_process_image).shuffle(1000).repeat().batch(TRAIN_BATCH_SIZE)
    validation = raw_validation.map(pre_process_image).repeat().batch(1000)


    standard_model = standard_cnn_model()
    train_standard_cnn_model(standard_model, train, validation, TRAIN_BATCH_SIZE)

    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    
    tl_model = tf_cnn_model(IMG_SHAPE)
    train_tl_cnn_model(tl_model, train, validation, TRAIN_BATCH_SIZE)