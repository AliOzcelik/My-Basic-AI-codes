import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

training_set = train_datagen.flow_from_directory(
    'heart disease\train',
    class_mode = 'binary',
    batch_size = 32,
    target_size = (64,64)
)

test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(
    'heart disease\test',
    class_mode = 'binary',
    batch_size = 32,
    target_size = (64,64)
)


cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

cnn.add(tf.keras.layers.Flatten())

tf.keras.layers.Dropout(rate = 0.2)

cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))   

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

def img(directory):
    test_image = image.load_img(directory, target_size = (64,64))
    test_image = image.img_to_array(np.array(test_image))
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)

    training_set.class_indices

    if result[0][0] == 1:
        prediction = "true"
    else:
        prediction = "false"
        
    return prediction

print(img('heart disease\test\true\1.png'))
print(img('heart disease\test\false\1.png'))




