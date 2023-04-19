import pickle
import os 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

test = []
file_list = []
all_data = []
all_label = []
absolute_path_to_files = os.listdir('cifar-10/')

# Function to unpack images
def unpack_images(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

# Adding directories to file
for filee in absolute_path_to_files:
    if filee.startswith('data'):
        file_list.append(f"cifar-10/{filee}")
    if filee.startswith('test'):
        test.append(f"cifar-10/{filee}")

# Unpacking one file (to be updated to all files)
# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

# All batch data into one list
for no_file in range(5):
    file_content = unpack_images(file_list[no_file])
    
    # Loading data and converting into desirable shape
    data = file_content[b'data']
    data_label = file_content[b'labels']
    data = data.reshape(len(data), 3, 32, 32).transpose(0,2,3,1)
    all_data.append(data)
    all_label.append(data_label)

all_data = np.concatenate(all_data, axis=0)
all_label = np.concatenate(all_label, axis=0)

# Test data
test_content = unpack_images(test[0])
test_data = test_content[b'data']
test_label = file_content[b'labels']
test_data = test_data.reshape(len(test_data), 3, 32, 32).transpose(0,2,3,1)

print(f"All data shape: {np.shape(all_data)}")
print(f"Test data shape: {np.shape(test_data)}")


# ----------------------
#        PLOTTING
# ----------------------
# plt.imshow(data[9999])
# plt.show()

# Plotting 25 images, starting from the first
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.imshow(test_data[i])
# plt.show()

# ----------------------
#          CNN
# ----------------------
x_train, y_train = all_data, all_label
x_test, y_test = test_data, test_label

# Normalize the pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert the labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the batch size and number of epochs
batch_size = 16
epochs = 2

# Define the model architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)

# Print the test accuracy
print('Test accuracy:', test_acc)