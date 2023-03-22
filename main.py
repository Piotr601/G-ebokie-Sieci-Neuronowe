import pickle
import os 
import time
import numpy as np
import matplotlib.pyplot as plt

file_list = []
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

# Unpacking one files (to be updated to all files)
# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
file_content = unpack_images(file_list[0])
print(file_content[b'data'][0].shape)

# Loading data and converting into desirable shape
data = file_content[b'data']
data = data.reshape(len(data), 3, 32, 32).transpose(0,2,3,1)

# Plotting first image
plt.imshow(data[0])
plt.show()

# Plotting 25 images, starting from the first
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(data[i])
plt.show()