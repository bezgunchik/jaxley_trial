import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import glob
from PIL import Image, ImageOps
from functools import partial
from sklearn.model_selection import train_test_split

char_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=', ' ']
image_width = 1840
image_height = 128

def load_image(file_path, slice_chars=False, specific_char_index=None):
	image = Image.open(file_path)
	
	image = ImageOps.pad(image, (image_width, image_height), centering=(0, 0))
	# 128 x 1840
	# min - 579, max - 1840
	# mean - 1259, median - 1197
	result = np.array(image)
	if slice_chars:
		char_height = len(char_order)
		if specific_char_index:
			result = result[specific_char_index]
		else:
			result = result[:char_height]
	return result

def pad_arrays(arrays):
	max_length = max(array.shape[1] for array in arrays)
	return np.stack([np.pad(array, (0, max_length - array.shape[0]), mode='constant') for array in arrays])

def load_dataset(train_size, test_size):
	# Set the path to your dataset
	dataset_foldername = 'math_equations_images_dataset_tiny'
	curr_file_path = os.path.abspath(os.path.dirname(__file__))
	# curr_file_path = r"C:\Users\david\Desktop\Code\ImSME-dataset"
	
	downloaded_from_kaggle = False
	if downloaded_from_kaggle:
		dataset_folder = os.path.join(curr_file_path, 'data', dataset_foldername, dataset_foldername)
	else:
		dataset_folder = os.path.join(curr_file_path, 'data', dataset_foldername)
	
	csv_file = glob.glob(os.path.join(dataset_folder, 'simple_math_equation_images__*.csv'))[0]
	
	# Load the CSV file
	df = pd.read_csv(os.path.join(dataset_folder, csv_file))
	
	# Number of samples to display
	num_samples_to_show = 2
	
	# num_sample_to_load = 16384
	num_sample_to_load = 110
	# Randomly sample rows from the dataframe
	sampled_rows = df.sample(n=num_sample_to_load, random_state=99)
	# sampled_rows = df.sample(n=num_samples_to_show)
	
	# # Create a figure with num_samples_to_show rows and 1 column
	# fig, axs = plt.subplots(num_samples_to_show, 1, figsize=(14, 3 * num_samples_to_show))
	
	# for i, (_, row) in enumerate(sampled_rows.iterrows()):
	# 	# Load the image
	# 	img_path = os.path.join(dataset_folder, 'equation_images', row['image_filename'])
	# 	img = Image.open(img_path)
	#
	# 	# Display the image
	# 	axs[i].imshow(img, cmap='gray')
	
	# plt.tight_layout()
	# plt.show()
	#
	selected_row = df[:1].iloc[0]
	
	# Load the equation image
	eq_img_path = os.path.join(dataset_folder, 'equation_images', selected_row['image_filename'])
	# eq_img = Image.open(eq_img_path)
	# eq_array = np.array(eq_img)
	# image_file_paths = sampled_rows.apply(lambda row: os.path.join(dataset_folder, 'equation_images', sampled_rows['image_filename']), axis=1)
	image_file_paths = sampled_rows['image_filename'].map(lambda x: os.path.join(dataset_folder, 'equation_images', x))
	
	eq_images_arrays = np.array(list(map(load_image, image_file_paths.tolist())))
	# eq_array = np.array(eq_img)
	
	# Load the labels image
	# label_img_path = os.path.join(dataset_folder, 'label_images', selected_row['image_filename'])
	label_file_paths = sampled_rows['image_filename'].map(lambda x: os.path.join(dataset_folder, 'label_images', x))
	load_image_with_slice = partial(load_image, slice_chars=True, specific_char_index=2)
	char_images_arrays = np.array(list(map(load_image_with_slice, label_file_paths.tolist())))
	
	
	return eq_images_arrays, char_images_arrays, sampled_rows