import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import glob
from PIL import Image

from sklearn.model_selection import train_test_split

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
	
	# Randomly sample rows from the dataframe
	sampled_rows = df[:2]
	# sampled_rows = df.sample(n=num_samples_to_show)
	
	# Create a figure with num_samples_to_show rows and 1 column
	fig, axs = plt.subplots(num_samples_to_show, 1, figsize=(14, 3 * num_samples_to_show))
	
	for i, (_, row) in enumerate(sampled_rows.iterrows()):
		# Load the image
		img_path = os.path.join(dataset_folder, 'equation_images', row['image_filename'])
		img = Image.open(img_path)
		
		# Display the image
		axs[i].imshow(img, cmap='gray')
		
		# # Set the title (simple description + '\n' + additional description)
		# axs[i].set_title(row['simple_description'] + '\n' + row['additional_description'], wrap=True)
	
	plt.tight_layout()
	plt.show()
	
	selected_row = df[:1].iloc[0]
	
	# Load the equation image
	eq_img_path = os.path.join(dataset_folder, 'equation_images', selected_row['image_filename'])
	eq_img = Image.open(eq_img_path)
	eq_array = np.array(eq_img)
	
	# Load the labels image
	label_img_path = os.path.join(dataset_folder, 'label_images', selected_row['image_filename'])
	label_img = Image.open(label_img_path)
	
	# Convert labels image to numpy array
	label_array = np.array(label_img)
	char_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=', ' ']
	char_height = len(char_order)
	char_image = label_array[:char_height]
	
	return eq_array, char_image, df[:1]