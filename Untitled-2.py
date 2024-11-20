
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import shutil

class ClothingClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clothing Classifier")
        self.root.geometry("400x300")
        
        # Directories for storing images
        self.dataset_dir = "dataset"
        self.model_dir = "clothing_model"

        self.create_directories()

        # Add buttons and labels to the GUI
        self.add_widgets()

    def create_directories(self):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Subdirectories for categories
        self.categories = ["shirts", "pants", "jackets"]
        for category in self.categories:
            category_path = os.path.join(self.dataset_dir, category)
            if not os.path.exists(category_path):
                os.makedirs(category_path)

    def add_widgets(self):
        # Add buttons for functionality
        self.upload_button = tk.Button(self.root, text="Upload Clothing Image", command=self.upload_image)
        self.upload_button.pack(pady=20)

        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            # Prompt user for category
            category = self.choose_category()
            if category:
                # Move image to the appropriate folder
                category_path = os.path.join(self.dataset_dir, category)
                shutil.copy(file_path, category_path)
                messagebox.showinfo("Info", f"Image uploaded to {category} category!")

    def choose_category(self):
        category = tk.simpledialog.askstring("Category", "Enter the category (shirt, pants, jacket):")
        if category in self.categories:
            return category
        else:
            messagebox.showerror("Error", "Invalid category. Please choose from 'shirt', 'pants', 'jacket'.")
            return None

    def train_model(self):
        # Create a simple CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(self.categories), activation='softmax')  # Number of categories
        ])
        
        # Compile the model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Data preprocessing
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )

        # Train the model
        model.fit(train_generator, epochs=10, validation_data=validation_generator)

        # Save the trained model
        model.save(os.path.join(self.model_dir, "clothing_model.h5"))
        messagebox.showinfo("Info", "Model trained and saved!")

# Create the app window
root = tk.Tk()
app = ClothingClassifierApp(root)
root.mainloop()
