import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the saved model
path = r"C:\Users\vedan\OneDrive\Desktop\visem_objectdetection\resnet101V2.h5"
loaded_model = tf.keras.models.load_model(path)

# Define class labels
class_labels = ["BioDegradable", "NonBioDegradable"]

# Function to preprocess input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match ResNet input size
    img = img.convert('RGB')  # Convert to RGB
    img = np.array(img)  # Convert to numpy array
    img = tf.keras.applications.resnet_v2.preprocess_input(img)  # Preprocess input
    return img

def classify_image(image_path):
    try:
        # Preprocess the input image
        input_image = preprocess_image(image_path)

        # Make prediction on the input image
        predictions = loaded_model.predict(np.expand_dims(input_image, axis=0))

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]

        # Display the input image with predicted class label
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo
        classify_label.configure(text="Prediction: " + predicted_class)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        classify_image(file_path)

# Create GUI
root = tk.Tk()
root.title("Image Classifier")

# Add an image to the GUI
image_path = r"C:\Users\vedan\OneDrive\Desktop\visem_objectdetection\74508e6d9dc9d69c2befb4fbfccf894b.jpg"
image = Image.open(image_path)
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=photo)
image_label.pack()

# Create a label for displaying prediction
classify_label = tk.Label(root, text="")
classify_label.pack()

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=5)

root.mainloop()
