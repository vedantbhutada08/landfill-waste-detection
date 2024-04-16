import tkinter as tk
from tkinter import filedialog, ttk, messagebox, Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# Define classes for biodegradable and non-biodegradable waste
biodegradable_classes = ['coloured_paper', 'corrugated_cardboard', 'mixed_cardboard_-cww-', 'white_paper']
non_biodegradable_classes = ['alu_foil', 'cat_food_cans', 'clear_plastic', 'food_cans', 'glass', 'hdpe_clear',
                             'hdpe_color', 'pet_clear-blue', 'pet_coloured', 'polypropelene',
                             'polystyrene_-Food trays-', 'polystyrene_-other-', 'tetra-pak', 'thermoforms',
                             'universal_beverage_cans', 'wte_-non-recyclable_plastic-']

# Load YOLO model with the "best.pt" file
model = YOLO(r"best (1).pt")

# Information about biodegradable and non-biodegradable waste
biodegradable_info = """
Biodegradable waste consists of organic materials that can decompose naturally. Examples include:
- Paper products (newspapers, cardboard, etc.)
- Food waste (fruits, vegetables, etc.)
- Yard waste (grass clippings, leaves, etc.)

Proper disposal methods for biodegradable waste include composting and recycling.
Biodegradable waste can decompose aerobically or anaerobically, producing either carbon dioxide and water or methane gas.
"""

non_biodegradable_info = """
Non-biodegradable waste consists of materials that do not decompose or decompose very slowly in the environment. Examples include:
- Plastics (bottles, bags, etc.)
- Metals (aluminum foil, cans, etc.)
- Glass
- Styrofoam

Proper disposal methods for non-biodegradable waste include recycling, incineration, and landfill disposal.
Non-biodegradable waste can persist in the environment for hundreds or even thousands of years, contributing to pollution and harming wildlife.
"""


def detect_waste(image_path):
    try:
        results = model.predict(image_path)
        return results
    except Exception as e:
        tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")
        return None


def process_image(image_path):
    results = detect_waste(image_path)
    if results:
        image = Image.open(image_path)
        fig, ax = plt.subplots()
        ax.imshow(image)

        biodegradable_count = 0
        non_biodegradable_count = 0

        for box in results[0].boxes:
            class_id = results[0].names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)

            x_min, y_min, x_max, y_max = cords
            width = x_max - x_min
            height = y_max - y_min

            if class_id in biodegradable_classes:
                biodegradable_count += 1
                category = 'Bio-degradable'
                border_color = 'g'
                text_color = 'g'
            elif class_id in non_biodegradable_classes:
                non_biodegradable_count += 1
                category = 'Non-biodegradable'
                border_color = 'r'
                text_color = 'r'
            else:
                category = 'Unknown'
                border_color = 'none'
                text_color = 'black'

            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=border_color,
                                     facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f"{category} ({conf})", color=text_color, verticalalignment='top')

        plt.show()

        # Display information about detected waste
        display_info(biodegradable_count, non_biodegradable_count)


def display_info(biodegradable_count, non_biodegradable_count):
    info_window = Toplevel()
    info_window.title("Waste Information")

    biodegradable_label = ttk.Label(info_window, text="Biodegradable Waste Count: " + str(biodegradable_count))
    biodegradable_label.pack()

    non_biodegradable_label = ttk.Label(info_window, text="Non-biodegradable Waste Count: " + str(non_biodegradable_count))
    non_biodegradable_label.pack()

    biodegradable_info_label = ttk.Label(info_window, text=biodegradable_info, justify=tk.LEFT, wraplength=400)
    biodegradable_info_label.pack()

    non_biodegradable_info_label = ttk.Label(info_window, text=non_biodegradable_info, justify=tk.LEFT, wraplength=400)
    non_biodegradable_info_label.pack()

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)


# Create GUI
root = tk.Tk()
root.title("Waste Detector")

# Add an image to the GUI
image_path = r"C:\Users\vedan\OneDrive\Desktop\visem_objectdetection\74508e6d9dc9d69c2befb4fbfccf894b.jpg"
image = Image.open(image_path)
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=photo)
image_label.pack()

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=5)

root.mainloop()
