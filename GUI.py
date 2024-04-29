# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# import predict  # Make sure predict.py is in the same directory or properly referenced
#
# def upload_image():
#     global img, img_path
#     img_path = filedialog.askopenfilename()
#     if img_path:
#         img = Image.open(img_path)
#         img.thumbnail((350, 350))
#         img = ImageTk.PhotoImage(img)
#         panel.config(image=img)
#         panel.image = img
#         predict_btn.config(state='normal')
#
# def predict_image():
#     predicted_class,probability = predict.predict_image_with_face_detection_gui(img_path)
#     if predicted_class != "Error: No face detected":
#         result_label.config(text=f"Predicted class: {predicted_class}, Probability: {probability:.4f}")
#     else:
#         result_label.config(text='no face detected')
#
# root = tk.Tk()
# root.title("Autism Detection from Facial Features")
#
# panel = tk.Label(root)
# panel.pack()
#
# upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
# upload_btn.pack()
#
# predict_btn = tk.Button(root, text="Predict", command=predict_image, state='disabled')
# predict_btn.pack()
#
# result_label = tk.Label(root, text="")
# result_label.pack()
#
# root.mainloop()


import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import predict  # Make sure predict.py is in the same directory or properly referenced

# Style Constants
BG_COLOR = "#2B2B2B"
BUTTON_COLOR = "#3C3F41"
BUTTON_FONT_COLOR = "#A9B7C6"
TEXT_COLOR = "#A9B7C6"
FRAME_COLOR = "#214283"
FONT = ("Segoe UI", 14, "normal")

def upload_image():
    global img, img_path
    img_path = filedialog.askopenfilename()
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((500, 500))  # Adjust size as needed
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        predict_btn.config(state='normal')

def predict_image():
    predicted_class, probability = predict.predict_image_with_face_detection_gui(img_path)
    if predicted_class != "Error: No face detected":
        result_label.config(text=f"Predicted class: {predicted_class}\nProbability: {probability:.2f}")
    else:
        messagebox.showerror("Prediction Error", predicted_class)

# GUI Layout
root = tk.Tk()
root.title("Autism Detection from Facial Features")
root.state('zoomed')  # Opens the window in full-screen mode
root.configure(bg=BG_COLOR)

# Main image display frame
image_frame = tk.Frame(root, bg=BG_COLOR)
image_frame.pack(fill='both', expand=True)

# Panel to display the image
panel = tk.Label(image_frame, bg=BG_COLOR)
panel.pack(pady=20)

# Bottom frame for buttons and results
bottom_frame = tk.Frame(root, bg=FRAME_COLOR)
bottom_frame.pack(fill='x', side='bottom')

upload_btn = tk.Button(bottom_frame, text="Upload Image", command=upload_image, bg=BUTTON_COLOR, fg=BUTTON_FONT_COLOR, font=FONT)
upload_btn.pack(side='left', padx=20, pady=10)

predict_btn = tk.Button(bottom_frame, text="Predict", command=predict_image, state='disabled', bg=BUTTON_COLOR, fg=BUTTON_FONT_COLOR, font=FONT)
predict_btn.pack(side='left', padx=20, pady=18)

result_label = tk.Label(bottom_frame, text="", bg=FRAME_COLOR, fg=TEXT_COLOR, font=FONT)
result_label.pack(side='bottom', padx=20, pady=10)

root.mainloop()
