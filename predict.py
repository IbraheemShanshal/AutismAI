import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

def detect_and_crop_face(image_path, margin=50):  # Added a margin parameter with a default value
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    # Crop to the first face found and add margin
    for (x, y, w, h) in faces:
        # Adjust x, y, w, h to include margin
        x_margin = max(x - margin, 0)
        y_margin = max(y - margin, 0)
        w_margin = min(w + 2 * margin, img.shape[1] - x_margin)
        h_margin = min(h + 2 * margin, img.shape[0] - y_margin)
        face_img_with_margin = img[y_margin:y_margin+h_margin, x_margin:x_margin+w_margin]
        face_img_rgb = cv2.cvtColor(face_img_with_margin, cv2.COLOR_BGR2RGB)
        return face_img_rgb

def preprocess_image_with_face_detection(image_path, target_size=(365, 365)):
    try:
        face_img_rgb = detect_and_crop_face(image_path)

        # Calculate aspect ratio
        h, w = face_img_rgb.shape[:2]
        aspect_ratio = w / h

        # Compute new dimensions that maintain aspect ratio
        if w > h:  # width is greater than height
            new_w = target_size[1]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = target_size[0]
            new_w = int(new_h * aspect_ratio)

        # Resize face image while maintaining aspect ratio
        face_img_resized = cv2.resize(face_img_rgb, (new_w, new_h))

        # If needed, pad the smaller dimension with black pixels
        if new_w != target_size[1] or new_h != target_size[0]:
            delta_w = target_size[1] - new_w
            delta_h = target_size[0] - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [0, 0, 0]  # Black padding
            face_img_resized = cv2.copyMakeBorder(face_img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # Convert the face image to grayscale
        face_img_gray = cv2.cvtColor(face_img_resized, cv2.COLOR_RGB2GRAY)

        # Convert grayscale image to array, expand dimensions and normalize
        img_array = image.img_to_array(face_img_gray)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        return img_array
    except ValueError as e:
        print(e)
        return None


# Predict a single image with face detection (No changes needed here, but included for completeness)
def predict_image_with_face_detection(image_path):
    processed_image = preprocess_image_with_face_detection(image_path)
    if processed_image is not None:
        plt.imshow(np.squeeze(processed_image), cmap = 'gray')  # Remove the batch dimension
        plt.axis('off')  # Hide the axis
        plt.show()
        model_path = 'models/autismAiModel.h5'  # Update this to the path where your model is saved
        model = load_model(model_path)
        prediction = model.predict(processed_image)
        predicted_class = 'Autistic' if prediction[0][0] < 0.2 else 'Non_Autistic'
        return predicted_class, prediction[0][0]
    else:
        return "Error: No face detected", None

# if __name__ == '__main__':
#     image_path = 'test pictures/thatKid.jpg'  # Update this to the path of the image you want to predict
#     predicted_class, probability = predict_image_with_face_detection(image_path)
#     if probability is not None:  # Ensure prediction was made
#         print(f"Predicted class: {predicted_class}, Probability: {probability}")

def predict_image_with_face_detection_gui(image_path):
    predicted_class, probability = predict_image_with_face_detection(image_path)
    if probability is not None:
         return predicted_class, probability
    else:
        return "Error: No face detected", None