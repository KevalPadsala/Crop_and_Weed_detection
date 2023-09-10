import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('crop_weed_detection_model.h5')

def browse_video():
    global video_path
    video_path = filedialog.askopenfilename()
    if video_path:
        process_video()

def process_video():
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (resize, normalize, etc.)
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0

        # Make a prediction
        predictions = model.predict(np.expand_dims(frame, axis=0))
        weed_detected = predictions[0][0] >= 0.5  # You can adjust the threshold here

        # Display the result on the frame
        if weed_detected:
            result_text = "Weed Detected"
            frame = cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            result_text = "No Weed Detected"
            frame = cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' key or ESC key
            break
        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break


    cap.release()
    cv2.destroyAllWindows()

def browse_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        process_image()

def process_image():
    global image_label
    global result_label

    # Load and preprocess the selected image
    test_image = cv2.imread(image_path)
    test_image = cv2.resize(test_image, (224, 224))
    test_image = test_image / 255.0

    # Make a prediction
    predictions = model.predict(np.expand_dims(test_image, axis=0))
    weed_detected = predictions[0][0] >= 0.5  # You can adjust the threshold here

    # Display the result in a tkinter Label
    result_text = "Weed Detected" if weed_detected else "No Weed Detected"
    result_label.config(text=result_text)

    # Display the selected image
    img = Image.open(image_path)
    img.thumbnail((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Keep a reference to the image to prevent it from being garbage collected

# Create the main GUI window
root = tk.Tk()
root.title("Weed Detection from Video and Image")

# Create GUI elements
browse_video_button = tk.Button(root, text="Browse Video", command=browse_video)
browse_image_button = tk.Button(root, text="Browse Image", command=browse_image)
image_label = tk.Label(root)
result_label = tk.Label(root, font=("Helvetica", 16))

# Pack GUI elements
browse_video_button.pack()
browse_image_button.pack()
image_label.pack()
result_label.pack()

root.mainloop()
