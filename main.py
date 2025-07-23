import tensorflow as tf
import numpy as np
import cv2
import string

# Load the trained model
model = tf.keras.models.load_model("model/asl_cnn_best.h5")

# Labels: A-Z + special signs
labels = list(string.ascii_uppercase) + ['del', 'nothing', 'space']

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Webcam not detected. Please check your camera.")
    exit()

print("📸 Webcam is live! Show your hand sign in the blue box 🟦")
print("Press 'q' to quit anytime.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed!")
        break

    # ✅ DO NOT flip the frame if you want it non-mirrored
    # If your webcam is mirrored by default, use this line instead:
    # frame = cv2.flip(frame, 1)

    # Define the Region of Interest (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Preprocess the ROI for prediction
    image = cv2.resize(roi, (64, 64)).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the class
    prediction = model.predict(image, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index]

    # Display prediction on the frame
    display_text = f"{predicted_label} ({confidence * 100:.1f}%)"
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the live frame
    cv2.imshow("Sign Language Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting. Hope your signs were seen!")
        break

cap.release()
cv2.destroyAllWindows()

