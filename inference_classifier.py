import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./cnnmodel.p', 'rb'))
model = model_dict['Classifier']

cap = cv2.VideoCapture(0)  # Change this to the correct camera index if necessary

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)  # Adjust to max_num_hands=2

# Label dictionary for all alphabets (replace with your full labels)
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z labels

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:  # Check if any hand landmarks are detected
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Process each hand separately
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize the coordinates by subtracting the minimum values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure input is consistent (84 values)
        if len(data_aux) == 84:
            # Create a blank image of size 128x128
            img = np.zeros((128, 128), dtype=np.float32)

            # Map normalized coordinates to image
            for i in range(0, len(data_aux), 2):  # Process in pairs (x, y)
                x = int(data_aux[i] * 127)  # Scale to 0-127
                y = int(data_aux[i + 1] * 127)  # Scale to 0-127
                if 0 <= x < 128 and 0 <= y < 128:
                    img[y, x] = 1.0  # Assign data to the image

            # Resize image to 128x128 if necessary
            img_resized = cv2.resize(img, (128, 128))

            # Expand dimensions to match model input shape (1, 128, 128, 1)
            data_array = np.expand_dims(img_resized, axis=(0, -1))
            prediction = model.predict(data_array)
            predicted_character = labels_dict.get(int(np.argmax(prediction[0])), "Unknown")

            # Drawing bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
