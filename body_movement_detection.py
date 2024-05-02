import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        hand_results = hands.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        # 2. Draw body pose
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # 3. Additional Hands
        if hand_results.multi_hand_landmarks:
            # Draw each hand
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks on the image.
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                          )
                # Connect hand landmarks to body pose landmarks
                for landmark in hand_landmarks.landmark:
                    x, y, _ = image.shape
                    # Connect wrists to shoulders
                    if landmark == hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]:
                        cv2.line(image, (int(landmark.x * x), int(landmark.y * y)),
                                 (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * x),
                                  int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * y)),
                                 (121, 22, 76), 2)

                        cv2.line(image, (int(landmark.x * x), int(landmark.y * y)),
                                 (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * x),
                                  int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * y)),
                                 (121, 22, 76), 2)

        cv2.imshow('Face Mesh & Body Lines', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()