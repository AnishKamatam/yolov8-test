import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Helper function to count fingers
# Returns number of fingers up (excluding thumb)
def count_fingers(hand_landmarks):
    tips_ids = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    fingers = []
    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for tip_id in tips_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Flip the frame for natural interaction
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        finger_count = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_count = count_fingers(hand_landmarks)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # Display finger count
        cv2.rectangle(frame, (0,0), (150, 80), (0,0,0), -1)
        cv2.putText(frame, f'Fingers: {finger_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        cv2.imshow('Finger Counter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows() 