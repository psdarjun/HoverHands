import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# Moving average filter for smoother movement
smooth_x, smooth_y = 0, 0
alpha = 0.3  # Increased smoothing factor for more stability

# Click state
was_closed = False
dragging = False
scrolling = False

# Time tracking for click prevention
click_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            pinky_tip = landmarks[20]
            ring_tip = landmarks[16]
            middle_tip = landmarks[12]

            # Convert index finger tip to screen coordinates
            target_x = int(index_tip.x * screen_width)
            target_y = int(index_tip.y * screen_height)

            # Apply smoothing filter
            smooth_x = alpha * target_x + (1 - alpha) * smooth_x
            smooth_y = alpha * target_y + (1 - alpha) * smooth_y

            # Move cursor
            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # Detect palm closure (for left click)
            fingers_bent = (
                index_tip.y > landmarks[6].y and
                middle_tip.y > landmarks[10].y and
                ring_tip.y > landmarks[14].y and
                pinky_tip.y > landmarks[18].y and
                thumb_tip.x > landmarks[3].x
            )

            if fingers_bent and not was_closed and (time.time() - click_time > 0.5):
                pyautogui.click()
                cv2.putText(frame, "Left Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                was_closed = True
                click_time = time.time()

            elif not fingers_bent:
                was_closed = False

            # Right-click gesture: Thumb and index finger should form a small circle
            distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
            if distance < 0.05 and (time.time() - click_time > 0.5):
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                click_time = time.time()

            # Scroll Gesture: All fingers curled, thumb determines direction
            fingers_curled = (
                index_tip.y > landmarks[6].y and
                middle_tip.y > landmarks[10].y and
                ring_tip.y > landmarks[14].y and
                pinky_tip.y > landmarks[18].y
            )

            if fingers_curled and not scrolling and not was_closed and not dragging:
                scrolling = True
            elif scrolling:
                if not fingers_curled:
                    scrolling = False
                else:
                    if thumb_tip.y < landmarks[2].y:  # Thumb up
                        scroll_speed = 20  # Adjust scroll speed as needed
                    elif thumb_tip.y > landmarks[2].y:  # Thumb down
                        scroll_speed = -20  # Adjust scroll speed as needed
                    else:
                        scroll_speed = 0

                    pyautogui.scroll(scroll_speed)
                    direction = "Up" if scroll_speed > 0 else "Down"
                    cv2.putText(frame, f"Scrolling {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Drag and Drop: Hold the pinch gesture
            if distance < 0.05 and not dragging:
                pyautogui.mouseDown()
                dragging = True
            elif distance > 0.08 and dragging:
                pyautogui.mouseUp()
                dragging = False

    # Show webcam feed
    cv2.imshow("Hand Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
