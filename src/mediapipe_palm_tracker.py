import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

def get_palm_info(landmarks):
    # Palm landmark indices
    palm_ids = [0, 5, 9, 13, 17]

    palm_points = np.array([
        [landmarks[i].x, landmarks[i].y, landmarks[i].z]
        for i in palm_ids
    ])

    # --- Palm center ---
    center = np.mean(palm_points, axis=0)

    # --- Orientation ---
    wrist = palm_points[0]
    index_mcp = palm_points[1]
    pinky_mcp = palm_points[4]

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist

    # Normal vector (palm facing direction)
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm != 0:
        normal = normal / norm

    # Local coordinate frame
    x_axis = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) != 0 else v1
    z_axis = normal
    y_axis = np.cross(z_axis, x_axis)

    return center, normal, x_axis, y_axis, z_axis


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand
    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # --- Compute palm info ---
            center, normal, x_axis, y_axis, z_axis = get_palm_info(landmarks)

            # Convert center to pixel coords
            cx, cy = int(center[0] * w), int(center[1] * h)

            # Draw palm center
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

            # Draw normal vector (orientation)
            scale = 100  # length of arrow

            nx = int(cx + normal[0] * scale)
            ny = int(cy + normal[1] * scale)

            cv2.arrowedLine(frame, (cx, cy), (nx, ny), (255, 0, 0), 3)

            # Draw X axis (red)
            xx = int(cx + x_axis[0] * scale)
            xy = int(cy + x_axis[1] * scale)
            cv2.arrowedLine(frame, (cx, cy), (xx, xy), (0, 0, 255), 2)

            # Draw Y axis (yellow)
            yx = int(cx + y_axis[0] * scale)
            yy = int(cy + y_axis[1] * scale)
            cv2.arrowedLine(frame, (cx, cy), (yx, yy), (0, 255, 255), 2)

            # Debug text
            cv2.putText(frame, f"Normal: {normal.round(2)}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

    cv2.imshow("Palm Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()


