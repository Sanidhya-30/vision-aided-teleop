import cv2
import mediapipe as mp
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from collections import deque

# ==========================================
# 1. HELPER FUNCTIONS & CV LOGIC
# ==========================================
def compute_palm_area(landmarks, w, h):
    palm_ids = [0, 5, 9, 13, 17]
    pts = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)]
        for i in palm_ids
    ], dtype=np.int32)
    hull = cv2.convexHull(pts)
    return cv2.contourArea(hull)

def compute_thumb_index_distance(landmarks, w, h):
    thumb = landmarks[4]
    index = landmarks[8]
    x1, y1 = int(thumb.x * w), int(thumb.y * h)
    x2, y2 = int(index.x * w), int(index.y * h)
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist, (x1, y1), (x2, y2)

def map_range(value, in_min, in_max, out_min, out_max):
    """Maps a value from one range to another."""
    # Clamp value to input range
    value = max(min(value, in_max), in_min)
    return out_min + (((value - in_min) / (in_max - in_min)) * (out_max - out_min))

# ==========================================
# 2. INITIALIZATION (CV & PYBULLET)
# ==========================================
# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)

# PyBullet Setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load Environment
p.loadURDF("plane.urdf")
table_height = 0.625
p.loadURDF("table/table.urdf")

# Load Panda Robot
pandaId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, table_height], useFixedBase=True)
end_effector_index = 11
arm_num_dofs = 7
finger_joints = [9, 10] # Panda gripper joints

# Move robot to ready position
for i, joint_val in enumerate([0, -0.785, 0, -2.356, 0, 1.571, 0.785]):
    p.resetJointState(pandaId, i, joint_val)

# Fix orientation pointing down
target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.6])

# Tracking Variables
area_history = deque(maxlen=10)
MIN_AREA = 2000
MAX_AREA = 20000
PINCH_THRESHOLD = 40

print("Starting... Show your hand to the camera.")
print("Press 'ESC' in the OpenCV window to exit.")

# ==========================================
# 3. MAIN REAL-TIME LOOP
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process Hand Landmarks
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Get Pinch Data (Gripper Control)
            dist, thumb_pt, index_pt = compute_thumb_index_distance(landmarks, w, h)
            
            # Get Palm Center (X, Y Control)
            cx_norm = landmarks[9].x # Using Middle Finger MCP as center reference
            cy_norm = landmarks[9].y
            cx, cy = int(cx_norm * w), int(cy_norm * h)

            # Get Palm Area (Z Control)
            area = compute_palm_area(landmarks, w, h)
            area_history.append(area)
            smooth_area = np.mean(area_history)

            # --- MAPPING MATH (Webcam -> Robot Workspace) ---
            # X-axis camera -> Y-axis robot (Left/Right)
            robot_y = map_range(cx_norm, 0.2, 0.8, 0.4, -0.4) 
            
            # Y-axis camera -> X-axis robot (Forward/Backward)
            robot_x = map_range(cy_norm, 0.2, 0.8, 0.7, 0.3)
            
            # Area (Z) -> Z-axis robot (Up/Down)
            robot_z = map_range(smooth_area, MIN_AREA, MAX_AREA, table_height + 0.1, table_height + 0.6)

            target_pos = [robot_x, robot_y, robot_z]

            # --- INVERSE KINEMATICS ---
            joint_poses = p.calculateInverseKinematics(
                bodyUniqueId=pandaId, 
                endEffectorLinkIndex=end_effector_index, 
                targetPosition=target_pos, 
                targetOrientation=target_orn,
                maxNumIterations=20,
                residualThreshold=1e-4
            )

            # Move Arm
            for i in range(arm_num_dofs):
                p.setJointMotorControl2(
                    pandaId, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], 
                    force=500, positionGain=0.1, velocityGain=1.0
                )

            # Move Gripper based on Pinch
            gripper_target = 0.0 if dist < PINCH_THRESHOLD else 0.04
            for finger_idx in finger_joints:
                p.setJointMotorControl2(
                    pandaId, finger_idx, p.POSITION_CONTROL, targetPosition=gripper_target, force=100
                )

            # --- VISUALIZATION (OpenCV) ---
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.line(frame, thumb_pt, index_pt, (255, 255, 0), 2)
            cv2.putText(frame, f"Robot XYZ: {robot_x:.2f}, {robot_y:.2f}, {robot_z:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if dist < PINCH_THRESHOLD:
                cv2.putText(frame, "GRIPPER CLOSED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "GRIPPER OPEN", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

    # Step simulation multiple times per camera frame to keep physics smooth
    for _ in range(8):
        p.stepSimulation()

    cv2.imshow("Hand Control Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()
p.disconnect()
