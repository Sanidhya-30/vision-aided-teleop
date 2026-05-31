"""
hand_robot_control.py
─────────────────────
Real-time Franka Panda control via MediaPipe hand tracking.

Hand → Robot mapping
  hand norm-X  (0–1, left→right)  →  robot Y  (-0.5 → +0.5)   (mirrored)
  hand norm-Y  (0–1, top→bottom)  →  robot Z  (table+0.6 → table+0.1)  (inverted)
  palm area    (depth proxy)      →  robot X  (0.3 → 0.7)
  pinch        (thumb-index < 40) →  gripper close / open

Dependencies
  pip install pybullet opencv-python mediapipe numpy
"""

import threading
import time
import math
import cv2
import mediapipe as mp
import numpy as np
import pybullet as p
import pybullet_data
from collections import deque

# ─────────────────────────────────────────────
# Shared state (written by vision thread,
#               read by simulation thread)
# ─────────────────────────────────────────────
class HandState:
    def __init__(self):
        self.lock = threading.Lock()
        # Normalized hand coordinates (all 0–1)
        self.norm_x   = 0.5   # left-right
        self.norm_y   = 0.5   # up-down
        self.norm_z   = 0.5   # depth (from palm area)
        self.pinching = False
        self.detected = False   # True while a hand is visible

hand_state = HandState()

# ─────────────────────────────────────────────
# Workspace bounds (robot base frame, metres)
# ─────────────────────────────────────────────
TABLE_HEIGHT  = 0.625
X_MIN, X_MAX  = 0.30, 0.70   # forward/backward  (mapped from depth)
Y_MIN, Y_MAX  = -0.50, 0.50  # left/right        (mapped from hand X)
Z_MIN, Z_MAX  = TABLE_HEIGHT + 0.10, TABLE_HEIGHT + 0.60   # height (mapped from hand Y)
PINCH_THRESH  = 40           # pixels

# Palm-area calibration – tune for your camera / distance
AREA_MIN, AREA_MAX = 2_000, 20_000


# ═══════════════════════════════════════════════════════════════════
# VISION THREAD  –  MediaPipe hand tracking
# ═══════════════════════════════════════════════════════════════════
def vision_thread():
    mp_hands   = mp.solutions.hands
    mp_draw    = mp.solutions.drawing_utils

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Vision] ERROR: cannot open webcam.")
        return

    area_history = deque(maxlen=20)   # smoothing buffer

    def palm_center(lm):
        ids = [0, 5, 9, 13, 17]
        pts = np.array([[lm[i].x, lm[i].y] for i in ids])
        return pts.mean(axis=0)   # (norm_x, norm_y)

    def palm_area(lm, w, h):
        ids = [0, 5, 9, 13, 17]
        pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in ids], dtype=np.int32)
        return cv2.contourArea(cv2.convexHull(pts))

    def thumb_index_dist(lm, w, h):
        t = (int(lm[4].x * w), int(lm[4].y * h))
        i = (int(lm[8].x * w), int(lm[8].y * h))
        return math.hypot(t[0]-i[0], t[1]-i[1]), t, i

    print("[Vision] Starting hand tracking... Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)                      # mirror so left=left
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            # --- palm center (norm coords) ---
            cx, cy = palm_center(lm)

            # --- depth from palm area ---
            area = palm_area(lm, w, h)
            area_history.append(area)
            smooth_area = float(np.mean(area_history))
            nz = np.clip((smooth_area - AREA_MIN) / (AREA_MAX - AREA_MIN), 0.0, 1.0)

            # --- pinch ---
            dist, thumb_pt, index_pt = thumb_index_dist(lm, w, h)
            pinch = dist < PINCH_THRESH

            # --- write to shared state ---
            with hand_state.lock:
                hand_state.norm_x   = float(cx)
                hand_state.norm_y   = float(cy)
                hand_state.norm_z   = float(nz)
                hand_state.pinching = pinch
                hand_state.detected = True

            # --- draw overlay ---
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS,
            )
            cx_px, cy_px = int(cx * w), int(cy * h)
            cv2.circle(frame, (cx_px, cy_px), 8, (0, 255, 0), -1)

            # pinch line
            cv2.line(frame, thumb_pt, index_pt, (255, 255, 0), 2)
            cv2.circle(frame, thumb_pt, 6, (0, 0, 255), -1)
            cv2.circle(frame, index_pt, 6, (0, 255, 0), -1)

            color = (0, 0, 255) if pinch else (255, 255, 255)
            label = "PINCH (close gripper)" if pinch else "open"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"norm x={cx:.2f}  y={cy:.2f}  depth={nz:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, f"palm area: {int(smooth_area)}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        else:
            with hand_state.lock:
                hand_state.detected = False
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        cv2.imshow("Hand Tracking → Robot Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    print("[Vision] Thread exiting.")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def lerp(v, src_lo, src_hi, dst_lo, dst_hi):
    """Linear remap, clamped."""
    t = (v - src_lo) / (src_hi - src_lo) if (src_hi - src_lo) != 0 else 0.5
    t = max(0.0, min(1.0, t))
    return dst_lo + t * (dst_hi - dst_lo)


def smooth_target(current, target, alpha=0.08):
    """Exponential low-pass filter (per axis)."""
    return [c + alpha * (t - c) for c, t in zip(current, target)]


# ═══════════════════════════════════════════════════════════════════
# MAIN  –  PyBullet simulation (runs on the main thread)
# ═══════════════════════════════════════════════════════════════════
def main():
    # ── PyBullet init ──────────────────────────────────────────────
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Floor + table
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf")

    # Panda robot
    panda_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0, 0, TABLE_HEIGHT],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
    )

    # ── Robot config ───────────────────────────────────────────────
    EE_INDEX    = 11   # end-effector link
    ARM_DOFS    = 7
    # Finger joint indices for the gripper
    FINGER_JOINTS = [9, 10]

    home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for i in range(ARM_DOFS):
        p.resetJointState(panda_id, i, home_joints[i])

    # End-effector always points downward
    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5, cameraYaw=45,
        cameraPitch=-30, cameraTargetPosition=[0, 0, 0.6],
    )

    fps       = 240.0
    time_step = 1.0 / fps
    p.setTimeStep(time_step)

    # Smoothed target (starts at a neutral pose above the table centre)
    smooth_pos = [0.5, 0.0, TABLE_HEIGHT + 0.35]

    # Debug sphere to show current IK target
    target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.02,
                                     rgbaColor=[1, 0.2, 0.2, 0.8])
    target_body = p.createMultiBody(baseVisualShapeIndex=target_vis,
                                    basePosition=smooth_pos)

    print("[Sim] Simulation running.")
    print("      Move your hand in front of the webcam to control the robot.")
    print("      Pinch thumb+index to close the gripper.")
    print("      Press ESC in the camera window (or CTRL+C) to exit.\n")

    try:
        while True:
            # ── Read hand state ────────────────────────────────────
            with hand_state.lock:
                detected = hand_state.detected
                nx       = hand_state.norm_x
                ny       = hand_state.norm_y
                nz       = hand_state.norm_z
                pinching = hand_state.pinching

            if detected:
                # Map normalised hand coords → robot workspace
                # hand X (0=left, 1=right) → robot Y (-0.5 → +0.5)
                robot_y = lerp(nx, 0.0, 1.0, Y_MAX, Y_MIN)   # mirrored
                # hand Y (0=top, 1=bottom) → robot Z (high → low)
                robot_z = lerp(ny, 0.0, 1.0, Z_MAX, Z_MIN)   # inverted
                # palm depth (0=far, 1=close) → robot X (far → near)
                robot_x = lerp(nz, 0.0, 1.0, X_MIN, X_MAX)

                raw_target = [robot_x, robot_y, robot_z]
            else:
                # No hand: hold last smooth position
                raw_target = smooth_pos[:]

            # ── Smooth the target to avoid jerky motion ────────────
            smooth_pos = smooth_target(smooth_pos, raw_target, alpha=0.08)

            # ── Update visual marker ───────────────────────────────
            p.resetBasePositionAndOrientation(
                target_body, smooth_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
            )

            # ── Inverse Kinematics ─────────────────────────────────
            joint_poses = p.calculateInverseKinematics(
                bodyUniqueId=panda_id,
                endEffectorLinkIndex=EE_INDEX,
                targetPosition=smooth_pos,
                targetOrientation=target_orn,
                maxNumIterations=100,
                residualThreshold=1e-5,
            )

            # ── Arm position control ───────────────────────────────
            for i in range(ARM_DOFS):
                p.setJointMotorControl2(
                    bodyIndex=panda_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=500,
                    positionGain=0.05,
                    velocityGain=1.0,
                )

            # ── Gripper control (pinch gesture) ────────────────────
            gripper_pos = 0.0 if pinching else 0.04   # 0 = closed, 0.04 = open
            for fj in FINGER_JOINTS:
                p.setJointMotorControl2(
                    bodyIndex=panda_id,
                    jointIndex=fj,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=gripper_pos,
                    force=20,
                )

            # ── Step physics ───────────────────────────────────────
            p.stepSimulation()
            time.sleep(time_step)

    except KeyboardInterrupt:
        print("\n[Sim] Stopped by user.")
    finally:
        p.disconnect()


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Start the vision thread as a daemon so it dies with the main thread
    v_thread = threading.Thread(target=vision_thread, daemon=True)
    v_thread.start()

    # Give the webcam a moment to warm up before the sim window opens
    time.sleep(1.5)

    main()