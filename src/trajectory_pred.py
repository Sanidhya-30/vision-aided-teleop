import cv2
import mediapipe as mp
import numpy as np
import pybullet as p
import pybullet_data
import math
from collections import deque

import matplotlib
matplotlib.use("MacOSX")   # best backend for macOS; fallback: "TkAgg"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. HELPER FUNCTIONS
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
    thumb  = landmarks[4]
    index  = landmarks[8]
    x1, y1 = int(thumb.x * w), int(thumb.y * h)
    x2, y2 = int(index.x * w), int(index.y * h)
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist, (x1, y1), (x2, y2)

def map_range(value, in_min, in_max, out_min, out_max):
    value = max(min(value, in_max), in_min)
    return out_min + (((value - in_min) / (in_max - in_min)) * (out_max - out_min))

# ==========================================
# 2. TRAJECTORY PREDICTION
# ==========================================
HISTORY_LEN     = 40
PREDICT_STEPS   = 20
POLY_DEGREE     = 2
MIN_POINTS_FIT  = 8

# Workspace bounds (robot space)
WS_X = (0.25, 0.75)
WS_Y = (-0.45, 0.45)
WS_Z = (0.63,  1.25)

def predict_trajectory(hx, hy, hz):
    n = len(hx)
    if n < MIN_POINTS_FIT:
        return None
    t        = np.arange(n, dtype=float)
    t_future = np.arange(n, n + PREDICT_STEPS, dtype=float)
    try:
        pred_x = np.clip(np.polyval(np.polyfit(t, hx, POLY_DEGREE), t_future), *WS_X)
        pred_y = np.clip(np.polyval(np.polyfit(t, hy, POLY_DEGREE), t_future), *WS_Y)
        pred_z = np.clip(np.polyval(np.polyfit(t, hz, POLY_DEGREE), t_future), *WS_Z)
    except np.linalg.LinAlgError:
        return None
    return pred_x, pred_y, pred_z

# ==========================================
# 3. MATPLOTLIB GRAPH  (manual draw, no FuncAnimation)
# ==========================================
plt.ion()
fig = plt.figure(figsize=(13, 5), facecolor="#0d0d14")
fig.suptitle("Robot End-Effector  ·  Trajectory Prediction",
             color="#e0e0ff", fontsize=13, fontweight="bold", y=0.97)

gs    = GridSpec(1, 3, figure=fig, wspace=0.38)
ax_xy = fig.add_subplot(gs[0], facecolor="#12121e")
ax_xz = fig.add_subplot(gs[1], facecolor="#12121e")
ax_3d = fig.add_subplot(gs[2], projection="3d")
ax_3d.set_facecolor("#12121e")

def _style_2d(ax, xlabel, ylabel, title, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, color="#8888bb", fontsize=8)
    ax.set_ylabel(ylabel, color="#8888bb", fontsize=8)
    ax.set_title(title,   color="#ccccff", fontsize=9, pad=6)
    ax.tick_params(colors="#555588", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#222244")

def _style_3d(ax):
    ax.set_xlabel("Y",  color="#8888bb", fontsize=7, labelpad=2)
    ax.set_ylabel("X",  color="#8888bb", fontsize=7, labelpad=2)
    ax.set_zlabel("Z",  color="#8888bb", fontsize=7, labelpad=2)
    ax.set_title("3-D View", color="#ccccff", fontsize=9, pad=4)
    ax.tick_params(colors="#555588", labelsize=6)
    ax.set_xlim(*WS_Y); ax.set_ylim(*WS_X); ax.set_zlim(*WS_Z)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(color="#1e1e33", linewidth=0.5)

GRAPH_EVERY    = 6   # redraw graph every N camera frames
_frame_counter = 0

def redraw_graph(hx, hy, hz, pred, pinching):
    dot_col    = "#ff4466" if pinching else "#44ffaa"
    hist_patch = mpatches.Patch(color="#3355ff", label="History")
    pred_patch = mpatches.Patch(color="#ff9933", label="Predicted")

    # ── Top view X-Y ──────────────────────────────
    ax_xy.cla(); ax_xy.set_facecolor("#12121e")
    _style_2d(ax_xy, "Y (left/right)", "X (fwd/back)", "Top View  X-Y", WS_Y, WS_X)
    if len(hx) > 1:
        ax_xy.plot(hy, hx, color="#3355ff", linewidth=1.4, alpha=0.85)
        ax_xy.scatter(hy[-1], hx[-1], color=dot_col, s=60, zorder=5)
    if pred is not None:
        px, py, pz = pred
        ax_xy.plot(py, px, "--", color="#ff9933", linewidth=1.4, alpha=0.9)
        ax_xy.scatter(py[-1], px[-1], color="#ffcc44", s=55, marker="*", zorder=6)
    ax_xy.legend(handles=[hist_patch, pred_patch], fontsize=7,
                 facecolor="#1a1a2e", labelcolor="#ccccff", loc="upper right")

    # ── Side view X-Z ──────────────────────────────
    ax_xz.cla(); ax_xz.set_facecolor("#12121e")
    _style_2d(ax_xz, "X (fwd/back)", "Z (height)", "Side View  X-Z", WS_X, WS_Z)
    if len(hx) > 1:
        ax_xz.plot(hx, hz, color="#33aaff", linewidth=1.4, alpha=0.85)
        ax_xz.scatter(hx[-1], hz[-1], color=dot_col, s=60, zorder=5)
    if pred is not None:
        px, py, pz = pred
        ax_xz.plot(px, pz, "--", color="#ff9933", linewidth=1.4, alpha=0.9)
        ax_xz.scatter(px[-1], pz[-1], color="#ffcc44", s=55, marker="*", zorder=6)
    ax_xz.legend(handles=[hist_patch, pred_patch], fontsize=7,
                 facecolor="#1a1a2e", labelcolor="#ccccff", loc="upper right")

    # ── 3-D view ───────────────────────────────────
    ax_3d.cla(); ax_3d.set_facecolor("#12121e"); _style_3d(ax_3d)
    if len(hx) > 1:
        ax_3d.plot(hy, hx, hz, color="#3355ff", linewidth=1.2, alpha=0.85)
        ax_3d.scatter(hy[-1], hx[-1], hz[-1], color=dot_col, s=50, zorder=5)
    if pred is not None:
        px, py, pz = pred
        ax_3d.plot(py, px, pz, "--", color="#ff9933", linewidth=1.2, alpha=0.9)
        ax_3d.scatter(py[-1], px[-1], pz[-1], color="#ffcc44", s=40, marker="*", zorder=6)
    ax_3d.legend(handles=[hist_patch, pred_patch], fontsize=7,
                 facecolor="#1a1a2e", labelcolor="#ccccff", loc="upper left")

    fig.canvas.draw()
    fig.canvas.flush_events()   # pumps the macOS event loop — no plt.pause() needed

# ==========================================
# 4. PYBULLET SETUP
# ==========================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")
table_height = 0.625
p.loadURDF("table/table.urdf")

pandaId = p.loadURDF("franka_panda/panda.urdf",
                     basePosition=[0, 0, table_height], useFixedBase=True)
end_effector_index = 11
arm_num_dofs  = 7
finger_joints = [9, 10]

for i, val in enumerate([0, -0.785, 0, -2.356, 0, 1.571, 0.785]):
    p.resetJointState(pandaId, i, val)

target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
                             cameraTargetPosition=[0.5, 0, 0.6])

glove_visual = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="duck.obj",
    meshScale=[0.1, 0.1, 0.1],
    rgbaColor=[0.8, 0.3, 0.3, 1]
)
glove_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=glove_visual,
                              basePosition=[0, 0, 0])

# ==========================================
# 5. CV SETUP
# ==========================================
mp_hands_mod   = mp.solutions.hands
hands_detector = mp_hands_mod.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)

# History buffers
hist_x    = deque(maxlen=HISTORY_LEN)
hist_y    = deque(maxlen=HISTORY_LEN)
hist_z    = deque(maxlen=HISTORY_LEN)
area_hist = deque(maxlen=10)

MIN_AREA        = 2000
MAX_AREA        = 20000
PINCH_THRESHOLD = 40

print("Starting… Show your hand to the camera.")
print("Press ESC to exit.")
plt.show(block=False)

# ==========================================
# 6. MAIN LOOP
# ==========================================
sim_connected = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    pinching = False
    pred     = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            dist, thumb_pt, index_pt = compute_thumb_index_distance(lm, w, h)
            pinching = dist < PINCH_THRESHOLD

            cx_norm = lm[9].x
            cy_norm = lm[9].y

            area = compute_palm_area(lm, w, h)
            area_hist.append(area)
            smooth_area = float(np.mean(area_hist))

            robot_y = map_range(cx_norm,     0.2, 0.8,  0.4, -0.4)
            robot_x = map_range(cy_norm,     0.2, 0.8,  0.7,  0.3)
            robot_z = map_range(smooth_area, MIN_AREA, MAX_AREA,
                                table_height + 0.1, table_height + 0.6)

            hist_x.append(robot_x)
            hist_y.append(robot_y)
            hist_z.append(robot_z)

            pred = predict_trajectory(list(hist_x), list(hist_y), list(hist_z))

            # ── PyBullet control ──────────────────────────────────────
            if sim_connected:
                try:
                    joint_poses = p.calculateInverseKinematics(
                        pandaId, end_effector_index,
                        [robot_x, robot_y, robot_z], target_orn,
                        maxNumIterations=20
                    )
                    for i in range(arm_num_dofs):
                        p.setJointMotorControl2(pandaId, i, p.POSITION_CONTROL,
                                                targetPosition=joint_poses[i], force=500)

                    gripper_target = 0.0 if pinching else 0.04
                    for fj in finger_joints:
                        p.setJointMotorControl2(pandaId, fj, p.POSITION_CONTROL,
                                                targetPosition=gripper_target, force=100)

                    link_state = p.getLinkState(pandaId, end_effector_index)
                    ee_pos, ee_orn = link_state[0], link_state[1]
                    final_pos, final_orn = p.multiplyTransforms(
                        ee_pos, ee_orn, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0])
                    )
                    p.resetBasePositionAndOrientation(glove_id, final_pos, final_orn)
                    p.changeVisualShape(glove_id, -1,
                                        rgbaColor=[1, 0, 0, 1] if pinching else [0, 1, 0, 1])
                except p.error:
                    sim_connected = False

            # ── OpenCV overlays ───────────────────────────────────────
            # Pinch indicator
            cv2.line(frame, thumb_pt, index_pt, (255, 255, 0), 2)
            cx_px = int(cx_norm * w)
            cy_px = int(cy_norm * h)
            cv2.circle(frame, (cx_px, cy_px), 8,
                       (0, 0, 255) if pinching else (0, 255, 0), -1)

            # History trail (blue fade) projected onto camera frame
            hx_list = list(hist_x)
            hy_list = list(hist_y)
            for k in range(1, len(hx_list)):
                # robot Y  → screen X:  0.4 (left) → 0.2*w,  -0.4 (right) → 0.8*w
                # robot X  → screen Y:  0.7 (top)  → 0.2*h,   0.3 (bot)   → 0.8*h
                sx1 = int(map_range(hy_list[k-1],  0.4, -0.4, 0.2*w, 0.8*w))
                sy1 = int(map_range(hx_list[k-1],  0.7,  0.3, 0.2*h, 0.8*h))
                sx2 = int(map_range(hy_list[k],    0.4, -0.4, 0.2*w, 0.8*w))
                sy2 = int(map_range(hx_list[k],    0.7,  0.3, 0.2*h, 0.8*h))
                fade = int(80 + 175 * (k / max(len(hx_list), 1)))
                cv2.line(frame, (sx1, sy1), (sx2, sy2), (fade//3, fade//2, fade), 1)

            # Predicted trajectory (orange dashed) projected onto camera frame
            if pred is not None and len(hx_list) > 0:
                px_arr, py_arr, _ = pred
                # Start from last known history point
                prev_sx = int(map_range(hy_list[-1], 0.4, -0.4, 0.2*w, 0.8*w))
                prev_sy = int(map_range(hx_list[-1], 0.7,  0.3, 0.2*h, 0.8*h))
                for k in range(len(px_arr)):
                    cur_sx = int(map_range(py_arr[k], 0.4, -0.4, 0.2*w, 0.8*w))
                    cur_sy = int(map_range(px_arr[k], 0.7,  0.3, 0.2*h, 0.8*h))
                    if k % 2 == 0:   # dashed effect
                        cv2.line(frame, (prev_sx, prev_sy), (cur_sx, cur_sy),
                                 (0, 165, 255), 2)
                    prev_sx, prev_sy = cur_sx, cur_sy
                # Star marker at predicted endpoint
                cv2.drawMarker(frame, (prev_sx, prev_sy), (0, 200, 255),
                               cv2.MARKER_STAR, 14, 2)

            # Label overlay
            label = "PINCHING" if pinching else "OPEN"
            color  = (0, 0, 255)  if pinching else (0, 255, 120)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)
            if pred is not None:
                cv2.putText(frame, "Trajectory: ON", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)

    # ── Step simulation (guarded) ─────────────────────────────────────
    if sim_connected:
        try:
            for _ in range(8):
                p.stepSimulation()
        except p.error:
            sim_connected = False

    # ── Redraw matplotlib every N frames ─────────────────────────────
    _frame_counter += 1
    if _frame_counter % GRAPH_EVERY == 0:
        redraw_graph(list(hist_x), list(hist_y), list(hist_z), pred, pinching)

    cv2.imshow("Hand Control Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================
# 7. CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()
if sim_connected:
    try:
        p.disconnect()
    except Exception:
        pass
plt.close("all")