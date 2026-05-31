"""
robot_sim.py  —  Headless PyBullet + Matplotlib arm visualiser
---------------------------------------------------------------
• PyBullet runs in p.DIRECT (no OpenGL) — safe on macOS M-series background threads
• Arm joint positions are read from PyBullet and drawn with matplotlib
• Shared state dict is written by hand_tracker.py (main thread)

Run via:  python main.py
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # works headlessly in a thread on macOS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── Workspace mapping ──────────────────────────────────────────────────────────
X_MIN, X_MAX = -0.4,  0.4
Y_MIN, Y_MAX =  0.2,  0.7
Z_MIN, Z_MAX =  0.1,  0.6

SIM_HZ  = 240
CTRL_HZ =  30
PLOT_HZ =  15


def map_range(value, in_min, in_max, out_min, out_max):
    value = np.clip(value, in_min, in_max)
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)


class RobotSim:
    def __init__(self, shared_state: dict, use_gui: bool = False):
        self.state = shared_state

        # ── Headless PyBullet ─────────────────────────────────────────────────
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / SIM_HZ)

        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        self.num_joints = p.getNumJoints(self.robot_id)
        self.ee_link    = 6

        self.ik_damping  = [0.1] * self.num_joints
        self.last_target = [0.0, 0.45, 0.35]
        self.last_pinch  = False

        # ── Matplotlib arm viewer ─────────────────────────────────────────────
        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.fig.patch.set_facecolor("#1a1a2e")
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self._style_axes()
        self.fig.canvas.manager.set_window_title("Robot Arm — PyBullet IK")

        self.arm_line,  = self.ax.plot([], [], [], "o-",
                                       color="#00d4ff", lw=3, ms=6, zorder=5)
        self.finger_l,  = self.ax.plot([], [], [], "-",
                                       color="#00ff99", lw=4)
        self.finger_r,  = self.ax.plot([], [], [], "-",
                                       color="#00ff99", lw=4)
        self.ee_dot,    = self.ax.plot([], [], [], "o",
                                       color="#ff4757", ms=10, zorder=10)
        self.status_txt = self.ax.text2D(
            0.02, 0.95, "Waiting…",
            transform=self.ax.transAxes,
            color="white", fontsize=9
        )

        print("[RobotSim] Headless PyBullet + matplotlib viewer ready.")

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(-0.6, 0.6);  ax.set_xlabel("X (m)", color="white")
        ax.set_ylim( 0.0, 0.9);  ax.set_ylabel("Y (m)", color="white")
        ax.set_zlim( 0.0, 0.8);  ax.set_zlabel("Z (m)", color="white")
        ax.set_title("Robot Arm — live IK", color="white", pad=8)
        ax.tick_params(colors="white")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#444466")
        ax.view_init(elev=20, azim=-60)

    def _move_to(self, target_pos, pinch: bool):
        target_orn = p.getQuaternionFromEuler([0, np.pi, 0])
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link,
            target_pos,
            target_orn,
            jointDamping=self.ik_damping,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        for j in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id, j,
                p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=250,
                maxVelocity=1.5
            )

    def _get_link_positions(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        positions   = [list(base_pos)]
        for i in range(self.num_joints):
            ls = p.getLinkState(self.robot_id, i)
            positions.append(list(ls[4]))
        return np.array(positions)

    def _redraw(self, positions, ee_pos, pinch):
        xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
        self.arm_line.set_data(xs, ys)
        self.arm_line.set_3d_properties(zs)

        self.ee_dot.set_data([ee_pos[0]], [ee_pos[1]])
        self.ee_dot.set_3d_properties([ee_pos[2]])

        spread = 0.02 if pinch else 0.06
        ee = np.array(ee_pos)
        for line, sign in zip((self.finger_l, self.finger_r), (-1, 1)):
            tip = ee + np.array([sign * spread, 0, -0.04])
            line.set_data([ee[0], tip[0]], [ee[1], tip[1]])
            line.set_3d_properties([ee[2], tip[2]])

        active = self.state.get("active", False)
        label  = (
            f"EE: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}) m\n"
            f"Gripper: {'CLOSED' if pinch else 'OPEN'}\n"
            f"{'TRACKING' if active else 'no hand'}"
        )
        self.status_txt.set_text(label)
        self.status_txt.set_color("#00ff99" if active else "#ff6b81")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        step_interval  = 1.0 / SIM_HZ
        ctrl_interval  = 1.0 / CTRL_HZ
        plot_interval  = 1.0 / PLOT_HZ
        last_ctrl_time = time.time()
        last_plot_time = time.time()

        try:
            while self.state.get("running", True):
                now = time.time()

                if now - last_ctrl_time >= ctrl_interval:
                    last_ctrl_time = now
                    active = self.state.get("active", False)

                    if active:
                        hx    = self.state.get("x", 0.5)
                        hy    = self.state.get("y", 0.5)
                        hz    = self.state.get("z", 0.5)
                        pinch = self.state.get("pinch", False)

                        rx = map_range(hx, 0, 1, X_MIN, X_MAX)
                        ry = map_range(hy, 0, 1, Y_MIN, Y_MAX)
                        rz = map_range(hz, 0, 1, Z_MIN, Z_MAX)

                        self.last_target = [rx, ry, rz]
                        self.last_pinch  = pinch

                    self._move_to(self.last_target, self.last_pinch)

                p.stepSimulation()

                if now - last_plot_time >= plot_interval:
                    last_plot_time = now
                    positions = self._get_link_positions()
                    ee_state  = p.getLinkState(self.robot_id, self.ee_link)
                    ee_pos    = ee_state[4]
                    self._redraw(positions, ee_pos, self.last_pinch)

                time.sleep(step_interval)

        except Exception as e:
            print(f"[RobotSim] Error: {e}")
        finally:
            p.disconnect()
            plt.close(self.fig)
            print("[RobotSim] Disconnected.")
