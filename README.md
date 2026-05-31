# Vision-Aided Feedback System for Remote Surgical Assistance

An interactive, vision-driven robotic teleoperation framework engineered to facilitate remote surgical mentorship and training. This system tracks a human operator's hand posture and gestures non-invasively via a monocular RGB camera, applies predictive filtering to eliminate network transmission lag, and maps coordinates in real-time onto both simulated environments and physical Universal Robots (UR).

**IE 549: Machine Vision in Intelligent Robotic Systems** | **Purdue University** **Authors:** Vaishnavi Satish, Madhura Kunachi, Sanidhya Shrivastava

---

## System Demonstrations

### 1. 3D Hand Tracking & Orientation
![Hand Tracking](https://github.com/Sanidhya-30/vision-aided-teleop/blob/main/media/Hand_detection.gif)

### 2. Trajectory Prediction for Latency Management
![Trajectory Prediction](https://github.com/Sanidhya-30/vision-aided-teleop/blob/main/media/traj_pred.gif)

### 3. Real-Time Robot Teleoperation
![Real time teleoperation](https://github.com/Sanidhya-30/vision-aided-teleop/blob/main/media/Robot arm.gif)

---

## System Architecture & Mathematics

This project eliminates the need for expensive wearable trackers or multi-camera setups by extracting 6-Degrees of Freedom (6-DoF) from a standard 2D webcam and applying predictive mathematics to stabilize remote teleoperation.

### 1. Palm Detection & Hand Tracking

The vision pipeline utilizes MediaPipe Hands to detect 21 localized hand landmarks at 30 FPS. To establish a stable tracking anchor, the system isolates five specific palm landmarks: the wrist (Landmark 0) and the four Metacarpophalangeal (MCP) joints (Landmarks 5, 9, 13, 17).

* **Palm Center (X, Y):** Calculated as the geometric centroid of these five landmarks in normalized camera space $[0, 1]$.
* **Orientation (Roll/Pitch/Yaw):** A local coordinate frame is established on the hand. Two vectors are defined: $\vec{v}_1$ (wrist to index MCP) and $\vec{v}_2$ (wrist to pinky MCP). The surface normal vector $\vec{n}$, representing the palm's tilt, is derived via their cross product:

$$\vec{n} = \frac{\vec{v}_1 \times \vec{v}_2}{||\vec{v}_1 \times \vec{v}_2||}$$


* **Gripper Actuation (Pinch):** The Euclidean pixel distance between the thumb tip (Landmark 4) and index fingertip (Landmark 8) is continuously measured. A distance dropping below a 40-pixel threshold registers as a "closed" gripper state.

### 2. Depth Estimation (The Palm Area Method)

Because monocular cameras lack native depth sensing, a custom proxy is used to calculate the Z-axis. The five palm landmarks are projected into 2D pixel space, and their convex hull area is calculated.

Assuming a pinhole camera model, the perceived area $A$ is inversely proportional to the square of the distance. As the hand moves closer, the polygon area expands. This raw area is smoothed over a 20-frame rolling average to eliminate jitter, then linearly mapped between empirical bounds (2,000 to 20,000 pixels) and inverted to translate into physical Z-axis depth for the robot.

### 3. Coordinate Mapping Mathematics

To map the normalized camera space directly to the physical robot workspace safely, the robot initializes its starting pose as a fixed "Home Anchor." The system captures the middle 60% of the camera frame ($0.2$ to $0.8$ normalized) and maps it linearly to a strict $\pm 15\text{ cm}$ physical bounding box around the anchor.

The linear transformation is defined as:


$$x_{robot} = x_{min} + \frac{x_{cam} - x_{cam}^{min}}{x_{cam}^{max} - x_{cam}^{min}} \cdot (x_{max} - x_{min})$$


*(This formula is applied independently across the mapped X and Y axes, while Z relies on the Palm Area depth mapping).*

### 4. Trajectory Prediction (Latency Compensation)

Network and computational delays naturally cause the robot to lag behind the operator's actual hand position. To achieve true real-time synchronization, the system maintains a rolling buffer of the $N=30$ most recent pose keypoints.

A Least Squares curve-fitting model calculates a polynomial trajectory to capture nonlinear dynamics (like deceleration near tissue). It solves for coefficients $a, b, c$:


$$\min_{a_x, b_x, c_x} \sum_{i=1}^{n} \left( x_i - (a_x t_i^2 + b_x t_i + c_x) \right)^2$$

Once the curve is fitted, the model anticipates the operator's future pose $K$ steps ahead (optimally tuned to $K=7$ frames, or ~233 ms look-ahead) and issues proactive movement commands to the robot:


$$\hat{x}(t+k) = a_x(t+k)^2 + b_x(t+k) + c_x$$


This extrapolation effectively cancels out ~150ms of network latency.

---

## Repository Structure

```text
.
├── media/
│   ├── Hand_orientation_detection (online-video-cutter.com).gif
│   ├── Screenshot 2026-03-26 at 5.07.45 PM.png
│   ├── Screenshot 2026-03-26 at 5.10.30 PM.png
│   ├── Screenshot 2026-03-26 at 5.10.55 PM.png
│   ├── Video_Robot arm.gif
│   └── traj_pred.gif
├── requirements.txt
└── src/
    ├── mediapipe_palm_tracker.py  # Standalone MediaPipe tracking & orientation
    ├── sim_pybullet.py            # Local PyBullet Physics simulation
    ├── sim.py                     # Multi-threaded simulation with vision logic
    ├── robot.py                   # UR Hardware execution with safety constraints
    ├── robot_with_sim.py          # Simultaneous UR execution and virtual sim
    └── trajectory_pred.py         # Least squares latency compensation module

```

---

## Installation & Usage

### 1. Prerequisites

Due to network socket bindings in the robot controller libraries, Python 3.10+ is recommended.

```bash
git clone https://github.com/your-username/vision-aided-teleoperation.git
cd vision-aided-teleoperation
pip install -r requirements.txt

```

### 2. Module Execution

Navigate to the root directory and execute the specific environment you wish to run:

* **Test Vision Pipeline:** Visualizes the hand tracking, 3D coordinate mapping, and surface normal arrows.
```bash
python src/mediapipe_palm_tracker.py

```


* **Run Local Simulation:** Runs the multithreaded MediaPipe tracking coupled to a virtual Franka Panda robotic arm inside PyBullet.
```bash
python src/sim.py

```


* **Hardware Deployment (Universal Robots):** Controls the physical UR10 arm. *Ensure your machine is on the same subnet as the UR controller (`192.168.1.10`)*.
```bash
python src/robot.py

```


* **Test Trajectory Prediction:** Visualizes the latency-compensation algorithm against real-time hand movements.
```bash
python src/trajectory_pred.py

```

> **Safety Limits:** When running `robot.py`, velocity and acceleration are strictly limited to $0.2\text{ m/s}$ and $0.2\text{ m/s}^2$ respectively. Hardware bounds will not allow the end-effector to deviate beyond $15\text{ cm}$ from the initialized home anchor.


---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

