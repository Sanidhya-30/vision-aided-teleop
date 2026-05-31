import cv2
import mediapipe as mp
import time
import numpy as np

# --- URX Patch for Python 3.10+ ---
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable 
import urx

# ==========================================
# 1. CONFIGURATION & SAFETY LIMITS
# ==========================================
ROBOT_IP = "192.168.1.10"

# Significantly lowered for tele-operation safety
VELOCITY = 0.2     
ACCELERATION = 0.2

# Bounding Box Limits (Relative to Home)
MAX_XY_BOUND = 0.15  # Maximum 15cm deviation from the home anchor in X/Y

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def map_range(value, in_min, in_max, out_min, out_max):
    """Maps a value from one range to another, clamping to the bounds."""
    value = max(min(value, in_max), in_min)
    return out_min + (((value - in_min) / (in_max - in_min)) * (out_max - out_min))

# ==========================================
# 3. MAIN ROUTINE
# ==========================================
def main():
    print(f"Connecting to UR Robot at {ROBOT_IP}...")
    
    try:
        rob = urx.Robot(ROBOT_IP)
        print("Robot Connected successfully.")
        
        # Establish the fixed "Home" anchor
        raw_home_pose = rob.getl()
        # Unpack into standard float list
        home_pose = [
            float(raw_home_pose[0]), # X
            float(raw_home_pose[1]), # Y
            float(raw_home_pose[2]), # Z
            float(raw_home_pose[3]), # Rx
            float(raw_home_pose[4]), # Ry
            float(raw_home_pose[5])  # Rz
        ]
        print(f"Home Anchor established at: {home_pose[:3]}")
        
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize Camera
        cap = cv2.VideoCapture(0)
        print("Camera initialized. Show your hand to control the robot.")
        print("Press ESC in the video window to exit.")

        offset_x = 0
        offset_y = 0
        cx_norm = cy_norm = 0

        while cap.isOpened():
            # 1. ALWAYS read the camera to keep the video buffer empty and feed smooth
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            frame = cv2.flip(frame, 1) # Mirror image for intuitive control
            h, w, _ = frame.shape
            # print(f"X: {offset_x:.4f}  Y: {offset_y:.4f}  (raw: cx={cx_norm:.3f}, cy={cy_norm:.3f})")

            # 2. GATEKEEPER: Only calculate and send a new position IF the robot is stationary
            if not rob.is_program_running():
                
                # Convert to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                
                if result.multi_hand_landmarks:
                    # Get the first hand detected
                    hand_landmarks = result.multi_hand_landmarks[0]
                    
                    # Use the palm center (landmark 9) as the tracking point
                    cx_norm = hand_landmarks.landmark[9].x
                    cy_norm = hand_landmarks.landmark[9].y
                    
                    # Draw a circle on the palm for visual feedback
                    cx, cy = int(cx_norm * w), int(cy_norm * h)
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                    # --- MAPPING LOGIC ---
                    # We map the middle 60% of the camera frame (0.2 to 0.8) 
                    # to the physical +/- 15cm bounding box.
                    
                    # Camera Left/Right (cx_norm) -> Robot Y-axis
                    # Note: You may need to swap -MAX_XY_BOUND and MAX_XY_BOUND 
                    # depending on which side of the table you are standing on
                    offset_y = map_range(cx_norm, 0.2, 0.8, MAX_XY_BOUND, -MAX_XY_BOUND)
                    
                    # Camera Up/Down (cy_norm) -> Robot X-axis
                    offset_x = map_range(cy_norm, 0.2, 0.8, -MAX_XY_BOUND, MAX_XY_BOUND)

                    # Create new target based on HOME anchor
                    target_pose = list(home_pose)
                    target_pose[0] += offset_x # X axis
                    target_pose[1] += offset_y # Y axis
                    
                    # Note: Z-axis (target_pose[2]), Rx, Ry, Rz remain perfectly locked to home_pose!

                    # Send the movement command (wait=False so our video loop doesn't freeze!)
                    rob.movel(target_pose, acc=ACCELERATION, vel=VELOCITY, wait=False)
                    
                    # Critical: Give the controller a split second to register that it is 
                    # "running" before the next loop iteration checks is_program_running()
                    time.sleep(0.1) 
                    
                    cv2.putText(frame, f"MOVING (X:{offset_x:.2f}, Y:{offset_y:.2f})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # If robot is currently moving, just show text on screen
                cv2.putText(frame, "WAITING FOR ROBOT...", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"MOVING (X:{offset_x:.2f}, Y:{offset_y:.2f})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

            # print(f"X: {offset_x:.4f}  Y: {offset_y:.4f}  (raw: cx={cx_norm:.3f}, cy={cy_norm:.3f})")

            # Display the video feed
            cv2.imshow("Tele-operation Camera Feed", frame)

            # Exit condition (ESC key)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # --- CLEANUP ---
        print("\nExiting loop. Returning to Home Position...")
        rob.movel(home_pose, acc=ACCELERATION, vel=VELOCITY, wait=False)
        time.sleep(0.2)
        while rob.is_program_running():
            time.sleep(0.05)
        print("Safely back at Home.")

    except KeyboardInterrupt:
        print("\nScript interrupted! Stopping robot...")
        if 'rob' in locals():
            rob.stopl(acc=0.5)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        if 'rob' in locals():
            rob.close()
            print("Connections closed safely.")

if __name__ == "__main__":
    main()