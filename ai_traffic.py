import cv2
import numpy as np
import time
import json
import os
import sys
import argparse
from ultralytics import YOLO
from q_learning_agent import AdaptiveQLearningAgent

# =================================================================================
# === CONFIGURATION                                                             ===
# =================================================================================

VIDEO_FILE = "my_video.mp4"  # Your video file

# Your custom lane polygons
LANE_POLYGONS = {
    "lane_1": np.array([[2124, 487], [2830, 514], [2103, 1657], [2829, 1592]], np.int32),
    "lane_2": np.array([[966, 1568], [1380, 1574], [1467, 2048], [830, 2085]], np.int32),
    # For a 4-way intersection, you would ideally have 4 lanes defined.
    # The agent is currently set up for 4 actions (one for each of 4 lanes).
    # Let's add placeholder lanes for now.
    "lane_3": np.array([[0,0], [1,1], [2,2], [3,3]], np.int32), # <<< UPDATE THIS
    "lane_4": np.array([[0,0], [1,1], [2,2], [3,3]], np.int32), # <<< UPDATE THIS
}
# A list of lane names for easy indexing, ensure this order matches your intersection logic
LANE_NAMES_ORDER = ["lane_1", "lane_2", "lane_3", "lane_4"] 

# --- RL Agent and Simulation Parameters ---
MAX_VEHICLES_PER_LANE = 20  # Used for normalizing the state
MAX_TIME_SINCE_CHANGE = 60 # Max seconds a light can be green, for normalization
EPISODE_LENGTH_IN_FRAMES = 1000 # How many frames to run before an episode ends

# --- Detection Parameters ---
CONF_THRESHOLD = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# =================================================================================
# === (No changes needed in the setup mode function)                            ===
# =================================================================================
def get_coordinates_mode(video_path):
    # This function remains unchanged.
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Point selected: [{x}, {y}]")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError
    except IOError:
        print(f"\n[ERROR] Could not open video file '{video_path}'.")
        sys.exit(1)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read the first frame from the video.")
        sys.exit(1)
    window_name = 'SETUP MODE: Click on lane corners, then press any key'
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("\n--- SETUP MODE ---")
    print("Click on the 4 corners of a lane. The coordinates will be printed here.")
    print("Press any key on the video window to quit setup mode.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    print("Setup mode finished.")

# =================================================================================
# === MAIN TRAINING AND DETECTION LOGIC                                         ===
# =================================================================================
def run_training_and_detection_mode(video_path, lane_polygons):
    """
    Runs the main training loop for the Q-Learning agent while processing the video.
    """
    print("[INFO] Initializing Q-Learning agent...")
    # The agent has 4 actions, one for making each of the 4 lanes green.
    agent = AdaptiveQLearningAgent(action_size=4)
    
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open video file '{video_path}'.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    # --- Training State Variables ---
    current_phase = 0  # Represents which light is green (0-3, corresponding to LANE_NAMES_ORDER)
    time_since_change = 0
    
    episode_num = 0
    total_reward_this_episode = 0
    frame_in_episode = 0
    
    previous_observation = None
    previous_action = None

    print("\n[INFO] RUN MODE: Starting training and detection... Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream. Restarting for continuous training.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue

        # --- Vehicle Detection ---
        results = model(frame, verbose=False)
        lane_counts = {name: 0 for name in LANE_NAMES_ORDER}
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            if class_id in VEHICLE_CLASSES and box.conf[0].item() > CONF_THRESHOLD:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                for lane_name, polygon in lane_polygons.items():
                    if cv2.pointPolygonTest(polygon, center_point, False) >= 0:
                        lane_counts[lane_name] += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        break
        
        # --- Q-Learning Agent Step ---
        
        # 1. Construct the state observation for the agent
        lane_counts_ordered = [lane_counts[name] for name in LANE_NAMES_ORDER]
        normalized_counts = [count / MAX_VEHICLES_PER_LANE for count in lane_counts_ordered]
        normalized_phase = current_phase / (agent.action_size - 1)
        normalized_time = time_since_change / MAX_TIME_SINCE_CHANGE
        
        current_observation = np.array(normalized_counts + [normalized_phase, normalized_time])

        # 2. Calculate reward: penalty for total waiting cars
        reward = -sum(lane_counts_ordered)
        total_reward_this_episode += reward
        
        # 3. On the second step onwards, the agent can learn from the previous action
        if previous_observation is not None:
            agent.learn(
                state=previous_observation,
                action=previous_action,
                reward=reward,
                next_state=current_observation,
                done=(frame_in_episode >= EPISODE_LENGTH_IN_FRAMES)
            )

        # 4. Agent chooses a new action based on the current state
        action = agent.choose_action(current_observation)

        # 5. Update the virtual traffic light state based on the agent's action
        if action != current_phase:
            current_phase = action # Change the light
            time_since_change = 0
        else:
            time_since_change += 1 / fps # Increment time

        # 6. Store current state and action for the next learning step
        previous_observation = current_observation
        previous_action = action
        
        frame_in_episode += 1
        
        # --- Episode Handling ---
        if frame_in_episode >= EPISODE_LENGTH_IN_FRAMES:
            print(f"--- Episode {episode_num} Finished ---")
            print(f"Total reward: {total_reward_this_episode:.2f}")
            agent.episode_finished(total_reward_this_episode, frame_in_episode)
            stats = agent.get_enhanced_statistics()
            print(f"Q-Table Size: {stats['q_table_size']}, Epsilon: {stats['current_epsilon']:.4f}")
            
            # Reset for next episode
            frame_in_episode = 0
            total_reward_this_episode = 0
            episode_num += 1
            
            # Save the model periodically
            if episode_num % 10 == 0:
                agent.save_model("traffic_agent")

        # --- Visualization ---
        for i, (lane_name, polygon) in enumerate(lane_polygons.items()):
            color = (0, 255, 0) if i == current_phase else (0, 0, 255) # Green for active lane, Red for others
            cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=4)
            cv2.putText(frame, f"{lane_name}: {lane_counts[lane_name]}", (polygon[0][0], polygon[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Display agent stats on screen
        info_text = f"Episode: {episode_num} | Frame: {frame_in_episode}/{EPISODE_LENGTH_IN_FRAMES} | Phase: {LANE_NAMES_ORDER[current_phase]}"
        cv2.putText(frame, info_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 6)
        cv2.putText(frame, info_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        
        # --- Visualization ---
        # ... (previous visualization code) ...

        # <<< ADD THIS CODE BLOCK TO RESIZE THE FRAME >>>
        # Define a more manageable display size
        display_width = 1280
        display_height = 720
        # Resize the frame for display
        display_frame = cv2.resize(frame, (display_width, display_height))
        # <<< END OF ADDED CODE >>>

        # NOW, DISPLAY THE RESIZED FRAME INSTEAD OF THE ORIGINAL
        cv2.imshow('AI Traffic Agent Training', display_frame) # Use 'display_frame' here

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    print("[INFO] Cleaning up...")
    agent.save_model("traffic_agent_final") # Save final progress
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Traffic Analysis Script")
    parser.add_argument('--setup', action='store_true', help="Run in setup mode to get lane coordinates.")
    args = parser.parse_args()

    if args.setup:
        get_coordinates_mode(VIDEO_FILE)
    else:
        if "YOUR_VIDEO_NAME.mp4" in VIDEO_FILE:
             print("\n[ERROR] Please configure the 'VIDEO_FILE' variable in the script first.")
             sys.exit(1)
        run_training_and_detection_mode(VIDEO_FILE, LANE_POLYGONS)