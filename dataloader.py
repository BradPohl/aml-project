import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import mediapipe as mp  # For pose estimation (using MediaPipe as an example)

# Define your class mappings
class_map = {'punch': 0, 'kick': 1, 'downtime': 2}

# Set up MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

class PunchingBagDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        
        # Collect all video files and labels based on folders
        for class_name, class_idx in class_map.items():
            class_folder = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_folder):
                if file_name.endswith('.mov'): #When expaning Dataset you will most likely need to check for more than .mov
                    self.data.append((os.path.join(class_folder, file_name), class_idx))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        
        # Load the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        pose_keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Optional: Resize or preprocess frame
            if self.transform:
                frame = self.transform(frame)
            
            # Convert the frame to RGB (if needed by the pose estimation library)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform pose estimation
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract keypoints (e.g., x, y coordinates of each landmark)
                keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
                pose_keypoints.append(keypoints)
            else:
                # If no pose detected, append a placeholder or skip frame
                pose_keypoints.append(None)
        
        cap.release()
        
        # Convert the list of keypoints into a tensor or other suitable format
        # Example: torch.tensor(pose_keypoints) if you plan to use them as input to a neural network
        return pose_keypoints, label
    

    def display_random_video_with_pose(self):
        video_path, label = random.choice(self.data)
        cap = cv2.VideoCapture(video_path)
        
        print(f"Displaying video: {video_path} (Label: {label})")

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            cv2.imshow("Pose Estimation", frame)
            
            # Exit the display by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage
data_directory = './dataset'
dataset = PunchingBagDataset(data_directory)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
for i in range(10):
    dataset.display_random_video_with_pose()
