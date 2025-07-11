import mediapipe as mp
import numpy as np
import cv2
import json
import os

# ========== CONFIG ==========
pose_name = "warrior"  # cobra✅ Use the correct pose name (you had a typo: "traingle")
image_path = r"C:\Users\SMART TECH\Downloads\fitness-ai-app\pose-backend\reference_images\warrior.jpg"
output_path = f"./reference_poses/{pose_name}.json"
# ============================

# Ensure output folder exists
os.makedirs("reference_poses", exist_ok=True)

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# Load image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")
print("✅ Image loaded.")

# Detect pose
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
if not results.pose_landmarks:
    raise ValueError("❌ No pose detected. Try a clearer image.")
print("✅ Pose detected.")

# Helpers
def get_landmark(idx):
    lm = results.pose_landmarks.landmark[idx]
    return [lm.x, lm.y]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# Key joints
JOINTS = {
    "left_elbow": [11, 13, 15],
    "right_elbow": [12, 14, 16],
    "left_knee": [23, 25, 27],
    "right_knee": [24, 26, 28],
    "left_shoulder": [13, 11, 23],
    "right_shoulder": [14, 12, 24],
    "left_hip": [11, 23, 25],
    "right_hip": [12, 24, 26]
}

# Calculate angles
pose_angles = {}
for joint, (a, b, c) in JOINTS.items():
    try:
        angle = calculate_angle(get_landmark(a), get_landmark(b), get_landmark(c))
        pose_angles[joint] = round(angle, 2)
    except:
        pose_angles[joint] = None

# Save to JSON
pose_data = {
    "name": pose_name,
    "angles": pose_angles
}

with open(output_path, "w") as f:
    json.dump(pose_data, f, indent=2)

print(f"✅ Saved reference pose to {output_path}")
