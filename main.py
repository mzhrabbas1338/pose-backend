from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mediapipe as mp
import numpy as np
import cv2
import json
import os
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Joint mappings
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

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

@app.post("/analyze-pose/")
async def analyze_pose(file: UploadFile = File(...), pose_name: str = Form(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return process_pose(image, pose_name)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze-live-frame/")
async def analyze_live_frame(request: Request):
    try:
        data = await request.json()
        pose_name = data.get("pose_name")
        base64_image = data.get("frame")

        if not base64_image or not pose_name:
            return JSONResponse(status_code=400, content={"error": "Missing image or pose name"})

        image_data = base64.b64decode(base64_image.split(",")[-1])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return process_pose(image, pose_name)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def process_pose(image, pose_name):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return {"score": 0, "feedback": ["❌ Pose not detected. Try again."], "pose_image": None}

    json_path = f"./reference_poses/{pose_name}.json"
    if not os.path.exists(json_path):
        return JSONResponse(status_code=404, content={"error": "❌ Reference pose not found."})

    with open(json_path, "r") as f:
        reference_data = json.load(f)
    reference_angles = reference_data["angles"]

    # Extract user angles
    user_angles = {}
    lm = results.pose_landmarks.landmark
    for joint, (a, b, c) in JOINTS.items():
        try:
            user_angle = calculate_angle([lm[a].x, lm[a].y], [lm[b].x, lm[b].y], [lm[c].x, lm[c].y])

            user_angles[joint] = round(user_angle, 2)
        except:
            user_angles[joint] = None

    # Compare and calculate score
    total_diff = 0
    count = 0
    feedback = []

    for joint, ref_angle in reference_angles.items():
        user_angle = user_angles.get(joint)
        if user_angle is not None and ref_angle is not None:
            diff = abs(ref_angle - user_angle)
            total_diff += diff
            count += 1
            if diff > 15:
                feedback.append(f"⚠️ Adjust your {joint.replace('_', ' ')}")
            else:
                feedback.append(f"✅ Good {joint.replace('_', ' ')}")

    avg_diff = total_diff / count if count else 100
    score = max(0, 100 - avg_diff)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    _, buffer = cv2.imencode(".jpg", image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "score": round(score, 2),
        "feedback": feedback,
        "pose_image": img_base64
    }
