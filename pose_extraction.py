import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# 初始化 MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    def get_landmark_list(landmarks, count, selected_indices=None):
        if landmarks:
            lm = landmarks.landmark
            if selected_indices is None:
                return [[l.x, l.y, l.z] for l in lm]
            else:
                return [[lm[i].x, lm[i].y, lm[i].z] if i < len(landmarks.landmark) else [0.0, 0.0, 0.0]for i in selected_indices]
        else:
            return [[0.0, 0.0, 0.0]] * count
    SELECTED_FACE_INDICES = [
        33, 133,
        263, 362,
        61, 291,
        13, 14,
        1
    ]
    pose = get_landmark_list(results.pose_landmarks, 33)
    left_hand = get_landmark_list(results.left_hand_landmarks, 21)
    right_hand = get_landmark_list(results.right_hand_landmarks, 21)
    face = get_landmark_list(results.face_landmarks, 9, SELECTED_FACE_INDICES)

    return pose + left_hand + right_hand + face

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(fps / 15)) if fps > 0 else 1
    frame_count = 0
    sequence = []

    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    cap.release()

    sequence = np.array(sequence)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, sequence)

def process_all_videos(video_dir, output_dir, ratio=1.0):
    for split in ['train', 'dev', 'test']:
    # for split in ['dev']:
        split_path = os.path.join(video_dir, split)
        save_split_path = os.path.join(output_dir, split)
        for subject in os.listdir(split_path):
            subject_path = os.path.join(split_path, subject)
            save_subject_path = os.path.join(save_split_path, subject)
            videos = sorted(os.listdir(subject_path))
            # os.makedirs(os.path.dirname(save_subject_path), exist_ok=True)
            n_selected = max(1, int(len(videos) * ratio))
            selected_videos = videos[:n_selected]

            for video_file in tqdm(selected_videos, desc=f"{split}/{subject}"):
                video_path = os.path.join(subject_path, video_file)
                name_string = video_file.split(".")
                save_video_path = os.path.join(save_subject_path, name_string[0]+ '.npy')
                process_video(video_path, save_video_path)

video_directory = '/root/autodl-tmp/data/CE-CSL/video/'
output_directory = '/root/autodl-tmp/data/pose/'
process_all_videos(video_directory, output_directory)
