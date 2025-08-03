import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

video_dir = Path("../data/video")
input_dir = Path("../data/pose")
output_dir = Path("../data/normpose")

def get_person_bbox_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return x, y, w, h

def normalize_pose(pose, bbox):
    """
    pose: np.array shape (num_keypoints, 3), 3为(x,y,z)或(x,y,visibility)
    bbox: (x, y, w, h)
    归一化 x,y 坐标到 [0,1] 区间
    """
    x, y, w, h = bbox
    pose_norm = pose.copy()
    if w == 0 or h == 0:
        return pose_norm

    pose_norm[:, 0] = (pose[:, 0] - x) / w
    pose_norm[:, 1] = (pose[:, 1] - y) / h

    return pose_norm

def process_video(video_path, pose_path, save_path):
    """
    video_path: 原视频路径
    pose_path: 对应npy pose路径，形状 (frames, keypoints, 3)
    save_path: 归一化后pose的保存路径
    """
    cap = cv2.VideoCapture(video_path)
    poses = np.load(pose_path)

    normalized_poses = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(poses):
            break

        bbox = get_person_bbox_from_frame(frame)
        if bbox is None:
            normalized_poses.append(poses[frame_idx])
        else:
            pose_norm = normalize_pose(poses[frame_idx], bbox)
            normalized_poses.append(pose_norm)

        frame_idx += 1

    cap.release()

    normalized_poses = np.array(normalized_poses)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, normalized_poses)

def process_all_videos(video_dir, pose_dir, output_dir, ratio=0.2):
    for split in ['train', 'dev', 'test']:
        split_path = os.path.join(video_dir, split)
        pose_split_path = os.path.join(pose_dir, split)
        save_split_path = os.path.join(output_dir, split)

        for subject in os.listdir(split_path):
            subject_path = os.path.join(split_path, subject)
            pose_subject_path = os.path.join(pose_split_path, subject)
            save_subject_path = os.path.join(save_split_path, subject)

            # videos = sorted(os.listdir(subject_path))
            # n_selected = max(1, int(len(videos) * ratio))
            # selected_videos = videos[:n_selected]
            poses = sorted(os.listdir(subject_path))

            for video_file in tqdm(selected_videos, desc=f"{split}/{subject}"):
                video_path = os.path.join(subject_path, video_file)
                name_string = video_file.split(".")[0]
                pose_path = os.path.join(pose_subject_path, name_string + '.npy')
                save_video_path = os.path.join(save_subject_path, name_string + '.npy')

                process_video(video_path, pose_path, save_video_path)

process_all_videos(video_dir, input_dir, output_dir)
