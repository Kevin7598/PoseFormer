import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from multiprocessing import Pool, cpu_count

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

def normalize_sequence(sequence):
    """
    Normalizes a pose sequence.
    1. Translates the sequence so the midpoint of the shoulders is the origin.
    2. Scales the sequence to have a unified mean and variance (Z-score normalization).
    """
    if sequence.shape[0] == 0:  # Handle empty sequences
        return sequence

    # Use shoulder landmarks for centering (indices 11 and 12 in MediaPipe pose)
    left_shoulder = sequence[:, 11]
    right_shoulder = sequence[:, 12]

    # Calculate the origin (midpoint of shoulders) for each frame
    origin = (left_shoulder + right_shoulder) / 2.0
    
    # Add a dimension to origin for broadcasting and translate
    sequence = sequence - origin[:, np.newaxis, :]

    # Find all non-zero keypoints to calculate stats, avoiding padding
    non_zero_mask = sequence.sum(axis=2) != 0
    if not np.any(non_zero_mask): # If all points are zero, return as is
        return sequence
        
    non_zero_points = sequence[non_zero_mask]

    # Calculate mean and std dev for scaling
    mean = np.mean(non_zero_points, axis=0)
    std = np.std(non_zero_points, axis=0)
    std[std == 0] = 1e-6 # Avoid division by zero

    # Apply scaling to all points
    sequence[non_zero_mask] = (non_zero_points - mean) / std
    
    return sequence

def process_video_worker(task):
    """
    Worker function to process a single video. Initializes MediaPipe here.
    """
    video_path, save_path = task
    
    # Initialize MediaPipe Holistic within each worker
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

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
    holistic.close()

    sequence = np.array(sequence, dtype=np.float32)
    
    # Apply normalization to the extracted sequence
    sequence = normalize_sequence(sequence)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, sequence)
    return f"Processed {video_path}"

def main():
    video_directory = '/root/autodl-tmp/data/CE-CSL/video/'
    output_directory = '/root/autodl-tmp/data/pose-norm/'
    ratio = 1.0

    # 1. Collect all video processing tasks
    tasks = []
    for split in ['train', 'dev', 'test']:
        split_path = os.path.join(video_directory, split)
        save_split_path = os.path.join(output_directory, split)
        for subject in os.listdir(split_path):
            subject_path = os.path.join(split_path, subject)
            save_subject_path = os.path.join(save_split_path, subject)
            videos = sorted(os.listdir(subject_path))
            n_selected = max(1, int(len(videos) * ratio))
            selected_videos = videos[:n_selected]

            for video_file in selected_videos:
                video_path = os.path.join(subject_path, video_file)
                name_string = video_file.split(".")[0]
                save_video_path = os.path.join(save_subject_path, name_string + '.npy')
                tasks.append((video_path, save_video_path))

    # 2. Use a multiprocessing Pool to process tasks in parallel
    # Using more workers than GPUs is fine, as the OS and CUDA will schedule them.
    # A good starting point is the number of CPU cores.
    num_workers = cpu_count()
    print(f"Starting pose extraction with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        # Use tqdm to show progress for the multiprocessing tasks
        list(tqdm(pool.imap_unordered(process_video_worker, tasks), total=len(tasks)))

    print("All videos have been processed and normalized.")

if __name__ == '__main__':
    main()