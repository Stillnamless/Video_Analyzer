import cv2

import mediapipe as mp

import numpy as np

from scipy.spatial import distance as dist

# from eye_blink import count_blinks

# from eye_direction import detect_invalid_eye_direction

import os

from ultralytics import YOLO

import time


model = YOLO("yolov8n.pt")  


# eye_capture_dir = "eye_captures"

# os.makedirs(eye_capture_dir, exist_ok=True)


multi_face_dir = "captures"

os.makedirs(multi_face_dir, exist_ok=True)


phone_detected_dir = "phone_detected_frames"

os.makedirs(phone_detected_dir, exist_ok=True)


head_movement_dir = "head_movement"

os.makedirs(head_movement_dir, exist_ok=True)




mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)


EAR_THRESHOLD = 0.23 

CONSEC_FRAMES = 3     


LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_EYE = [33, 160, 158, 133, 153, 144]




NOSE_TIP_IDX = 1  # MediaPipe FaceMesh nose tip index

MOVEMENT_THRESHOLD = 15  # Pixels

CONSECUTIVE_MOVING_FRAMES = 10  # Number of moving frames before triggering






def euclidean_distance(p1, p2):

    return np.linalg.norm(np.array(p1) - np.array(p2))






def calculate_ear(landmarks, eye_indices, image_w, image_h):

    coords = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]

    vertical1 = dist.euclidean(coords[1], coords[5])

    vertical2 = dist.euclidean(coords[2], coords[4])

    horizontal = dist.euclidean(coords[0], coords[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear




def calculate_everything(video_path):

    if not os.path.exists(video_path):

        print(f"[ERROR] Video file {video_path} does not exist.")

        return None

    cap = cv2.VideoCapture(video_path)

    blink_count = 0

    is_eye_direction = False

    is_head_movement = False

    is_multiple_faces = False

    is_mobile = False

    blink_frame_counter = 0 

    total_frames = 0       

    invalid_gaze_count = 0

    prev_nose = None

    moving_counter = 0

    phone_detected = False

    capture = False

    capture_multiple = False  

    face_detected = False


    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            break


        total_frames += 1


        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print(rgb,"this is rgb")
        

        results = face_mesh.process(rgb)


        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 1 and capture is False :
            print(results.multi_face_landmarks, "multiple")

            capture = True

            timestamp = int(time.time() * 1000)

            img_path = os.path.join(multi_face_dir, f"multi_face_{timestamp}.jpg")

            cv2.imwrite(img_path, frame)

            print(f"[INFO] Captured frame saved at {img_path}")

            face_detected = True

        else:

            print(f" Single or no face detected.")
            
        if results.multi_face_landmarks:

            print("it is working")

            landmarks = results.multi_face_landmarks[0].landmark


            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)

            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)

            ear = (left_ear + right_ear) / 2.0


            if ear < EAR_THRESHOLD:

                blink_frame_counter += 1

            else:

                if blink_frame_counter >= CONSEC_FRAMES:

                    blink_count += 1

                blink_frame_counter = 0

    cap.release()

    return  face_detected, blink_count


if __name__ == "__main__":

    blink = calculate_everything(r"C:\Users\Dell\Downloads\5588091-hd_1080_1920_30fps.mp4") 

    print(f"Total blinks detected: {blink}")