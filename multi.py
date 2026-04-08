import cv2

import mediapipe as mp

import os

import time

mp_face_mesh = mp.solutions.face_mesh
print(mp_face_mesh,"this is mash face mash")

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)

output_dir = "multi_face_captures"

os.makedirs(output_dir, exist_ok=True)

def capture_on_multiple_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    print(cap.isOpened(),"cap output")
    frame_number = 0

    face_detected = False

    capture = False  

    while cap.isOpened():

        ret, frame = cap.read()
        print("yes i am working")
        if not ret:

            break

        frame_number += 1


        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print(rgb,"this is rgb")

        results = face_mesh.process(rgb) 
        # print(len(results.multi_face_landmarks) ,"this is length of multi_face_landmarks")   

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 1 and capture is False:

            
            capture = True

            timestamp = int(time.time() * 1000)

            img_path = os.path.join(output_dir, f"multi_face_{timestamp}.jpg")

            cv2.imwrite(img_path, frame)

            print(f"[INFO] Captured frame saved at {img_path}")

            face_detected = True

        else:

            print(f"[Frame {frame_number}] Single or no face detected.")


    cap.release()

    return face_detected


if __name__ == "__main__":

    video_path = r"C:\Users\Dell\Downloads\0_Business_Meeting_3840x2160.mp4" 

    result = capture_on_multiple_faces(video_path)

    print("Multiple face detection result:", result)