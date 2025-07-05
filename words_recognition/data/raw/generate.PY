import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

INPUT_DIR = 'dataset2'
OUTPUT_DIR = 'keypoints'

os.makedirs(OUTPUT_DIR, exist_ok=True)

with mp_holistic.Holistic(static_image_mode=False) as holistic:
    for label in os.listdir(INPUT_DIR):
        class_in = os.path.join(INPUT_DIR, label)
        class_out = os.path.join(OUTPUT_DIR, label)
        os.makedirs(class_out, exist_ok=True)

        for vid in os.listdir(class_in):
            cap = cv2.VideoCapture(os.path.join(class_in, vid))
            frames = []
            while True:
                ret, img = cap.read()
                if not ret: break
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = holistic.process(img_rgb)

                # concat pose+left hand+ right hand landmarks
                keypts = []
                if res.pose_landmarks:
                    for lm in res.pose_landmarks.landmark:
                        keypts += [lm.x, lm.y, lm.z]
                else:
                    keypts += [0]*33*3

                if res.left_hand_landmarks:
                    for lm in res.left_hand_landmarks.landmark:
                        keypts += [lm.x, lm.y, lm.z]
                else:
                    keypts += [0]*21*3

                if res.right_hand_landmarks:
                    for lm in res.right_hand_landmarks.landmark:
                        keypts += [lm.x, lm.y, lm.z]
                else:
                    keypts += [0]*21*3

                frames.append(keypts)

            cap.release()
            arr = np.array(frames)  # shape: (n_frames,  (33+21+21)*3 )
            np.save(os.path.join(class_out, vid.replace('.mp4', '.npy')), arr)
