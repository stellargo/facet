import numpy as np
import mediapipe as mp
import cv2
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2
import sys
from multiprocessing import Pool


# Note: If using president dataset, set zoomin to False
zoomin = True

num_threads = 16
NUM_FACES = 1
mesh_dir = '../dataset/zoomin/mesh'
if not zoomin:
    mesh_dir = '../dataset/president/mesh'


def process_video_file(filename):

    input_file = os.path.join(directory, filename)
    output_file = os.path.join(mesh_dir, filename[:-4] + '.pkl')
    cap = cv2.VideoCapture(input_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmark_results = []
    part = 0

    output_file = os.path.join(mesh_dir, filename[:-4] + '_$_' + str(part) + '.pkl')
    if os.path.exists(output_file):
        return landmark_results

    part += 1
    with open(output_file, 'wb') as f:
        pickle.dump(landmark_results, f)

    with FaceLandmarker.create_from_options(options) as landmarker:

        for i in range(0, length):
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            face_landmarker_result = landmarker.detect(mp_image)
            landmark_results.append(face_landmarker_result)

            if (i % 5000 == 0):
                output_file = os.path.join(mesh_dir, filename[:-4] + '_$_' + str(part) + '.pkl')
                part += 1
                with open(output_file, 'wb') as f:
                    pickle.dump(landmark_results, f)
                landmark_results = []
            
    output_file = os.path.join(mesh_dir, filename[:-4] + '_$_' + str(part) + '.pkl')
    part += 1
    with open(output_file, 'wb') as f:
        pickle.dump(landmark_results, f)
    landmark_results = []
    cap.release()
    return landmark_results


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=NUM_FACES)




if __name__ == "__main__":
    # get directory from command line
    directory = sys.argv[1]

    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    video_files = [filename for filename in os.listdir(directory) if filename.endswith(".mp4")]

    with Pool(processes=num_threads) as pool:
        results = list(tqdm(pool.imap(process_video_file, video_files), total=len(video_files)))