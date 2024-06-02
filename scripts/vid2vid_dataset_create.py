# %%
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

# update below with the directory containing the mp4 videos
vid_directory = "/replace/with/directory_with_mp4_videos"
# Note: If using president dataset, set zoomin to False
zoomin = True

csv_file = "../dataset/zoomin_info.csv"
if not zoomin:
    csv_file = "../dataset/pres_info.csv"

save_directory = "../dataset/zoomin/dezoom/"
mediapipe_directory = "../dataset/zoomin/processed/"
csv_file = pd.read_csv(csv_file)


def visualize_landmarks(rgb_image, np_landmarks):
    annotated_images = []
    
    for frame_landmarks in np_landmarks:
        annotated_image = np.copy(rgb_image)

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in frame_landmarks
            ])
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
    
        annotated_images.append(annotated_image)

    return annotated_images


# %%
files_to_starts = {}

for filename in tqdm(os.listdir(mediapipe_directory)):
    vid_id, _, start = filename.split("_$_")
    start = int(start.split(".")[0])

    fps = int(csv_file[csv_file["vid_id"] == vid_id]["fps"].fillna(0).values[0])
    view = csv_file[csv_file["vid_id"] == vid_id]["view"].values[0]
    # if fps != 25 or view != "off":
    if fps != 25 or view != "on":
        continue

    if vid_id not in files_to_starts:
        files_to_starts[vid_id] = []

    files_to_starts[vid_id].append(start)

# sort the starts and remove duplicates
for vid_id in tqdm(files_to_starts):
    files_to_starts[vid_id] = list(set(files_to_starts[vid_id]))
    files_to_starts[vid_id].sort()

# list of all keys
files_to_process = list(files_to_starts.keys()) 


# %%
def process_video(vid_id):
    starts = files_to_starts[vid_id]

    video_name = csv_file[csv_file["vid_id"] == vid_id]["file_name"].values[0]
    video_path = os.path.join(vid_directory, video_name)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # load frames
    for start in starts:

        # load mediapipe data
        mp_left_path = os.path.join(mediapipe_directory, vid_id + "_$_left_$_" + str(start) + ".npy")
        mp_right_path = os.path.join(mediapipe_directory, vid_id + "_$_right_$_" + str(start) + ".npy")
        mp_left = np.load(mp_left_path)
        mp_right = np.load(mp_right_path)
        size = mp_left.shape[0]

        # left bounding box is the widest box in the left mediapipe data
        x_left = np.min(mp_left[:, :, 0]) * width
        x_right = np.max(mp_left[:, :, 0]) * width
        y_top = np.min(mp_left[:, :, 1]) * height
        y_bottom = np.max(mp_left[:, :, 1]) * height
        x_left = max(0, x_left)
        x_right = min(width, x_right)
        y_top = max(0, y_top)
        y_bottom = min(height, y_bottom)
        left_bbox = [x_left, x_right, y_top, y_bottom]

        # right bounding box is the widest box in the right mediapipe data
        x_left = np.min(mp_right[:, :, 0]) * width
        x_right = np.max(mp_right[:, :, 0]) * width
        y_top = np.min(mp_right[:, :, 1]) * height
        y_bottom = np.max(mp_right[:, :, 1]) * height
        x_left = max(0, x_left)
        x_right = min(width, x_right)
        y_top = max(0, y_top)
        y_bottom = min(height, y_bottom)
        right_bbox = [x_left, x_right, y_top, y_bottom]

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        # read the next size frames
        left_frames = []
        right_frames = []
        left_frame_shape = (int(left_bbox[3] - left_bbox[2]), int(left_bbox[1] - left_bbox[0]), 3)
        right_frame_shape = (int(right_bbox[3] - right_bbox[2]), int(right_bbox[1] - right_bbox[0]), 3)
        for i in range(size):
            _, frame = cap.read()
            left_frame = frame[int(left_bbox[2]):int(left_bbox[3]), int(left_bbox[0]):int(left_bbox[1])]
            right_frame = frame[int(right_bbox[2]):int(right_bbox[3]), int(right_bbox[0]):int(right_bbox[1])]
            left_frames.append(left_frame)
            right_frames.append(right_frame)

        mp_left = visualize_landmarks(np.zeros((height//4, width//4, 3)), mp_left)
        mp_right = visualize_landmarks(np.zeros((height//4, width//4, 3)), mp_right)

        # save the data
        save_path_left_images = os.path.join(save_directory, "images", vid_id + "_$_left_$_" + str(start))
        save_path_right_images = os.path.join(save_directory, "images", vid_id + "_$_right_$_" + str(start))
        os.makedirs(save_path_left_images, exist_ok=True)
        os.makedirs(save_path_right_images, exist_ok=True)

        save_path_left_mp = os.path.join(save_directory, "mesh_images", vid_id + "_$_left_$_" + str(start))
        save_path_right_mp = os.path.join(save_directory, "mesh_images", vid_id + "_$_right_$_" + str(start))
        os.makedirs(save_path_left_mp, exist_ok=True)
        os.makedirs(save_path_right_mp, exist_ok=True)

        error = False

        for i in range(size):
            try:
                cv2.imwrite(os.path.join(save_path_left_images, str(i).zfill(5) + ".jpg"), left_frames[i])
                cv2.imwrite(os.path.join(save_path_right_images, str(i).zfill(5) + ".jpg"), right_frames[i])

                # save mp as images
                left_frame = mp_left[i][int(left_bbox[2]/4):int(left_bbox[3]/4), int(left_bbox[0]/4):int(left_bbox[1]/4)]
                right_frame = mp_right[i][int(right_bbox[2]/4):int(right_bbox[3]/4), int(right_bbox[0]/4):int(right_bbox[1]/4)]
                cv2.imwrite(os.path.join(save_path_left_mp, str(i).zfill(5) + ".jpg"), left_frame)
                cv2.imwrite(os.path.join(save_path_right_mp, str(i).zfill(5) + ".jpg"), right_frame)

            except Exception as e:
                error = True
                print(e)

        if error:
            print("Error in ", vid_id, start)
            print("Left: ", left_bbox, "Right: ", right_bbox, "Width: ", width, "Height: ", height)

    cap.release()

# %%
if __name__ == '__main__':
    num_threads = 24
    with Pool(num_threads) as pool:
        list(tqdm(pool.imap(process_video, files_to_process), total=len(files_to_process)))





