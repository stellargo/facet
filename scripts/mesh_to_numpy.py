import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import math
from multiprocessing import Pool
import sys
import re


# Note: If using president dataset, set zoomin to False
zoomin = True

csv_file = "../dataset/zoomin_info.csv"
if not zoomin:
    csv_file = "../dataset/pres_info.csv"
nose = 1
left_ear = 234
right_ear = 454
default_inter_ear_distance = 0.14717439
stride = 192
perform_rotation = False
normalize = False
chunk_size = 176
stride_25 = 176
stride_30 = 211
num_threads = 24
mesh_dir = "../dataset/zoomin/mesh/"
if not zoomin:
    mesh_dir = "../dataset/president/mesh/"
output_dir = "../dataset/zoomin/processed/"
if not zoomin:
    output_dir = "../dataset/president/processed/"


def inter_ear_distance(np_landmarks):
   return np.linalg.norm(np_landmarks[left_ear] - np_landmarks[right_ear])


def landmarks_to_np(landmarks):
  np_landmarks = np.zeros((478, 3)).astype(np.float32)

  for i, landmark in enumerate(landmarks):
    np_landmarks[i][0] = float(landmark.x)
    np_landmarks[i][1] = float(landmark.y)
    np_landmarks[i][2] = float(landmark.z)

  return np_landmarks


def blendshapes_to_np(blendshapes):
  np_blendshapes = np.zeros((52, 1)).astype(np.float32)

  for i, blendshape in enumerate(blendshapes):
    np_blendshapes[i][0] = float(blendshape.score)

  return np_blendshapes



def landmarks_normalize(np_landmarks, np_median_landmarks, median_transformation):
    
  # Extract rotation vector and find its inverse
  inv_median_transform = np.linalg.inv(median_transformation[:3,:3])

  # Perform inverse rotation
  if perform_rotation:
    np_landmarks = np.matmul(np_landmarks, inv_median_transform)

  # Scale to inter-ear distance
  scaling_factor = default_inter_ear_distance / inter_ear_distance(np_median_landmarks)
  np_landmarks *= scaling_factor
  
  # Perform inverse translation on median position and use it for translation to 
  # the origin using nose coordinates.
  if perform_rotation:
    np_median_landmarks = np.matmul(np_median_landmarks, inv_median_transform)
  np_median_landmarks *= scaling_factor
  np_landmarks -= np.expand_dims(np_median_landmarks[nose], 0)
  np_landmarks += np.array([0.5, 0.5, 0.5])
  
  return np_landmarks


def process_blendshapes(file_landmarks, chunk_size, save_dir, file_id, fps):
    i = 0
    stride = stride_25 if fps == 25 else stride_30
    ratio_num = 1 if fps == 25 else 6
    ratio_den = 1 if fps == 25 else 5

    while i < len(file_landmarks) - stride:
        chunk_left = []
        chunk_right = []
        np_blendshapes_left = []
        np_blendshapes_right = []
        contiguous = True

        for j in range(i, i + stride):
            if len(file_landmarks[j].face_landmarks) != 2:
                contiguous = False
                break

            if file_landmarks[j].face_landmarks[0][nose].x < file_landmarks[j].face_landmarks[1][nose].x:
                chunk_left.append(file_landmarks[j].face_blendshapes[0])
                chunk_right.append(file_landmarks[j].face_blendshapes[1])
            else:
                chunk_left.append(file_landmarks[j].face_blendshapes[1])
                chunk_right.append(file_landmarks[j].face_blendshapes[0])
        
        if not contiguous:
            i += 1
            continue

        for j in range(chunk_size):
            j_eq = (j * ratio_num) / ratio_den
            floor = math.floor(j_eq)
            ceil = math.ceil(j_eq)

            # If floor is close to j_eq, it should be weighted more.
            # This means that we can look at how close ceil is to j_eq
            floor_weight = round(ceil - j_eq, 1) # for removing numerical instability (probably useless)
            ceil_weight = round(j_eq - floor, 1) # for removing numerical instability (probably useless)

            # In case ceil and floor are the same, just use floor.
            if floor == ceil:
                floor_weight = 1

            # Calculate landmarks weighted by ratio
            blendshapes_left = blendshapes_to_np(chunk_left[floor]) * floor_weight + blendshapes_to_np(chunk_left[ceil]) * ceil_weight
            blendshapes_right = blendshapes_to_np(chunk_right[floor]) * floor_weight + blendshapes_to_np(chunk_right[ceil]) * ceil_weight

            np_blendshapes_left.append(blendshapes_left)
            np_blendshapes_right.append(blendshapes_right)

        np.save(os.path.join(save_dir, file_id + '_$_left_$_' + str(i) + '.npy'), 
                np.stack(np_blendshapes_left, axis=0))
        np.save(os.path.join(save_dir, file_id + '_$_right_$_' + str(i) + '.npy'), 
                np.stack(np_blendshapes_right, axis=0))

        i += stride


def process_landmarks_v2(file_landmarks, chunk_size, save_dir, file_id, fps):
    i = 0
    stride = stride_25 if fps == 25 else stride_30
    ratio_num = 1 if fps == 25 else 6
    ratio_den = 1 if fps == 25 else 5

    while i < len(file_landmarks) - stride:
        chunk_left = []
        chunk_right = []
        np_landmarks_left = []
        np_landmarks_right = []
        contiguous = True
        for j in range(i, i + stride):
            if len(file_landmarks[j].face_landmarks) != 2:
                contiguous = False
                break

            if file_landmarks[j].face_landmarks[0][nose].x < file_landmarks[j].face_landmarks[1][nose].x:
                chunk_left.append(file_landmarks[j].face_landmarks[0])
                chunk_right.append(file_landmarks[j].face_landmarks[1])
            else:
                chunk_left.append(file_landmarks[j].face_landmarks[1])
                chunk_right.append(file_landmarks[j].face_landmarks[0])
        
        if not contiguous:
            i += 1
            continue

        for j in range(chunk_size):
            j_eq = (j * ratio_num) / ratio_den
            floor = math.floor(j_eq)
            ceil = math.ceil(j_eq)

            # If floor is close to j_eq, it should be weighted more.
            # This means that we can look at how close ceil is to j_eq
            floor_weight = round(ceil - j_eq, 1) # for removing numerical instability (probably useless)
            ceil_weight = round(j_eq - floor, 1) # for removing numerical instability (probably useless)

            # In case ceil and floor are the same, just use floor.
            if floor == ceil:
                floor_weight = 1

            # Calculate landmarks weighted by ratio
            landmarks_left = landmarks_to_np(chunk_left[floor]) * floor_weight + landmarks_to_np(chunk_left[ceil]) * ceil_weight
            landmarks_right = landmarks_to_np(chunk_right[floor]) * floor_weight + landmarks_to_np(chunk_right[ceil]) * ceil_weight

            np_landmarks_left.append(landmarks_left)
            np_landmarks_right.append(landmarks_right)

        np.save(os.path.join(save_dir, file_id + '_$_left_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks_left, axis=0))
        np.save(os.path.join(save_dir, file_id + '_$_right_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks_right, axis=0))

        i += stride


def process_landmarks_president(file_landmarks, chunk_size, save_dir, file_id, fps):
    i = 0

    stride = 176 if fps == 24 else 220
    ratio_num = 1 if fps == 24 else 5
    ratio_den = 1 if fps == 24 else 4

    if fps != 24 and fps != 30:
        print(file_id, " Invalid fps: " + str(fps))
        return

    while i < len(file_landmarks) - stride:
        chunk = []
        np_landmarks = []
        contiguous = True
        for j in range(i, i + stride):
            if len(file_landmarks[j].face_landmarks) != 1:
                contiguous = False
                break

            chunk.append(file_landmarks[j].face_landmarks[0])
        
        if not contiguous:
            i += 1
            continue

        for j in range(chunk_size):
            j_eq = (j * ratio_num) / ratio_den
            floor = math.floor(j_eq)
            ceil = math.ceil(j_eq)

            # If floor is close to j_eq, it should be weighted more.
            # This means that we can look at how close ceil is to j_eq
            floor_weight = round(ceil - j_eq, 1) # for removing numerical instability (probably useless)
            ceil_weight = round(j_eq - floor, 1) # for removing numerical instability (probably useless)

            # In case ceil and floor are the same, just use floor.
            if floor == ceil:
                floor_weight = 1

            # Calculate landmarks weighted by ratio
            landmarks = landmarks_to_np(chunk[floor]) * floor_weight + landmarks_to_np(chunk[ceil]) * ceil_weight

            np_landmarks.append(landmarks)

        np.save(os.path.join(save_dir, file_id + '_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks, axis=0))

        i += stride


def process_landmarks_single_frame(file_landmarks, chunk_size, save_dir, file_id):
    i = 0

    while i < len(file_landmarks):
        np_landmarks_left = []
        np_landmarks_right = []
        j = i
        while j < len(file_landmarks) and len(np_landmarks_left) < chunk_size:
            if len(file_landmarks[j].face_landmarks) != 2:
                j += 1
                continue

            if file_landmarks[j].face_landmarks[0][nose].x < file_landmarks[j].face_landmarks[1][nose].x:
                np_landmarks_left.append(landmarks_to_np(file_landmarks[j].face_landmarks[0]))
                np_landmarks_right.append(landmarks_to_np(file_landmarks[j].face_landmarks[1]))
            else:
                np_landmarks_left.append(landmarks_to_np(file_landmarks[j].face_landmarks[1]))
                np_landmarks_right.append(landmarks_to_np(file_landmarks[j].face_landmarks[0]))
            
            j += 1

        if len(np_landmarks_left) < chunk_size:
            return

        np.save(os.path.join(save_dir, file_id + '_$_left_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks_left, axis=0))
        np.save(os.path.join(save_dir, file_id + '_$_right_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks_right, axis=0))

        i = j


def process_landmarks_inverse(file_landmarks, chunk_size, save_dir, file_id, fps):
    i = 0
    stride = stride_25 if fps == 25 else stride_30
    ratio_num = 1 if fps == 25 else 6
    ratio_den = 1 if fps == 25 else 5

    while i < len(file_landmarks) - stride:
        chunk_left = []
        chunk_right = []
        chunk_left_transformation = []
        chunk_right_transformation = []
        np_landmarks_left = []
        np_landmarks_right = []
        contiguous = True
        for j in range(i, i + stride):
            if len(file_landmarks[j].face_landmarks) != 2:
                contiguous = False
                break

            if file_landmarks[j].face_landmarks[0][nose].x < file_landmarks[j].face_landmarks[1][nose].x:
                chunk_left.append(file_landmarks[j].face_landmarks[0])
                chunk_left_transformation.append(file_landmarks[j].facial_transformation_matrixes[0])
                chunk_right.append(file_landmarks[j].face_landmarks[1])
                chunk_right_transformation.append(file_landmarks[j].facial_transformation_matrixes[1])
            else:
                chunk_left.append(file_landmarks[j].face_landmarks[1])
                chunk_left_transformation.append(file_landmarks[j].facial_transformation_matrixes[1])
                chunk_right.append(file_landmarks[j].face_landmarks[0])
                chunk_right_transformation.append(file_landmarks[j].facial_transformation_matrixes[0])
        
        if not contiguous:
            i += 1
            continue

        for j in range(chunk_size):
            j_eq = (j * ratio_num) / ratio_den
            floor = math.floor(j_eq)
            ceil = math.ceil(j_eq)

            # If floor is close to j_eq, it should be weighted more.
            # This means that we can look at how close ceil is to j_eq
            floor_weight = round(ceil - j_eq, 1) # for removing numerical instability (probably useless)
            ceil_weight = round(j_eq - floor, 1) # for removing numerical instability (probably useless)

            # In case ceil and floor are the same, just use floor.
            if floor == ceil:
                floor_weight = 1

            left_floor = landmarks_to_np(chunk_left[floor])
            left_ceil = landmarks_to_np(chunk_left[ceil])
            right_floor = landmarks_to_np(chunk_right[floor])
            right_ceil = landmarks_to_np(chunk_right[ceil])

            # Extract rotation vector and find its inverse, then perform inverse rotation
            inv_left_floor_transform = np.linalg.inv(chunk_left_transformation[floor][:3,:3])
            inv_left_ceil_transform = np.linalg.inv(chunk_left_transformation[ceil][:3,:3])
            inv_right_floor_transform = np.linalg.inv(chunk_right_transformation[floor][:3,:3])
            inv_right_ceil_transform = np.linalg.inv(chunk_right_transformation[ceil][:3,:3])

            left_floor = np.matmul(left_floor, inv_left_floor_transform)
            left_ceil = np.matmul(left_ceil, inv_left_ceil_transform)
            right_floor = np.matmul(right_floor, inv_right_floor_transform)
            right_ceil = np.matmul(right_ceil, inv_right_ceil_transform)

            # Calculate landmarks weighted by ratio
            landmarks_left = left_floor * floor_weight + left_ceil * ceil_weight
            landmarks_right = right_floor * floor_weight + right_ceil * ceil_weight

            np_landmarks_left.append(landmarks_left)
            np_landmarks_right.append(landmarks_right)

        np.save(os.path.join(save_dir, file_id + '_$_left_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks_left, axis=0))
        np.save(os.path.join(save_dir, file_id + '_$_right_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks_right, axis=0))

        i += stride


def process_landmarks(file_landmarks, chunk_size, save_dir, file_id):
    i = 0
    while i < len(file_landmarks) - chunk_size:
        chunk = []
        contiguous = True
        for j in range(i, i + chunk_size):
            if len(file_landmarks[j].face_landmarks) != 2:
                contiguous = False
                break

            chunk.append(file_landmarks[j])
        
        if not contiguous:
            i += 1
            continue
        
        np_landmarks_left = []
        np_landmarks_right = []

        if file_landmarks[i].face_landmarks[0][nose].x < file_landmarks[i].face_landmarks[1][nose].x:
          median_transformation_matrix_left = file_landmarks[i].facial_transformation_matrixes[0]
          median_landmarks_left = landmarks_to_np(file_landmarks[i].face_landmarks[0])
          median_transformation_matrix_right = file_landmarks[i].facial_transformation_matrixes[1]
          median_landmarks_right = landmarks_to_np(file_landmarks[i].face_landmarks[1])

        else:
          median_transformation_matrix_left = file_landmarks[i].facial_transformation_matrixes[1]
          median_landmarks_left = landmarks_to_np(file_landmarks[i].face_landmarks[1])
          median_transformation_matrix_right = file_landmarks[i].facial_transformation_matrixes[0]
          median_landmarks_right = landmarks_to_np(file_landmarks[i].face_landmarks[0])

        for j in range(i, i + chunk_size):
            if file_landmarks[j].face_landmarks[0][nose].x < file_landmarks[j].face_landmarks[1][nose].x:
              landmarks_left = landmarks_to_np(file_landmarks[j].face_landmarks[0])
              landmarks_right = landmarks_to_np(file_landmarks[j].face_landmarks[1])
            else:
              landmarks_left = landmarks_to_np(file_landmarks[j].face_landmarks[1])
              landmarks_right = landmarks_to_np(file_landmarks[j].face_landmarks[0])

            if normalize:
              landmarks_left = landmarks_normalize(landmarks_left, median_landmarks_left, median_transformation_matrix_left)
              landmarks_right = landmarks_normalize(landmarks_right, median_landmarks_right, median_transformation_matrix_right)

            np_landmarks_left.append(landmarks_left)
            np_landmarks_right.append(landmarks_right)

        np.save(os.path.join(save_dir, file_id + '_$_left_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks_left, axis=0))
        np.save(os.path.join(save_dir, file_id + '_$_right_$_' + str(i) + '.npy'), 
                np.stack(np_landmarks_right, axis=0))

        i += stride

def process_file(filename):
    if not filename.endswith(".mp4"):
        return
    
    if zoomin:
        file_id = re.findall(r'\[(.*?)\]', filename)[0]
    else:
        file_id = filename.split("_@@@_")[1]
        file_id = file_id.split(".mp4")[0]
    
    # Skip if not 2 participants
    if zoomin and int(csv_data[csv_data["vid_id"] == file_id]["participants"].fillna(0).values[0]) != 2:
        return

    i = 0
    file_landmarks = []
    
    pkl_file = os.path.join(mesh_dir, filename[:-4] + "_$_" + str(i) + ".pkl")
    while (os.path.isfile(pkl_file)):
        with open(pkl_file, 'rb') as f:
            mp_list = pickle.load(f)
        file_landmarks.extend(mp_list)

        i += 1
        pkl_file = os.path.join(mesh_dir, filename[:-4] + "_$_" + str(i) + ".pkl")

    fps = int(csv_data[csv_data["vid_id"] == file_id]["fps"].fillna(0).values[0])

    if zoomin:
        process_landmarks_v2(file_landmarks, chunk_size, output_dir, file_id, fps)
    else:
        process_landmarks_president(file_landmarks, chunk_size, output_dir, file_id, fps)


if __name__ == '__main__':

    vid_directory = sys.argv[1]

    # create output_dir if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_data = pd.read_csv(csv_file)
    files_to_process = [filename for filename in os.listdir(vid_directory)]

    with Pool(num_threads) as pool:
        list(tqdm(pool.imap(process_file, files_to_process), total=len(files_to_process)))

