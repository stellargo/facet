from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import cv2
import pandas as pd


def visualize_landmarks(rgb_image, np_landmarks, invert=False):
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
        
        if invert:
            annotated_image = 255 - annotated_image

            _YELLOW = (207, 207, 0)
            # convert all yellows to blue
            annotated_image[np.where((annotated_image == _YELLOW).all(axis=2))] = [0, 0, 255]

        # if main_fig:
        #     _WHITE = (224, 224, 224)
        #     _GREEN = (48, 255, 48)
        #     _BLUE = (48, 48, 255)

        #     annotated_image[np.where((annotated_image == _WHITE).all(axis=2))] = [0,0,0]
        #     annotated_image[np.where((annotated_image == _GREEN).all(axis=2))] = np.subtract((255,255,255), _GREEN)
        #     annotated_image[np.where((annotated_image == _BLUE).all(axis=2))] = np.subtract((255,255,255), _BLUE)
    
        annotated_images.append(annotated_image)

    return annotated_images


def landmarks_to_video(np_landmarks, output_path='./viz.mp4', width=1080, height=1080, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    landmark_frames = visualize_landmarks(np.zeros((height, width, 3)).astype(np.uint8), np_landmarks)
    for frame in landmark_frames:
        out.write(frame)
    out.release()


def create_video(frames, output_path='./viz.mp4', width=1080, height=1080, fps=25, title=""):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # add title text to all frames
    for i in range(len(frames)):
        cv2.putText(frames[i], title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    for frame in frames:
        out.write(frame)


    out.release()


def visualize_video(zebra, fake_horse, cycle_zebra,
                    horse, fake_zebra, cycle_horse,
                    dual_channel = False):

    if not dual_channel:
        fake_horse_samples = []
        fake_zebra_samples = []
        true_horse_samples = []
        true_zebra_samples = []
        cycle_horse_samples = []
        cycle_zebra_samples = []
    else:
        fake_horse_samples_left = []
        fake_zebra_samples_left = []
        true_horse_samples_left = []
        true_zebra_samples_left = []
        cycle_horse_samples_left = []
        cycle_zebra_samples_left = []
        fake_horse_samples_right = []
        fake_zebra_samples_right = []
        true_horse_samples_right = []
        true_zebra_samples_right = []
        cycle_horse_samples_right = []
        cycle_zebra_samples_right = []
    
    for i in range(176):
        if not dual_channel:
            fake_horse_samples.append(fake_horse[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples.append(fake_zebra[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            true_horse_samples.append(horse[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples.append(zebra[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            cycle_horse_samples.append(cycle_horse[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            cycle_zebra_samples.append(cycle_zebra[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
        else:
            fake_horse_samples_left.append(fake_horse[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples_left.append(fake_zebra[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            true_horse_samples_left.append(horse[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples_left.append(zebra[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            cycle_horse_samples_left.append(cycle_horse[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            cycle_zebra_samples_left.append(cycle_zebra[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            fake_horse_samples_right.append(fake_horse[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples_right.append(fake_zebra[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            true_horse_samples_right.append(horse[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples_right.append(zebra[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            cycle_horse_samples_right.append(cycle_horse[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            cycle_zebra_samples_right.append(cycle_zebra[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)

    if not dual_channel:
        fake_horse_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples,
        )
        fake_zebra_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples,
        )
        true_horse_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples,
        )
        true_zebra_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples,
        )
        cycle_horse_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_horse_samples,
        )
        cycle_zebra_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_zebra_samples,
        )
        fake_horse_video = np.array(fake_horse_video).transpose((0, 3, 1, 2))
        fake_zebra_video = np.array(fake_zebra_video).transpose((0, 3, 1, 2))
        true_horse_video = np.array(true_horse_video).transpose((0, 3, 1, 2))
        true_zebra_video = np.array(true_zebra_video).transpose((0, 3, 1, 2))
        cycle_horse_video = np.array(cycle_horse_video).transpose((0, 3, 1, 2))
        cycle_zebra_video = np.array(cycle_zebra_video).transpose((0, 3, 1, 2))
    else:
        fake_horse_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples_left,
        )
        fake_zebra_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples_left,
        )
        true_horse_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples_left,
        )
        true_zebra_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples_left,
        )
        cycle_horse_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_horse_samples_left,
        )
        cycle_zebra_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_zebra_samples_left,
        )
        fake_horse_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples_right,
        )
        fake_zebra_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples_right,
        )
        true_horse_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples_right,
        )
        true_zebra_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples_right,
        )
        cycle_horse_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_horse_samples_right,
        )
        cycle_zebra_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_zebra_samples_right,
        )
        fake_horse_video_left = np.array(fake_horse_video_left).transpose((0, 3, 1, 2))
        fake_zebra_video_left = np.array(fake_zebra_video_left).transpose((0, 3, 1, 2))
        true_horse_video_left = np.array(true_horse_video_left).transpose((0, 3, 1, 2))
        true_zebra_video_left = np.array(true_zebra_video_left).transpose((0, 3, 1, 2))
        cycle_horse_video_left = np.array(cycle_horse_video_left).transpose((0, 3, 1, 2))
        cycle_zebra_video_left = np.array(cycle_zebra_video_left).transpose((0, 3, 1, 2))
        fake_horse_video_right = np.array(fake_horse_video_right).transpose((0, 3, 1, 2))
        fake_zebra_video_right = np.array(fake_zebra_video_right).transpose((0, 3, 1, 2))
        true_horse_video_right = np.array(true_horse_video_right).transpose((0, 3, 1, 2))
        true_zebra_video_right = np.array(true_zebra_video_right).transpose((0, 3, 1, 2))
        cycle_horse_video_right = np.array(cycle_horse_video_right).transpose((0, 3, 1, 2))
        cycle_zebra_video_right = np.array(cycle_zebra_video_right).transpose((0, 3, 1, 2))

    if not dual_channel:
        true_horse_video = np.concatenate((true_horse_video, fake_zebra_video, cycle_horse_video), axis=3)
        true_zebra_video = np.concatenate((true_zebra_video, fake_horse_video, cycle_zebra_video), axis=3)
    else:
        true_horse_video = np.concatenate((true_horse_video_left, true_horse_video_right, 
                                           fake_zebra_video_left, fake_zebra_video_right, 
                                           cycle_horse_video_left, cycle_horse_video_right), axis=3)
        true_zebra_video = np.concatenate((true_zebra_video_left, true_zebra_video_right,
                                           fake_horse_video_left, fake_horse_video_right,
                                           cycle_zebra_video_left, cycle_zebra_video_right), axis=3)

    return true_horse_video, true_zebra_video


def visualize_video_v1(zebra, fake_horse, cycle_zebra,
                       horse, fake_zebra, cycle_horse,
                       dual_channel = False, temporal_width = 176):

    if not dual_channel:
        fake_horse_samples = []
        fake_zebra_samples = []
        true_horse_samples = []
        true_zebra_samples = []
        cycle_horse_samples = []
        cycle_zebra_samples = []
    else:
        fake_horse_samples_left = []
        fake_zebra_samples_left = []
        true_horse_samples_left = []
        true_zebra_samples_left = []
        cycle_horse_samples_left = []
        cycle_zebra_samples_left = []
        fake_horse_samples_right = []
        fake_zebra_samples_right = []
        true_horse_samples_right = []
        true_zebra_samples_right = []
        cycle_horse_samples_right = []
        cycle_zebra_samples_right = []
    
    for i in range(temporal_width):
        if not dual_channel:
            fake_horse_samples.append(fake_horse[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples.append(fake_zebra[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            true_horse_samples.append(horse[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples.append(zebra[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            cycle_horse_samples.append(cycle_horse[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            cycle_zebra_samples.append(cycle_zebra[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
        else:
            fake_horse_samples_left.append(fake_horse[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples_left.append(fake_zebra[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            true_horse_samples_left.append(horse[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples_left.append(zebra[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            cycle_horse_samples_left.append(cycle_horse[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            cycle_zebra_samples_left.append(cycle_zebra[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            fake_horse_samples_right.append(fake_horse[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples_right.append(fake_zebra[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            true_horse_samples_right.append(horse[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples_right.append(zebra[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            cycle_horse_samples_right.append(cycle_horse[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            cycle_zebra_samples_right.append(cycle_zebra[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)

    if not dual_channel:
        fake_horse_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples,
        )
        fake_zebra_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples,
        )
        true_horse_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples,
        )
        true_zebra_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples,
        )
        cycle_horse_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_horse_samples,
        )
        cycle_zebra_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_zebra_samples,
        )
        fake_horse_video = np.array(fake_horse_video).transpose((0, 3, 1, 2))
        fake_zebra_video = np.array(fake_zebra_video).transpose((0, 3, 1, 2))
        true_horse_video = np.array(true_horse_video).transpose((0, 3, 1, 2))
        true_zebra_video = np.array(true_zebra_video).transpose((0, 3, 1, 2))
        cycle_horse_video = np.array(cycle_horse_video).transpose((0, 3, 1, 2))
        cycle_zebra_video = np.array(cycle_zebra_video).transpose((0, 3, 1, 2))
    else:
        fake_horse_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples_left,
        )
        fake_zebra_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples_left,
        )
        true_horse_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples_left,
        )
        true_zebra_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples_left,
        )
        cycle_horse_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_horse_samples_left,
        )
        cycle_zebra_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_zebra_samples_left,
        )
        fake_horse_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples_right,
        )
        fake_zebra_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples_right,
        )
        true_horse_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples_right,
        )
        true_zebra_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples_right,
        )
        cycle_horse_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_horse_samples_right,
        )
        cycle_zebra_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            cycle_zebra_samples_right,
        )
        fake_horse_video_left = np.array(fake_horse_video_left).transpose((0, 3, 1, 2))
        fake_zebra_video_left = np.array(fake_zebra_video_left).transpose((0, 3, 1, 2))
        true_horse_video_left = np.array(true_horse_video_left).transpose((0, 3, 1, 2))
        true_zebra_video_left = np.array(true_zebra_video_left).transpose((0, 3, 1, 2))
        cycle_horse_video_left = np.array(cycle_horse_video_left).transpose((0, 3, 1, 2))
        cycle_zebra_video_left = np.array(cycle_zebra_video_left).transpose((0, 3, 1, 2))
        fake_horse_video_right = np.array(fake_horse_video_right).transpose((0, 3, 1, 2))
        fake_zebra_video_right = np.array(fake_zebra_video_right).transpose((0, 3, 1, 2))
        true_horse_video_right = np.array(true_horse_video_right).transpose((0, 3, 1, 2))
        true_zebra_video_right = np.array(true_zebra_video_right).transpose((0, 3, 1, 2))
        cycle_horse_video_right = np.array(cycle_horse_video_right).transpose((0, 3, 1, 2))
        cycle_zebra_video_right = np.array(cycle_zebra_video_right).transpose((0, 3, 1, 2))

    if not dual_channel:
        true_horse_video = np.concatenate((true_horse_video, fake_zebra_video, cycle_horse_video), axis=3)
        true_zebra_video = np.concatenate((true_zebra_video, fake_horse_video, cycle_zebra_video), axis=3)
    else:
        true_horse_video = np.concatenate((true_horse_video_left, true_horse_video_right, 
                                           fake_zebra_video_left, fake_zebra_video_right, 
                                           cycle_horse_video_left, cycle_horse_video_right), axis=3)
        true_zebra_video = np.concatenate((true_zebra_video_left, true_zebra_video_right,
                                           fake_horse_video_left, fake_horse_video_right,
                                           cycle_zebra_video_left, cycle_zebra_video_right), axis=3)

    return true_horse_video, true_zebra_video


def visualize_video_v3(zebra, fake_horse, horse, fake_zebra,
                       fake_horse_t = [], fake_zebra_t = [],
                       dual_channel = False, temporal_width = 176):

    if not dual_channel:
        fake_horse_samples = []
        fake_zebra_samples = []
        true_horse_samples = []
        true_zebra_samples = []
    else:
        fake_horse_samples_left = []
        fake_zebra_samples_left = []
        true_horse_samples_left = []
        true_zebra_samples_left = []
        fake_horse_samples_right = []
        fake_zebra_samples_right = []
        true_horse_samples_right = []
        true_zebra_samples_right = []
    
    for i in range(temporal_width):
        if not dual_channel:
            fake_horse_samples.append(fake_horse[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples.append(fake_zebra[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            true_horse_samples.append(horse[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples.append(zebra[i].cpu().detach().numpy().reshape((478, 3)) * 3 + 0.5)
        else:
            fake_horse_samples_left.append(fake_horse[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples_left.append(fake_zebra[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            true_horse_samples_left.append(horse[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples_left.append(zebra[i].cpu().detach().numpy()[:478 * 3].reshape((478, 3)) * 3 + 0.5)
            fake_horse_samples_right.append(fake_horse[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            fake_zebra_samples_right.append(fake_zebra[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            true_horse_samples_right.append(horse[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)
            true_zebra_samples_right.append(zebra[i].cpu().detach().numpy()[478 * 3:].reshape((478, 3)) * 3 + 0.5)

    if not dual_channel:
        fake_horse_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples, invert=True,
        )
        fake_zebra_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples, invert=True,
        )
        true_horse_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples, invert=True,
        )
        true_zebra_video = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples, invert=True,
        )
        fake_horse_video = np.array(fake_horse_video).transpose((0, 3, 1, 2))
        fake_zebra_video = np.array(fake_zebra_video).transpose((0, 3, 1, 2))
        true_horse_video = np.array(true_horse_video).transpose((0, 3, 1, 2))
        true_zebra_video = np.array(true_zebra_video).transpose((0, 3, 1, 2))
    else:
        fake_horse_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples_left, invert=True,
        )
        fake_zebra_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples_left, invert=True,
        )
        true_horse_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples_left, invert=True,
        )
        true_zebra_video_left = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples_left, invert=True,
        )
        fake_horse_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_horse_samples_right, invert=True,
        )
        fake_zebra_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            fake_zebra_samples_right, invert=True,
        )
        true_horse_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_horse_samples_right, invert=True,
        )
        true_zebra_video_right = visualize_landmarks(
            np.zeros((1080, 1080, 3)).astype(np.uint8),
            true_zebra_samples_right, invert=True,
        )
        fake_horse_video_left = np.array(fake_horse_video_left).transpose((0, 3, 1, 2))
        fake_zebra_video_left = np.array(fake_zebra_video_left).transpose((0, 3, 1, 2))
        true_horse_video_left = np.array(true_horse_video_left).transpose((0, 3, 1, 2))
        true_zebra_video_left = np.array(true_zebra_video_left).transpose((0, 3, 1, 2))
        fake_horse_video_right = np.array(fake_horse_video_right).transpose((0, 3, 1, 2))
        fake_zebra_video_right = np.array(fake_zebra_video_right).transpose((0, 3, 1, 2))
        true_horse_video_right = np.array(true_horse_video_right).transpose((0, 3, 1, 2))
        true_zebra_video_right = np.array(true_zebra_video_right).transpose((0, 3, 1, 2))

    if not dual_channel:
        true_horse_video = np.concatenate((true_horse_video, fake_zebra_video), axis=3) # [176, 3, 1080, 2160]
        true_zebra_video = np.concatenate((true_zebra_video, fake_horse_video), axis=3) # [176, 3, 1080, 2160]
    else:
        true_horse_video = np.concatenate((true_horse_video_left, true_horse_video_right, 
                                           fake_zebra_video_left, fake_zebra_video_right), axis=3)
        true_zebra_video = np.concatenate((true_zebra_video_left, true_zebra_video_right,
                                           fake_horse_video_left, fake_horse_video_right), axis=3)

    # at all frames indexed by true_horse_t and true_zebra_t, add a white dot of 10px at the corner
    # lasting for +/- 10 frames around the index

    for i in range(len(fake_horse_t)):
        rand_color = np.random.randint(0, 255, 3)
        rand_color = np.tile(rand_color, (30, 30, 1))
        rand_color = np.transpose(rand_color, (2, 0, 1))
        for j in range(-10, 11):
            if fake_horse_t[i] + j >= 0 and fake_horse_t[i] + j < temporal_width:                                                                                                                                                       
                true_zebra_video[fake_horse_t[i] + j, :, 30:60, 30:60] = rand_color
                # add broadcast

    for i in range(len(fake_zebra_t)):
        rand_color = np.random.randint(0, 255, 3)
        rand_color = np.tile(rand_color, (30, 30, 1))
        rand_color = np.transpose(rand_color, (2, 0, 1))
        for j in range(-10, 11):
            if fake_zebra_t[i] + j >= 0 and fake_zebra_t[i] + j < temporal_width:
                true_horse_video[fake_zebra_t[i] + j, :, 30:60, 30:60] = rand_color

    # add 30 empty frames at the end
    true_zebra_video = np.concatenate((true_zebra_video, np.ones((30, 3, 1080, 2160))), axis=0)
    true_horse_video = np.concatenate((true_horse_video, np.ones((30, 3, 1080, 2160))), axis=0)

    return true_horse_video, true_zebra_video


def visualize_video_vae(inputs, outputs, temporal_width = 176):

    inputs = inputs.cpu().detach().numpy().reshape((temporal_width, 478, 3)) * 3 + 0.5
    outputs = outputs.cpu().detach().numpy().reshape((temporal_width, 478, 3)) * 3 + 0.5

    inputs = visualize_landmarks(
        np.zeros((1080, 1080, 3)).astype(np.uint8),
        inputs,
    )
    outputs = visualize_landmarks(
        np.zeros((1080, 1080, 3)).astype(np.uint8),
        outputs,
    )

    inputs = np.array(inputs).transpose((0, 3, 1, 2))
    outputs = np.array(outputs).transpose((0, 3, 1, 2))
    
    final = np.concatenate((inputs, outputs), axis=3)

    return final


def visualize_video_vae(inputs, temporal_width = 176, dims=1080, transpose=True):
    inputs = inputs.cpu().detach().numpy().reshape((temporal_width, 478, 3)) * 3 + 0.5
    inputs = visualize_landmarks(
        np.zeros((dims, dims, 3)).astype(np.uint8),
        inputs,
        invert=True,
    )
    if transpose:
        inputs = np.array(inputs).transpose((0, 3, 1, 2))
    return inputs


def get_dataframes(horse_a, zebra_a, chunks):
    chunks = horse_a.shape[0]
    horse_a = horse_a.reshape(-1, 2)
    zebra_a = zebra_a.reshape(-1, 2)
    indices = np.array([[i, j] for i in range(chunks) for j in range(12)])
    columns = ['Chunk', 'VAE_index', 'alpha', 'zi']
    df_horse = pd.DataFrame(np.column_stack([indices, horse_a]), columns=columns)
    df_zebra = pd.DataFrame(np.column_stack([indices, zebra_a]), columns=columns)
    df_horse[['Chunk', 'VAE_index']] = df_horse[['Chunk', 'VAE_index']].astype(int)
    df_zebra[['Chunk', 'VAE_index']] = df_zebra[['Chunk', 'VAE_index']].astype(int)
    df_horse = df_horse.round({'alpha': 1, 'zi': 1})
    df_zebra = df_zebra.round({'alpha': 1, 'zi': 1})
    return df_horse, df_zebra