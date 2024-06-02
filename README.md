# FacET: How Video Meetings Change Your Expression

### ECCV 2024

### [Project Page](https://facet.cs.columbia.edu/)  | [Paper](TODO)

[How Video Meetings Change Your Expression](https://facet.cs.columbia.edu/)  
 [Sumit Sarin](https://stellargo.github.io/), [Utkarsh Mall](https://www.cs.columbia.edu/~utkarshm/), [Purva Tendulkar](https://purvaten.github.io/), [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/) <br>
Columbia University

##  Usage

### Requirements

Create a new environment, and install the required packages.

```
conda create --name facet python=3.10.13
conda activate facet
pip install -r requirements.txt
```

### Dataset

The datasets are available at the following links:
- [ZoomIn](dataset/zoomin_info.csv)
- [Presidents](dataset/pres_info.csv)

The data contains the following column fields:
- `vid_id`: unique identifier for the videos
- `fps`: frame-rate.
- `vid_url`: URL for the video.
- `view`: (only for ZoomIn dataset) whether video is F2F (`off`) or VC (`on`).
- `president`: (only     for Presidents dataset) whether video is Trump (`trump`) or Obama (`obama`). 
- `participants`: number of participants in the conversation.
- `file_name`: the name of the video file to be be used for saving.

Place the videos in a `<directory_with_mp4_videos>` with file name same as `file_name` column from the csv file.

In the following sections, we only give instructions for the ZoomIn dataset, but it is straightforward to change configurations accordingly and use the Presidents dataset.

### Data Pre-processing

1.  Download the [landmarker model](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python#model) made available by Mediapipe and save it as `facet/scripts/model/face_landmarker_v2_with_blendshapes.task`.

2. `cd scripts`

3. `python vid_to_mesh.py <directory_with_mp4_videos>` (will take some time to run, you can adjust num_threads accordingly).

4. `python mesh_to_numpy.py <directory_with_mp4_videos>`

### Training Models

```
cd ..
python train-facet.py --wandb --variable_alpha --variable_changepoints --chunks=2
```
(Wandb can be disabled by not passing the `--wandb` flag.)

Note: The above scripts uses the [`beta`-VAE](./model/trained/beta_vae.pth) we trained on the ZoomIn dataset. To train a new model from scratch:

```
python train-vae.py
```

### De-zoomification

We use [imaginaire's vid2vid](https://github.com/NVlabs/imaginaire/blob/master/projects/vid2vid/README.md) off the shelf. Please follow setup instructions and use it directly.

For reference, we provide the script which can be used for building training/validation data for running vid2vid. Note that this basically involves putting RGB video frames from an F2F video in folder `mesh_images` and the MediaPipe keypoints in folder `mesh`. To run the script, update the variable `vid_directory` and run:

```
cd scripts
python vid2vid_dataset_create.py
```

Once that is done, you can update the [config file](./vid2vid_config/ampO1.yaml) which we used for running vid2vid for reference with correct paths for training/validation data (just use any F2F video from the output of previous command in `dataset/zoomin/dezoom`) and then run vid2vid as explained in the repo with the updated config file path as input:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py --config <facet-base-path>/facet/vid2vid_config/ampO1.yaml --logdir=./logs
```

The above process trains the vid2vid model. To effectively de-zoom a video, we need to test the model on keypoints that have been de-zoomed by FacET (which was obtained using `train-facet.py`). This involves, for a VC video,

1. Get its keypoints through MediaPipe (refer [vid2vid_dataset_create.py](scripts/vid2vid_dataset_create.py))
2. Obtain its latents using trained beta-VAE encoder (refer [train-facet.py](train-facet.py))
3. Passing them through FacET (refer [train-facet.py](train-facet.py))
4. Obtain keypoints back through beta-VAE decoder (refer [train-facet.py](train-facet.py))
5. Save the keypoints (refer [vid2vid_dataset_create.py](scripts/vid2vid_dataset_create.py))

The saved keypoints path can then be updated in the config file, and vid2vid can be runned in test mode by running the following in the imaginaire repository:

```
python -m torch.distributed.launch --nproc_per_node=8 inference.py --config <facet-base-path>/facet/vid2vid_config/ampO1.yaml --logdir=./logs
```

Note that we train vid2vid on one RGB video and its corresponding keypoints pair, and then test it on keypoints from other videos. This can need callibration because the distance of the keypoints from the camera can change. Ideally, you want that the keypoints should roughly be at the same distance from camera in both train and test videos. This can be easily adjusted by cropping the frame accordingly.


##  Acknowledgement
This research is based on work partially supported by the DARPA CCU program under contract HR001122C0034 and the National Science Foundation AI Institute for Artificial and Natural Intelligence (ARNI). PT is supported by the Apple PhD fellowship. 


##  Citation
```
@misc{sarin2023facet,
      title={TODO}, 
      author={Sumit Sarin and Utkarsh Mall and Purva Tendulkar and Carl Vondrick},
      year={2024},
      eprint={2303.11328},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
