# FACaP

This is  the official implementation for the paper 
***Floorplan-Aware Camera Poses Refinement***.

## Introduction


For many indoor scenes, there exists an image of a technical
floorplan that contains information about the geometry and 
main structural elements of the scene, such as walls, 
partitions, and doors. We argue that such a floorplan is 
a useful source of spatial information, which can guide 
a 3D model optimization.

The standard RGB-D 3D reconstruction pipeline consists of
a tracking module applied to an RGB-D sequence and a bundle
adjustment (BA) module that takes the posed RGB-D sequence and
corrects the camera poses to improve consistency. We propose
a novel optimization algorithm expanding conventional BA that
leverages the prior knowledge about the scene structure in
the form of a floorplan. Our experiments on the Redwood
dataset and our self-captured data demonstrate that utilizing
floorplan improves accuracy of 3D reconstructions.

![](imgs/pipeline.png)

## Data structure
All scans should be preprocessed to the next structure:
```
scan
│   floorplan.npy
│   db.h5
│
└───arcore
│   │   cam_params-0001.txt
│   │   ...
│   │   depth-0001.png
│   │   ...
│   │   frame-0001.png
│   │   ...
│   │   pose-0001.txt
│   │   ...
│   │
│
└───segmentation
│   │   frame-0001_wall.png
│   │   ...
│   │   frame-0001_floor.png
│   │   ...
│  

```

Here:
- `floorplan.npy` is an array with the shape `n x 4`. Each element is a segment of the floorplan.
- `db.h5` features a database in COLMAP format, which is used to map covisible points. 
- `cam_params-0001.txt` intrinsics of the corresponding camera (w, h, f1, f1, p1, p2).
- `pose-0001.txt` extrinsic matrix of the corresponding camera
- `depth-0001.png` depth map
- `frame-0001.png` RGB frame
- `frame-0001_wall.png` rotated mask of walls for the corresponding frame
- `frame-0001_floor.png` rotated mask of the floor for the corresponding frame

For more details please see the file `facap/data/scan.py`.

## Usage

To run an experiment you should create a config file run experiment. 

```python
python scripts/run_experimnt.py --config path_to_config --device "cuda:0"
```

The example of the config can be found in the path `experiments/config.yaml`.
