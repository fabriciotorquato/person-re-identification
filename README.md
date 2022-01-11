# Person Re-identification

## Requirements

### Software

* Python == 3.8 (conda)
* NVIDIA driver version: 460.x
* NVIDIA CUDA version: 11.2
* NVIDIA cuDNN version: 8.1.0

> $ apt-get install ffmpeg

### Recommended hardware

* RAM size: >= 12 GB;
* Hard disk free space: >= 50 GB;

## Repository organization

    .
    ├── data                        # Directory with the dataset
    |   └── ground_truth            # Wisenet Ground Truth Files
    |   └── output                  # Report tracking of person re-identification
    |   └── wisenet_dataset         # Dataset
    ├── libs                        # External libs
    |   └── facenet                 # https://github.com/davidsandberg/facenet
    ├── notebooks                   # Notebooks codes
    ├── output                      # Results of metrics evaluation
    ├── scripts                     # Shell scripts for run project
    ├── src                         # Source code


## Build

The person re-identification project is composed of two face recognition neural networks, Facenet and Mobilnet. To run the project, you will need two different environments, 

### Base Environment

   > $ apt-get install ffmpeg

### MobileNet Environment

   > $ conda create -n tcc-25 python=3.6 anaconda

   > $ conda activate tcc-25

   > $ conda install -c anaconda cudatoolkit==11.2 -y

   > $ conda install -c anaconda cudnn -y

   > $ pip install -r requirements_mobilenet.txt 

### FaceNet Environment

   > $ conda create -n tcc-17 python=3.6 anaconda

   > $ conda activate tcc-17

   > $ conda install -c anaconda cudatoolkit==9.0 -y

   > $ conda install -c anaconda cudnn -y

   > $ pip install -r requirements_facenet.txt

## Run

1. Download Wisenet dataset, in [official repository](https://data.4tu.nl/articles/dataset/WiseNET_Multi-camera_dataset/12714416/1)

```
    .
    ├── data                        # Directory with the dataset
    |   └── wisenet_dataset         # Dataset
    |   |   └── video               # Directory with sets
    |   |   |   └── set_1           # Set 1
    |   |   |   └── set_2           # Set 2
    |   |   |   └── set_3           # Set 3  
    |   |   |   └── set_4           # Set 4
```

3. Run the script that will convert all videos into frames and save only those that have a face detected by MTCNN.

   > $ cd scripts

   > $ get_faces.sh

   > Noted: Use the FaceNet environment

4. In directory `data/wisenet_dataset/videos_frames`, the human faces resulting from the `get_faces.sh` script will be saved, make the separation of the faces by user, being that unknown user will be called `UNK`.
```
    .
    ├── data                      # Directory with the dataset
    |   └── wisenet_dataset       # Dataset
    |   |   └── database_frames   # Directory faces for trainig model
    |   |   |   └── ID1           # Faces of person 1
    |   |   |   └── ID2           # Faces of person 2
    |   |   |   └── ID3           # Faces of person 3
    |   |   |   └── UNK           # Faces of person unknown
```
   > Noted: Select only images from videos that will not be used in the experiment.

5. Run notebook `notebooks/train/train_mobilenet.ipynb` for training mobileNet model, or download the models [pre-trained](https://drive.google.com/drive/folders/1u7buMV-dhHH-ux9znJRa5Z2QrWsQv3aY)
```
    .
    ├── models                      # Directory with the models
    |   └── mobilenet     
    |   |   └── assets   
    |   |   └── variables  
    |   |   └── keras_metada.pb  
    |   |   └── saved_model.pb  
    |   └── labels.txt
```
   > Noted: Use the MobileNet environment

6. Download VGGFace2 pre-trained model trained by [David Sandberg](https://github.com/davidsandberg/facenet).
```
    .
    ├── models                      # Directory with the models
    |   └── facenet     
    |   |   └── 20180402-114759.pb   
    |   |   └── model-20180402-114759.ckpt-275.data-00000-of-00001  
    |   |   └── model-20180402-114759.ckpt-275.index
    |   |   └── model-20180402-114759.meta
```
7. Run notebook `notebooks/train/train_facenet.ipynb` for training FaceNet model, or download the models [pre-trained](https://drive.google.com/drive/folders/1uVBXT71XRVK6aKtZP_-2tpmrSrjqN2Me).
```
    .
    ├── models                      # Directory with the models
    |   └── facenet     
    |   |   └── 20180402-114759.pb   
    |   |   └── model-20180402-114759.ckpt-275.data-00000-of-00001  
    |   |   └── model-20180402-114759.ckpt-275.index
    |   |   └── model-20180402-114759.meta
    |   |   └── one_shot_classifier.pkl
```
   > Noted: Use the FaceNet environment

8. Run the script that will do the person re-identification used MobileNet.

   > $ cd scripts

   > $ multi_mobilenet.sh `number_of_set_videos`

   > Example: $ multi_mobilenet.sh 1
```
    .
    ├── data                      
    |   └── output       
    |   |   └── set_1   
    |   |   |   └── tracking_predict_db_mobilenet.json
```
   > Noted: Use the MobileNet environment

9. Run the script that will do the person re-identification used FaceNet.

   > $ cd scripts

   > $ multi_facenet.sh `number_of_set_videos`

   > Example: $ multi_facenet.sh 1
```
    .
    ├── data                      
    |   └── output       
    |   |   └── set_1   
    |   |   |   └── tracking_predict_db_facenet.json   
```
   > Noted: Use the FaceNet environment

10. Execute metrics evaluate of MobileNet report tracking.

    > $ cd scripts

    > $ evaluate_mobilenet.sh `number_of_set_videos`

    > Example: $ evaluate_mobilenet.sh 1
```
     .
     ├── output                      
     |   └── set_1       
     |   |   └── mobilenet   
     |   |   |   └── metrics.csv   
```
    > Noted: Use the MobileNet environment

11. Execute metrics evaluate of FaceNet report tracking.

   > $ cd scripts

   > $ evaluate_facenet.sh `number_of_set_videos`

   > Example: $ evaluate_facenet.sh 1
```
    .
    ├── output                      
    |   └── set_1       
    |   |   └── facenet   
    |   |   |   └── metrics.csv   
```
   > Noted: Use the MobileNet environment