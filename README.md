# Face Recognition Using Graph Neural Networks

A computer vision project exploring facial recognition using Graph Neural Networks (GNNs) and MediaPipe Face Mesh. Rather than learning directly from raw image pixels, the system extracts 3D facial landmarks and represents facial structure as a graph for binary classification.

The project implements a complete machine learning pipeline, from facial landmark extraction and graph construction to model training, evaluation, and real-time testing.

## Overview

Traditional facial recognition systems often use convolutional neural networks (CNNs) to learn visual features directly from images. This project investigates an alternative approach in which facial landmarks are represented as nodes in a graph, with edges describing relationships between facial features.

This representation allows the model to learn structural relationships between areas such as the eyes, nose, and mouth, while reducing its reliance on raw image pixels.

The system is designed to distinguish between my face and an unknown face, exploring how graph-based learning can be applied to facial verification.



## Key Features

* Facial landmark extraction: Uses MediaPipe Face Mesh to extract 468 facial landmarks.
* Graph representation: Represents facial landmarks as graph nodes connected by edges.
* Graph Neural Network: Combines Graph Convolutional Network (GCN) and Graph Attention Network (GAT) layers.
* Model training: Implements a PyTorch training pipeline with loss optimisation and gradient clipping.
* Performance evaluation: Tracks precision, recall, F1 score, loss, and confusion matrices.
* Hyperparameter comparison: Investigates how class weighting affects false positives and false negatives.
* Real-time testing: Demonstrates recognition of a known face and rejection of an unseen face.

## Model Architecture

The model uses a graph-based representation of facial structure.

1. Landmark extraction: MediaPipe Face Mesh extracts 3D facial landmark coordinates from an input image.
2. Graph construction: Each landmark becomes a node, with edges representing relationships between facial landmarks.
3. Graph convolution: Two GCN layers learn features by aggregating information from neighbouring nodes.
4. Graph attention: A GAT layer learns to assign different levels of importance to neighbouring nodes.
5. Normalisation and activation: GraphNorm and ReLU are used to support stable learning and nonlinear feature transformations.
6. Classification: Two fully connected layers produce predictions for the two classes.

The model is implemented using PyTorch and PyTorch Geometric.

## Dataset Setup

1. Download the Dataset

Download the not_my_face dataset from the following Google Drive link:

https://drive.google.com/drive/folders/16ZqUais6Kf29N4qX3Nyhga0vEyFa_ILZ?usp=share_link

Extract the downloaded dataset and keep the not_my_face folder.

2. Add Your Own Face Images

To enable the system to recognise your face, collect approximately 100 images of your face. Place these images in the appropriate input directory and run face_feature.py to extract your facial features.

The extracted features should be saved in a folder named my_faceCropped.

3. Organise the Dataset

Create a folder named face_features in your project directory. Move both the not_my_face folder and the my_faceCropped folder containing your extracted facial features into face_features.

Your directory structure should look like this:

``` text
project/
├── face_feature.py
├── face_features/
│   ├── not_my_face/
│   │   └── ... (dataset images/features)
│   └── my_faceCropped/
│       └── ... (your extracted facial features)
└── F211828_coursework_jupyter.ipynb
```
Important: Ensure that both folders are inside face_features before running the notebook.

4. Run the Project

Once the dataset is downloaded, your face features have been extracted, and the folders are organised correctly, open and run F211828_coursework_jupyter.ipynb to execute the face recognition system.

## Dataset and Preprocessing

The project uses facial images organised into two categories:

* Known face: Images of the individual the model is trained to recognise.
* Unknown face: Images of other individuals used to distinguish unfamiliar faces from the known identity.

MediaPipe converts the images into facial landmark representations, which are then processed into graph data for training and evaluation.

Note: The original dataset and generated facial feature files are not included in this repository. You will need to prepare your own data and generate the corresponding features before reproducing the experiments.

## Training and Evaluation

The notebook implements a training and evaluation pipeline in PyTorch. The model is trained using batches of graph data, with its performance monitored on the validation set.

Evaluation includes:

* Precision and recall
* F1 score
* Training and validation loss
* Training and validation macro F1
* Confusion matrix
* Comparison of true and predicted class distributions

The notebook also investigates class weighting as a way to manage the trade-off between false positives and false negatives.

In the reported comparison, class weighting reduced false positives from 13 to 11 and increased precision for the known-face class from 0.86 to 0.88. However, recall decreased from 0.87 to 0.83, illustrating the trade-off between rejecting unfamiliar faces and recognising genuine matches.

## Real-Time Demonstration

The notebook includes a real-time test using the trained model. The demonstration shows the system identifying the known face and classifying an unseen face as unknown.

This illustrates how graph-based facial representations can be applied beyond offline model evaluation. However, the demonstration is an initial experiment and does not establish production-level reliability or robustness across different environments.

## Repository Structure

```text 
gnn-face-recognition/
├── dataset.py
├── face_features.py
├── face_recognition_pipeline.ipynb
├── face_recognition_pipeline.html
├── models/
│   └── trained_model.pt
├── requirements.txt
├── .gitignore
└── README.md

```

The structure above describes the intended organisation of the repository; some directories may be empty or omitted depending on the files published.

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* MediaPipe Face Mesh
* NumPy
* scikit-learn
* Matplotlib
* Jupyter Notebook

# Getting Started

1. Clone the repository
``` 
git clone <your-repository-url>
cd gnn-face-recognition
```
2. Create a virtual environment
```
python -m venv .venv
```
Activate it on macOS or Linux:
```
source .venv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
PyTorch and PyTorch Geometric installation requirements can vary depending on your operating system and hardware. Consult their official installation instructions if you encounter compatibility issues.

4. Prepare the data

Prepare your own known-face and unknown-face image datasets, then generate the facial landmark features required by the pipeline.

The dataset and feature paths may need to be adjusted to match your local directory structure.

5. Run the notebook

Open the Jupyter notebook and follow the pipeline through preprocessing, graph construction, model training, evaluation, and real-time testing.

The included HTML export provides a static, executed version of the notebook for readers who want to explore the implementation without running the code.

## Limitations

* Performance depends on the quality and consistency of facial landmark detection.
* Changes in lighting, pose, occlusion, and camera conditions may affect recognition.
* The model is trained for a specific known-face versus unknown-face task; it is not a general-purpose facial recognition system.
* Results from a limited dataset do not establish generalisation to a wider population.
* Real-time demonstrations alone are insufficient to establish the reliability required for security-critical applications.

## Future Improvements

Potential extensions include:

* Evaluating performance across different lighting conditions, head poses, and image quality.
* Testing generalisation on a larger and more diverse dataset.
* Comparing GNN performance with a CNN-based baseline.
* Exploring alternative graph architectures and attention mechanisms.
* Measuring inference latency and computational requirements.
* Improving unknown-face rejection through systematic threshold calibration.

## Learning Objectives

This project explores practical applications of graph-based machine learning, including:

* Building a machine learning pipeline in PyTorch.
* Representing structured visual information as graph data.
* Training GNNs for binary classification.
* Evaluating classification performance using multiple metrics.
* Investigating hyperparameters and their effect on model behaviour.
* Applying a trained model to real-time input.

### References

* PyTorch
* PyTorch Geometric
* MediaPipe
* PyTorch Quickstart Tutorial
