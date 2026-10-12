import cv2
import mediapipe as mp
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm  

# --- Setup Mediapipe ---
mp_face = mp.solutions.face_mesh
#face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,     
    max_num_faces=1,
    refine_landmarks=True,       
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# Facial mesh edges for GNN
FACEMESH_EDGES = list(mp_face.FACEMESH_TESSELATION)

# --- Paths ---
input_dir = Path("data")
output_dir = Path("face_features")
output_dir.mkdir(exist_ok=True)

cap_per_class = {
    "My_faceCropped": None,      
    "not_my_face": 100    
}

# create more images of my_face
def augment_image(img):
    augmented = []

    # original
    augmented.append(img)

    # horizontal flip
    augmented.append(cv2.flip(img, 1))

    # rotate small angles
    for angle in [-10, 10]:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)

    # brightness change
    brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)

    augmented.append(brighter)
    augmented.append(darker)

    return augmented
# --- Normalization / feature functions ---
def normalize_landmarks(landmarks):
    """
    landmarks: np.array of shape (N, 3) -> x, y, z
    """
    nose = landmarks[1]  # nose tip
    landmarks_centered = landmarks - nose
    
    # Use inter-eye distance in xy-plane only
    eye_distance = np.linalg.norm(landmarks[33][:2] - landmarks[263][:2])
    landmarks_normalized = landmarks_centered / (eye_distance + 1e-8)
    
    return landmarks_normalized



def build_edges(edges, num_nodes):
    edge_pairs = []

    for i, j in edges:
        edge_pairs.append((i, j))
        edge_pairs.append((j, i))
    
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    
    return edge_index.clamp(0, num_nodes - 1)


# --- Process each folder ---
for class_folder in ["My_faceCropped", "not_my_face"]:
    input_class_dir = input_dir / class_folder
    output_class_dir = output_dir / class_folder
    output_class_dir.mkdir(exist_ok=True)
    
    image_paths = list(input_class_dir.glob("*.jpg"))  


    cap = cap_per_class[class_folder]
    if cap is not None:
        image_paths = image_paths[:cap]  
    
    for img_path in tqdm(image_paths, desc=f"Processing {class_folder}"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping {img_path} (cannot read)")
            continue
        augmented_images = augment_image(img)
        for aug_img in augmented_images:
            # Convert BGR -> RGB
            results = face_mesh.process(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print(f"No face detected in {img_path}")
                continue
            
            # Extract landmarks
            lm_coords = np.array([(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark])
            landmarks = normalize_landmarks(lm_coords)
            #Gaussian noise to landmarks
            if np.random.rand() < 0.5:
                landmarks = landmarks + np.random.normal(0, 0.01, landmarks.shape)
            edge_index = build_edges(FACEMESH_EDGES, num_nodes=468)

           
            x = torch.tensor(landmarks, dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index)
            data.y = torch.tensor([0 if class_folder=="My_faceCropped" else 1], dtype=torch.long)
            output_file = output_class_dir / f"{img_path.stem}_{np.random.randint(100000)}.pt"
            torch.save(data, output_file)


     
