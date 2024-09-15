import cv2
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from source.classification.data_loader_clf import image_transformation
from source.classification.resnet_50_modify import Modified_Resnet_50

# Load models and configurations
def load_models(cfg_path, label_cfg_path, detection_model_path, classification_model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(cfg_path) as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    with open(label_cfg_path) as label_cfg_file:
        label_cfg = yaml.safe_load(label_cfg_file)

    yolomodel = YOLO(detection_model_path)
    clfmodel = Modified_Resnet_50()
    checkpoint = torch.load(classification_model_path, map_location=device)
    clfmodel.load_state_dict(checkpoint['state_dict'])
    clfmodel.to(device).eval()

    return yolomodel, clfmodel, cfg, label_cfg, device

# Find key in YAML file
def find_key(yaml_file, value):
    for key, val in yaml_file.items():
        if val == value:
            return key

# Analyze face in the image
def analyze_face(image, yolomodel, clfmodel, cfg, label_cfg, device):
    results = yolomodel.predict(image, imgsz=640)
    xyxys = results[0].boxes.xyxy.cpu().numpy()
    class_names = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    face_data = []

    face_list = []
    for bbox, c, conf in zip(xyxys, class_names, confs):
        cropped_frame = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        cropped_frame = Image.fromarray(cropped_frame)
        trans_cropped_frame = image_transformation(cropped_frame, cfg)
        face_list.append(trans_cropped_frame)

    if face_list:
        batch_face = torch.stack(face_list, dim=0)
        with torch.no_grad():
            clf_pred = clfmodel(batch_face.to(device))
            ages, genders, emotions = clf_pred
            ages = torch.argmax(ages, dim=1)
            genders = torch.argmax(genders, dim=1)
            emotions = torch.argmax(emotions, dim=1)

            for i, (bbox, age, gender, emotion) in enumerate(zip(xyxys, ages, genders, emotions)):
                age = find_key(label_cfg["age"], age.item())
                gender = find_key(label_cfg["gender"], gender.item())
                emotion = find_key(label_cfg["emotion"], emotion.item())
                face_data.append({
                    "face_id": i + 1,
                    "age": age,
                    "gender": gender,
                    "emotion": emotion,
                    "bbox": bbox.tolist()  # Thêm thông tin bounding box
                })

    return face_data
