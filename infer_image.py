import cv2
import argparse
import torch
import yaml
import random
import os
from PIL import Image
from ultralytics import YOLO
from source.classification.data_loader_clf import image_transformation
from source.classification.resnet_50_modify import Modified_Resnet_50

def get_args():
    parsers = argparse.ArgumentParser(description='Face')
    parsers.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='path to the input image'
    )
    parsers.add_argument(
        '--result_path',
        type=str,
        default='results',
        help='path to the result image'
    )
    parsers.add_argument(
        "--cfg",
        type=str,
        default="cfg/classifier.yaml",
    )
    parsers.add_argument(
        "--label_cfg",
        type=str,
        default="cfg/labels.yaml",
    )
    args = parsers.parse_args()
    return args

def find_key(yaml_file, value):
    for key, val in yaml_file.items():
        if val == value:
            return key

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (255, 165, 0), (255, 192, 203),
    (128, 0, 128), (0, 128, 128), (255, 255, 0), (75, 0, 130),
    (255, 127, 80), (230, 230, 250)
]

def main(args):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.cfg) as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    with open(args.label_cfg) as label_cfg_file:
        label_cfg = yaml.safe_load(label_cfg_file)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    yolomodel = YOLO('checkpoint/detection.pt')
    clfmodel = Modified_Resnet_50()
    checkpoint = torch.load('checkpoint/classifier_best.pt', map_location=device)
    clfmodel.load_state_dict(checkpoint['state_dict'])
    clfmodel.to(device).eval()

    image_name = args.image_path.split('/')[-1]

    frame = cv2.imread(args.image_path)
    if frame is None:
        raise ValueError(f"Image not found at {args.image_path}")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = yolomodel.predict(frame, imgsz=640)

    xyxys = results[0].boxes.xyxy.cpu().numpy()
    class_names = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    face_list = []
    for (bbox, c, conf) in zip(xyxys, class_names, confs):
        cropped_frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
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
                age = find_key(label_cfg['age'], age.item())
                gender = find_key(label_cfg['gender'], gender.item())
                emotion = find_key(label_cfg['emotion'], emotion.item())
                
                info = 'khoidepzai'
                (text_width, text_height), _ = cv2.getTextSize(info, font, font_scale, thickness)
                color = random.choice(colors)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[2]), int(bbox[1] - 1)), (int(bbox[2] + text_width), int(bbox[1] + 3*text_height+10)),
                            color, -1)
                cv2.putText(frame, age, (int(bbox[2]), int(bbox[1]+text_height)), font, font_scale, (0, 0, 0), thickness,
                            cv2.LINE_AA)
                cv2.putText(frame, gender, (int(bbox[2]), int(bbox[1]+2*text_height+5)), font, font_scale, (0, 0, 0), thickness,
                            cv2.LINE_AA)
                cv2.putText(frame, emotion, (int(bbox[2]), int(bbox[1]+3*text_height+7)), font, font_scale, (0, 0, 0), thickness,
                            cv2.LINE_AA)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{args.result_path}/results_{image_name}.jpg', frame)
    print(f'Results saved to {args.result_path}/results_{image_name}.jpg')
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    print(f"Start inference {args.image_path}")
    main(args)

