from ultralytics import YOLO
import torch
import cv2
import matplotlib

# model_path = 'face_detection.pt'
# model = YOLO(model_path)

# # Path to your image
# image_path = "people.jpeg"

# # Read the image
# ori_img = cv2.imread(image_path)

# # Get the expected input dimensions from the model (if possible)
# # This part might require checking the model documentation or experimentation
# expected_width, expected_height = 1024, 1024  # Assuming 1024 based on error message

# # Resize the image to match the expected dimensions
# if ori_img.shape[0] != expected_height or ori_img.shape[1] != expected_width:
#     img = cv2.resize(ori_img, (expected_width, expected_height))
# # cap = cv2.VideoCapture(2)
# #
# # while True:
# #     flag, frame = cap.read()
# #     if not flag:
# #         break
# #
# #     cv2.imshow('Camera', frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# results = model.predict(source=img)

# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 0.5
# thickness = 1
# names = model.names

# for result in results:
#     xyxys = result.boxes.xyxy.cpu().numpy()
#     class_names = result.boxes.cls.cpu().numpy()
#     confs = result.boxes.conf.cpu().numpy()
#     for (bbox, c, conf) in zip(xyxys, class_names, confs):
#         name = names[int(c)] + ' ' + f'{conf:.2f}%'
#         (text_width, text_height), _ = cv2.getTextSize(name, font, font_scale, thickness)
#         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 51), 2)
#         cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - text_height - 12)), (int(bbox[0] + text_width), int(bbox[1])),
#                       (255, 255, 51), -1)
#         cv2.putText(img, name, (int(bbox[0]), int(bbox[1] - text_height / 2)), font, font_scale, (0, 0, 0), thickness,
#                     cv2.LINE_AA)

# img = cv2.resize(img, (ori_img.shape[1], ori_img.shape[0]))
# cv2.imshow('image', img)
# cv2.waitKey(0)

class Detection:
    def __init__(self, source):
        self.model = self.load_model()
        self.source = source
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self, model_file):
        self.model = YOLO(model_file)
        return self.model

    
    def predict(self, frame):
        img_height, img_width, _ = frame.shape
        if img_height != 1024 or img_width != 1024:
            resized_img = cv2.resize(frame, (1024, 1024))
            results = self.model(frame)
        img = cv2.resize(resized_img, (img_width, img_height))
        return results, img
    
    def plot_bboxes(self, results, frame):
        
        xyxy_list = []
        conf_list = []
        class_list = []
        names = self.model.names
        for result in results:
            xyxys = result.boxes.xyxy.cpu().numpy()
            class_names = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for (bbox, c, conf) in zip(xyxys, class_names, confs):
                name = names[int(c)] + ' ' + f'{conf:.2f}%'
                (text_width, text_height), _ = cv2.getTextSize(name, font, font_scale, thickness)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 51), 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - text_height - 12)), (int(bbox[0] + text_width), int(bbox[1])),
                            (255, 255, 51), -1)
                cv2.putText(img, name, (int(bbox[0]), int(bbox[1] - text_height / 2)), font, font_scale, (0, 0, 0), thickness,
                            cv2.LINE_AA)
        

    
