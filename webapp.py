from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64
from infer_webcam import load_models, analyze_face

app = Flask(__name__)

# Load models and configurations
yolomodel, clfmodel, cfg, label_cfg, device = load_models(
    "cfg/classifier.yaml",
    "cfg/labels.yaml",
    "checkpoint/detection.pt",
    "checkpoint/classifier_best.pt"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/infer', methods=['POST'])
def infer():
    try:
        # Nhận dữ liệu hình ảnh từ yêu cầu
        data = request.json
        image_data = data['image']
        
        # Kiểm tra nếu dữ liệu hình ảnh trống
        if not image_data:
            raise ValueError("No image data received")
        
        # Chuyển đổi dữ liệu hình ảnh từ base64 sang numpy array
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        
        # Giải mã hình ảnh
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Kiểm tra nếu giải mã thất bại
        if img is None:
            raise ValueError("Image decoding failed")
        
        # Thực hiện phân tích gương mặt
        face_data = analyze_face(img, yolomodel, clfmodel, cfg, label_cfg, device)
        
    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({"error": str(e)}), 400

    return jsonify(face_data)
        

def generate_frames():
    camera = cv2.VideoCapture(0)  # Sử dụng 0 cho webcam

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)