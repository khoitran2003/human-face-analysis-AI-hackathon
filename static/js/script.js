const video = document.getElementById('video');
const image = document.getElementById('image');
const fileInput = document.getElementById('file-input');
const faceTableBody = document.getElementById('face-table').getElementsByTagName('tbody')[0];
let webcamStream;
let inferInterval;
let originalImageSrc; // Biến để lưu trữ ảnh gốc

window.onload = function() {
    fileInput.value = ''; // Xóa tên file ảnh hoặc video đã chọn khi tải lại trang
};

function startWebcam() {
    stopWebcam();
    image.style.display = 'none';
    video.style.display = 'block';
    fileInput.value = ''; // Xóa tên file ảnh hoặc video đã chọn
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            webcamStream = stream;
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Error accessing webcam: ", err);
        });
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (inferInterval) {
        clearInterval(inferInterval);
        inferInterval = null;
    }
}

function loadMedia(event) {
    const file = event.target.files[0];
    if (file) {
        stopWebcam();
        const fileType = file.type.split('/')[0];
        if (fileType === 'image') {
            loadImage(file);
        } else if (fileType === 'video') {
            loadVideo(file);
        }
    }
}

function loadImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        video.style.display = 'none';
        image.style.display = 'block';
        image.src = e.target.result;
        originalImageSrc = e.target.result; // Lưu trữ ảnh gốc
    };
    reader.readAsDataURL(file);
}

function loadVideo(file) {
    const url = URL.createObjectURL(file);
    video.style.display = 'block';
    image.style.display = 'none';
    video.src = url;
    video.load(); // Ensure the video is loaded
    video.play();
}

function infer() {
    if (video.style.display === 'block') {
        inferInterval = setInterval(() => {
            processFrame(video);
        }, 1000); // Xử lý mỗi giây một lần
    } else if (image.style.display === 'block') {
        processFrame(image);
    }
}

function processFrame(element) {
    let canvas = document.createElement('canvas');
    let context = canvas.getContext('2d');
    canvas.width = element.videoWidth || element.width;
    canvas.height = element.videoHeight || element.height;

    // Nếu là ảnh, vẽ lại ảnh gốc trước khi vẽ bounding box mới
    if (element.tagName.toLowerCase() === 'img') {
        let img = new Image();
        img.src = originalImageSrc;
        img.onload = function() {
            context.drawImage(img, 0, 0, canvas.width, canvas.height);
            processAndDrawBoundingBox(context, canvas, element);
        };
    } else {
        context.drawImage(element, 0, 0, canvas.width, canvas.height);
        processAndDrawBoundingBox(context, canvas, element);
    }
}

function processAndDrawBoundingBox(context, canvas, element) {
    // Chuyển đổi khung hình sang base64
    let imageData = canvas.toDataURL('image/jpeg').split(',')[1];

    // Gửi yêu cầu phân tích gương mặt đến server
    fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        // Xóa các dòng cũ trong bảng
        faceTableBody.innerHTML = '';

        // Thêm các dòng mới vào bảng
        data.forEach(face => {
            let row = faceTableBody.insertRow();
            row.insertCell(0).textContent = face.face_id;
            row.insertCell(1).textContent = face.age;
            row.insertCell(2).textContent = face.gender;
            row.insertCell(3).textContent = face.emotion;

            // Vẽ bounding box
            context.strokeStyle = 'red';
            context.lineWidth = 2;
            context.strokeRect(face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]);

            // Vẽ face_id bên trong góc trái trên của bounding box
            context.fillStyle = 'white';
            context.font = '20px Arial';
            context.fillText(face.face_id, face.bbox[0] + 2, face.bbox[1] + 18);
        });

        // Nếu là ảnh, chỉ xử lý một lần
        if (element.tagName.toLowerCase() === 'img') {
            clearInterval(inferInterval);
            inferInterval = null;
            // Cập nhật lại hình ảnh với bounding box
            image.src = canvas.toDataURL('image/jpeg');
        } else {
            // Cập nhật lại video frame với bounding box
            video.srcObject = null;
            video.src = canvas.toDataURL('image/jpeg');
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}