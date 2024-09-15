# Phân Tích Khuôn Mặt - AI Hackathon Challenge

## Language / Ngôn Ngữ

- [English](README-en.md)      
- [Tiếng Việt](README-vi.md)


![Face Analysis](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQhHJnSMXcxrEpTLsPA25PAnwiLar6cHYUk6Q&s)

Chạy trên VSCode:

<a href="https://code.visualstudio.com/download">
<img src= "https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" width=80>
</a>

Tác giả:
- Github: khoitran2003
- Email: anhkhoi246813579@gmail.com

## Tổng Quan

Chào mừng bạn đến với dự án Phân Tích Khuôn Mặt! Công cụ này sử dụng các kỹ thuật học máy và thị giác máy tính tiên tiến để phát hiện và phân tích khuôn mặt người trong hình ảnh hoặc video. Cho dù bạn muốn phát hiện cảm xúc, ước lượng tuổi, nhận diện giới tính, hay trích xuất các đặc điểm khuôn mặt, dự án này đều có thể đáp ứng.

## Tính Năng

- **Phát Hiện Cảm Xúc**: Nhận diện và phân tích cảm xúc của con người từ biểu cảm khuôn mặt.
- **Ước Lượng Tuổi**: Dự đoán tuổi của cá nhân dựa trên các đặc điểm khuôn mặt.
- **Nhận Diện Giới Tính**: Xác định giới tính của cá nhân từ khuôn mặt.
- **Trích Xuất Đặc Điểm Khuôn Mặt**: Trích xuất và phân tích các đặc điểm chính của khuôn mặt để có thêm thông tin chi tiết.

## Bắt Đầu

### Yêu Cầu

- [Visual Studio Code](https://code.visualstudio.com/download)
- Python 3.7+
- Các thư viện Python cần thiết (được liệt kê trong `requirements.txt`)

### Cài Đặt

1. Clone repository:
    ```bash
    git clone https://github.com/khoitran2003/human-face-analysis-AI-hackathon.git
    cd human-face-analysis-AI-hackathon
    ```

2. Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

3. Tải mô hình đã được huấn luyện từ Google Drive:
    [Tải Checkpoint](https://drive.google.com/uc?id=YOUR_CHECKPOINT_ID&export=download)

4. Di chuyển checkpoint đã tải vào thư mục `checkpoint`:
    ```bash
    mv path/to/downloaded/checkpoint checkpoint/
    ```

5. Chạy ứng dụng:
    ```bash
    python webapp.py
    ```

## Giới Thiệu

Phân tích khuôn mặt là một công nghệ sử dụng học máy và thị giác máy tính để phát hiện và phân tích khuôn mặt người trong hình ảnh hoặc video. Nó có thể được sử dụng cho nhiều ứng dụng khác nhau như phát hiện cảm xúc, ước lượng tuổi, nhận diện giới tính và trích xuất đặc điểm khuôn mặt. Dự án này nhằm cung cấp một công cụ đơn giản nhưng hiệu quả cho việc phân tích khuôn mặt, giúp người dùng có được những thông tin chi tiết từ dữ liệu hình ảnh.


## Ảnh Chụp Màn Hình

![image](results/Screenshot%20from%202024-09-15%2020-06-14.png)

![image](results/Screenshot%20from%202024-09-15%2020-07-13.png)

![image](results/Screenshot%20from%202024-09-15%2020-26-43.png)


## Giấy Phép

Dự án này được cấp phép theo giấy phép MIT - xem tệp [LICENSE](LICENSE) để biết thêm chi tiết.

## Lời Cảm Ơn

- Đặc biệt cảm ơn cộng đồng mã nguồn mở vì những đóng góp vô giá của họ.
- Lấy cảm hứng từ các bài báo và dự án nghiên cứu phân tích khuôn mặt khác nhau.

