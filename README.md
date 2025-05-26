Dự báo thời tiết Hà Nội

Ứng dụng web sử dụng Flask và TensorFlow để dự đoán thời tiết tại Hà Nội, bao gồm nhiệt độ và xác suất mưa trong 1 giờ tới

Môi trường triển khai

Ubuntu 20.04

Python 3

API Key chính: d2d021ceb18c28747d865fd2b33d71ea

API Key dự phòng: b19d206dbc49beb17d642bcba1d7ed5b

Để huấn luyện lại mô hình hoặc tạo dữ liệu mới, cần xóa các file từ lần chạy trước. Danh sách các file:

weather_dataset.csv: Lưu dữ liệu thời tiết (nhiệt độ, độ ẩm, áp suất, xác suất mưa, lượng mưa, thời gian) từ API OpenWeatherMap.

temp_model.h5: Mô hình TensorFlow/Keras dự đoán nhiệt độ sau 1 giờ.

rain_model.h5: Mô hình TensorFlow/Keras dự đoán xác suất mưa.

scaler.pkl: Đối tượng MinMaxScaler để chuẩn hóa dữ liệu cho mô hình.

loss_history.pkl: Lịch sử huấn luyện (loss) của mô hình nhiệt độ.

temperature_plot.png: Biểu đồ so sánh nhiệt độ thực tế và dự đoán (lưu trong thư mục static/).

rain_plot.png: Biểu đồ so sánh xác suất mưa thực tế và dự đoán (lưu trong static/).

loss_plot.png (tùy chọn): Biểu đồ loss huấn luyện và xác thực (lưu trong static/, nếu có lịch sử huấn luyện).

Thư mục static/: Chứa các file hình ảnh biểu đồ, tự động tạo nếu chưa tồn tại.

Để tạo dữ liệu mới và huấn luyện lại mô hình, xóa các file từ lần chạy trước:

rm weather_dataset.csv temp_model.h5 rain_model.h5 scaler.pkl loss_history.pkl
rm -rf static/*

Lưu ý: Khi huấn luyện lại, cần khoảng 15 phút để thu thập dữ liệu mới từ API.

Cài đặt môi trường


Cập nhật hệ thống và Python 3, pip:

sudo apt update
sudo apt install python3 python3-pip

Cài đặt các thư viện :

pip3 install requests pandas numpy tensorflow scikit-learn matplotlib flask

Chạy chương trình:

python3 weather_forecast_app.py

Truy cập ứng dụng:
http://localhost:5000

Mạng ổn định để lấy dữ liệu
