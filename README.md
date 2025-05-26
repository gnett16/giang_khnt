Dự báo thời tiết Hà Nội

Ứng dụng web sử dụng Flask và TensorFlow để dự đoán thời tiết tại Hà Nội, bao gồm nhiệt độ và xác suất mưa trong 1 giờ tới, dựa trên dữ liệu từ API OpenWeatherMap. Ứng dụng tạo các biểu đồ trực quan và cảnh báo nguy cơ ngập lụt.

Môi trường triển khai





Hệ điều hành: Ubuntu 20.04



Ngôn ngữ: Python 3



API: OpenWeatherMap





API Key chính: d2d021ceb18c28747d865fd2b33d71ea



API Key dự phòng: b19d206dbc49beb17d642bcba1d7ed5b

Các file dữ liệu được tạo

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

Lưu ý: Khi huấn luyện lại, cần khoảng 15 phút để thu thập dữ liệu mới từ API.

Cài đặt môi trường





Cập nhật hệ thống và cài Python 3, pip:

sudo apt update
sudo apt install python3 python3-pip



Cài đặt các thư viện cần thiết:

pip3 install requests pandas numpy tensorflow scikit-learn matplotlib flask

Chạy ứng dụng





Chạy chương trình:

python3 weather_forecast_app.py



Truy cập ứng dụng:





Mở trình duyệt và truy cập: http://localhost:5000



Ứng dụng sẽ hiển thị dự báo thời tiết, biểu đồ và cảnh báo ngập lụt.

Tạo dữ liệu mới và huấn luyện lại mô hình

Để tạo dữ liệu mới và huấn luyện lại mô hình, xóa các file từ lần chạy trước:

rm weather_dataset.csv temp_model.h5 rain_model.h5 scaler.pkl loss_history.pkl
rm -rf static/*

Sau đó, chạy lại chương trình:

python3 weather_forecast_app.py

Ghi chú





Đảm bảo kết nối mạng ổn định để lấy dữ liệu từ API OpenWeatherMap.



Nếu gặp lỗi liên quan đến API, thử sử dụng API Key dự phòng bằng cách thay đổi giá trị API_KEY trong file weather_forecast_app.py.



Mã nguồn và giao diện web được thiết kế để dễ dàng mở rộng và bảo trì.
