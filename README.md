# giang_khnt

README : Thực hiện trên ubuntu 20.04

/**///////dưới đây là các file dữ liệu đã được tạo từ lần thực hiện trước, để làm lại mô hình mới cần xóa đi để tạo lại

weather_dataset.csv: Lưu dữ liệu thời tiết (nhiệt độ, độ ẩm, áp suất, xác suất mưa, lượng mưa, thời gian) từ API OpenWeatherMap.

temp_model.h5: Mô hình TensorFlow/Keras dự đoán nhiệt độ sau 1 giờ.

rain_model.h5: Mô hình TensorFlow/Keras dự đoán xác suất mưa.

scaler.pkl: Đối tượng MinMaxScaler để chuẩn hóa dữ liệu cho mô hình.

loss_history.pkl: Lịch sử huấn luyện (loss) của mô hình nhiệt độ.

temperature_plot.png: Biểu đồ so sánh nhiệt độ thực tế và dự đoán (lưu trong thư mục static/).

rain_plot.png: Biểu đồ so sánh xác suất mưa thực tế và dự đoán (lưu trong static/).

loss_plot.png (tùy chọn): Biểu đồ loss huấn luyện và xác thực (lưu trong static/, nếu có lịch sử huấn luyện).

Thư mục static/: Chứa các file hình ảnh biểu đồ, tự động tạo nếu chưa tồn tại./////**



Cài python3 và pip:

sudo apt update

sudo apt install python3 python3-pip

cài thư viện :

pip3 install requests pandas numpy tensorflow scikit-learn matplotlib flask

chạy:

python3 weather_forecast_app.py

Để tạo dữ liệu mới và huấn luyện lại:

rm weather_dataset.csv temp_model.h5 rain_model.h5 scaler.pkl loss_history.pkl rm -rf static/*

*khi huấn luyện lại thời gian khoảng 15p để lấy dữ liệu mới

api key từ openweather: d2d021ceb18c28747d865fd2b33d71ea

key dự phòng: b19d206dbc49beb17d642bcba1d7ed5b
