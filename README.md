Dự báo thời tiết Hà Nội bao gồm nhiệt độ và xác suất mưa trong 1 giờ tới

Thực hiện trên Ubuntu 20.04 + py3

API Key chính: d2d021ceb18c28747d865fd2b33d71ea

API Key dự phòng: b19d206dbc49beb17d642bcba1d7ed5b

Để huấn luyện lại mô hình hoặc tạo dữ liệu mới, cần xóa các file từ lần chạy trước:

rm weather_dataset.csv temp_model.h5 rain_model.h5 scaler.pkl loss_history.pkl

rm -rf static/*

Yêu cầu để chạy hệ thống:

sudo apt update
sudo apt install python3 python3-pip

Cài thư viện :

pip3 install requests pandas numpy tensorflow scikit-learn matplotlib flask

Chạy :

python3 weather_forecast_app.py

Sau đó mở link localhost trên terminal để đến trang wed hiện thị dự báo.
