import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
from datetime import datetime
from flask import Flask, render_template
import os
import pickle

# === Cấu hình Flask ===
app = Flask(__name__)

# === Đảm bảo thư mục static tồn tại ===
if not os.path.exists('static'):
    os.makedirs('static')

# === Cấu hình API ===
API_KEY = "d2d021ceb18c28747d865fd2b33d71ea"
CITY = "Hanoi"
FORECAST_URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

# === Hàm lấy dữ liệu thời tiết từ API ===
def fetch_weather():
    try:
        response = requests.get(FORECAST_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('cod') != '200':
            raise Exception(f"API error: {data.get('message')}")
        forecast = data['list'][0]  # Lấy dự báo gần nhất
        main = forecast['main']
        weather = {
            "temp": max(main['temp'], 15),  # Giới hạn nhiệt độ tối thiểu 15°C
            "feels_like": main['feels_like'],
            "humidity": main['humidity'],
            "pressure": main['pressure'],
            "pop": forecast.get('pop', 0),  # Xác suất mưa (0-1)
            "rain": forecast.get('rain', {}).get('3h', 0),  # Lượng mưa trong 3 giờ (mm)
            "timestamp": datetime.fromtimestamp(forecast['dt']).strftime("%Y-%m-%d %H:%M:%S")  # Sửa lỗi
        }
        return weather
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu thời tiết: {e}")
        return None

# === Tạo dữ liệu huấn luyện mẫu ===
def collect_data(n=200):
    rows = []
    print("Đang thu thập dữ liệu thời tiết...")
    for i in range(n):
        weather = fetch_weather()
        if weather:
            current_hour = datetime.now().hour
            # Nhiệt độ tương lai
            future_temp = weather['temp'] + np.random.normal(0, 1.0)  # Tăng nhiễu để mô phỏng biến động lớn
            future_temp = max(min(future_temp, 40), 15)  # Giới hạn 15-40°C
            if abs(future_temp - weather['temp']) > 10:  # Giới hạn chênh lệch 10°C
                future_temp = weather['temp'] + (10 if future_temp > weather['temp'] else -10)
            # Xác suất mưa tương lai
            future_pop = min(max(weather['pop'] + np.random.normal(0, 0.05), 0), 1)
            row = [weather['temp'], weather['humidity'], weather['pressure'], weather['pop'], weather['rain'],
                   future_temp, future_pop, weather['timestamp']]
            rows.append(row)
        else:
            print(f"Không thể lấy dữ liệu lần {i+1}")
        time.sleep(2)  # Tránh vượt giới hạn API
    df = pd.DataFrame(rows, columns=["temp", "humidity", "pressure", "pop", "rain", "future_temp", "future_pop", "timestamp"])
    if df.empty or df.isna().sum().sum() > 0:
        print("Dữ liệu thu thập không hợp lệ hoặc chứa giá trị NaN")
        return pd.DataFrame(columns=["temp", "humidity", "pressure", "pop", "rain", "future_temp", "future_pop", "timestamp"])
    # Kiểm tra dữ liệu bất thường
    if (df['temp'] < 15).any() or (df['future_temp'] < 15).any():  # Cập nhật ngưỡng kiểm tra
        print("Dữ liệu chứa nhiệt độ bất thường (<15°C)")
        return pd.DataFrame(columns=["temp", "humidity", "pressure", "pop", "rain", "future_temp", "future_pop", "timestamp"])
    print("Dữ liệu thu thập:")
    print(df.describe())
    print("Mẫu dữ liệu đầu tiên:")
    print(df.head())
    return df

# === Huấn luyện mô hình AI ===
def train_model(df):
    scaler = MinMaxScaler()
    X = df[["temp", "humidity", "pressure", "pop", "rain"]]
    y_temp = df["future_temp"]
    y_rain = df["future_pop"] >= 0.5  # Phân loại: mưa (True) hoặc không mưa (False)

    X_scaled = scaler.fit_transform(X)
    y_temp = np.array(y_temp)
    y_rain = np.array(y_rain)

    X_train, X_test, y_temp_train, y_temp_test, y_rain_train, y_rain_test = train_test_split(
        X_scaled, y_temp, y_rain, test_size=0.2, random_state=42)

    # Mô hình dự đoán nhiệt độ
    temp_model_path = 'temp_model.h5'
    scaler_path = 'scaler.pkl'
    history_path = 'loss_history.pkl'

    if os.path.exists(temp_model_path):
        try:
            temp_model = tf.keras.models.load_model(temp_model_path)
            print("Tải mô hình nhiệt độ từ temp_model.h5")
        except Exception as e:
            print(f"Lỗi khi tải mô hình nhiệt độ: {e}")
            temp_model = None
    else:
        temp_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        temp_model.compile(optimizer='adam', loss='mse')
        temp_history = temp_model.fit(X_train, y_temp_train, epochs=500, verbose=1, validation_split=0.2,
                                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)])
        temp_model.save(temp_model_path)
        print("Đã lưu mô hình nhiệt độ vào temp_model.h5")
        with open(history_path, 'wb') as f:
            pickle.dump(temp_history.history, f)
        print("Đã lưu history vào loss_history.pkl")

    # Mô hình dự đoán mưa
    rain_model_path = 'rain_model.h5'
    if os.path.exists(rain_model_path):
        try:
            rain_model = tf.keras.models.load_model(rain_model_path)
            print("Tải mô hình mưa từ rain_model.h5")
        except Exception as e:
            print(f"Lỗi khi tải mô hình mưa: {e}")
            rain_model = None
    else:
        rain_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        rain_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        rain_model.fit(X_train, y_rain_train, epochs=500, verbose=1, validation_split=0.2,
                       callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)])
        rain_model.save(rain_model_path)
        print("Đã lưu mô hình mưa vào rain_model.h5")

    # Lưu scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Đã lưu scaler vào scaler.pkl")

    # Đánh giá mô hình
    temp_loss = temp_model.evaluate(X_test, y_temp_test, verbose=0) if temp_model else None
    rain_accuracy = rain_model.evaluate(X_test, y_rain_test, verbose=0)[1] if rain_model else None
    print(f"MSE nhiệt độ trên tập kiểm tra: {temp_loss:.4f}" if temp_loss else "Không thể đánh giá mô hình nhiệt độ")
    print(f"Độ chính xác mưa trên tập kiểm tra: {rain_accuracy:.4f}" if rain_accuracy else "Không thể đánh giá mô hình mưa")

    # Tải history nếu tồn tại
    history = None
    if os.path.exists(history_path):
        try:
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            print("Tải history từ loss_history.pkl")
        except Exception as e:
            print(f"Lỗi khi tải history: {e}")

    return temp_model, rain_model, scaler, history

# === Dự đoán và lưu biểu đồ ===
def predict_and_visualize(temp_model, rain_model, scaler, df, history):
    current = fetch_weather()
    if not current:
        return None, None, None, None, None, False

    X_input = [[current['temp'], current['humidity'], current['pressure'], current['pop'], current['rain']]]
    try:
        X_input_scaled = scaler.transform(X_input)
        print("X_input:", X_input)
        print("X_input_scaled:", X_input_scaled)
        if (X_input_scaled < 0).any() or (X_input_scaled > 1).any():
            print("X_input_scaled ngoài khoảng [0, 1]")
    except Exception as e:
        print(f"Lỗi khi chuẩn hóa dữ liệu đầu vào: {e}")
        return None, None, None, None, None, False

    # Dự đoán nhiệt độ
    temp_prediction = temp_model.predict(X_input_scaled, verbose=0)[0][0]
    temp_prediction = max(min(temp_prediction, 40), 15)  # Giới hạn 15-40°C
    if abs(temp_prediction - current['temp']) > 10:  # Giới hạn chênh lệch 10°C
        temp_prediction = current['temp'] + (10 if temp_prediction > current['temp'] else -10)
    print("Dự đoán nhiệt độ:", temp_prediction)

    # Dự đoán mưa
    rain_prediction = rain_model.predict(X_input_scaled, verbose=0)[0][0]
    rain_status = "Có mưa" if rain_prediction >= 0.5 else "Không mưa"
    print("Dự đoán mưa:", rain_status, f"(Xác suất: {rain_prediction*100:.2f}%)")

    # Cảnh báo ngập lụt
    flood_warning = None
    if current['rain'] > 10 or rain_prediction * 100 > 70:
        flood_warning = "Cảnh báo: Nguy cơ ngập lụt cao!"
    elif current['rain'] > 5 or rain_prediction * 100 > 50:
        flood_warning = "Cảnh báo: Nguy cơ ngập lụt trung bình."
    else:
        flood_warning = "Không có nguy cơ ngập lụt."
    print("Cảnh báo ngập lụt:", flood_warning)

    # Biểu đồ nhiệt độ
    plt.figure(figsize=(10, 6))
    timestamps = df['timestamp'].iloc[-10:].tolist() + [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    temps = df['temp'].iloc[-10:].tolist() + [current['temp']]
    predicted_temps = df['future_temp'].iloc[-10:].tolist() + [temp_prediction]
    plt.plot(timestamps, temps, marker='o', label='Nhiệt độ thực tế (°C)', color='#1f77b4')
    plt.plot(timestamps, predicted_temps, marker='x', linestyle='--', label='Nhiệt độ dự đoán (°C)', color='#ff7f0e')
    plt.xticks(rotation=45)
    plt.xlabel('Thời gian')
    plt.ylabel('Nhiệt độ (°C)')
    plt.title('So sánh Nhiệt độ Thực tế và Dự đoán tại Hà Nội')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/temperature_plot.png')
    plt.close()

    # Biểu đồ xác suất mưa
    plt.figure(figsize=(10, 6))
    pops = df['pop'].iloc[-10:].tolist() + [current['pop']]
    predicted_pops = df['future_pop'].iloc[-10:].tolist() + [rain_prediction]
    plt.plot(timestamps, pops, marker='o', label='Xác suất mưa thực tế', color='#1f77b4')
    plt.plot(timestamps, predicted_pops, marker='x', linestyle='--', label='Xác suất mưa dự đoán', color='#ff7f0e')
    plt.xticks(rotation=45)
    plt.xlabel('Thời gian')
    plt.ylabel('Xác suất mưa')
    plt.title('So sánh Xác suất Mưa Thực tế và Dự đoán tại Hà Nội')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/rain_plot.png')
    plt.close()

    # Biểu đồ loss
    loss_plot_exists = False
    if history is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(history['loss'], label='Loss huấn luyện', color='#1f77b4')
        plt.plot(history['val_loss'], label='Loss xác thực', color='#ff7f0e')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Loss trong Quá trình Huấn luyện')
        plt.legend()
        plt.savefig('static/loss_plot.png')
        plt.close()
        loss_plot_exists = True

    return current, temp_prediction, rain_prediction, rain_status, flood_warning, loss_plot_exists

# === Route chính cho web ===
@app.route('/')
def index():
    if os.path.exists('weather_dataset.csv'):
        df = pd.read_csv('weather_dataset.csv')
        print("Tải dữ liệu từ weather_dataset.csv")
        print(df.describe())
        print("Mẫu dữ liệu đầu tiên:")
        print(df.head())
        if (df['temp'] < 15).any() or (df['future_temp'] < 15).any():  # Cập nhật ngưỡng kiểm tra
            print("Dữ liệu CSV chứa nhiệt độ bất thường (<15°C). Xóa và tạo lại...")
            os.remove('weather_dataset.csv')
            df = collect_data(n=200)
    else:
        df = collect_data(n=200)

    if df.empty:
        error = "Không thu thập được dữ liệu. Kiểm tra API key hoặc kết nối mạng."
        return render_template('index.html', error=error)
    df.to_csv("weather_dataset.csv", index=False)
    print("Đã lưu dữ liệu vào weather_dataset.csv")

    temp_model, rain_model, scaler, history = train_model(df)
    current, temp_prediction, rain_prediction, rain_status, flood_warning, loss_plot_exists = predict_and_visualize(
        temp_model, rain_model, scaler, df, history)

    if current is None:
        error = "Không thể lấy dữ liệu thời tiết hiện tại để dự đoán."
        return render_template('index.html', error=error)

    return render_template('index.html',
                           temp=current['temp'],
                           humidity=current['humidity'],
                           pressure=current['pressure'],
                           feels_like=current['feels_like'],
                           prediction=temp_prediction,
                           rain_prediction=rain_prediction,
                           rain_status=rain_status,
                           flood_warning=flood_warning,
                           mse=history['loss'][-1] if history else None,
                           timestamp=current['timestamp'],
                           loss_plot_exists=loss_plot_exists)

# === Main ===
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)