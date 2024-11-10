from flask import Flask, jsonify, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Đọc dữ liệu và tiền xử lý
df = pd.read_csv("KNN_Dataset.csv", header=0, na_values="NA", comment='\t', sep=',', skipinitialspace=True)

# Tách dữ liệu thành các biến độc lập (X) và biến mục tiêu (y)
X = df.drop(columns=["Outcome"])  # Các cột đầu vào (ví dụ: Pregnancies, Glucose, ...)
y = df["Outcome"]  # Cột kết quả (Outcome)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ yêu cầu POST
    data = request.get_json()

    # Tạo DataFrame từ dữ liệu đầu vào
    input_data = pd.DataFrame([[
        data.get("Pregnancies"),
        data.get("Glucose"),
        data.get("BloodPressure"),
        data.get("SkinThickness"),
        data.get("Insulin"),
        data.get("BMI"),
        data.get("DiabetesPedigreeFunction"),
        data.get("Age")
    ]])

    # Chuẩn hóa dữ liệu đầu vào
    input_data_scaled = scaler.transform(input_data)

    # Dự đoán bằng mô hình KNN
    prediction = knn.predict(input_data_scaled)

    # Trả về kết quả dự đoán
    return jsonify({
        "prediction": int(prediction[0])  # Kết quả là 0 hoặc 1
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Đảm bảo cổng phù hợp với Render


