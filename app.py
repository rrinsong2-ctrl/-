from flask import Flask, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# AI 모델 불러오기
model = joblib.load("methanol_model.pkl")

@app.route("/")

def home():

    # 센서 데이터 읽기
    df = pd.read_csv("sensor_data.csv")

    # 마지막 데이터 사용
    sample = df.iloc[-1:].values

    # AI 예측
    prediction = model.predict(sample)

    gas = prediction[0]

    danger = False

    if gas == "Methanol":
        danger = True

    return render_template(
        "index.html",
        gas=gas,
        danger=danger
    )

if __name__ == "__main__":
    app.run(debug=True)
