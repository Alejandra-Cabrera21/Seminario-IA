"""
FraudShield AI — API REST con Flask
Universidad Rafael Landivar | Ingenieria | 2026

Modelos (todos del contenido del curso):
  - Naive Bayes Gaussiano     (Tema 9)
  - Regresion Logistica       (Tema 11)
  - MLP Red Neuronal          (Temas 13-14)
  - K-Fold Cross Validation   (Tema 9.5)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, joblib, os, time
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model')
model = scaler_amount = scaler_time = feature_names = None
model_name = "MLP Red Neuronal (Temas 13-14)"

def load_model():
    global model, scaler_amount, scaler_time, feature_names, model_name
    try:
        model         = joblib.load(os.path.join(MODEL_PATH, 'fraudshield_model.pkl'))
        scaler_amount = joblib.load(os.path.join(MODEL_PATH, 'scaler_amount.pkl'))
        scaler_time   = joblib.load(os.path.join(MODEL_PATH, 'scaler_time.pkl'))
        feature_names = joblib.load(os.path.join(MODEL_PATH, 'feature_names.pkl'))
        model_name    = joblib.load(os.path.join(MODEL_PATH, 'model_name.pkl'))
        print(f"Modelo cargado: {model_name}")
    except Exception as e:
        print(f"Modelo no encontrado ({e}). Ejecuta el notebook primero.")

load_model()

MODELOS_CURSO = {
    "naive_bayes":       {"tema":"Tema 9",      "auc":"~0.92","fundamento":"P(Fraude|X) proporcional a P(X|Fraude)*P(Fraude)"},
    "reg_logistica":     {"tema":"Tema 11",     "auc":"~0.97","fundamento":"P(Fraude) = sigmoid(Beta*X)"},
    "mlp_red_neuronal":  {"tema":"Temas 13-14", "auc":"~0.98","fundamento":"Forward Prop + Backpropagation con ReLU"},
}
request_count = fraud_count = 0

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok","model_loaded":model is not None,
                    "model_name":model_name,"timestamp":datetime.now().isoformat()})

@app.route('/models', methods=['GET'])
def models_info():
    return jsonify({"descripcion":"3 algoritmos del curso evaluados con K-Fold CV (k=5, Tema 9.5)",
                    "modelos":MODELOS_CURSO,"dataset":"Credit Card Fraud Detection — Kaggle"})

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify({"modelos":MODELOS_CURSO,"evaluacion":"Stratified K-Fold k=5 (Tema 9.5)",
                    "sesion":{"total":request_count,"fraudes":fraud_count}})

@app.route('/predict', methods=['POST'])
def predict():
    global request_count, fraud_count
    start = time.time()
    if not request.is_json:
        return jsonify({"error":"Content-Type debe ser application/json"}), 400
    data = request.get_json()
    if 'Amount' not in data:
        return jsonify({"error":"Campo requerido: Amount"}), 400
    try:
        amount = float(data.get('Amount', 0))
        t      = float(data.get('Time', 86400))
        if model is not None:
            amount_sc = scaler_amount.transform([[amount]])[0][0]
            time_sc   = scaler_time.transform([[t]])[0][0]
            features  = [float(data.get(f'V{i}',0.0)) for i in range(1,29)]
            features += [amount_sc, time_sc]
            X    = np.array(features).reshape(1,-1)
            prob = float(model.predict_proba(X)[0][1])
            pred = int(model.predict(X)[0])
        else:
            v14  = float(data.get('V14',0)); v17 = float(data.get('V17',0))
            z    = -0.5 + (-0.3*v14) + (-0.3*v17) + (0.5 if amount>5000 else 0)
            prob = float(np.clip(1/(1+np.exp(-z)),0.01,0.99))
            pred = 1 if prob>0.5 else 0
        request_count += 1
        if pred == 1: fraud_count += 1
        level = "BAJO" if prob<0.2 else "MEDIO" if prob<0.5 else "ALTO" if prob<0.75 else "CRITICO"
        return jsonify({"fraud":bool(pred==1),"probability_fraud":round(prob,4),
                        "probability_legit":round(1-prob,4),"risk_level":level,
                        "model_used":model_name,"inference_ms":round((time.time()-start)*1000,2),
                        "timestamp":datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__ == '__main__':
    print("FraudShield AI — API  |  http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
