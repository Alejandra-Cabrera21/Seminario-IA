# FraudShield AI
## Sistema de Deteccion de Fraude Bancario — Universidad Rafael Landivar

**Curso: Inteligencia Artificial | Primer Semestre 2026 | Facultad de Ingenieria**

---

## Descripcion

FraudShield AI aplica los tres algoritmos de clasificacion supervisada del curso de IA para detectar fraude bancario en tiempo real. El preprocesamiento sigue la metodologia ETL del Laboratorio 1 con Pandas.

| Algoritmo | Tema del Curso | CV AUC-ROC (k=5) |
|-----------|---------------|-----------------|
| Naive Bayes Gaussiano | Tema 9 | ~0.92 |
| Regresion Logistica | Tema 11 | ~0.97 |
| **MLP Red Neuronal** *(seleccionado)* | **Temas 13-14** | **~0.98** |

**Dataset:** Credit Card Fraud Detection (Kaggle) — 284,807 transacciones reales.

---

## Estructura

```
fraudshield-ai/
├── notebooks/FraudShield_AI_Training.ipynb  <- Pipeline ML completo
├── api/app.py                                <- API REST Flask
├── web/FraudShield_AI_Prototipo.html         <- App web funcional
├── docs/FraudShield_AI_Documento.docx        <- Documento academico
├── model/                                    <- .pkl generados por notebook
├── data/                                     <- Coloca creditcard.csv aqui
└── requirements.txt
```

---

## Instalacion Rapida

```bash
git clone https://github.com/tu-usuario/fraudshield-ai.git
cd fraudshield-ai
pip install -r requirements.txt

# 1. Descarga creditcard.csv de Kaggle y ponlo en data/
# 2. Ejecuta el notebook para entrenar los 3 modelos
jupyter notebook notebooks/FraudShield_AI_Training.ipynb

# 3. Lanza la API
cd api && python app.py

# 4. Abre web/FraudShield_AI_Prototipo.html en el navegador
```

---

## Algoritmos del Curso Implementados

```python
# Tema 9 — Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()  # P(Fraude|X) prop. P(X|Fraude)*P(Fraude)

# Tema 11 — Regresion Logistica
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1, class_weight='balanced')
# sigmoid(z) = 1/(1+e^{-z}), optimizacion gradiente descendente, reg. L2

# Temas 13-14 — MLP Red Neuronal
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu',
                    solver='adam', alpha=0.001, learning_rate='adaptive')
# ReLU (T.14.2), Forward Prop (T.14.3), BCE Loss (T.14.4), Backprop (T.14.5)

# Tema 9.5 — K-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold, cross_val_score
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X_train, y_train, cv=kfold, scoring='roc_auc')
```

---

## API REST

```bash
# Prediccion
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"Amount":125.50,"Time":86400,"V14":-3.5,"V17":-4.2}'

# Respuesta
{"fraud":true,"probability_fraud":0.89,"risk_level":"CRITICO",
 "model_used":"MLP Red Neuronal (Temas 13-14)","inference_ms":8.2}
```

---

## Equipo

| Integrante | Rol |
|-----------|-----|
| Integrante 1 | Lider / ML Engineer |
| Integrante 2 | Data Engineer — ETL con Pandas (Lab 1) |
| Integrante 3 | Backend Developer — Flask API |
| Integrante 4 | Frontend Developer — App Web |
| Integrante 5 | Documentacion y Presentacion |

---

## Referencias

- Russell & Norvig (2020). *Artificial Intelligence: A Modern Approach* (4a ed.)
- Goodfellow et al. (2016). *Deep Learning*. MIT Press.
- Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Scikit-learn: https://scikit-learn.org/

**Universidad Rafael Landivar | Inteligencia Artificial | 2026**
