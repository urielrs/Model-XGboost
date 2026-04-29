from flask import Flask, render_template, jsonify, request
from sklearn.datasets import (
    load_breast_cancer, load_iris, load_wine, 
    load_digits, load_diabetes, fetch_california_housing,
    make_classification
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb
import numpy as np

app = Flask(__name__)

MODELOS = {
    "cancer": {"nombre": "Cáncer de Mama", "descripcion": "Clasificación de tumores.", "tipo": "clasificacion", "dataset": "breast_cancer"},
    "iris": {"nombre": "Flores Iris", "descripcion": "Especies de flores.", "tipo": "clasificacion", "dataset": "iris"},
    "wine": {"nombre": "Tipos de Vino", "descripcion": "Clasificación de vinos.", "tipo": "clasificacion", "dataset": "wine"},
    "digits": {"nombre": "Dígitos", "descripcion": "Números escritos a mano.", "tipo": "clasificacion", "dataset": "digits"},
    "diabetes": {"nombre": "Diabetes", "descripcion": "Regresión de enfermedad.", "tipo": "regresion", "dataset": "diabetes"},
    "california": {"nombre": "California Housing", "descripcion": "Precios de viviendas.", "tipo": "regresion", "dataset": "california"}
}

def cargar_dataset(nombre):
    if nombre == "breast_cancer": d = load_breast_cancer()
    elif nombre == "iris": d = load_iris()
    elif nombre == "wine": d = load_wine()
    elif nombre == "digits":
        d = load_digits()
        return d.data, d.target, [f"pixel_{i}" for i in range(64)], [str(i) for i in d.target_names]
    elif nombre == "diabetes": d = load_diabetes()
    elif nombre == "california": d = fetch_california_housing()
    return d.data, d.target, d.feature_names, getattr(d, 'target_names', ["valor"])

def top_features(modelo, feature_names):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[-10:][::-1]
    return [str(feature_names[i]) for i in indices], [round(float(importancias[i]), 4) for i in indices]

@app.route("/")
def index():
    return render_template("index.html", modelos=MODELOS)

@app.route("/modelo/<modelo_id>")
def modelo(modelo_id):
    info = MODELOS[modelo_id]
    X, y, features, target_names = cargar_dataset(info["dataset"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if info["tipo"] == "clasificacion":
        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, eval_metric="mlogloss", random_state=42).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f_names, f_vals = top_features(model, features)
        resultados = {
            "tipo": "clasificacion",
            "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
            "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
            "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)) * 100, 2),
            "matriz": confusion_matrix(y_test, y_pred).tolist(),
            "features": f_names, "importancias": f_vals,
            "clases": [str(c) for c in target_names],
            "conteo_predicciones": [int(np.sum(y_pred == i)) for i in range(len(target_names))]
        }
    else:
        model = xgb.XGBRegressor(n_estimators=120, max_depth=4, objective="reg:squarederror", random_state=42).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f_names, f_vals = top_features(model, features)
        resultados = {
            "tipo": "regresion",
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 3),
            "mse": round(float(mean_squared_error(y_test, y_pred)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
            "r2": round(float(r2_score(y_test, y_pred)) * 100, 2),
            "features": f_names, "importancias": f_vals,
            "reales": np.round(y_test[:20], 2).tolist(),
            "predichos": np.round(y_pred[:20], 2).tolist()
        }
    return jsonify({"nombre": info["nombre"], "descripcion": info["descripcion"], "resultados": resultados})

@app.route("/evaluar_credito", methods=["POST"])
def evaluar_credito():
    data = request.get_json()
    age = float(data.get("age", 0))
    if age < 18 or age > 80:
        return jsonify({"error": "La edad debe estar entre 18 y 80 años"}), 400

    X_s, y_s = make_classification(n_samples=2000, n_features=6, n_informative=4, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, eval_metric="logloss").fit(X_s, y_s)
    
    entrada = np.array([[
        float(data['limit_bal'])/100000, age/100, float(data['pay_0']), 
        float(data['pay_2']), float(data['bill_amt1'])/100000, float(data['pay_amt1'])/10000
    ]])
    
    pred = int(model.predict(entrada)[0])
    prob = float(model.predict_proba(entrada)[0][1]) * 100
    return jsonify({"riesgo": "Riesgo alto" if pred == 1 else "Riesgo bajo", "probabilidad": round(prob, 2)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
