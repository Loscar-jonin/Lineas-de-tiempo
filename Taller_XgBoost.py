import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb

def graficar_regresion(y_real, y_pred, titulo):
    y_real = y_real.values.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    modelo = LinearRegression()
    modelo.fit(y_real, y_pred)
    y_fit = modelo.predict(y_real)

    r2 = r2_score(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_real, y_pred, color='black', s=30)
    plt.plot(y_real, y_fit, color='gray', linewidth=2)
    plt.title(titulo, fontsize=12)
    plt.xlabel("Observed", fontsize=10)
    plt.ylabel("Predicted", fontsize=10)

    texto = f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nn = {len(y_real)}"
    plt.text(0.05, 0.95, texto, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def procesar_hoja(nombre_hoja, nombre_columnas, titulo_proceso, split_ratio):
    df = pd.read_excel("Hoja De Datos.xlsx", sheet_name=nombre_hoja, skiprows=1)
    df.columns = nombre_columnas
    df = df.sort_values("Fecha").reset_index(drop=True)

    X = df[[nombre_columnas[1]]]
    y = df[nombre_columnas[2]]

    split_index = int(len(df) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # --- Métricas ---
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)

    print(f"\n📘 MODELO PARA: {titulo_proceso}")
    print(f"📊 Total de filas: {len(df)}")
    print(f"🟩 Entrenamiento: {len(X_train)} filas ({int(split_ratio*100)}%)")
    print(f"🟦 Prueba: {len(X_test)} filas ({int((1-split_ratio)*100)}%)")
    print(f"✅ RMSE Entrenamiento: {rmse_train:.4f}")
    print(f"✅ RMSE Prueba: {rmse_test:.4f}")
    print(f"📈 R² Entrenamiento: {r2_train:.4f}")
    print(f"📈 R² Prueba: {r2_test:.4f}")

    # -------- GRAFICA 1: Observado vs Predicho --------
    y_total_pred = np.concatenate([y_train_pred, y_test_pred])
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(y)), y.values, label='Real', color='red', marker='o', markersize=4)
    plt.plot(np.arange(len(y_total_pred)), y_total_pred, label='Prediccion', color='yellow', linestyle='--', marker='x', markersize=4)
    plt.title(f"{titulo_proceso}: Real vs Prediccion", fontsize=12)
    plt.xlabel("Data Index", fontsize=10)
    plt.ylabel(nombre_columnas[2], fontsize=10)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # -------- GRAFICA 2: Regresión lineal --------
    graficar_regresion(y, y_total_pred, f"Tendencia de {titulo_proceso.lower()}")

    # -------- GRAFICA 3: Tabla resumen (con título bien ubicado) --------
    tabla = pd.DataFrame({
        "Observado": y.values,
        "Predicho": y_total_pred
    }).round(4)

    tabla_mostrada = tabla.head(10)

    fig, ax = plt.subplots(figsize=(6, 3))  # Altura suficiente
    ax.axis('off')
    tabla_plot = ax.table(cellText=tabla_mostrada.values,
                          colLabels=tabla_mostrada.columns,
                          loc='center',
                          cellLoc='center')
    tabla_plot.scale(1.2, 1.5)

    plt.tight_layout()  # Ajusta primero
    fig.subplots_adjust(top=0.75)  # Luego deja espacio arriba
    fig.suptitle(f"Resumen de Predicción - {titulo_proceso} (Primeros 10 valores)", fontsize=12)
    plt.show()

# --- Et0: 60% entrenamiento ---
procesar_hoja(
    nombre_hoja="Evapotranspiration (Et0)",
    nombre_columnas=["Fecha", "Forecasting", "Observed"],
    titulo_proceso="Evapotranspiración",
    split_ratio=0.6
)

# --- Humedad del Suelo: 70% entrenamiento ---
procesar_hoja(
    nombre_hoja="SoilMoisture",
    nombre_columnas=["Fecha", "Forecasting", "Observed"],
    titulo_proceso="Humedad del Suelo",
    split_ratio=0.7
)
