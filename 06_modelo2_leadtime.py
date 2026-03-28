import numpy as np
import sys
import joblib
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================================
# -- 0. GUARDAR ANÁLISIS EN TXT
# ============================================================

Tk().withdraw()
ruta_txt = asksaveasfilename(
    defaultextension=".txt",
    filetypes=[("Archivo de texto", "*.txt")],
    title="Guardar análisis ejercicio 6 como..."
)
sys.stdout = open(ruta_txt, "w", encoding="utf-8")

# ============================================================
# -- 1. CARGAR DATASET (SOLO ENVÍOS NORMALES)
# ============================================================

df = pd.read_csv('datos_modelo2.csv')

print(f'Registros cargados : {len(df):,}')
print(f'Columnas           : {df.shape[1]}')

TARGET = 'LEAD_TIME_DIAS'
EXCLUIR = ['LEAD_TIME_DIAS', 'UMBRAL_RUTA', 'TASA_ATASCO_RUTA']  # eliminamos la feature peligrosa

features = [c for c in df.columns if c not in EXCLUIR]

X = df[features]
y = df[TARGET]

print('\n=== ESTRUCTURA DE LA FUENTE ===')
print(f'Features usadas : {len(features)}')
print(f'Lead time medio : {y.mean():.2f} días')
print(f'Desv. estándar  : {y.std():.2f}')

# ============================================================
# -- 1b. Correlación de features con target
# ============================================================

print('\n=== CORRELACIÓN FEATURES vs TARGET ===')
corrs = df[features + [TARGET]].corr()[TARGET].sort_values(ascending=False)
for feat, corr in corrs.items():
    if feat != TARGET:
        print(f'{feat:25} : {corr:.3f}')

# ============================================================
# -- 2. SPLIT TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'\n=== SPLIT TRAIN / TEST ===')
print(f'  Train : {len(X_train):,} registros')
print(f'  Test  : {len(X_test):,} registros')

# ============================================================
# -- 3. ESCALADO
# ============================================================

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ============================================================
# -- 4. ENTRENAR MODELO DE REGRESIÓN
# ============================================================

reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print(f'\n=== ENTRENANDO MODELO DE REGRESIÓN ===')
reg.fit(X_train_sc, y_train)
print('Entrenamiento completado ✓')

# ============================================================
# -- 5. PREDICCIONES
# ============================================================

y_pred = reg.predict(X_test_sc)

# ============================================================
# -- 6. MÉTRICAS DE REGRESIÓN
# ============================================================

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f'\n=== MÉTRICAS DEL MODELO ===')
print(f'MAE  (error medio absoluto) : {mae:.2f} días')
print(f'RMSE (penaliza errores grandes) : {rmse:.2f} días')
print(f'R²   (explicación del modelo)   : {r2:.4f}')

print('\nInterpretación logística:')
print(f'- Error medio ≈ {mae:.1f} días → bastante preciso operativamente')
print(f'- RMSE mayor que MAE indica presencia de algunos errores grandes')
print(f'- R² cercano a 1 = buen ajuste')

# ============================================================
# -- 7. VALIDACIÓN CRUZADA
# ============================================================

print(f'\n=== VALIDACIÓN CRUZADA (5 folds) ===')
cv_scores = cross_val_score(reg, X_train_sc, y_train,
                           cv=5, scoring='neg_mean_absolute_error',
                           n_jobs=-1)

cv_mae = -cv_scores

print(f'MAE por fold : {[f"{s:.2f}" for s in cv_mae]}')
print(f'Media        : {cv_mae.mean():.2f}')
print(f'Desv. std    : {cv_mae.std():.2f}')
print('(std bajo = modelo estable)')

# ============================================================
# -- 8. IMPORTANCIA DE FEATURES
# ============================================================

importancias = pd.Series(reg.feature_importances_, index=features)
top20 = importancias.sort_values(ascending=False).head(20)

print(f'\n=== TOP 20 FEATURES MÁS IMPORTANTES ===')
for feat, imp in top20.items():
    barra = '█' * int(imp * 200)
    print(f'{feat:55} {imp:.4f} {barra}')

# ============================================================
# -- 9. GUARDAR MODELO
# ============================================================

joblib.dump(reg, 'modelo2_regresion.pkl')
joblib.dump(scaler, 'modelo2_scaler.pkl')
joblib.dump(features, 'modelo2_features.pkl')

print('\nGuardado:')
print('modelo2_regresion.pkl')
print('modelo2_scaler.pkl')
print('modelo2_features.pkl')

sys.stdout.close()

# ============================================================
# -- 10. GRÁFICOS
# ============================================================

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Real')
plt.ylabel('Predicho')
plt.title('Modelo 2 — Predicción Lead Time')

# línea perfecta
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val])

plt.savefig(r"C:\Users\vchan\Downloads\modelo2_pred_vs_real.png")