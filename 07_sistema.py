# ============================================================
# EJERCICIO 7: Ensamblaje + Validacion cruzada K-Fold
# Script: 07_sistema.py
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection  import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble         import RandomForestClassifier, RandomForestRegressor
import joblib

# ── 1. VALIDACION CRUZADA MODELO 1 ───────────────────────
print('=== VALIDACION CRUZADA K-FOLD (K=5) — MODELO 1 ===')
df1 = pd.read_csv('datos_modelo1.csv')
X1  = df1.drop(columns=['LEAD_TIME_DIAS','ATASCADO'], errors='ignore')
y1  = df1['ATASCADO']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=100,
                              class_weight='balanced', random_state=42)

scores_roc = cross_val_score(clf, X1, y1, cv=skf, scoring='roc_auc')
scores_rec = cross_val_score(clf, X1, y1, cv=skf, scoring='recall')

print(f'ROC-AUC por fold: {scores_roc.round(3)}')
print(f'ROC-AUC media:    {scores_roc.mean():.3f} +/- {scores_roc.std():.3f}')
print(f'Recall   media:   {scores_rec.mean():.3f} +/- {scores_rec.std():.3f}')
print('Desviacion baja = modelo estable y generalizable')

# ── 2. VALIDACION CRUZADA MODELO 2 ───────────────────────
print('\n=== VALIDACION CRUZADA K-FOLD (K=5) — MODELO 2 ===')
df2 = pd.read_csv('datos_modelo2.csv')
X2  = df2.drop(columns=['LEAD_TIME_DIAS','ATASCADO'], errors='ignore')
y2  = df2['LEAD_TIME_DIAS']

kf  = KFold(n_splits=5, shuffle=True, random_state=42)
reg = RandomForestRegressor(n_estimators=200, random_state=42)

scores_mae = cross_val_score(reg, X2, y2, cv=kf,
                             scoring='neg_mean_absolute_error')
scores_r2  = cross_val_score(reg, X2, y2, cv=kf, scoring='r2')

print(f'MAE por fold:  {(-scores_mae).round(3)}')
print(f'MAE media:     {-scores_mae.mean():.3f} +/- {scores_mae.std():.3f} dias')
print(f'R2  media:     {scores_r2.mean():.3f} +/- {scores_r2.std():.3f}')

# ── 3. SISTEMA COMPLETO ─────────────────────────────────
print('\n=== SISTEMA COMPLETO: CARGA Y PRUEBA ===')

# Cargar modelos y artefactos
m1 = joblib.load('modelo1_clasificador.pkl')
m2 = joblib.load('modelo2_regresion.pkl')

sc1 = joblib.load('modelo1_scaler.pkl')
sc2 = joblib.load('modelo2_scaler.pkl')

features_m1 = joblib.load('modelo1_features.pkl')
features_m2 = joblib.load('modelo2_features.pkl')

# ── FUNCION AUXILIAR ────────────────────────────────────
def preparar_features(X, features_modelo):
    X = X.copy()

    # Añadir columnas faltantes
    for col in features_modelo:
        if col not in X.columns:
            X[col] = 0

    # Mantener solo las necesarias y en orden correcto
    X = X[features_modelo]

    return X

# ── FUNCION PRINCIPAL ───────────────────────────────────
def predecir_envio(X_nuevo_m1, X_nuevo_m2, umbral=0.35):

    # Preparar datos
    X1 = preparar_features(X_nuevo_m1, features_m1)
    X2 = preparar_features(X_nuevo_m2, features_m2)

    # Escalar
    X1_sc = sc1.transform(X1)
    X2_sc = sc2.transform(X2)

    # Modelo 1: clasificación
    prob_atasco = m1.predict_proba(X1_sc)[0, 1]
    es_riesgo   = prob_atasco >= umbral

    # Modelo 2: regresión con intervalo
    preds = np.array([t.predict(X2_sc) for t in m2.estimators_])

    p10 = float(np.percentile(preds, 10))
    p50 = float(np.percentile(preds, 50))
    p90 = float(np.percentile(preds, 90))

    return {
        'riesgo'             : 'ALTO' if es_riesgo else 'BAJO',
        'prob_atasco_pct'    : round(prob_atasco * 100, 1),
        'dias_minimo'        : round(p10, 1),
        'dias_estimado'      : round(p50, 1),
        'dias_maximo'        : round(p90, 1),
    }

# ── PRUEBA ──────────────────────────────────────────────
df1_test = pd.read_csv('datos_modelo1.csv').tail(5)
df2_test = pd.read_csv('datos_modelo2.csv').tail(5)

X1t = df1_test.drop(columns=['LEAD_TIME_DIAS','ATASCADO'], errors='ignore')
X2t = df2_test.drop(columns=['LEAD_TIME_DIAS','ATASCADO'], errors='ignore')

for i in range(5):
    r = predecir_envio(
        X1t.iloc[[i]],
        X2t.iloc[[i]]
    )

    print(f'Envio {i+1}: {r["riesgo"]:4s} riesgo ({r["prob_atasco_pct"]}%) | '
          f'{r["dias_minimo"]}-{r["dias_maximo"]} dias (est: {r["dias_estimado"]})')

print('\nEjercicio 7 completado. Sistema listo para produccion.')