import numpy as np
import sys
import joblib
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
import pandas as pd
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, ConfusionMatrixDisplay)
from sklearn.preprocessing   import StandardScaler
import matplotlib.pyplot as plt

# -- 0. GUARDAR ANÁLISIS EN TXT
Tk().withdraw()
ruta_txt = asksaveasfilename(
    defaultextension=".txt",
    filetypes=[("Archivo de texto", "*.txt")],
    title="Guardar análisis ejercicio 5 como..."
)
sys.stdout = open(ruta_txt, "w", encoding="utf-8")


# ============================================================
# -- 1. CARGAR DATASET
# ============================================================
df = pd.read_csv('datos_modelo1.csv')

print(f'Registros cargados : {len(df):,}')
print(f'Columnas           : {df.shape[1]}')

TARGET  = 'ATASCADO'
EXCLUIR = ['ATASCADO', 'LEAD_TIME_DIAS']
features = [c for c in df.columns if c not in EXCLUIR]

X = df[features]
y = df[TARGET]

print('\n=== ESTRUCTURA DE LA FUENTE ===')
print(f'Features usadas    : {len(features)}')
print(f'  BAJO RIESGO (0) : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)')
print(f'  ALTO RIESGO (1) : {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)')


# ============================================================
# -- 2. SPLIT TRAIN / TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\n=== SPLIT TRAIN / TEST ===')
print(f'  Train : {len(X_train):,} registros  ({y_train.sum():,} atascos — {y_train.mean()*100:.1f}%)')
print(f'  Test  : {len(X_test):,}  registros  ({y_test.sum():,} atascos — {y_test.mean()*100:.1f}%)')


# ============================================================
# -- 3. ESCALADO (fit solo sobre train — evitar data leakage)
# ============================================================
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ============================================================
# -- 4. ENTRENAR EL MODELO
# ============================================================
clf = RandomForestClassifier(
    n_estimators = 300,
    max_depth    = 15,
    class_weight = 'balanced',
    random_state = 42,
    n_jobs       = -1
)

print(f'\n=== ENTRENANDO MODELO ===')
print(f'  RandomForestClassifier — n_estimators=300, max_depth=15, class_weight=balanced')
clf.fit(X_train_sc, y_train)
print(f'  Entrenamiento completado ✓')


# ============================================================
# -- 5. EVALUAR CON UMBRAL POR DEFECTO (0.50)
# ============================================================
y_proba = clf.predict_proba(X_test_sc)[:, 1]
y_pred_50 = clf.predict(X_test_sc)
roc_auc   = roc_auc_score(y_test, y_proba)

print(f'\n=== MÉTRICAS — UMBRAL POR DEFECTO (0.50) ===')
print(f'  ROC-AUC : {roc_auc:.4f}')
print(f'  (0.5=azar  |  >0.85=muy bueno  |  1.0=perfecto)\n')
print(classification_report(y_test, y_pred_50,
                             target_names=['BAJO RIESGO', 'ALTO RIESGO']))

cm_50 = confusion_matrix(y_test, y_pred_50)
vn, fp, fn, vp = cm_50.ravel()
print(f'  Verdaderos negativos (normal  → normal) : {vn:,}')
print(f'  Falsos positivos     (normal  → atasco) : {fp:,}  ← alarmas falsas')
print(f'  Falsos negativos     (atasco  → normal) : {fn:,}  ← atascos no detectados ⚠')
print(f'  Verdaderos positivos (atasco  → atasco) : {vp:,}')


# ============================================================
# -- 6. EVALUAR CON UMBRAL AJUSTADO (0.30)
#
#    Bajamos el umbral de decisión de 0.50 a 0.30:
#    el modelo alerta si la probabilidad de atasco supera el 30%.
#    Efecto: detectamos más atascos (más recall) a costa de
#    generar más alarmas falsas (menos precisión).
#    En logística es preferible: mejor avisar de más que
#    dejar pasar un atasco real sin detectar.
# ============================================================
UMBRAL_DECISION = 0.30
y_pred_30 = (y_proba >= UMBRAL_DECISION).astype(int)

print(f'\n=== MÉTRICAS — UMBRAL AJUSTADO ({UMBRAL_DECISION}) ===')
print(f'  ROC-AUC : {roc_auc:.4f}  (no cambia — es independiente del umbral)\n')
print(classification_report(y_test, y_pred_30,
                             target_names=['BAJO RIESGO', 'ALTO RIESGO']))

cm_30 = confusion_matrix(y_test, y_pred_30)
vn30, fp30, fn30, vp30 = cm_30.ravel()
print(f'  Verdaderos negativos (normal  → normal) : {vn30:,}')
print(f'  Falsos positivos     (normal  → atasco) : {fp30:,}  ← alarmas falsas')
print(f'  Falsos negativos     (atasco  → normal) : {fn30:,}  ← atascos no detectados ⚠')
print(f'  Verdaderos positivos (atasco  → atasco) : {vp30:,}')

print(f'\n=== COMPARATIVA UMBRAL 0.50 vs 0.30 ===')
print(f'  {"Métrica":<30} {"0.50":>8} {"0.30":>8} {"Cambio":>8}')
print(f'  {"─"*54}')
recall_50 = vp  / (vp  + fn)  if (vp  + fn)  > 0 else 0
recall_30 = vp30/ (vp30+ fn30) if (vp30+ fn30) > 0 else 0
prec_50   = vp  / (vp  + fp)  if (vp  + fp)  > 0 else 0
prec_30   = vp30/ (vp30+ fp30) if (vp30+ fp30) > 0 else 0
print(f'  {"Recall atascos detectados":<30} {recall_50:>7.1%} {recall_30:>7.1%} {recall_30-recall_50:>+7.1%}')
print(f'  {"Precisión alertas fiables":<30} {prec_50:>7.1%} {prec_30:>7.1%} {prec_30-prec_50:>+7.1%}')
print(f'  {"Atascos no detectados (FN)":<30} {fn:>8,} {fn30:>8,} {fn30-fn:>+8,}')
print(f'  {"Alarmas falsas (FP)":<30} {fp:>8,} {fp30:>8,} {fp30-fp:>+8,}')
print(f'\n  Conclusión: con umbral 0.30 detectamos más atascos reales')
print(f'  a cambio de más alarmas falsas — trade-off aceptable en logística')


# ============================================================
# -- 7. VALIDACIÓN CRUZADA
# ============================================================
print(f'\n=== VALIDACIÓN CRUZADA (5 folds) ===')
cv_scores = cross_val_score(clf, X_train_sc, y_train,
                             cv=5, scoring='roc_auc', n_jobs=-1)
print(f'  ROC-AUC por fold : {[f"{s:.4f}" for s in cv_scores]}')
print(f'  Media            : {cv_scores.mean():.4f}')
print(f'  Desv. estándar   : {cv_scores.std():.4f}')
print(f'  (std bajo = modelo estable)')


# ============================================================
# -- 8. IMPORTANCIA DE FEATURES — top 20
# ============================================================
importancias = pd.Series(clf.feature_importances_, index=features)
top20 = importancias.sort_values(ascending=False).head(20)

print(f'\n=== TOP 20 FEATURES MÁS IMPORTANTES ===')
for feat, imp in top20.items():
    barra = '█' * int(imp * 200)
    print(f'  {feat:55} {imp:.4f}  {barra}')


# ============================================================
# -- 9. GUARDAR MODELO, SCALER Y UMBRAL DE DECISIÓN
# ============================================================
joblib.dump(clf,              'modelo1_clasificador.pkl')
joblib.dump(scaler,           'modelo1_scaler.pkl')
joblib.dump(features,         'modelo1_features.pkl')
joblib.dump(UMBRAL_DECISION,  'modelo1_umbral_decision.pkl')

print(f'\nGuardado: modelo1_clasificador.pkl')
print(f'Guardado: modelo1_scaler.pkl')
print(f'Guardado: modelo1_features.pkl')
print(f'Guardado: modelo1_umbral_decision.pkl  (umbral={UMBRAL_DECISION})')

sys.stdout.close()


# ============================================================
# -- 10. GRÁFICOS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Modelo 1 — Clasificador de Riesgo de Atasco', fontsize=13, fontweight='bold')

# Gráfico 1 — Matriz de confusión umbral 0.50
ConfusionMatrixDisplay(confusion_matrix=cm_50,
                       display_labels=['BAJO RIESGO', 'ALTO RIESGO']
                       ).plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusión — Umbral 0.50\nROC-AUC = {roc_auc:.4f}')

# Gráfico 2 — Matriz de confusión umbral 0.30
ConfusionMatrixDisplay(confusion_matrix=cm_30,
                       display_labels=['BAJO RIESGO', 'ALTO RIESGO']
                       ).plot(ax=axes[1], colorbar=False, cmap='Oranges')
axes[1].set_title(f'Confusión — Umbral 0.30\n(más recall, más alarmas falsas)')

# Gráfico 3 — Importancia de features top 20
top20.sort_values().plot(kind='barh', ax=axes[2], color='steelblue')
axes[2].set_title('Top 20 Features más importantes')
axes[2].set_xlabel('Importancia')

plt.tight_layout()
plt.savefig(r"C:\Users\vchan\Downloads\05_modelo1_clasificador.png",
            dpi=120, bbox_inches='tight')
plt.show()