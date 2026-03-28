import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title='Predictor Lead Time',
    page_icon='🚚',
    layout='centered'
)

# ── CARGA ───────────────────────────────────────────────
@st.cache_resource
def cargar():
    m1 = joblib.load('modelo1_clasificador.pkl')
    m2 = joblib.load('modelo2_regresion.pkl')

    sc1 = joblib.load('modelo1_scaler.pkl')
    sc2 = joblib.load('modelo2_scaler.pkl')

    f1 = joblib.load('modelo1_features.pkl')
    f2 = joblib.load('modelo2_features.pkl')

    df = pd.read_csv('datos_modelo1.csv')

    return m1, m2, sc1, sc2, f1, f2, df

m1, m2, sc1, sc2, f1, f2, df_ref = cargar()

# ── FEATURES ALIGN ──────────────────────────────────────
def preparar(X, features):
    for col in features:
        if col not in X:
            X[col] = 0
    return X[features]

# ── LISTAS DINÁMICAS REALES ─────────────────────────────
transportistas = sorted(
    [c.replace('Courier name_', '') 
     for c in df_ref.columns if c.startswith('Courier name_')]
)

paises = sorted(
    [c.replace('Country code of delivery country_', '') 
     for c in df_ref.columns if c.startswith('Country code of delivery country_')]
)

# ── INTERFAZ ────────────────────────────────────────────
st.title('🚚 Predictor Lead Time')
st.divider()

col1, col2 = st.columns(2)

with col1:
    transportista = st.selectbox('Transportista', transportistas)
    mes = st.selectbox('Mes', list(range(1,13)))

with col2:
    pais_destino = st.selectbox('País destino', paises)

umbral = st.slider('Sensibilidad riesgo', 0.1, 0.6, 0.25)

# ── VALIDAR RUTA REAL ───────────────────────────────────
def ruta_valida(df, transportista, pais):
    col = f'CARRIER_PAIS_{transportista}_{pais}'
    return col in df.columns

# ── CONSTRUIR INPUT ─────────────────────────────────────
def construir(df, transportista, pais, mes):
    fila = pd.DataFrame(0, index=[0], columns=df.columns)

    # variables temporales
    fila['MES'] = mes
    fila['TRIMESTRE'] = (mes-1)//3 + 1
    fila['ES_TEMPORADA_ALTA'] = int(mes in [11,12,1])

    # one-hot carrier
    col_t = f'Courier name_{transportista}'
    if col_t in fila:
        fila[col_t] = 1

    # one-hot país
    col_p = f'Country code of delivery country_{pais}'
    if col_p in fila:
        fila[col_p] = 1

    # relación carrier-país
    col_cp = f'CARRIER_PAIS_{transportista}_{pais}'
    if col_cp in fila:
        fila[col_cp] = 1

    return fila.drop(columns=['LEAD_TIME_DIAS','ATASCADO'], errors='ignore')

# ── BOTÓN ───────────────────────────────────────────────
if st.button('Predecir'):

    if not ruta_valida(df_ref, transportista, pais_destino):
        st.error('❌ Este carrier no opera en ese país')
        st.stop()

    X = construir(df_ref, transportista, pais_destino, mes)

    X1 = preparar(X.copy(), f1)
    X2 = preparar(X.copy(), f2)

    X1 = sc1.transform(X1)
    X2 = sc2.transform(X2)

    # modelo riesgo
    prob = m1.predict_proba(X1)[0,1]
    riesgo = prob >= umbral

    # modelo regresión (ensemble)
    preds = np.array([t.predict(X2) for t in m2.estimators_])

    p10 = np.percentile(preds,10)
    p50 = np.percentile(preds,50)
    p90 = np.percentile(preds,90)

    st.divider()

    # resultado riesgo
    if riesgo:
        st.error(f'🔴 Alto riesgo ({prob*100:.1f}%)')
    else:
        st.success(f'🟢 Bajo riesgo ({prob*100:.1f}%)')

    # lead time
    st.subheader('Lead time')
    c1,c2,c3 = st.columns(3)

    c1.metric('P10', f'{p10:.1f} días')
    c2.metric('P50', f'{p50:.1f} días')
    c3.metric('P90', f'{p90:.1f} días')