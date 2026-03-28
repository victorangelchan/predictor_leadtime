import pandas as pd
import numpy as np
import sys
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename

# -- 0. GUARDAR ANÁLISIS EN TXT
Tk().withdraw()
ruta_txt = asksaveasfilename(
    defaultextension=".txt",
    filetypes=[("Archivo de texto", "*.txt")],
    title="Guardar análisis ejercicio 4 como..."
)
sys.stdout = open(ruta_txt, "w", encoding="utf-8")


# -- 1. CARGAR DATASETS DEL EJERCICIO 3
df_todos    = pd.read_csv('datos_con_etiqueta.csv',
                          parse_dates=['Shipping date', 'Delivery date'])
df_normales = pd.read_csv('datos_normales.csv',
                          parse_dates=['Shipping date', 'Delivery date'])

print('\nREGISTROS CARGADOS A PARTIR DE AMBAS FUENTES')
print(f'Cargado datos_con_etiqueta : {len(df_todos):,} registros')
print(f'Cargado datos_normales     : {len(df_normales):,} registros')

# -- 2. HISTÓRICO DE VOLUMEN MENSUAL (fuente: datos reales 2025)
VOLUMEN_MENSUAL = {
     1: 16350,   # Enero
     2: 13934,   # Febrero  ← mínimo del año
     3: 16150,   # Marzo
     4: 21691,   # Abril
     5: 16057,   # Mayo
     6: 41730,   # Junio    ← PICO MÁXIMO (final temporada RM)
     7: 21870,   # Julio
     8: 19814,   # Agosto
     9: 15796,   # Septiembre
    10: 13089,   # Octubre  ← segundo mínimo
    11: 24523,   # Noviembre (Black Friday)
    12: 36580,   # Diciembre (Navidad)
}
MEDIA_ANUAL = sum(VOLUMEN_MENSUAL.values()) / 12

print('\nDATOS HISTÓRICOS 2025')
print(f'Media mensual histórica    : {MEDIA_ANUAL:,.0f} envíos')
print(f'Mes de mayor volumen       : Junio ({VOLUMEN_MENSUAL[6]:,} envíos — índice {VOLUMEN_MENSUAL[6]/MEDIA_ANUAL:.2f}x)')
print(f'Mes de menor volumen       : Octubre ({VOLUMEN_MENSUAL[10]:,} envíos — índice {VOLUMEN_MENSUAL[10]/MEDIA_ANUAL:.2f}x)')


# -- 3. FUNCIÓN DE FEATURE ENGINEERING
def aplicar_features(df, nombre=''):
    df = df.copy()

    # ── FEATURE EXTRACTION desde fecha de envío ───────────────
    df['MES']              = df['Shipping date'].dt.month
    df['DIA_SEMANA']       = df['Shipping date'].dt.dayofweek
    df['SEMANA_ANO']       = df['Shipping date'].dt.isocalendar().week.astype(int)
    df['TRIMESTRE']        = df['Shipping date'].dt.quarter
    df['ES_FIN_DE_SEMANA'] = (df['DIA_SEMANA'] >= 5).astype(int)
    df['ES_TEMPORADA_ALTA'] = df['MES'].isin([6, 11, 12, 1]).astype(int)
    df['INDICE_VOLUMEN_MES'] = df['MES'].map(VOLUMEN_MENSUAL) / MEDIA_ANUAL

    # ── FEATURE INTERACTION: carrier + país destino ───────────
    df['CARRIER_PAIS'] = (
        df['Courier name'] + '_' +
        df['Country code of delivery country']
    )

    # ── VARIABLE NUMÉRICA: peso ───────────────────────────────
    df['PESO'] = pd.to_numeric(df['Shipment weight'], errors='coerce').fillna(1)

    # ── AGRUPAR CATEGORÍAS CON < 30 REGISTROS como OTROS ──────
    for col in ['Courier name', 'Country code of delivery country', 'CARRIER_PAIS']:
        conteo  = df[col].value_counts()
        raros   = conteo[conteo < 30].index
        n_raros = len(raros)
        df[col] = df[col].replace(raros, f'{col}_OTROS')
        if nombre:
            print(f'  [{nombre}] {col}: {n_raros} categorías agrupadas como OTROS')

    # ── ONE-HOT ENCODING ──────────────────────────────────────
    cols_cat = ['Courier name', 'Country code of delivery country', 'CARRIER_PAIS']
    df = pd.get_dummies(df, columns=cols_cat)

    # ── ELIMINAR COLUMNAS QUE NO SON FEATURES ─────────────────
    # UMBRAL_RUTA y TASA_ATASCO_RUTA se conservan — son features numéricas nuevas
    cols_drop = [
        'Order ID', 'Secondary order ID', 'OTN',
        'Courier tracking numbers',
        'Address of delivery point', 'Postcode of delivery point',
        'NIF delivery', 'Delivery city', 'Delivery country',
        'Name of contact person', 'Phone of contact person',
        'Email of contact person',
        'Products', 'Comments',
        'Alias', 'CMS', 'Branding', 'Name of warehouse',
        'Original payment method', 'Outvio payment method',
        'Order status',
        'Shipping cost', 'Order total',
        'Shipping date', 'Delivery date', 'Order date',
        'Shipment weight',
    ]
    df = df.drop(columns=cols_drop, errors='ignore')

    return df


# -- 4. APLICAR A LOS DOS DATASETS
print('\nAGRUPACIÓN DE CATEGORÍAS RARAS')
df_m1 = aplicar_features(df_todos,    nombre='M1')
df_m2 = aplicar_features(df_normales, nombre='M2')

print('\nCOLUMNAS EN NUEVOS DATASETS')
print(f'Dataset Modelo 1 (clasificación — todos):  {df_m1.shape[0]:,} filas, {df_m1.shape[1]} columnas')
print(f'Dataset Modelo 2 (regresión — normales):   {df_m2.shape[0]:,} filas, {df_m2.shape[1]} columnas')

# -- 5. VERIFICACIÓN
print('\nCOLUMNAS FINALES MODELO 1')
for i, col in enumerate(df_m1.columns.tolist()):
    print(f'  {i+1:3}. {col}')

print('\nMUESTRA DE FEATURES NUMÉRICAS (5 primeras filas)')
cols_num = ['MES', 'DIA_SEMANA', 'SEMANA_ANO', 'TRIMESTRE',
            'ES_FIN_DE_SEMANA', 'ES_TEMPORADA_ALTA',
            'INDICE_VOLUMEN_MES', 'PESO',
            'UMBRAL_RUTA', 'TASA_ATASCO_RUTA',
            'LEAD_TIME_DIAS', 'ATASCADO']
print(df_m1[[c for c in cols_num if c in df_m1.columns]].head(5).to_string())

print('\nVALORES NULOS TRAS FEATURE ENGINEERING')
nulos = df_m1.isnull().sum()
nulos_presentes = nulos[nulos > 0]
if len(nulos_presentes) > 0:
    print(nulos_presentes)
else:
    print('  Sin valores nulos ✓')

# -- 6. GUARDAR
df_m1.to_csv('datos_modelo1.csv', index=False)
df_m2.to_csv('datos_modelo2.csv', index=False)

print(f'\nGuardado: datos_modelo1.csv ({len(df_m1):,} registros, {df_m1.shape[1]} columnas)')
print(f'Guardado: datos_modelo2.csv ({len(df_m2):,} registros, {df_m2.shape[1]} columnas)')
print('\nEjercicio 4 completado.')

sys.stdout.close()