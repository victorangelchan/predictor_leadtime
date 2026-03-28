import pandas as pd
import matplotlib.pyplot as plt
import sys
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
import numpy as np


# -- 0. DESCARGAR TXT ANALISIS
Tk().withdraw()
ruta_txt = asksaveasfilename(
    defaultextension=".txt",
    filetypes=[("Archivo de texto", "*.txt")],
    title="Guardar archivo como..."
)
sys.stdout = open(ruta_txt, "w", encoding="utf-8")

# -- 1. CARGA
df = pd.read_csv(r"C:\PROGRAMAS_CURSO_IA\TRANSITO_ARCHIVOS\OTUVIO.csv",
                 sep='\t', dtype=str)

# -- 2. COMPROBAR REGISTROS
print(f'Registros: {len(df)}')

# -- 3. TRANSFORMAR FECHAS
df['Shipping date'] = pd.to_datetime(df['Shipping date'], format='%d-%m-%Y %H:%M:%S')
df['Delivery date'] = pd.to_datetime(df['Delivery date'], format='%d-%m-%Y %H:%M:%S')

# -- 4. CALCULAR LEAD TIME
df['LEAD_TIME_DIAS'] = (
    df['Delivery date'].dt.normalize() - df['Shipping date'].dt.normalize()
).dt.days

# -- 5. LIMPIEZA
df['Name of warehouse'] = df['Name of warehouse'].str.strip().str.rstrip('.')
df_modelo = df[df['LEAD_TIME_DIAS'].notna()].copy()
df_modelo = df_modelo[df_modelo['Name of warehouse'] != 'FIFA STORE P']
df_modelo.loc[df_modelo['LEAD_TIME_DIAS'] == 0, 'LEAD_TIME_DIAS'] = 1


# ============================================================
# -- 6. UMBRAL HÍBRIDO POR RUTA
#
#    Si la ruta tiene >= 30 registros → IQR local
#    Si tiene < 30 registros          → umbral global como fallback
# ============================================================

MIN_REGISTROS_RUTA = 30

Q1_global  = df_modelo['LEAD_TIME_DIAS'].quantile(0.25)
Q3_global  = df_modelo['LEAD_TIME_DIAS'].quantile(0.75)
IQR_global = Q3_global - Q1_global
UMBRAL_GLOBAL = Q3_global + 1.5 * IQR_global

print(f'\n=== UMBRAL GLOBAL (fallback para rutas con pocos datos) ===')
print(f'  Q1={Q1_global:.1f}  Q3={Q3_global:.1f}  IQR={IQR_global:.1f}  Umbral={UMBRAL_GLOBAL:.1f} días')

# Columna de ruta
df_modelo['RUTA'] = (
    df_modelo['Courier name'] + '_' +
    df_modelo['Country code of delivery country']
)

# Calcular umbral y tasa de atasco por ruta en dos pasadas:
# Primera pasada: calcular umbrales
umbrales_ruta = {}
rutas_con_umbral_propio = []
rutas_con_umbral_global = []

for ruta, grupo in df_modelo.groupby('RUTA'):
    n = len(grupo)
    if n >= MIN_REGISTROS_RUTA:
        q1  = grupo['LEAD_TIME_DIAS'].quantile(0.25)
        q3  = grupo['LEAD_TIME_DIAS'].quantile(0.75)
        iqr = q3 - q1
        umbral_ruta = q3 + 1.5 * iqr
        umbrales_ruta[ruta] = umbral_ruta
        rutas_con_umbral_propio.append({
            'Ruta': ruta, 'N': n,
            'Q1': q1, 'Q3': q3, 'IQR': iqr, 'Umbral': umbral_ruta
        })
    else:
        umbrales_ruta[ruta] = UMBRAL_GLOBAL
        rutas_con_umbral_global.append({'Ruta': ruta, 'N': n})

df_umbrales = pd.DataFrame(rutas_con_umbral_propio).sort_values('Umbral', ascending=False)
print(f'\n=== UMBRALES POR RUTA (mín. {MIN_REGISTROS_RUTA} registros) ===')
print(df_umbrales[['Ruta', 'N', 'Q1', 'Q3', 'IQR', 'Umbral']].to_string(index=False))
print(f'\n  Rutas con umbral propio : {len(rutas_con_umbral_propio)}')
print(f'  Rutas con umbral global : {len(rutas_con_umbral_global)}')
print(f'  Total rutas             : {len(umbrales_ruta)}')

# Aplicar umbral híbrido
df_modelo['UMBRAL_RUTA'] = df_modelo['RUTA'].map(umbrales_ruta)
df_modelo['ATASCADO']    = (df_modelo['LEAD_TIME_DIAS'] > df_modelo['UMBRAL_RUTA']).astype(int)

# Segunda pasada: calcular tasa histórica de atasco por ruta
# IMPORTANTE: se calcula DESPUÉS de etiquetar, con los datos reales
# Esta tasa le dice al modelo cuán problemática es históricamente cada ruta
TASA_GLOBAL = df_modelo['ATASCADO'].mean()

tasa_por_ruta = (
    df_modelo.groupby('RUTA')['ATASCADO']
    .mean()
    .to_dict()
)

df_modelo['TASA_ATASCO_RUTA'] = df_modelo['RUTA'].map(tasa_por_ruta).fillna(TASA_GLOBAL)

print(f'\n=== TASA HISTÓRICA DE ATASCO POR RUTA (top 15) ===')
tasas_df = (pd.Series(tasa_por_ruta)
              .sort_values(ascending=False)
              .head(15)
              .reset_index())
tasas_df.columns = ['Ruta', 'Tasa']
tasas_df['Tasa_pct'] = (tasas_df['Tasa'] * 100).round(1)
print(tasas_df.to_string(index=False))


# -- 7. BALANCE DE CLASES
conteo = df_modelo['ATASCADO'].value_counts()
pct    = df_modelo['ATASCADO'].value_counts(normalize=True) * 100

print(f'\n=== BALANCE DE CLASES (umbral híbrido) ===')
print(f'  BAJO RIESGO (0): {conteo[0]:,} envíos ({pct[0]:.1f}%)')
print(f'  ALTO RIESGO (1): {conteo[1]:,} envíos ({pct[1]:.1f}%)')

atascado_global  = (df_modelo['LEAD_TIME_DIAS'] > UMBRAL_GLOBAL).sum()
atascado_hibrido = conteo[1]
print(f'\n  Con umbral global  : {atascado_global:,} atascados')
print(f'  Con umbral híbrido : {atascado_hibrido:,} atascados')
print(f'  Diferencia         : {atascado_hibrido - atascado_global:+,} envíos reclasificados')


# -- 8. CONCENTRACIÓN DE ATASCOS
print('\n=== TASA DE ATASCO POR TRANSPORTISTA ===')
tasa_t = (df_modelo.groupby('Courier name')['ATASCADO']
            .agg(['sum', 'mean', 'count'])
            .rename(columns={'sum': 'atascos', 'mean': 'tasa', 'count': 'total'})
            .sort_values('tasa', ascending=False))
tasa_t['tasa_pct'] = (tasa_t['tasa'] * 100).round(1)
print(tasa_t[['total', 'atascos', 'tasa_pct']].to_string())

print('\n=== TOP 10 RUTAS CON MÁS ATASCOS ===')
tasa_r = (df_modelo.groupby('RUTA')['ATASCADO']
            .agg(['sum', 'mean', 'count'])
            .rename(columns={'sum': 'atascos', 'mean': 'tasa', 'count': 'total'})
            .query('total >= 30')
            .sort_values('tasa', ascending=False)
            .head(10))
tasa_r['tasa_pct'] = (tasa_r['tasa'] * 100).round(1)
print(tasa_r[['total', 'atascos', 'tasa_pct']].to_string())


# -- 9. GUARDAR
# UMBRAL_RUTA y TASA_ATASCO_RUTA se conservan — son features para el modelo
# RUTA se elimina porque en el ejercicio 4 se reconstruye desde Courier+País
df_modelo = df_modelo.drop(columns=['RUTA'])

df_modelo.to_csv('datos_con_etiqueta.csv', index=False)

df_normales = df_modelo[df_modelo['ATASCADO'] == 0].copy()
df_normales.to_csv('datos_normales.csv', index=False)

print(f'\nGuardado: datos_con_etiqueta.csv ({len(df_modelo):,} registros)')
print(f'Guardado: datos_normales.csv ({len(df_normales):,} registros)')
print('\nEjercicio 3 completado.')

sys.stdout.close()