import pandas as pd
import matplotlib.pyplot as plt
import sys
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
import numpy as np

# -- 0. DESCARGAR TXT ANALISIS
Tk().withdraw()
ruta_txt = asksaveasfilename(
    defaultextension = ".txt",
    filetypes =  [("Archivo de texto", "*.txt")],
    title = "Guardar archivo como..."
)

sys.stdout = open(ruta_txt, "w", encoding="utf-8")


# -- 1. CARGA
df=pd.read_csv(r"C:\PROGRAMAS_CURSO_IA\TRANSITO_ARCHIVOS\OTUVIO.csv",
               sep='\t',
               dtype=str
               )


# -- 2. COMPROBAR REGISTROS EN DATASET
print(f'Registros:{len(df)}')



# -- 3. TRANSFORMAR FECHAS
df['Shipping date'] = pd.to_datetime(df['Shipping date'], format = '%d-%m-%Y %H:%M:%S')
df['Delivery date'] = pd.to_datetime(df['Delivery date'], format = '%d-%m-%Y %H:%M:%S')



# -- 4. CALCULAR LEAD TIME
df['LEAD_TIME_DIAS'] = (df['Delivery date'].dt.normalize() - df['Shipping date'].dt.normalize()).dt.days

print('\n=== PRIMERAS FILAS ===\n')
print(df[['Shipping date', 'Delivery date', 'LEAD_TIME_DIAS']].head(10))





# -- 5. ESTADÍSTICAS BÁSICAS

negativos = (df['LEAD_TIME_DIAS']<0).sum()
cero = (df['LEAD_TIME_DIAS'] == 0).sum()
mas30 = (df['LEAD_TIME_DIAS']>30).sum()

print('\n=== DISTRIBUCIÓN DEL LEAD TIME ===')
print(f'\nLead times negativos (errores de datos):{negativos}')
print(f'Lead times = 0 (mismo día) : {cero}')
print(f'Lead times > 30 (mismo día) : {mas30}\n')
print(df['LEAD_TIME_DIAS'].describe().round(2))
print('\n=== VALORES NULOS ===')
nulos = df.isnull().sum()
pct = (nulos / len(df) * 100).round(2)
print(pd.DataFrame({'Nulos': nulos, 'Pct': pct})[nulos > 0])



# -- 6. CARDINALIDAD

print('\n=== VALORES ÚNICOS ===\n')

for col in ['Courier name', 'Country code of delivery country', 'Name of warehouse']:
    print(f'{col}: {df[col].nunique()} valores únicos')
    print(df[col].value_counts().head(10).to_string())
    print('\n')



    # -- 7. LIMPIEZA

print('\n=== TABLA TRAS LIMPIEZA ===')

# Unificar nombre de almacén con punto
df['Name of warehouse'] = df['Name of warehouse'].str.strip().str.rstrip('.')

# Excluir registros sin fecha de entrega (en tránsito)
df_modelo = df[df['LEAD_TIME_DIAS'].notna()].copy()
df_modelo = df_modelo[df_modelo['Name of warehouse'] != 'FIFA STORE P']
df_modelo.loc[df_modelo['LEAD_TIME_DIAS'] == 0, 'LEAD_TIME_DIAS'] = 1

print('\n-ELIMINACIÓN LEAD TIME NULOS-')
print(f'Registros originales  : {len(df):,}')
print(f'Registros para modelo : {len(df_modelo):,}')
print(f'Excluidos (en tránsito): {len(df) - len(df_modelo):,}')

print('\n-CONSOLIDACION DE ALMACENES-')
print(f'Registros tras eliminar FIFA STORE P: {len(df_modelo):,}')
print(df_modelo['Name of warehouse'].value_counts())


negativos = (df_modelo['LEAD_TIME_DIAS']<0).sum()
cero = (df_modelo['LEAD_TIME_DIAS'] == 0).sum()
mas30 = (df_modelo['LEAD_TIME_DIAS']>30).sum()

print('\n-TRANSFORMACIÓN LEAD TIMES 0 A 1')
print(f'Lead times negativos (errores de datos):{negativos}')
print(f'Lead times = 0 (mismo día) : {cero}')
print(f'Lead times > 30 (mismo día) : {mas30}')


# -- 8. PERCENTILES CLAVE
print('\n=== PERCENTILES DEL LEAD TIME ===\n')
for p in [25, 50, 75, 90, 95, 97, 99]:
    val = df_modelo['LEAD_TIME_DIAS'].quantile(p/100)
    print(f'  p{p:5.1f} --> {val:.1f} días')

print('\nEjercicio 1 completado.')


# -- 9. RESUMEN
print('\n\n=== RESUMEN FINAL ===\n')
print('\n-UMBRAL DE ATASCO-')
for p in [95]:
    val = df_modelo['LEAD_TIME_DIAS'].quantile(p/100)
    print(f'  EL VALOR DEL PERCENTIL {p:5.1f} ES DE {val:.1f} DÍAS O MÁS')

print('\n-ESTRUCTURA DE LAS PRINCIPALES VARIABLES-')
for col in ['Courier name']:
    print(f'  {col}: {df_modelo[col].nunique()} Transportistas en el modelo')
for col in ['Country code of delivery country']:
    print(f'  {col}: {df_modelo[col].nunique()} Países en el modelo')
    print('\n')

print('\n-REGISTROS CON ERROR EN LEADTIME EXCLUIDIOS-')
print('-SE HA GUARDADO UN GRÁFICOS EN LAS DESCARGAS PARA EXAMINAR LA FORMA DE LA COLA-')

sys.stdout.close()



# -- 10. GRAFICA
plt.figure()
df_modelo['LEAD_TIME_DIAS'].hist(bins=30)
plt.title('Distribución del Lead Time')
plt.xlabel('Días')
plt.ylabel('Frecuencia')
plt.xticks(np.arange(0, df_modelo['LEAD_TIME_DIAS'].max() + 5, 5))


plt.savefig(r"C:\Users\vchan\Downloads\grafico_leadtime.png")