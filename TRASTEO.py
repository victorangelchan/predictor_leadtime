import pandas as pd
import matplotlib.pyplot as plt
import os


# -- 1. CARGA
df = pd.read_csv(r"C:\PROGRAMAS_CURSO_IA\PROGRAMA_TEMA_1\datos_modelo2.csv",
                 sep=',',
                 dtype=str
                 )



#df=pd.read_csv(r"C:\PROGRAMAS_CURSO_IA\TRANSITO_ARCHIVOS\RECORD_2025.csv",
#               sep=',',
#               dtype=str
#               )

# -- 2. IMPRIMIR TODAS LAS COLUMNAS
print("\n=== NOMBRE DE COLUMNAS ===")
print(df.columns.to_list())  # lista de todas las columnas


# -- 2. COMPROBAR REGISTROS EN DATASET
#print(f'Registros:{len(df)}')



# -- 3. TRANSFORMAR FECHAS
#df['FECHA_LIMPIA'] = pd.to_datetime(
#    df['FECHA_LIMPIA'],
#    errors='coerce'
#)


# -- 4. CREAR COLUMNA MES-AÑO
#df['MES_ANIO'] = df['FECHA_LIMPIA'].dt.to_period('M')

# -- 5. RESUMEN POR MES
#resumen_meses = df.groupby('MES_ANIO').size().reset_index(name='TOTAL_ENVIOS')

# Ordenar por fecha
#resumen_meses = resumen_meses.sort_values('MES_ANIO')

# -- 6. MOSTRAR RESULTADO
#print('\n=== RESUMEN DE ENVÍOS POR MES ===\n')
#print(resumen_meses.to_string(index=False))

