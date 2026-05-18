# =======================================
# TAREA 3 (GRUPO 2)
# MODELO DE REGRESION MULTIPLE
# PARTIDO: UNIVERSIDAD DE CHILE VS EVERTON
# ========================================

# ENUNCIADO:
#PARTE 1: aplicar un modelo de regresion (el más conveniente)
#para predecir el resultado del partido entre Universidad de Chile vs Everton (se utilizaron 10 partidos de cada equipo: 5 local y 5 de visita)
#recomendaciones: utilizar regresión múltiple, colocar partidos anteriores y todos los datos posibles para una mayor presicion.
#PARTE 2: predecir 4 variables del partido Universidad de Chile vs Everton usando regresión


#datos:

#ultimos 5 partidos de Universidad de chile:
#Ñublense vs Universidad de Chile: 1-0
#Universidad de chile vs Deportes la serena: 4-0
#Audax Italiano vs Universidad de Chile: 1-3
#Universidad de Chile vs Union la calera: 0-1
#Universidad de Chile vs Deportes La serena: 2-2

#ultimos 5 partidos de visita Universidad de chile:
#Ñublense vs Universidad de Chile: 1-0
#Audax Italiano vs Universidad de Chile: 1-3
#Coquimbo Unido vs Universidad de Chile: 0-1
#Colo colo vs Universidad de Chile: 0-1
#Palestino vs Universidad de Chile: 0-0

#ultimos 5 partidos de Everton CD:
#DEportes la serena vs Everton CD: 1-0
#Everton CD vs ñublense: 0-0
#Everton CD vs Deportes Limache: 1-2
#Everton CD vs O'Higgins: 1-2
#Palestino vs Everton CD: 1-2

#ultimos 5 partidos de local Everton CD:
#Everton CD vs ñublense: 0-0
#Everton CD vs Deportes Limache: 1-2
#Everton CD vs O'Higgins: 1-2
#Everton CD vs Deportes Limache: 1-0
#Everton CD vs Huachipato: 0-3

#ultmos 5 partidos entre Everton CD vs Universidad de Chile:
#Universidad de Chile vs Everton CD: 2-0
#Everton CD vs Universidad de Chile: 2-0
#Universidad de Chile vs Everton CD: 1-1
#Unniversidad de Chile vs Everton CD: 1-0
#Everton CD vs Universidad de Chile: 2-1

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from statistics import mean, median, mode, StatisticsError

data = [
    # DATOS DE UNIVERSIDAD DE CHILE BPSAI
    #ultimos 5 partido
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 1}, # Ñublense vs U de Chile: 1-0
    {"equipo": "UdeChile", "local": 1, "gf": 4, "gc": 0}, # U de Chile vs La Serena: 4-0
    {"equipo": "UdeChile", "local": 0, "gf": 3, "gc": 1}, # Audax vs U de Chile: 1-3
    {"equipo": "UdeChile", "local": 1, "gf": 0, "gc": 1}, # U de Chile vs Unión La Calera: 0-1
    {"equipo": "UdeChile", "local": 1, "gf": 2, "gc": 2}, # U de Chile vs LA serena: 2-2

    #ultimos 5 de visita
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 1}, # Ñublense vs U de Chile: 1-0
    {"equipo": "UdeChile", "local": 0, "gf": 3, "gc": 1}, # Audax vs U de Chile: 1-3
    {"equipo": "UdeChile", "local": 0, "gf": 1, "gc": 0}, # Coquimbo vs U de Chile: 0-1
    {"equipo": "UdeChile", "local": 0, "gf": 1, "gc": 0}, # Colo colo vs U de Chile: 0-1
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 0}, # Palestino vs U de Chile: 0-0

    # DATOS DE everton
    #ultimos 5 partidos
    {"equipo": "Everton", "local": 0, "gf": 0, "gc": 1}, # La Serena vs Everton: 1-0
    {"equipo": "Everton", "local": 1, "gf": 0, "gc": 0}, # Everton vs Ñublense: 0-0
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2}, # Everton vs Deportes Limache: 1-2
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2}, # Everton vs O'Higgins: 1-2
    {"equipo": "Everton", "local": 0, "gf": 2, "gc": 1}, # Palestino vs Everton: 1-2

    #ultimos 5 de local
    {"equipo": "Everton", "local": 1, "gf": 0, "gc": 0}, # Everton vs Ñublense: 0-0
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2}, # Everton vs Deportes Limache: 1-2
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2}, # Everton vs O'Higgins: 1-2
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 0}, # Everton vs Deportes Limache: 1-0
    {"equipo": "Everton", "local": 1, "gf": 0, "gc": 3}, # Everton vs Huachipato: 0-3
]
# Convertir a data frame
df = pd.DataFrame(data)

# funcion aux

def estadisticas_equipo(equipo_name, local=None):
    #calcular el promedio de goles a favor y en contra
    #local=1 -> solo local
    #local=0 -> visita
    equipo_df = df[df['equipo'] == equipo_name].copy()
    if local is not None:
        equipo_df = equipo_df[equipo_df['local'] == local]
    return {
        "gf_promedio": equipo_df['gf'].mean(),
        "gc_promedio": equipo_df['gc'].mean(),
        "partidos": len(equipo_df)
    }

def calcular_media(datos):
    #Calcula la media (promedio) de un conjunto de datos
    if len(datos) == 0:
        return 0
    return sum(datos) / len(datos)

def calcular_mediana(datos):
    #Calcula la mediana de un conjunto de datos
    if len(datos) == 0:
        return 0
    return median(datos)

def calcular_moda(datos):
    #Calcula la moda de un conjunto de datos
    if len(datos) == 0:
        return 0
    try:
        return mode(datos)
    except StatisticsError:
        # Si no hay moda única, retorna el promedio
        return calcular_media(datos)

def calcular_r2(y_actual, y_predicho):
    #Calcula el coeficiente de determinación R² manualmente
    y_actual = np.array(y_actual)
    y_predicho = np.array(y_predicho)
    y_media = np.mean(y_actual)
    sst = np.sum((y_actual - y_media) ** 2)
    ssr = np.sum((y_actual - y_predicho) ** 2)
    if sst == 0:
        return 0
    r2 = 1 - (ssr / sst)
    return r2

# historial ultimos 5 partidos entre ambos equipos

h2_data = [
    {"gf_everton": 0, "gf_udechile": 2}, # U de Chile vs Everton: 2-0
    {"gf_everton": 2, "gf_udechile": 0}, # Everton vs U de Chile: 2-0
    {"gf_everton": 1, "gf_udechile": 1}, # U de Chile vs Everton: 1-1
    {"gf_everton": 0, "gf_udechile": 1}, # U de Chile vs Everton: 1-0
    {"gf_everton": 2, "gf_udechile": 1}, # Everton vs U de Chile: 2-1
]

h2_df = pd.DataFrame(h2_data)

h2_everton_promedio = h2_df["gf_everton"].mean()
h2_udechile_promedio = h2_df["gf_udechile"].mean()

#dataset para regresion multiple
rows = []
for _, row in df.iterrows():
    equipo = row['equipo']
    local = row['local']

    equipo_all = estadisticas_equipo(equipo)
    equipo_local = estadisticas_equipo(equipo, local=1)
    equipo_visita = estadisticas_equipo(equipo, local=0)

    rows.append({
        "equipo": equipo,
        "local": local,
        "equipo_gf_promedio": equipo_all['gf_promedio'],
        "equipo_gc_promedio": equipo_all['gc_promedio'],
        "equipo_gf_promedio_local": equipo_local['gf_promedio'],
        "equipo_gc_promedio_local": equipo_local['gc_promedio'],
        "equipo_gf_promedio_visita": equipo_visita['gf_promedio'],
        "equipo_gc_promedio_visita": equipo_visita['gc_promedio'],
        "gf_target": row['gf']
    })

model_df = pd.DataFrame(rows)

everton_df = model_df[model_df['equipo'] == 'Everton'].copy()
udechile_df = model_df[model_df['equipo'] == 'UdeChile'].copy()

features = [
    "local",
    "equipo_gf_promedio",
    "equipo_gc_promedio",
    "equipo_gf_promedio_local",
    "equipo_gc_promedio_local",
    "equipo_gf_promedio_visita",
    "equipo_gc_promedio_visita"
]

x_udechile = udechile_df[features]
y_udechile = udechile_df["gf_target"]

x_everton = everton_df[features]
y_everton = everton_df["gf_target"]

# entrenamiento modelos de regresión lineal

model_udechile = LinearRegression()
model_udechile.fit(x_udechile, y_udechile)

model_everton = LinearRegression()
model_everton.fit(x_everton, y_everton)

# modelos de arbol binario

tree_udechile = DecisionTreeRegressor(max_depth=3, random_state=42)

tree_everton = DecisionTreeRegressor(max_depth=3, random_state=42)

#MODELOS RANDOM FOREST 
rf_udechile = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
rf_everton = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

#Se entrenan los modelos    
tree_udechile.fit(x_udechile, y_udechile)
tree_everton.fit(x_everton, y_everton)

# Entrenar Random Forest
rf_udechile.fit(x_udechile, y_udechile)
rf_everton.fit(x_everton, y_everton)

# se evalua regresion lineal

# Predicciones
pred_train_udechile = model_udechile.predict(x_udechile)
pred_train_everton = model_everton.predict(x_everton)

# metricas u de chile

mse_udechile = mean_squared_error(y_udechile, pred_train_udechile)
r2_udechile = r2_score(y_udechile, pred_train_udechile)

# metricas everton

mse_everton = mean_squared_error(y_everton, pred_train_everton)
r2_everton = r2_score(y_everton, pred_train_everton)


# evaluacion arbol binario

pred_tree_udechile = tree_udechile.predict(x_udechile)
pred_tree_everton = tree_everton.predict(x_everton)

mse_tree_udechile = mean_squared_error(y_udechile, pred_tree_udechile)

r2_tree_udechile = r2_score(y_udechile, pred_tree_udechile)

mse_tree_everton = mean_squared_error(y_everton, pred_tree_everton)

r2_tree_everton = r2_score(y_everton, pred_tree_everton)

#EVALUACION RANDOM FOREST

pred_rf_udechile = rf_udechile.predict(x_udechile)
pred_rf_everton = rf_everton.predict(x_everton)

mse_rf_udechile = mean_squared_error(y_udechile, pred_rf_udechile)
r2_rf_udechile = calcular_r2(y_udechile, pred_rf_udechile)  # Usando función personalizada

mse_rf_everton = mean_squared_error(y_everton, pred_rf_everton)
r2_rf_everton = calcular_r2(y_everton, pred_rf_everton)  # Usando función personalizada

print("=" * 50)
print("METRICAS DE RANDOM FOREST")
print("=" * 50)
print(f"Universidad de Chile - MSE: {mse_rf_udechile:.4f}, R²: {r2_rf_udechile:.4f}")
print(f"Everton - MSE: {mse_rf_everton:.4f}, R²: {r2_rf_everton:.4f}")
print()

# datos a predecir

udechile_stats = estadisticas_equipo("UdeChile")
udechile_local = estadisticas_equipo("UdeChile", local=1)
udechile_visita = estadisticas_equipo("UdeChile", local=0)

everton_stats = estadisticas_equipo("Everton")
everton_local = estadisticas_equipo("Everton", local=1)
everton_visita = estadisticas_equipo("Everton", local=0)

# partido everton vs udechile

everton_partido = pd.DataFrame([{
    "local": 1,
    "equipo_gf_promedio": everton_stats['gf_promedio'],
    "equipo_gc_promedio": everton_stats['gc_promedio'],
    "equipo_gf_promedio_local": everton_local['gf_promedio'],
    "equipo_gc_promedio_local": everton_local['gc_promedio'],
    "equipo_gf_promedio_visita": everton_visita['gf_promedio'],
    "equipo_gc_promedio_visita": everton_visita['gc_promedio']
}])

udechile_partido = pd.DataFrame([{
    "local": 0,
    "equipo_gf_promedio": udechile_stats['gf_promedio'],
    "equipo_gc_promedio": udechile_stats['gc_promedio'],
    "equipo_gf_promedio_local": udechile_local['gf_promedio'],
    "equipo_gc_promedio_local": udechile_local['gc_promedio'],
    "equipo_gf_promedio_visita": udechile_visita['gf_promedio'],
    "equipo_gc_promedio_visita": udechile_visita['gc_promedio']
}])

#predecir goles
prediccion_everton = model_everton.predict(everton_partido)[0]
prediccion_udechile = model_udechile.predict(udechile_partido)[0]

#Predicciones con Random Forest
prediccion_everton_rf = rf_everton.predict(everton_partido)[0]
prediccion_udechile_rf = rf_udechile.predict(udechile_partido)[0]
prediccion_everton = (prediccion_everton + h2_everton_promedio) / 2
prediccion_udechile = (prediccion_udechile + h2_udechile_promedio) / 2

#Promedio con Random Forest
prediccion_everton_rf = (prediccion_everton_rf + h2_everton_promedio) / 2
prediccion_udechile_rf = (prediccion_udechile_rf + h2_udechile_promedio) / 2

goles_everton = max(0, round(prediccion_everton))
goles_udechile = max(0, round(prediccion_udechile))

goles_everton_rf = max(0, round(prediccion_everton_rf))
goles_udechile_rf = max(0, round(prediccion_udechile_rf))

# Calculando estadísticas para goles en la historia
goles_everton_historial = h2_df["gf_everton"].tolist()
goles_udechile_historial = h2_df["gf_udechile"].tolist()

print("ESTADISTICAS DE ENCUENTROS ANTERIORES")
print(f"Everton:")
print(f"  Media de goles: {calcular_media(goles_everton_historial):.2f}")
print(f"  Mediana de goles: {calcular_mediana(goles_everton_historial):.2f}")
print(f"  Moda de goles: {calcular_moda(goles_everton_historial):.2f}")
print()
print(f"Universidad de Chile:")
print(f"  Media de goles: {calcular_media(goles_udechile_historial):.2f}")
print(f"  Mediana de goles: {calcular_mediana(goles_udechile_historial):.2f}")
print(f"  Moda de goles: {calcular_moda(goles_udechile_historial):.2f}")
print()

# posibles tiros al arco y amarilla

def tiros_arco(goles_esperados):
    return round(max(1, goles_esperados * 4))

def amarillas(goles_esperados, equipo):
    base = 2
    if equipo == "Everton":
        return base + 1
    return base

tiros_everton = tiros_arco(prediccion_everton)
tiros_udechile = tiros_arco(prediccion_udechile)

amarillas_everton = amarillas(prediccion_everton, "Everton")
amarillas_udechile = amarillas(prediccion_udechile, "UdeChile")

#resultado final
print(f"Predicción del partido Everton vs Universidad de Chile:")
print(f"Everton:{prediccion_everton:.2f} goles esperados")
print(f"Universidad de Chile:{prediccion_udechile:.2f} goles esperados")
print("")

print(f"Marcador estimado: Everton {goles_everton} - {goles_udechile} Universidad de Chile")

if goles_everton > goles_udechile:
    print("Predicción: Everton gana")
elif goles_everton < goles_udechile:
    print("Predicción: Universidad de Chile gana")
else:
    print("Predicción: Empate")

#PREDICCIONES CON RANDOM FOREST
print()
print("PREDICCIONES CON RANDOM FOREST")
print(f"Everton: {prediccion_everton_rf:.2f} goles esperados")
print(f"Universidad de Chile: {prediccion_udechile_rf:.2f} goles esperados")
print()
print(f"Marcador estimado (RF): Everton {goles_everton_rf} - {goles_udechile_rf} Universidad de Chile")

if goles_everton_rf > goles_udechile_rf:
    print("Predicción (RF): Everton gana")
elif goles_everton_rf < goles_udechile_rf:
    print("Predicción (RF): Universidad de Chile gana")
else:
    print("Predicción (RF): Empate")

print()
#FIN RANDOM FOREST

print(f"Posibles tiros al arco:")
print(f"Everton: {tiros_everton} tiros al arco")
print(f"Universidad de Chile: {tiros_udechile} tiros al arco")
print("")
print(f"Posibles tarjetas amarillas:")
print(f"Everton: {amarillas_everton} tarjetas amarillas")
print(f"Universidad de Chile: {amarillas_udechile} tarjetas amarillas")
