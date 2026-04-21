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

# ====================
# PARTIDOS UTILIZADOS
# ====================

# ULTIMOS PARTIDOS DE VISITA UNIVERSIDAD DE CHILE:
# Ñublense vs Universidad de Chile: 1-0
# Coquimbo Unido vs Universidad de Chile: 0-1
# Colo-Colo vs Universidad de Chile: 0-1
# Palestino vs Universidad de Chile: 0-0

# ULTIMOS PARTIDOS DE LOCAL UNIVERSIDAD DE CHILE:
# Universidad de Chile vs La Serena: 4-0
# Universidad de Chile vs U. de Concepción: 1-1
# Universidad de Chile vs Deportes Limache: 2-2
# Universidad de Chile vs Audax Italiano: 0-0
# Universidad de Chile vs Coquimbo Unido: 1-1

# ULTIMOS PARTIDOS DE VISITA EVERTON:
# Deportes La Serena vs Everton: 1-0
# Audax Italiano vs Everton: 1-0
# Colo-Colo vs Everton: 2-0
# U. de Concepción vs Everton: 0-3
# Universidad Católica vs Everton: 2-2

# ULTIMOS PARTIDOS DE LOCAL EVERTON:
# Everton vs Ñublense: 0-0
# Everton vs Deportes Limache: 1-2
# Everton vs O'Higgins: 1-2
# Everton vs Deportes Limache: 1-0
# Everton vs Huachipato: 0-3

# ULTIMOS ENFRENTAMIENTOS ENTRE AMBOS:
# Universidad de Chile vs Everton: 2-0
# Everton vs Universidad de Chile: 2-0
# Universidad de Chile vs Everton: 1-1
# Universidad de Chile vs Everton: 1-0
# Everton vs Universidad de Chile: 2-1

import pandas as pd
from sklearn.linear_model import LinearRegression


data = [
        # UNIVERSIDAD DE CHILE
    
     # Visita
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 1, "posesion": 63, "amarillas": 3},  # Ñublense 1-0 U
    {"equipo": "UdeChile", "local": 0, "gf": 1, "gc": 0, "posesion": 50, "amarillas": 2},  # Coquimbo 0-1 U
    {"equipo": "UdeChile", "local": 0, "gf": 1, "gc": 0, "posesion": 42, "amarillas": 2},  # Colo-Colo 0-1 U
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 0, "posesion": 59, "amarillas": 1},  # Palestino 0-0 U

    # Local
    {"equipo": "UdeChile", "local": 1, "gf": 4, "gc": 0, "posesion": 49, "amarillas": 4},  # U 4-0 La Serena
    {"equipo": "UdeChile", "local": 1, "gf": 1, "gc": 1, "posesion": 67, "amarillas": 2},  # U 1-1 U. de Concepción
    {"equipo": "UdeChile", "local": 1, "gf": 2, "gc": 2, "posesion": 65, "amarillas": 1},  # U 2-2 Limache
    {"equipo": "UdeChile", "local": 1, "gf": 0, "gc": 0, "posesion": 39, "amarillas": 2},  # U 0-0 Audax
    {"equipo": "UdeChile", "local": 1, "gf": 1, "gc": 1, "posesion": 62, "amarillas": 3},  # U 1-1 Coquimbo
    
   
        # EVERTON
   
    # Visita
    {"equipo": "Everton", "local": 0, "gf": 0, "gc": 1, "posesion": 48, "amarillas": 3},  # La Serena 1-0 Everton
    {"equipo": "Everton", "local": 0, "gf": 0, "gc": 1, "posesion": 47, "amarillas": 4},  # Audax 1-0 Everton
    {"equipo": "Everton", "local": 0, "gf": 0, "gc": 2, "posesion": 47, "amarillas": 4},  # Colo-Colo 2-0 Everton
    {"equipo": "Everton", "local": 0, "gf": 3, "gc": 0, "posesion": 46, "amarillas": 2},  # U. de Concepción 0-3 Everton
    {"equipo": "Everton", "local": 0, "gf": 2, "gc": 2, "posesion": 34, "amarillas": 2},  # U. Católica 2-2 Everton

    # Local
    {"equipo": "Everton", "local": 1, "gf": 0, "gc": 0, "posesion": 49, "amarillas": 3},  # Everton 0-0 Ñublense
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2, "posesion": 54, "amarillas": 3},  # Everton 1-2 Limache
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2, "posesion": 50, "amarillas": 3},  # Everton 1-2 O'Higgins
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 0, "posesion": 47, "amarillas": 2},  # Everton 1-0 Limache
    {"equipo": "Everton", "local": 1, "gf": 0, "gc": 3, "posesion": 54, "amarillas": 2},  # Everton 0-3 Huachipato
]


df=pd.DataFrame(data)

# funcion aux
def estadisticas_equipo(equipo_name, local=None):
    #calcular el promedio de goles a favor y en contra
    #local=1 -> solo local
    #local=0 -> visita
    equipo_df = df[df["equipo"] == equipo_name].copy()
    if local is not None:
        equipo_df = equipo_df[equipo_df['local'] == local]
    return{
        "gf_promedio": equipo_df['gf'].mean(),
        "gc_promedio": equipo_df['gc'].mean(),
        "posesion_promedio": equipo_df["posesion"].mean(),
        "amarillas_promedio": equipo_df["amarillas"].mean(),
        "partidos": len(equipo_df)
    }

# historial ultimos 5 partidos entre ambos equipos
h2_data = [
    {"gf_everton": 0, "gf_udechile": 2},
    {"gf_everton": 2, "gf_udechile": 0},
    {"gf_everton": 1, "gf_udechile": 1},
    {"gf_everton": 0, "gf_udechile": 1},
    {"gf_everton": 2, "gf_udechile": 1},
]

h2_df = pd.DataFrame(h2_data)

h2_everton_promedio=h2_df["gf_everton"].mean()
h2_udechile_promedio=h2_df["gf_udechile"].mean()

#datasetpara regresion multiple
rows = []
for _, row in df.iterrows():
    equipo= row["equipo"]
    local = row["local"]

    equipo_all= estadisticas_equipo(equipo)
    equipo_local= estadisticas_equipo(equipo, local=1)
    equipo_visita= estadisticas_equipo(equipo, local=0)
    
    if equipo == "UdeChile":
        h2_promedio = h2_udechile_promedio
    else:
        h2_promedio = h2_everton_promedio

    rows.append({
        "equipo": equipo,
        "local": local,
        "equipo_gf_promedio": equipo_all['gf_promedio'],
        "equipo_gc_promedio": equipo_all['gc_promedio'],
        "equipo_posesion_promedio": equipo_all["posesion_promedio"],
        "equipo_amarillas_promedio": equipo_all["amarillas_promedio"],
        "equipo_gf_promedio_local": equipo_local['gf_promedio'],
        "equipo_gc_promedio_local": equipo_local['gc_promedio'],
        "equipo_gf_promedio_visita": equipo_visita['gf_promedio'],
        "equipo_gc_promedio_visita": equipo_visita['gc_promedio'],
        "h2_goles_promedio": h2_promedio,
        "gf_target": row["gf"],
        "posesion_target": row["posesion"],
        "amarillas_target": row["amarillas"]
    })

model_df = pd.DataFrame(rows)

everton_df = model_df[model_df['equipo'] == 'Everton'].copy()
udechile_df = model_df[model_df['equipo'] == 'UdeChile'].copy()

features = [
    "local",
    "equipo_gf_promedio",
    "equipo_gc_promedio",
    "equipo_posesion_promedio",
    "equipo_amarillas_promedio",
    "equipo_gf_promedio_local",
    "equipo_gc_promedio_local",
    "equipo_gf_promedio_visita",
    "equipo_gc_promedio_visita",
    "h2_goles_promedio"
]



# ENTRENAMIENTO DE MODELOS

# U de Chile
x_udechile = udechile_df[features]

y_udechile_gf = udechile_df["gf_target"]
y_udechile_posesion = udechile_df["posesion_target"]
y_udechile_amarillas = udechile_df["amarillas_target"]

model_udechile_gf = LinearRegression()
model_udechile_gf.fit(x_udechile, y_udechile_gf)

model_udechile_posesion = LinearRegression()
model_udechile_posesion.fit(x_udechile, y_udechile_posesion)

model_udechile_amarillas = LinearRegression()
model_udechile_amarillas.fit(x_udechile, y_udechile_amarillas)

# Everton
x_everton = everton_df[features]

y_everton_gf = everton_df["gf_target"]
y_everton_posesion = everton_df["posesion_target"]
y_everton_amarillas = everton_df["amarillas_target"]

model_everton_gf = LinearRegression()
model_everton_gf.fit(x_everton, y_everton_gf)

model_everton_posesion = LinearRegression()
model_everton_posesion.fit(x_everton, y_everton_posesion)

model_everton_amarillas = LinearRegression()
model_everton_amarillas.fit(x_everton, y_everton_amarillas)



# datos a predecir

udechile_stats = estaditicas_equipo("UdeChile")
udechile_local=estaditicas_equipo("UdeChile", local=1)
udechile_visita=estaditicas_equipo("UdeChile", local=0)

everton_stats = estaditicas_equipo("Everton")
everton_local=estaditicas_equipo("Everton", local=1)
everton_visita=estaditicas_equipo("Everton", local=0)

# partido everton vs udechile

partido_udechile = pd.DataFrame([{
    "local": 1,
    "equipo_gf_promedio": udechile_stats["gf_promedio"],
    "equipo_gc_promedio": udechile_stats["gc_promedio"],
    "equipo_posesion_promedio": udechile_stats["posesion_promedio"],
    "equipo_amarillas_promedio": udechile_stats["amarillas_promedio"],
    "equipo_gf_promedio_local": udechile_local["gf_promedio"],
    "equipo_gc_promedio_local": udechile_local["gc_promedio"],
    "equipo_gf_promedio_visita": udechile_visita["gf_promedio"],
    "equipo_gc_promedio_visita": udechile_visita["gc_promedio"],
    "h2_goles_promedio": h2_udechile_promedio
}])

partido_everton = pd.DataFrame([{
    "local": 0,
    "equipo_gf_promedio": everton_stats["gf_promedio"],
    "equipo_gc_promedio": everton_stats["gc_promedio"],
    "equipo_posesion_promedio": everton_stats["posesion_promedio"],
    "equipo_amarillas_promedio": everton_stats["amarillas_promedio"],
    "equipo_gf_promedio_local": everton_local["gf_promedio"],
    "equipo_gc_promedio_local": everton_local["gc_promedio"],
    "equipo_gf_promedio_visita": everton_visita["gf_promedio"],
    "equipo_gc_promedio_visita": everton_visita["gc_promedio"],
    "h2_goles_promedio": h2_everton_promedio
}])

# predicciones

pred_udechile_gf = model_udechile_gf.predict(partido_udechile)[0]
pred_everton_gf = model_everton_gf.predict(partido_everton)[0]

pred_udechile_posesion = model_udechile_posesion.predict(partido_udechile)[0]
pred_everton_posesion = model_everton_posesion.predict(partido_everton)[0]

pred_udechile_amarillas = model_udechile_amarillas.predict(partido_udechile)[0]
pred_everton_amarillas = model_everton_amarillas.predict(partido_everton)[0]

# resultados finales

gf_udechile_final = round(pred_udechile_gf)
gf_everton_final = round(pred_everton_gf)

posesion_udechile_final = round(pred_udechile_posesion)
posesion_everton_final = round(pred_everton_posesion)

amarillas_udechile_final = round(pred_udechile_amarillas)
amarillas_everton_final = round(pred_everton_amarillas)


# resultados finales

gf_udechile_final = round(pred_udechile_gf)
gf_everton_final = round(pred_everton_gf)

posesion_udechile_final = round(pred_udechile_posesion)
amarillas_udechile_final = round(pred_udechile_amarillas)

print("Prediccion del partido: U de Chile vs Everton")
print("Resultado:", gf_udechile_final, "-", gf_everton_final)

print("Posesion U de Chile:", posesion_udechile_final, "%")
print("Amarillas U de Chile:", amarillas_udechile_final)

if gf_udechile_final > gf_everton_final:
    print("Gana U de Chile")
elif gf_udechile_final < gf_everton_final:
    print("Gana Everton")
else:
    print("Empate")
