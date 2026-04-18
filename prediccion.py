#aplicar un modelo de regrsion (el más conveniente)
#para encontrar el resultado del partido entre Unidersidad de Chile vs Everton CD
#recomendaciones: utilizar regresión múltiple, colocar partidos anteriores y todos los datos posibles para una mayor presicion.

#datos:
# ultimos 5 partidos de Universidad de chile:

# Ñublense vs Universidad de Chile: 1-0
# Universidad de chile vs Deportes la serena: 4-0
# Audax Italiano vs Universidad de Chile: 1-3
# Universidad de Chile vs Union la calera: 0-1
# Universidad de Chile vs Deportes La serena: 2-2

# ultimos 5 partidos de visita Universidad de chile:

# Ñublense vs Universidad de Chile: 1-0
# Audax Italiano vs Universidad de Chile: 1-3
# Coquimbo Unido vs Universidad de Chile: 0-1
# Colo colo vs Universidad de Chile: 0-1
# Palestino vs Universidad de Chile: 0-0

# ultimos 5 partidos de Everton CD:

# DEportes la serena vs Everton CD: 1-0
# Everton CD vs ñublense: 0-0
# Everton CD vs Deportes Limache: 1-2
# Everton CD vs O'Higgins: 1-2
# Palestino vs Everton CD: 1-2

# ultimos 5 partidos de local Everton CD:

# Everton CD vs ñublense: 0-0
# Everton CD vs Deportes Limache: 1-2
# Everton CD vs O'Higgins: 1-2
# Everton CD vs Deportes Limache: 1-0
# Everton CD vs Huachipato: 0-3

#ultmos 5 partidos entre Everton CD vs Universidad de Chile:

# Universidad de Chile vs Everton CD: 2-0
# Everton CD vs Universidad de Chile: 2-0
# Universidad de Chile vs Everton CD: 1-1
# Unniversidad de Chile vs Everton CD: 1-0
# Everton CD vs Universidad de Chile: 2-1

import pandas as pd
from sklearn.linear_model import LinearRegression


data = [
    # DATOS DE UNIVERSIDAD DE CHILE BPSAI
    #ultimos 5 partido
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 1}, # Ñublense vs U de Chile: 1-0
    {"equipo": "UdeChile", "local": 1, "gf": 4, "gc": 0}, # U de Chile vs La Serena: 4-0
    {"equipo": "UdeChile", "local": 0, "gf": 3, "gc": 1},# Audax vs U de Chile: 1-3
    {"equipo": "UdeChile", "local": 1, "gf": 0, "gc": 1},# U de Chile vs Unión La Calera: 0-1
    {"equipo": "UdeChile", "local": 1, "gf": 2, "gc": 2},# U de Chile vs LA serena: 2-2

    #ultimos 5 de visita
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 1}, # Ñublense vs U de Chile: 1-0
    {"equipo": "UdeChile", "local": 0, "gf": 3, "gc": 1},# Audax vs U de Chile: 1-3
    {"equipo": "UdeChile", "local": 0, "gf": 1, "gc": 0},# Coquimbo vs U de Chile: 0-1
    {"equipo": "UdeChile", "local": 0, "gf": 1, "gc": 0},# Colo colo vs U de Chile: 0-1
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 0},# Palestino vs U de Chile: 0-0

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

df=pd.DataFrame(data)

# funcion aux
def estaditicas_equipo(equipo_name, local=None):
    #calcular el promedio de goles a favor y en contra
    #local=1 -> solo local
    #local=0 -> visita
    equipo_df = df[df['equipo'] == equipo_name].copy()
    if local is not None:
        equipo_df = equipo_df[equipo_df['local'] == local]
    return{
        "gf_promedio": equipo_df['gf'].mean(),
        "gc_promedio": equipo_df['gc'].mean(),
        "partidos": len(equipo_df)
    }

# historial ultimos 5 partidos entre ambos equipos
h2_data = [
    {"gf_everton":0, "gf_udechile":2}, # U de Chile vs Everton: 2-0
    {"gf_everton":2, "gf_udechile":0}, # Everton vs U de Chile: 2-0
    {"gf_everton":1, "gf_udechile":1}, # U de Chile vs Everton: 1-1
    {"gf_everton":0, "gf_udechile":1}, # U de Chile vs Everton: 1-0
    {"gf_everton":2, "gf_udechile":1}, # Everton vs U de Chile: 2-1
]

h2_df = pd.DataFrame(h2_data)

h2_everton_promedio=h2_df["gf_everton"].mean()
h2_udechile_promedio=h2_df["gf_udechile"].mean()

#datasetpara regresion multiple
rows = []
for _, row in df.iterrows():
    equipo= row['equipo']
    local = row['local']

    equipo_all= estaditicas_equipo(equipo)
    equipo_local= estaditicas_equipo(equipo, local=1)
    equipo_visita= estaditicas_equipo(equipo, local=0)

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

# entrenamiento0 modelos de regresión lineal
model_udechile = LinearRegression()
model_udechile.fit(x_udechile, y_udechile)

model_everton = LinearRegression()
model_everton.fit(x_everton, y_everton)

# datos a predecir

udechile_stats = estaditicas_equipo("UdeChile")
udechile_local=estaditicas_equipo("UdeChile", local=1)
udechile_visita=estaditicas_equipo("UdeChile", local=0)

everton_stats = estaditicas_equipo("Everton")
everton_local=estaditicas_equipo("Everton", local=1)
everton_visita=estaditicas_equipo("Everton", local=0)

# partido everton vs udechil

everton_partido=pd.DataFrame([{
    "local": 1,
    "equipo_gf_promedio": everton_stats['gf_promedio'],
    "equipo_gc_promedio": everton_stats['gc_promedio'],
    "equipo_gf_promedio_local": everton_local['gf_promedio'],
    "equipo_gc_promedio_local": everton_local['gc_promedio'],
    "equipo_gf_promedio_visita": everton_visita['gf_promedio'],
    "equipo_gc_promedio_visita": everton_visita['gc_promedio']
}])

udechile_partido=pd.DataFrame([{
    "local": 0,
    "equipo_gf_promedio": udechile_stats['gf_promedio'],
    "equipo_gc_promedio": udechile_stats['gc_promedio'],
    "equipo_gf_promedio_local": udechile_local['gf_promedio'],
    "equipo_gc_promedio_local": udechile_local['gc_promedio'],
    "equipo_gf_promedio_visita": udechile_visita['gf_promedio'],
    "equipo_gc_promedio_visita": udechile_visita['gc_promedio']
}])

#predecir goles}

prediccion_everton = model_everton.predict(everton_partido)[0]
prediccion_udechile = model_udechile.predict(udechile_partido)[0]

prediccion_everton=(prediccion_everton+h2_everton_promedio)/2
prediccion_udechile=(prediccion_udechile+h2_udechile_promedio)/2

prediccion_everton=max(0, round(prediccion_everton))
prediccion_udechile=max(0, round(prediccion_udechile))

goles_everton=round(prediccion_everton)
goles_udechile=round(prediccion_udechile)

# posibles tiros al arco y amarlla

def tiros_arco(goles_esperados):
    return round (max(1, goles_esperados * 4))

def amarillas(goles_esperados,equipo):
    base=2
    if equipo=="Everton":
        return base+1
    return base

tiros_everton=tiros_arco(prediccion_everton)
tiros_udechile=tiros_arco(prediccion_udechile)

amarillas_everton=amarillas(prediccion_everton, "Everton")
amarillas_udechile=amarillas(prediccion_udechile, "UdeChile")


#resultado final
print(f"Predicción del partido Everton vs Universidad de Chile:")
print(f"Everton:{prediccion_everton:.2f} goles esperados")
print(f"Universidad de Chile:{prediccion_udechile:.2f} goles esperados")
print("")

print(f"Marcador estiamdo: Everton {goles_everton} - {goles_udechile} Universidad de Chile")

if goles_everton > goles_udechile:
    print("Predicción: Everton gana")
elif goles_everton < goles_udechile:
    print("Predicción: Universidad de Chile gana")
else:
    print("Predicción: Empate")

print("")
print(f"Posibles tiros al arco:")
print(f"Everton: {tiros_everton} tiros al arco")
print(f"Universidad de Chile: {tiros_udechile} tiros al arco")
print("")
print(f"Posibles tarjetas amarillas:")
print(f"Everton: {amarillas_everton} tarjetas amarillas")
print(f"Universidad de Chile: {amarillas_udechile} tarjetas amarillas")
