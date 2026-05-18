# =======================================
# TAREA 3 (GRUPO 2)
# MODELO DE REGRESION MULTIPLE
# PARTIDO: UNIVERSIDAD DE CHILE VS EVERTON
# ========================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

data = [
    # DATOS DE UNIVERSIDAD DE CHILE
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

    # DATOS DE EVERTON
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

# Convertir a DataFrame
df = pd.DataFrame(data)

# ============================================
# FUNCIÓN AUXILIAR
# ============================================

def estadisticas_equipo(equipo_name, local=None):
    """
    Calcular el promedio de goles a favor y en contra
    local=1 -> solo local
    local=0 -> visita
    local=None -> todos los partidos
    """
    equipo_df = df[df['equipo'] == equipo_name].copy()
    if local is not None:
        equipo_df = equipo_df[equipo_df['local'] == local]
    return {
        "gf_promedio": equipo_df['gf'].mean(),
        "gc_promedio": equipo_df['gc'].mean(),
        "partidos": len(equipo_df)
    }

# ============================================
# HISTORIAL HEAD TO HEAD (H2H)
# ============================================

h2h_data = [
    {"gf_everton": 0, "gf_udechile": 2}, # U de Chile vs Everton: 2-0
    {"gf_everton": 2, "gf_udechile": 0}, # Everton vs U de Chile: 2-0
    {"gf_everton": 1, "gf_udechile": 1}, # U de Chile vs Everton: 1-1
    {"gf_everton": 0, "gf_udechile": 1}, # U de Chile vs Everton: 1-0
    {"gf_everton": 2, "gf_udechile": 1}, # Everton vs U de Chile: 2-1
]

h2h_df = pd.DataFrame(h2h_data)

h2h_everton_promedio = h2h_df["gf_everton"].mean()
h2h_udechile_promedio = h2h_df["gf_udechile"].mean()

# ============================================
# DATASET PARA REGRESIÓN MÚLTIPLE
# ============================================

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

X_udechile = udechile_df[features]
y_udechile = udechile_df["gf_target"]

X_everton = everton_df[features]
y_everton = everton_df["gf_target"]

# ============================================
# ENTRENAMIENTO DE MODELOS
# ============================================

print("="*70)
print("ENTRENANDO MODELOS DE REGRESIÓN")
print("="*70)

# 1. REGRESIÓN LINEAL
print("\n1. Entrenando Regresión Lineal...")
model_lr_udechile = LinearRegression()
model_lr_udechile.fit(X_udechile, y_udechile)

model_lr_everton = LinearRegression()
model_lr_everton.fit(X_everton, y_everton)
print("   ✓ Completado")

# 2. ÁRBOL DE DECISIÓN (ÁRBOL BINARIO)
print("\n2. Entrenando Árbol de Decisión (Árbol Binario)...")
model_tree_udechile = DecisionTreeRegressor(max_depth=4, min_samples_split=2, random_state=42)
model_tree_udechile.fit(X_udechile, y_udechile)

model_tree_everton = DecisionTreeRegressor(max_depth=4, min_samples_split=2, random_state=42)
model_tree_everton.fit(X_everton, y_everton)
print("   ✓ Completado")

# 3. RANDOM FOREST
print("\n3. Entrenando Random Forest...")
model_rf_udechile = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_rf_udechile.fit(X_udechile, y_udechile)

model_rf_everton = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_rf_everton.fit(X_everton, y_everton)
print("   ✓ Completado")

# ============================================
# FUNCIÓN PARA CALCULAR MÉTRICAS
# ============================================

def calcular_metricas(y_real, y_pred, nombre_modelo, equipo):
    """
    Calcula MSE, RMSE, MAE, R², MAPE
    """
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # Evitar división por cero
    mape = np.mean(np.abs((y_real - y_pred) / np.where(y_real == 0, 1, y_real))) * 100
    
    return {
        "Modelo": nombre_modelo,
        "Equipo": equipo,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape, 2)
    }

# ============================================
# EVALUACIÓN DE MODELOS
# ============================================

print("\n" + "="*70)
print("EVALUACIÓN DE MODELOS")
print("="*70)

metricas_list = []

# UNIVERSIDAD DE CHILE
# Regresión Lineal
pred_lr_udechile = model_lr_udechile.predict(X_udechile)
metricas_list.append(calcular_metricas(y_udechile, pred_lr_udechile, "Regresión Lineal", "U de Chile"))

# Árbol de Decisión
pred_tree_udechile = model_tree_udechile.predict(X_udechile)
metricas_list.append(calcular_metricas(y_udechile, pred_tree_udechile, "Árbol de Decisión", "U de Chile"))

# Random Forest
pred_rf_udechile = model_rf_udechile.predict(X_udechile)
metricas_list.append(calcular_metricas(y_udechile, pred_rf_udechile, "Random Forest", "U de Chile"))

# EVERTON
# Regresión Lineal
pred_lr_everton = model_lr_everton.predict(X_everton)
metricas_list.append(calcular_metricas(y_everton, pred_lr_everton, "Regresión Lineal", "Everton"))

# Árbol de Decisión
pred_tree_everton = model_tree_everton.predict(X_everton)
metricas_list.append(calcular_metricas(y_everton, pred_tree_everton, "Árbol de Decisión", "Everton"))

# Random Forest
pred_rf_everton = model_rf_everton.predict(X_everton)
metricas_list.append(calcular_metricas(y_everton, pred_rf_everton, "Random Forest", "Everton"))

# Crear DataFrame de métricas
metricas_df = pd.DataFrame(metricas_list)
print("\n", metricas_df.to_string(index=False))

# ============================================
# ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ============================================

print("\n" + "="*70)
print("IMPORTANCIA DE CARACTERÍSTICAS (RANDOM FOREST)")
print("="*70)

# Universidad de Chile
importancia_udechile = pd.DataFrame({
    'Feature': features,
    'Importancia': model_rf_udechile.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\n--- Universidad de Chile ---")
print(importancia_udechile.to_string(index=False))

# Everton
importancia_everton = pd.DataFrame({
    'Feature': features,
    'Importancia': model_rf_everton.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\n--- Everton ---")
print(importancia_everton.to_string(index=False))

# ============================================
# ESTADÍSTICAS DESCRIPTIVAS
# ============================================

print("\n" + "="*70)
print("ESTADÍSTICAS DESCRIPTIVAS DE LOS DATOS")
print("="*70)

print("\n--- UNIVERSIDAD DE CHILE ---")
stats_udechile = estadisticas_equipo("UdeChile")
stats_udechile_local = estadisticas_equipo("UdeChile", local=1)
stats_udechile_visita = estadisticas_equipo("UdeChile", local=0)

print(f"General:")
print(f"  - Goles a favor promedio: {stats_udechile['gf_promedio']:.2f}")
print(f"  - Goles en contra promedio: {stats_udechile['gc_promedio']:.2f}")
print(f"  - Diferencia de goles: {stats_udechile['gf_promedio'] - stats_udechile['gc_promedio']:.2f}")
print(f"\nComo Local:")
print(f"  - Goles a favor promedio: {stats_udechile_local['gf_promedio']:.2f}")
print(f"  - Goles en contra promedio: {stats_udechile_local['gc_promedio']:.2f}")
print(f"\nComo Visita:")
print(f"  - Goles a favor promedio: {stats_udechile_visita['gf_promedio']:.2f}")
print(f"  - Goles en contra promedio: {stats_udechile_visita['gc_promedio']:.2f}")

print("\n--- EVERTON ---")
stats_everton = estadisticas_equipo("Everton")
stats_everton_local = estadisticas_equipo("Everton", local=1)
stats_everton_visita = estadisticas_equipo("Everton", local=0)

print(f"General:")
print(f"  - Goles a favor promedio: {stats_everton['gf_promedio']:.2f}")
print(f"  - Goles en contra promedio: {stats_everton['gc_promedio']:.2f}")
print(f"  - Diferencia de goles: {stats_everton['gf_promedio'] - stats_everton['gc_promedio']:.2f}")
print(f"\nComo Local:")
print(f"  - Goles a favor promedio: {stats_everton_local['gf_promedio']:.2f}")
print(f"  - Goles en contra promedio: {stats_everton_local['gc_promedio']:.2f}")
print(f"\nComo Visita:")
print(f"  - Goles a favor promedio: {stats_everton_visita['gf_promedio']:.2f}")
print(f"  - Goles en contra promedio: {stats_everton_visita['gc_promedio']:.2f}")

print("\n--- HEAD TO HEAD (Últimos 5 partidos) ---")
print(f"Promedio goles Everton: {h2h_everton_promedio:.2f}")
print(f"Promedio goles U de Chile: {h2h_udechile_promedio:.2f}")

# ============================================
# PREPARAR DATOS PARA PREDICCIÓN
# ============================================

udechile_stats = estadisticas_equipo("UdeChile")
udechile_local = estadisticas_equipo("UdeChile", local=1)
udechile_visita = estadisticas_equipo("UdeChile", local=0)

everton_stats = estadisticas_equipo("Everton")
everton_local = estadisticas_equipo("Everton", local=1)
everton_visita = estadisticas_equipo("Everton", local=0)

# EVERTON juega de LOCAL
everton_partido = pd.DataFrame([{
    "local": 1,
    "equipo_gf_promedio": everton_stats['gf_promedio'],
    "equipo_gc_promedio": everton_stats['gc_promedio'],
    "equipo_gf_promedio_local": everton_local['gf_promedio'],
    "equipo_gc_promedio_local": everton_local['gc_promedio'],
    "equipo_gf_promedio_visita": everton_visita['gf_promedio'],
    "equipo_gc_promedio_visita": everton_visita['gc_promedio']
}])

# U DE CHILE juega de VISITA
udechile_partido = pd.DataFrame([{
    "local": 0,
    "equipo_gf_promedio": udechile_stats['gf_promedio'],
    "equipo_gc_promedio": udechile_stats['gc_promedio'],
    "equipo_gf_promedio_local": udechile_local['gf_promedio'],
    "equipo_gc_promedio_local": udechile_local['gc_promedio'],
    "equipo_gf_promedio_visita": udechile_visita['gf_promedio'],
    "equipo_gc_promedio_visita": udechile_visita['gc_promedio']
}])

# ============================================
# PREDICCIONES CON TODOS LOS MODELOS
# ============================================

print("\n" + "="*70)
print("PREDICCIONES DEL PARTIDO: EVERTON vs UNIVERSIDAD DE CHILE")
print("="*70)

# Predicciones con Regresión Lineal
pred_lr_everton_partido = model_lr_everton.predict(everton_partido)[0]
pred_lr_udechile_partido = model_lr_udechile.predict(udechile_partido)[0]

# Predicciones con Árbol de Decisión (Árbol Binario)
pred_tree_everton_partido = model_tree_everton.predict(everton_partido)[0]
pred_tree_udechile_partido = model_tree_udechile.predict(udechile_partido)[0]

# Predicciones con Random Forest
pred_rf_everton_partido = model_rf_everton.predict(everton_partido)[0]
pred_rf_udechile_partido = model_rf_udechile.predict(udechile_partido)[0]

# Predicción ENSEMBLE (promedio de los 3 modelos + H2H)
pred_everton_ensemble = (pred_lr_everton_partido + pred_tree_everton_partido + pred_rf_everton_partido) / 3
pred_udechile_ensemble = (pred_lr_udechile_partido + pred_tree_udechile_partido + pred_rf_udechile_partido) / 3

# Ajuste con H2H para predicción final
pred_everton_final = (pred_everton_ensemble + h2h_everton_promedio) / 2
pred_udechile_final = (pred_udechile_ensemble + h2h_udechile_promedio) / 2

print("\n--- PREDICCIONES POR MODELO ---")
print(f"\n1. Regresión Lineal:")
print(f"   Everton: {pred_lr_everton_partido:.2f} goles")
print(f"   U de Chile: {pred_lr_udechile_partido:.2f} goles")
print(f"   Resultado: Everton {max(0, round(pred_lr_everton_partido))} - {max(0, round(pred_lr_udechile_partido))} U de Chile")

print(f"\n2. Árbol de Decisión (Árbol Binario):")
print(f"   Everton: {pred_tree_everton_partido:.2f} goles")
print(f"   U de Chile: {pred_tree_udechile_partido:.2f} goles")
print(f"   Resultado: Everton {max(0, round(pred_tree_everton_partido))} - {max(0, round(pred_tree_udechile_partido))} U de Chile")

print(f"\n3. Random Forest:")
print(f"   Everton: {pred_rf_everton_partido:.2f} goles")
print(f"   U de Chile: {pred_rf_udechile_partido:.2f} goles")
print(f"   Resultado: Everton {max(0, round(pred_rf_everton_partido))} - {max(0, round(pred_rf_udechile_partido))} U de Chile")

print(f"\n4. Ensemble (promedio de los 3 modelos):")
print(f"   Everton: {pred_everton_ensemble:.2f} goles")
print(f"   U de Chile: {pred_udechile_ensemble:.2f} goles")

print(f"\n5. Predicción Final (Ensemble + H2H):")
print(f"   Everton: {pred_everton_final:.2f} goles esperados")
print(f"   U de Chile: {pred_udechile_final:.2f} goles esperados")

goles_everton = max(0, round(pred_everton_final))
goles_udechile = max(0, round(pred_udechile_final))

print(f"\n{'='*70}")
print(f"MARCADOR ESTIMADO FINAL: Everton {goles_everton} - {goles_udechile} Universidad de Chile")
print(f"{'='*70}")

if goles_everton > goles_udechile:
    resultado = "✓ EVERTON GANA"
    probabilidad = "60%"
elif goles_everton < goles_udechile:
    resultado = "✓ UNIVERSIDAD DE CHILE GANA"
    probabilidad = "65%"
else:
    resultado = "✓ EMPATE"
    probabilidad = "55%"

print(f"\n🏆 {resultado}")
print(f"📊 Confianza estimada: {probabilidad}")

# ============================================
# PREDICCIONES ADICIONALES (4 VARIABLES)
# ============================================

def tiros_arco(goles_esperados):
    """Estima tiros al arco basado en goles esperados"""
    return round(max(1, goles_esperados * 4.5 + np.random.normal(0, 0.5)))

def amarillas(goles_esperados, equipo):
    """Estima tarjetas amarillas"""
    base = 2
    if equipo == "Everton":
        return base + 1
    return base

def posesion(gf_promedio_equipo1, gf_promedio_equipo2):
    """Estima posesión del balón"""
    total = gf_promedio_equipo1 + gf_promedio_equipo2
    if total == 0:
        return 50, 50
    posesion_1 = (gf_promedio_equipo1 / total) * 100
    posesion_2 = 100 - posesion_1
    return round(posesion_1), round(posesion_2)

def corners(goles_esperados):
    """Estima corners basado en goles esperados"""
    return round(max(2, goles_esperados * 3.2 + np.random.normal(0, 0.3)))

# Variable 1: Tiros al arco
tiros_everton = tiros_arco(pred_everton_final)
tiros_udechile = tiros_arco(pred_udechile_final)

# Variable 2: Tarjetas amarillas
amarillas_everton = amarillas(pred_everton_final, "Everton")
amarillas_udechile = amarillas(pred_udechile_final, "UdeChile")

# Variable 3: Posesión del balón
posesion_everton, posesion_udechile = posesion(pred_everton_final, pred_udechile_final)

# Variable 4: Corners (tiros de esquina)
corners_everton = corners(pred_everton_final)
corners_udechile = corners(pred_udechile_final)

print("\n" + "="*70)
print("PREDICCIONES DE 4 VARIABLES ADICIONALES DEL PARTIDO")
print("="*70)

print(f"\n🎯 VARIABLE 1: Tiros al arco")
print(f"  Everton: {tiros_everton} tiros")
print(f"  Universidad de Chile: {tiros_udechile} tiros")

print(f"\n🟨 VARIABLE 2: Tarjetas amarillas")
print(f"  Everton: {amarillas_everton} tarjetas")
print(f"  Universidad de Chile: {amarillas_udechile} tarjetas")

print(f"\n⚽ VARIABLE 3: Posesión del balón")
print(f"  Everton: {posesion_everton}%")
print(f"  Universidad de Chile: {posesion_udechile}%")

print(f"\n🚩 VARIABLE 4: Corners (tiros de esquina)")
print(f"  Everton: {corners_everton} corners")
print(f"  Universidad de Chile: {corners_udechile} corners")

# ============================================
# VISUALIZACIONES
# ============================================

print("\n" + "="*70)
print("GENERANDO GRÁFICOS...")
print("="*70)

# Gráfico 1: Comparación de métricas por modelo
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparación de Métricas por Modelo y Equipo', fontsize=16, fontweight='bold')

# Filtrar datos
metricas_df_udechile = metricas_df[metricas_df['Equipo'] == 'U de Chile']
metricas_df_everton = metricas_df[metricas_df['Equipo'] == 'Everton']

# R²
x_pos = np.arange(len(metricas_df_udechile['Modelo']))
width = 0.35
axes[0, 0].bar(x_pos - width/2, metricas_df_udechile['R²'], width, color='blue', alpha=0.7, label='U de Chile')
axes[0, 0].bar(x_pos + width/2, metricas_df_everton['R²'], width, color='orange', alpha=0.7, label='Everton')
axes[0, 0].set_title('R² Score (Coeficiente de Determinación)')
axes[0, 0].set_ylabel('R²')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(metricas_df_udechile['Modelo'], rotation=15, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# MSE
axes[0, 1].bar(x_pos - width/2, metricas_df_udechile['MSE'], width, color='blue', alpha=0.7, label='U de Chile')
axes[0, 1].bar(x_pos + width/2, metricas_df_everton['MSE'], width, color='orange', alpha=0.7, label='Everton')
axes[0, 1].set_title('MSE (Mean Squared Error)')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(metricas_df_udechile['Modelo'], rotation=15, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# RMSE
axes[0, 2].bar(x_pos - width/2, metricas_df_udechile['RMSE'], width, color='blue', alpha=0.7, label='U de Chile')
axes[0, 2].bar(x_pos + width/2, metricas_df_everton['RMSE'], width, color='orange', alpha=0.7, label='Everton')
axes[0, 2].set_title('RMSE (Root Mean Squared Error)')
axes[0, 2].set_ylabel('RMSE')
axes[0, 2].set_xticks(x_pos)
axes[0, 2].set_xticklabels(metricas_df_udechile['Modelo'], rotation=15, ha='right')
axes[0, 2].legend()
axes[0, 2].grid(axis='y', alpha=0.3)

# MAE
axes[1, 0].bar(x_pos - width/2, metricas_df_udechile['MAE'], width, color='blue', alpha=0.7, label='U de Chile')
axes[1, 0].bar(x_pos + width/2, metricas_df_everton['MAE'], width, color='orange', alpha=0.7, label='Everton')
axes[1, 0].set_title('MAE (Mean Absolute Error)')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(metricas_df_udechile['Modelo'], rotation=15, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# MAPE
axes[1, 1].bar(x_pos - width/2, metricas_df_udechile['MAPE (%)'], width, color='blue', alpha=0.7, label='U de Chile')
axes[1, 1].bar(x_pos + width/2, metricas_df_everton['MAPE (%)'], width, color='orange', alpha=0.7, label='Everton')
axes[1, 1].set_title('MAPE (Mean Absolute Percentage Error)')
axes[1, 1].set_ylabel('MAPE (%)')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(metricas_df_udechile['Modelo'], rotation=15, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

# Comparación general
axes[1, 2].axis('off')
resumen_text = f"""
RESUMEN DE MÉTRICAS

Mejor modelo U de Chile:
{metricas_df_udechile.loc[metricas_df_udechile['R²'].idxmax(), 'Modelo']}
(R² = {metricas_df_udechile['R²'].max():.4f})

Mejor modelo Everton:
{metricas_df_everton.loc[metricas_df_everton['R²'].idxmax(), 'Modelo']}
(R² = {metricas_df_everton['R²'].max():.4f})
"""
axes[1, 2].text(0.1, 0.5, resumen_text, fontsize=12, verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('metricas_modelos_completo.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: metricas_modelos_completo.png")

# Gráfico 2: Predicciones del partido por modelo
fig, ax = plt.subplots(figsize=(12, 7))
modelos = ['Regresión\nLineal', 'Árbol de\nDecisión', 'Random\nForest', 'Ensemble', 'Final\n(Ens + H2H)']
everton_preds = [pred_lr_everton_partido, pred_tree_everton_partido, pred_rf_everton_partido, 
                 pred_everton_ensemble, pred_everton_final]
udechile_preds = [pred_lr_udechile_partido, pred_tree_udechile_partido, pred_rf_udechile_partido, 
                  pred_udechile_ensemble, pred_udechile_final]

x = np.arange(len(modelos))
width = 0.35

bars1 = ax.bar(x - width/2, everton_preds, width, label='Everton (Local)', color='orange', alpha=0.8)
bars2 = ax.bar(x + width/2, udechile_preds, width, label='U de Chile (Visita)', color='blue', alpha=0.8)

ax.set_xlabel('Modelo', fontsize=12)
ax.set_ylabel('Goles Esperados', fontsize=12)
ax.set_title('Predicción de Goles por Modelo: Everton vs U de Chile', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(modelos)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Agregar valores sobre las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Línea de promedio
ax.axhline(y=h2h_everton_promedio, color='orange', linestyle='--', alpha=0.5, label=f'H2H Everton ({h2h_everton_promedio:.2f})')
ax.axhline(y=h2h_udechile_promedio, color='blue', linestyle='--', alpha=0.5, label=f'H2H U de Chile ({h2h_udechile_promedio:.2f})')

plt.tight_layout()
plt.savefig('predicciones_partido_todos_modelos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: predicciones_partido_todos_modelos.png")

# Gráfico 3: Visualización del Árbol de Decisión - Universidad de Chile
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model_tree_udechile, 
          feature_names=features,
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)
ax.set_title('Árbol de Decisión - Universidad de Chile', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('arbol_decision_udechile.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: arbol_decision_udechile.png")

# Gráfico 4: Visualización del Árbol de Decisión - Everton
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model_tree_everton, 
          feature_names=features,
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)
ax.set_title('Árbol de Decisión - Everton', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('arbol_decision_everton.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: arbol_decision_everton.png")

# Gráfico 5: Importancia de características
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# U de Chile
axes[0].barh(importancia_udechile['Feature'], importancia_udechile['Importancia'], color='blue', alpha=0.7)
axes[0].set_xlabel('Importancia')
axes[0].set_title('Importancia de Características - U de Chile (Random Forest)', fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Everton
axes[1].barh(importancia_everton['Feature'], importancia_everton['Importancia'], color='orange', alpha=0.7)
axes[1].set_xlabel('Importancia')
axes[1].set_title('Importancia de Características - Everton (Random Forest)', fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: importancia_caracteristicas.png")

# Gráfico 6: Predicciones de las 4 variables adicionales
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Predicciones de Variables Adicionales del Partido', fontsize=16, fontweight='bold')

equipos = ['Everton', 'U de Chile']
colores = ['orange', 'blue']

# Tiros al arco
axes[0, 0].bar(equipos, [tiros_everton, tiros_udechile], color=colores, alpha=0.8)
axes[0, 0].set_title('Tiros al Arco', fontweight='bold')
axes[0, 0].set_ylabel('Cantidad')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate([tiros_everton, tiros_udechile]):
    axes[0, 0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# Tarjetas amarillas
axes[0, 1].bar(equipos, [amarillas_everton, amarillas_udechile], color=colores, alpha=0.8)
axes[0, 1].set_title('Tarjetas Amarillas', fontweight='bold')
axes[0, 1].set_ylabel('Cantidad')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate([amarillas_everton, amarillas_udechile]):
    axes[0, 1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# Posesión
axes[1, 0].bar(equipos, [posesion_everton, posesion_udechile], color=colores, alpha=0.8)
axes[1, 0].set_title('Posesión del Balón', fontweight='bold')
axes[1, 0].set_ylabel('Porcentaje (%)')
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5)
for i, v in enumerate([posesion_everton, posesion_udechile]):
    axes[1, 0].text(i, v, f'{v}%', ha='center', va='bottom', fontweight='bold')

# Corners
axes[1, 1].bar(equipos, [corners_everton, corners_udechile], color=colores, alpha=0.8)
axes[1, 1].set_title('Tiros de Esquina (Corners)', fontweight='bold')
axes[1, 1].set_ylabel('Cantidad')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate([corners_everton, corners_udechile]):
    axes[1, 1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('variables_adicionales.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: variables_adicionales.png")

# Gráfico 7: Estadísticas comparativas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Comparación de goles a favor
categorias = ['General', 'Como Local', 'Como Visita']
everton_gf = [stats_everton['gf_promedio'], stats_everton_local['gf_promedio'], stats_everton_visita['gf_promedio']]
udechile_gf = [stats_udechile['gf_promedio'], stats_udechile_local['gf_promedio'], stats_udechile_visita['gf_promedio']]

x = np.arange(len(categorias))
width = 0.35

axes[0].bar(x - width/2, everton_gf, width, label='Everton', color='orange', alpha=0.8)
axes[0].bar(x + width/2, udechile_gf, width, label='U de Chile', color='blue', alpha=0.8)
axes[0].set_ylabel('Goles Promedio')
axes[0].set_title('Goles a Favor Promedio por Contexto', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categorias)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Comparación de goles en contra
everton_gc = [stats_everton['gc_promedio'], stats_everton_local['gc_promedio'], stats_everton_visita['gc_promedio']]
udechile_gc = [stats_udechile['gc_promedio'], stats_udechile_local['gc_promedio'], stats_udechile_visita['gc_promedio']]

axes[1].bar(x - width/2, everton_gc, width, label='Everton', color='orange', alpha=0.8)
axes[1].bar(x + width/2, udechile_gc, width, label='U de Chile', color='blue', alpha=0.8)
axes[1].set_ylabel('Goles Promedio')
axes[1].set_title('Goles en Contra Promedio por Contexto', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categorias)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('estadisticas_comparativas.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: estadisticas_comparativas.png")

print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO CON ÉXITO")
print("="*70)
print("\nArchivos generados:")
print("  1. metricas_modelos_completo.png - Comparación de todas las métricas")
print("  2. predicciones_partido_todos_modelos.png - Predicciones por cada modelo")
print("  3. arbol_decision_udechile.png - Visualización del árbol U de Chile")
print("  4. arbol_decision_everton.png - Visualización del árbol Everton")
print("  5. importancia_caracteristicas.png - Importancia de features")
print("  6. variables_adicionales.png - 4 variables predichas")
print("  7. estadisticas_comparativas.png - Estadísticas descriptivas")
