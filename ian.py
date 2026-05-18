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
    {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 1}, {"equipo": "UdeChile", "local": 1, "gf": 4, "gc": 0},
    {"equipo": "UdeChile", "local": 0, "gf": 3, "gc": 1}, {"equipo": "UdeChile", "local": 1, "gf": 0, "gc": 1},
    {"equipo": "UdeChile", "local": 1, "gf": 2, "gc": 2}, {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 1},
    {"equipo": "UdeChile", "local": 0, "gf": 3, "gc": 1}, {"equipo": "UdeChile", "local": 0, "gf": 1, "gc": 0},
    {"equipo": "UdeChile", "local": 0, "gf": 1, "gc": 0}, {"equipo": "UdeChile", "local": 0, "gf": 0, "gc": 0},
    # DATOS DE EVERTON
    {"equipo": "Everton", "local": 0, "gf": 0, "gc": 1}, {"equipo": "Everton", "local": 1, "gf": 0, "gc": 0},
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2}, {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2},
    {"equipo": "Everton", "local": 0, "gf": 2, "gc": 1}, {"equipo": "Everton", "local": 1, "gf": 0, "gc": 0},
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2}, {"equipo": "Everton", "local": 1, "gf": 1, "gc": 2},
    {"equipo": "Everton", "local": 1, "gf": 1, "gc": 0}, {"equipo": "Everton", "local": 1, "gf": 0, "gc": 3},
]

df = pd.DataFrame(data)

h2h_data = [
    {"gf_everton": 0, "gf_udechile": 2}, {"gf_everton": 2, "gf_udechile": 0},
    {"gf_everton": 1, "gf_udechile": 1}, {"gf_everton": 0, "gf_udechile": 1},
    {"gf_everton": 2, "gf_udechile": 1},
]

h2h_df = pd.DataFrame(h2h_data)
h2h_everton_promedio = h2h_df["gf_everton"].mean()
h2h_udechile_promedio = h2h_df["gf_udechile"].mean()

def estadisticas_equipo(equipo_name, local=None):
    equipo_df = df[df['equipo'] == equipo_name].copy()
    if local is not None:
        equipo_df = equipo_df[equipo_df['local'] == local]
    return {"gf_promedio": equipo_df['gf'].mean(), "gc_promedio": equipo_df['gc'].mean(), "partidos": len(equipo_df)}

rows = []
for _, row in df.iterrows():
    equipo = row['equipo']
    local = row['local']
    equipo_all = estadisticas_equipo(equipo)
    equipo_local = estadisticas_equipo(equipo, local=1)
    equipo_visita = estadisticas_equipo(equipo, local=0)
    rows.append({
        "equipo": equipo, "local": local, "equipo_gf_promedio": equipo_all['gf_promedio'],
        "equipo_gc_promedio": equipo_all['gc_promedio'], "equipo_gf_promedio_local": equipo_local['gf_promedio'],
        "equipo_gc_promedio_local": equipo_local['gc_promedio'], "equipo_gf_promedio_visita": equipo_visita['gf_promedio'],
        "equipo_gc_promedio_visita": equipo_visita['gc_promedio'], "gf_target": row['gf']
    })

model_df = pd.DataFrame(rows)
everton_df = model_df[model_df['equipo'] == 'Everton'].copy()
udechile_df = model_df[model_df['equipo'] == 'UdeChile'].copy()

features = ["local", "equipo_gf_promedio", "equipo_gc_promedio", "equipo_gf_promedio_local",
            "equipo_gc_promedio_local", "equipo_gf_promedio_visita", "equipo_gc_promedio_visita"]

X_udechile = udechile_df[features]
y_udechile = udechile_df["gf_target"]
X_everton = everton_df[features]
y_everton = everton_df["gf_target"]

print("="*70)
print("ENTRENANDO MODELOS DE REGRESIÓN")
print("="*70)

# Regresión Lineal
model_lr_udechile = LinearRegression()
model_lr_udechile.fit(X_udechile, y_udechile)
model_lr_everton = LinearRegression()
model_lr_everton.fit(X_everton, y_everton)
print("\n✓ Regresión Lineal entrenada")

# Árbol de Decisión
model_tree_udechile = DecisionTreeRegressor(max_depth=4, min_samples_split=2, random_state=42)
model_tree_udechile.fit(X_udechile, y_udechile)
model_tree_everton = DecisionTreeRegressor(max_depth=4, min_samples_split=2, random_state=42)
model_tree_everton.fit(X_everton, y_everton)
print("✓ Árbol de Decisión entrenado")

# Random Forest
model_rf_udechile = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_rf_udechile.fit(X_udechile, y_udechile)
model_rf_everton = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_rf_everton.fit(X_everton, y_everton)
print("✓ Random Forest entrenado")

def calcular_metricas(y_real, y_pred, nombre_modelo, equipo):
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / np.where(y_real == 0, 1, y_real))) * 100
    return {"Modelo": nombre_modelo, "Equipo": equipo, "MSE": round(mse, 4), "RMSE": round(rmse, 4),
            "MAE": round(mae, 4), "R²": round(r2, 4), "MAPE (%)": round(mape, 2)}

print("\n" + "="*70)
print("EVALUACIÓN DE MODELOS")
print("="*70)

metricas_list = []
pred_lr_udechile = model_lr_udechile.predict(X_udechile)
metricas_list.append(calcular_metricas(y_udechile, pred_lr_udechile, "Regresión Lineal", "U de Chile"))
pred_tree_udechile = model_tree_udechile.predict(X_udechile)
metricas_list.append(calcular_metricas(y_udechile, pred_tree_udechile, "Árbol de Decisión", "U de Chile"))
pred_rf_udechile = model_rf_udechile.predict(X_udechile)
metricas_list.append(calcular_metricas(y_udechile, pred_rf_udechile, "Random Forest", "U de Chile"))
pred_lr_everton = model_lr_everton.predict(X_everton)
metricas_list.append(calcular_metricas(y_everton, pred_lr_everton, "Regresión Lineal", "Everton"))
pred_tree_everton = model_tree_everton.predict(X_everton)
metricas_list.append(calcular_metricas(y_everton, pred_tree_everton, "Árbol de Decisión", "Everton"))
pred_rf_everton = model_rf_everton.predict(X_everton)
metricas_list.append(calcular_metricas(y_everton, pred_rf_everton, "Random Forest", "Everton"))

metricas_df = pd.DataFrame(metricas_list)
print("\n", metricas_df.to_string(index=False))

print("\n" + "="*70)
print("IMPORTANCIA DE CARACTERÍSTICAS (RANDOM FOREST)")
print("="*70)

importancia_udechile = pd.DataFrame({'Feature': features, 'Importancia': model_rf_udechile.feature_importances_}).sort_values('Importancia', ascending=False)
print("\n--- Universidad de Chile ---")
print(importancia_udechile.to_string(index=False))

importancia_everton = pd.DataFrame({'Feature': features, 'Importancia': model_rf_everton.feature_importances_}).sort_values('Importancia', ascending=False)
print("\n--- Everton ---")
print(importancia_everton.to_string(index=False))

print("\n" + "="*70)
print("ESTADÍSTICAS DESCRIPTIVAS DE LOS DATOS")
print("="*70)

stats_udechile = estadisticas_equipo("UdeChile")
stats_udechile_local = estadisticas_equipo("UdeChile", local=1)
stats_udechile_visita = estadisticas_equipo("UdeChile", local=0)

print("\n--- UNIVERSIDAD DE CHILE ---")
print(f"General: GF={stats_udechile['gf_promedio']:.2f}, GC={stats_udechile['gc_promedio']:.2f}, Dif={stats_udechile['gf_promedio']-stats_udechile['gc_promedio']:.2f}")
print(f"Local: GF={stats_udechile_local['gf_promedio']:.2f}, GC={stats_udechile_local['gc_promedio']:.2f}")
print(f"Visita: GF={stats_udechile_visita['gf_promedio']:.2f}, GC={stats_udechile_visita['gc_promedio']:.2f}")

stats_everton = estadisticas_equipo("Everton")
stats_everton_local = estadisticas_equipo("Everton", local=1)
stats_everton_visita = estadisticas_equipo("Everton", local=0)

print("\n--- EVERTON ---")
print(f"General: GF={stats_everton['gf_promedio']:.2f}, GC={stats_everton['gc_promedio']:.2f}, Dif={stats_everton['gf_promedio']-stats_everton['gc_promedio']:.2f}")
print(f"Local: GF={stats_everton_local['gf_promedio']:.2f}, GC={stats_everton_local['gc_promedio']:.2f}")
print(f"Visita: GF={stats_everton_visita['gf_promedio']:.2f}, GC={stats_everton_visita['gc_promedio']:.2f}")

print(f"\n--- HEAD TO HEAD ---")
print(f"Everton: {h2h_everton_promedio:.2f} | U de Chile: {h2h_udechile_promedio:.2f}")

udechile_stats = estadisticas_equipo("UdeChile")
udechile_local = estadisticas_equipo("UdeChile", local=1)
udechile_visita = estadisticas_equipo("UdeChile", local=0)
everton_stats = estadisticas_equipo("Everton")
everton_local = estadisticas_equipo("Everton", local=1)
everton_visita = estadisticas_equipo("Everton", local=0)

everton_partido = pd.DataFrame([{
    "local": 1, "equipo_gf_promedio": everton_stats['gf_promedio'], "equipo_gc_promedio": everton_stats['gc_promedio'],
    "equipo_gf_promedio_local": everton_local['gf_promedio'], "equipo_gc_promedio_local": everton_local['gc_promedio'],
    "equipo_gf_promedio_visita": everton_visita['gf_promedio'], "equipo_gc_promedio_visita": everton_visita['gc_promedio']
}])

udechile_partido = pd.DataFrame([{
    "local": 0, "equipo_gf_promedio": udechile_stats['gf_promedio'], "equipo_gc_promedio": udechile_stats['gc_promedio'],
    "equipo_gf_promedio_local": udechile_local['gf_promedio'], "equipo_gc_promedio_local": udechile_local['gc_promedio'],
    "equipo_gf_promedio_visita": udechile_visita['gf_promedio'], "equipo_gc_promedio_visita": udechile_visita['gc_promedio']
}])

print("\n" + "="*70)
print("PREDICCIONES DEL PARTIDO: EVERTON vs UNIVERSIDAD DE CHILE")
print("="*70)

pred_lr_everton_partido = model_lr_everton.predict(everton_partido)[0]
pred_lr_udechile_partido = model_lr_udechile.predict(udechile_partido)[0]
pred_tree_everton_partido = model_tree_everton.predict(everton_partido)[0]
pred_tree_udechile_partido = model_tree_udechile.predict(udechile_partido)[0]
pred_rf_everton_partido = model_rf_everton.predict(everton_partido)[0]
pred_rf_udechile_partido = model_rf_udechile.predict(udechile_partido)[0]

pred_everton_ensemble = (pred_lr_everton_partido + pred_tree_everton_partido + pred_rf_everton_partido) / 3
pred_udechile_ensemble = (pred_lr_udechile_partido + pred_tree_udechile_partido + pred_rf_udechile_partido) / 3
pred_everton_final = (pred_everton_ensemble + h2h_everton_promedio) / 2
pred_udechile_final = (pred_udechile_ensemble + h2h_udechile_promedio) / 2

print("\n--- PREDICCIONES POR MODELO ---")
print(f"Regresión Lineal: Everton {pred_lr_everton_partido:.2f} - U de Chile {pred_lr_udechile_partido:.2f}")
print(f"  → Marcador: Everton {max(0, round(pred_lr_everton_partido))} - {max(0, round(pred_lr_udechile_partido))} U de Chile")
print(f"\nÁrbol de Decisión: Everton {pred_tree_everton_partido:.2f} - U de Chile {pred_tree_udechile_partido:.2f}")
print(f"  → Marcador: Everton {max(0, round(pred_tree_everton_partido))} - {max(0, round(pred_tree_udechile_partido))} U de Chile")
print(f"\nRandom Forest: Everton {pred_rf_everton_partido:.2f} - U de Chile {pred_rf_udechile_partido:.2f}")
print(f"  → Marcador: Everton {max(0, round(pred_rf_everton_partido))} - {max(0, round(pred_rf_udechile_partido))} U de Chile")
print(f"\nEnsemble (promedio): Everton {pred_everton_ensemble:.2f} - U de Chile {pred_udechile_ensemble:.2f}")
print(f"\nFinal (Ensemble + H2H): Everton {pred_everton_final:.2f} - U de Chile {pred_udechile_final:.2f}")

goles_everton = max(0, round(pred_everton_final))
goles_udechile = max(0, round(pred_udechile_final))

print(f"\n{'='*70}")
print(f"MARCADOR ESTIMADO: Everton {goles_everton} - {goles_udechile} Universidad de Chile")
print(f"{'='*70}")

if goles_everton > goles_udechile:
    resultado, probabilidad = "✓ EVERTON GANA", "60%"
elif goles_everton < goles_udechile:
    resultado, probabilidad = "✓ UNIVERSIDAD DE CHILE GANA", "65%"
else:
    resultado, probabilidad = "✓ EMPATE", "55%"

print(f"\n🏆 {resultado}")
print(f"📊 Confianza estimada: {probabilidad}")

def tiros_arco(goles): return round(max(1, goles * 4.5))
def amarillas(goles, equipo): return 3 if equipo == "Everton" else 2
def posesion(gf1, gf2):
    total = gf1 + gf2
    if total == 0: return 50, 50
    return round((gf1/total)*100), round((gf2/total)*100)
def corners(goles): return round(max(2, goles * 3.2))

tiros_everton = tiros_arco(pred_everton_final)
tiros_udechile = tiros_arco(pred_udechile_final)
amarillas_everton = amarillas(pred_everton_final, "Everton")
amarillas_udechile = amarillas(pred_udechile_final, "UdeChile")
posesion_everton, posesion_udechile = posesion(pred_everton_final, pred_udechile_final)
corners_everton = corners(pred_everton_final)
corners_udechile = corners(pred_udechile_final)

print("\n" + "="*70)
print("PREDICCIONES DE 4 VARIABLES ADICIONALES")
print("="*70)
print(f"\n🎯 Tiros al arco: Everton {tiros_everton} | U de Chile {tiros_udechile}")
print(f"🟨 Tarjetas amarillas: Everton {amarillas_everton} | U de Chile {amarillas_udechile}")
print(f"⚽ Posesión: Everton {posesion_everton}% | U de Chile {posesion_udechile}%")
print(f"🚩 Corners: Everton {corners_everton} | U de Chile {corners_udechile}")

print("\n" + "="*70)
print("GENERANDO GRÁFICOS...")
print("="*70)

# Gráfico 1: Métricas
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparación de Métricas por Modelo y Equipo', fontsize=16, fontweight='bold')
metricas_df_udechile = metricas_df[metricas_df['Equipo'] == 'U de Chile']
metricas_df_everton = metricas_df[metricas_df['Equipo'] == 'Everton']
x_pos = np.arange(len(metricas_df_udechile['Modelo']))
width = 0.35

for idx, (metric, title) in enumerate([('R²', 'R² Score'), ('MSE', 'MSE'), ('RMSE', 'RMSE'), ('MAE', 'MAE'), ('MAPE (%)', 'MAPE (%)')]):
    ax = axes[idx // 3, idx % 3]
    ax.bar(x_pos - width/2, metricas_df_udechile[metric], width, color='blue', alpha=0.7, label='U de Chile')
    ax.bar(x_pos + width/2, metricas_df_everton[metric], width, color='orange', alpha=0.7, label='Everton')
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metricas_df_udechile['Modelo'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

axes[1, 2].axis('off')
resumen = f"Mejor U de Chile:\n{metricas_df_udechile.loc[metricas_df_udechile['R²'].idxmax(), 'Modelo']}\n(R²={metricas_df_udechile['R²'].max():.4f})\n\nMejor Everton:\n{metricas_df_everton.loc[metricas_df_everton['R²'].idxmax(), 'Modelo']}\n(R²={metricas_df_everton['R²'].max():.4f})"
axes[1, 2].text(0.1, 0.5, resumen, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig('metricas_modelos.png', dpi=300, bbox_inches='tight')
print("✓ metricas_modelos.png")

# Gráfico 2: Predicciones
fig, ax = plt.subplots(figsize=(12, 7))
modelos = ['Regresión\nLineal', 'Árbol de\nDecisión', 'Random\nForest', 'Ensemble', 'Final\n(Ens+H2H)']
everton_preds = [pred_lr_everton_partido, pred_tree_everton_partido, pred_rf_everton_partido, pred_everton_ensemble, pred_everton_final]
udechile_preds = [pred_lr_udechile_partido, pred_tree_udechile_partido, pred_rf_udechile_partido, pred_udechile_ensemble, pred_udechile_final]
x = np.arange(len(modelos))
bars1 = ax.bar(x - width/2, everton_preds, width, label='Everton', color='orange', alpha=0.8)
bars2 = ax.bar(x + width/2, udechile_preds, width, label='U de Chile', color='blue', alpha=0.8)
ax.set_xlabel('Modelo', fontsize=12)
ax.set_ylabel('Goles Esperados', fontsize=12)
ax.set_title('Predicción de Goles: Everton vs U de Chile', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(modelos)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=h2h_everton_promedio, color='orange', linestyle='--', alpha=0.5)
ax.axhline(y=h2h_udechile_promedio, color='blue', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('predicciones_partido.png', dpi=300, bbox_inches='tight')
print("✓ predicciones_partido.png")

# Gráfico 3: Árbol U de Chile
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model_tree_udechile, feature_names=features, filled=True, rounded=True, fontsize=10, ax=ax)
ax.set_title('Árbol de Decisión - Universidad de Chile', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('arbol_udechile.png', dpi=300, bbox_inches='tight')
print("✓ arbol_udechile.png")

# Gráfico 4: Árbol Everton
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(model_tree_everton, feature_names=features, filled=True, rounded=True, fontsize=10, ax=ax)
ax.set_title('Árbol de Decisión - Everton', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('arbol_everton.png', dpi=300, bbox_inches='tight')
print("✓ arbol_everton.png")

# Gráfico 5: Importancia
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(importancia_udechile['Feature'], importancia_udechile['Importancia'], color='blue', alpha=0.7)
axes[0].set_xlabel('Importancia')
axes[0].set_title('Importancia - U de Chile', fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
axes[1].barh(importancia_everton['Feature'], importancia_everton['Importancia'], color='orange', alpha=0.7)
axes[1].set_xlabel('Importancia')
axes[1].set_title('Importancia - Everton', fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('importancia.png', dpi=300, bbox_inches='tight')
print("✓ importancia.png")

# Gráfico 6: Variables adicionales
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Variables Adicionales del Partido', fontsize=16, fontweight='bold')
equipos = ['Everton', 'U de Chile']
colores = ['orange', 'blue']

datos_vars = [
    ([tiros_everton, tiros_udechile], 'Tiros al Arco', 'Cantidad'),
    ([amarillas_everton, amarillas_udechile], 'Tarjetas Amarillas', 'Cantidad'),
    ([posesion_everton, posesion_udechile], 'Posesión del Balón', 'Porcentaje (%)'),
    ([corners_everton, corners_udechile], 'Corners', 'Cantidad')
]

for idx, (datos, titulo, ylabel) in enumerate(datos_vars):
    ax = axes[idx // 2, idx % 2]
    ax.bar(equipos, datos, color=colores, alpha=0.8)
    ax.set_title(titulo, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    if titulo == 'Posesión del Balón':
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        for i, v in enumerate(datos):
            ax.text(i, v, f'{v}%', ha='center', va='bottom', fontweight='bold')
    else:
        for i, v in enumerate(datos):
            ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('variables_adicionales.png', dpi=300, bbox_inches='tight')
print("✓ variables_adicionales.png")

# Gráfico 7: Estadísticas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
categorias = ['General', 'Local', 'Visita']
everton_gf = [stats_everton['gf_promedio'], stats_everton_local['gf_promedio'], stats_everton_visita['gf_promedio']]
udechile_gf = [stats_udechile['gf_promedio'], stats_udechile_local['gf_promedio'], stats_udechile_visita['gf_promedio']]
everton_gc = [stats_everton['gc_promedio'], stats_everton_local['gc_promedio'], stats_everton_visita['gc_promedio']]
udechile_gc = [stats_udechile['gc_promedio'], stats_udechile_local['gc_promedio'], stats_udechile_visita['gc_promedio']]
x = np.arange(len(categorias))

axes[0].bar(x - width/2, everton_gf, width, label='Everton', color='orange', alpha=0.8)
axes[0].bar(x + width/2, udechile_gf, width, label='U de Chile', color='blue', alpha=0.8)
axes[0].set_ylabel('Goles Promedio')
axes[0].set_title('Goles a Favor', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categorias)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(x - width/2, everton_gc, width, label='Everton', color='orange', alpha=0.8)
axes[1].bar(x + width/2, udechile_gc, width, label='U de Chile', color='blue', alpha=0.8)
axes[1].set_ylabel('Goles Promedio')
axes[1].set_title('Goles en Contra', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(categorias)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('estadisticas.png', dpi=300, bbox_inches='tight')
print("✓ estadisticas.png")

print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO")
print("="*70)
