import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mplsoccer import VerticalPitch
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import numpy as np

# === Load and combine all CSVs from 'FEN/' folder ===
folder_path = 'Event-data/Women/WEURI 2025/FEN'
all_dfs = []

for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)
        df['source_file'] = file  # optional: track file origin
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

# === Filter: only blocked shots ===
df_blocked = df_all[df_all['shot_type'] == 'blocked'].copy()
df_blocked['danger_score_fenwick'] = df_blocked['danger_score_fenwick'].fillna(0)

# === Nonlinear scaling for size (sqrt scaling to reduce impact of outliers) ===
danger_scores = df_blocked['danger_score_fenwick'].values.reshape(-1, 1)
danger_scores_scaled = np.sqrt(danger_scores)  # sqrt transform

scaler = MinMaxScaler(feature_range=(80, 600))
df_blocked['size'] = scaler.fit_transform(danger_scores_scaled)

# === Set up vertical pitch with grid lines ===
pitch = VerticalPitch(pitch_type='opta', half=True,
                      pitch_color='white', line_color='#222222',
                      line_zorder=3)
fig, ax = pitch.draw(figsize=(7, 11))

# === Scatter plot with alpha, edge width varies by danger score ===
max_danger = df_blocked['danger_score_fenwick'].max()
min_danger = df_blocked['danger_score_fenwick'].min()

# Normalize edge width: thicker edges for higher danger
edge_width = 0.5 + 2 * (df_blocked['danger_score_fenwick'] - min_danger) / (max_danger - min_danger + 1e-6)

scatter = pitch.scatter(df_blocked['x'], df_blocked['y'],
                        s=df_blocked['size'],
                        c=df_blocked['danger_score_fenwick'],
                        cmap='Reds',
                        alpha=0.7,
                        edgecolors='black',
                        linewidth= edge_width,
                        ax=ax,
                        zorder=4)

# === Add pitch zones or heatmap grid (optional) ===
# For simplicity, add a faint grid on pitch for tactical context
for x in np.linspace(0, 120, 7):
    ax.axvline(x=x, color='grey', linestyle='--', alpha=0.15, zorder=1)
for y in np.linspace(0, 80, 5):
    ax.axhline(y=y, color='grey', linestyle='--', alpha=0.15, zorder=1)

# === Colorbar enhanced ===
cbar_ax = fig.add_axes([0.2, 0.20, 0.6, 0.02])
norm = mpl.colors.Normalize(vmin=min_danger, vmax=max_danger)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='Reds'),
                    cax=cbar_ax,
                    orientation='horizontal',
                    ticks=np.linspace(min_danger, max_danger, 5))
cbar.set_label('Danger Score (Fenwick)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# === Title and subtitle with summary stats ===
total_blocked = len(df_blocked)
avg_danger = df_blocked['danger_score_fenwick'].mean()

plt.suptitle('Blocked Shots - Size & Color by Danger Score (Fenwick)', fontsize=16, y=0.7, weight='bold')
plt.title(f'Total Blocked Shots: {total_blocked} | Average Danger Score: {avg_danger:.3f}', fontsize=12, y=23)

plt.tight_layout(rect=[0, 0, 1, 0.92])
# === Save figure as PNG with white background ===
fig.savefig('blocked_shots_fenwick.png', dpi=300, facecolor='white', edgecolor='white', transparent=False)

plt.show()
