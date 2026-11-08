"""
Generates a synthetic but realistic crop recommendation dataset.
Each crop has parameter ranges based on typical agronomic requirements
(N, P, K in kg/ha; temperature in C; humidity in %; ph; rainfall in mm).

Why synthetic: keeps the project fully reproducible offline without
depending on a third-party download link. Ranges are drawn from
commonly cited agronomic references for each crop.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# (crop, N_range, P_range, K_range, temp_range, humidity_range, ph_range, rainfall_range)
CROP_PROFILES = [
    ("rice",        (60, 100), (30, 60),  (30, 50),  (20, 32), (70, 95), (5.5, 7.0), (150, 300)),
    ("maize",       (60, 100), (35, 65),  (15, 30),  (18, 30), (50, 75), (5.5, 7.5), (60, 120)),
    ("chickpea",    (10, 40),  (55, 80),  (75, 100), (15, 28), (14, 30), (6.0, 8.0), (60, 100)),
    ("kidneybeans", (10, 40),  (55, 80),  (15, 25),  (15, 25), (18, 25), (5.5, 6.5), (60, 110)),
    ("pigeonpeas",  (10, 40),  (55, 80),  (15, 25),  (18, 35), (30, 70), (4.5, 7.5), (60, 150)),
    ("mothbeans",   (10, 40),  (30, 60),  (15, 25),  (24, 35), (30, 60), (3.5, 9.9), (30, 65)),
    ("mungbean",    (10, 40),  (30, 60),  (15, 25),  (25, 35), (60, 90), (6.0, 7.5), (40, 70)),
    ("blackgram",   (10, 40),  (55, 80),  (15, 25),  (24, 35), (60, 75), (6.0, 7.5), (60, 90)),
    ("lentil",      (10, 40),  (55, 80),  (15, 25),  (15, 25), (60, 75), (5.5, 7.0), (30, 60)),
    ("pomegranate", (10, 40),  (10, 30),  (30, 45),  (18, 25), (85, 95), (5.5, 7.0), (100, 130)),
    ("banana",      (80, 120), (70, 100), (45, 55),  (25, 30), (75, 90), (5.5, 6.5), (100, 180)),
    ("mango",       (10, 40),  (10, 30),  (25, 35),  (25, 35), (45, 60), (4.5, 7.0), (60, 100)),
    ("grapes",      (10, 40),  (110,145), (190,205), (10, 25), (75, 85), (5.5, 6.5), (60, 90)),
    ("watermelon",  (80, 120), (10, 30),  (45, 55),  (24, 32), (55, 75), (5.5, 6.8), (35, 55)),
    ("muskmelon",   (80, 120), (10, 30),  (45, 55),  (25, 32), (85, 95), (6.0, 6.8), (20, 35)),
    ("apple",       (10, 40),  (110,145), (190,205), (12, 22), (85, 95), (5.5, 6.5), (100, 130)),
    ("orange",      (10, 40),  (10, 30),  (5, 15),   (15, 30), (85, 95), (6.0, 7.5), (100, 130)),
    ("papaya",      (30, 70),  (45, 70),  (45, 55),  (23, 35), (85, 95), (6.0, 7.0), (40, 130)),
    ("coconut",     (10, 40),  (5, 30),   (25, 35),  (25, 32), (90, 100),(5.5, 7.0), (130, 220)),
    ("cotton",      (100,140), (35, 65),  (15, 25),  (22, 30), (65, 85), (5.5, 8.0), (60, 100)),
    ("jute",        (60, 100), (35, 60),  (35, 45),  (23, 32), (70, 90), (5.9, 7.5), (150, 200)),
    ("coffee",      (80, 120), (15, 35),  (25, 35),  (18, 28), (50, 70), (6.0, 7.5), (150, 220)),
]

N_SAMPLES_PER_CROP = 100
rows = []

for crop, n_r, p_r, k_r, t_r, h_r, ph_r, r_r in CROP_PROFILES:
    for _ in range(N_SAMPLES_PER_CROP):
        rows.append({
            "N": round(np.random.uniform(*n_r), 1),
            "P": round(np.random.uniform(*p_r), 1),
            "K": round(np.random.uniform(*k_r), 1),
            "temperature": round(np.random.uniform(*t_r), 2),
            "humidity": round(np.random.uniform(*h_r), 2),
            "ph": round(np.random.uniform(*ph_r), 2),
            "rainfall": round(np.random.uniform(*r_r), 2),
            "label": crop,
        })

import os
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Crop_recommendation.csv")

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUT_PATH, index=False)
print(f"Generated {len(df)} rows across {len(CROP_PROFILES)} crops -> {OUT_PATH}")
