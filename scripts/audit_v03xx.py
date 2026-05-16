"""
Аудит данных предсказаний v03xx: поиск аномалий по всем моделям,
наборам признаков и таргетам.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

RESULTS_DIR = Path("/Users/tascan/Desktop/диссер/results")
VERSIONS    = ["v0301","v0302","v0303","v0304","v0305","v0306",
               "v0307","v0308","v0309","v0310"]
FSETS       = ["EMG", "EMG_NIRS", "EMG_NIRS_HRV"]
TARGETS     = ["lt1", "lt2"]
DT          = 5.0  # сек на окно

# ─────────────────────── helpers ───────────────────────
def load(ver, tgt, fset):
    d = RESULTS_DIR / ver
    yp = d / f"ypred_{tgt}_{fset}.npy"
    yt = d / f"ytrue_{tgt}_{fset}.npy"
    if not yp.exists() or not yt.exists():
        return None, None
    return np.load(yp), np.load(yt)

def subject_splits(ytrue):
    diffs = np.diff(ytrue)
    bounds = np.where(diffs > 10)[0] + 1
    return np.split(ytrue, bounds), np.split(np.arange(len(ytrue)), bounds)

# ─────────────────────── сбор всех данных ───────────────────────
records = []   # строка = (ver, tgt, fset, subj_idx, ...)

for ver in VERSIONS:
    for tgt in TARGETS:
        for fset in FSETS:
            yp, yt = load(ver, tgt, fset)
            if yp is None:
                continue
            segs_yt, segs_idx = subject_splits(yt)
            for i, (seg_yt, seg_idx) in enumerate(zip(segs_yt, segs_idx)):
                seg_yp = yp[seg_idx]
                records.append(dict(
                    ver=ver, tgt=tgt, fset=fset,
                    subj=i,
                    n_windows=len(seg_yt),
                    ytrue_start=seg_yt[0],
                    ytrue_end=seg_yt[-1],
                    ypred_mean=float(seg_yp.mean()),
                    ypred_std=float(seg_yp.std()),
                    ypred_min=float(seg_yp.min()),
                    ypred_max=float(seg_yp.max()),
                    mae=float(np.abs(seg_yp - seg_yt).mean()),
                    has_nan=bool(np.isnan(seg_yp).any()),
                    has_inf=bool(np.isinf(seg_yp).any()),
                ))

df = pd.DataFrame(records)
print(f"Всего записей (ver×tgt×fset×subj): {len(df)}")
print(f"Версий: {df['ver'].nunique()}  |  Субъектов на модель: {df.groupby(['ver','tgt','fset'])['subj'].count().describe()[['min','max','mean']].to_dict()}")

issues = []

# ─────────────────────── 1. NaN / Inf ───────────────────────
bad = df[df['has_nan'] | df['has_inf']]
print(f"\n{'='*60}")
print(f"[1] NaN / Inf в ypred: {len(bad)} случаев")
if len(bad):
    print(bad[['ver','tgt','fset','subj','has_nan','has_inf']].to_string(index=False))
    issues.append("NaN/Inf найдены")

# ─────────────────────── 2. Число субъектов ───────────────────────
print(f"\n{'='*60}")
print("[2] Число субъектов на конфиг (ожидается одинаково):")
n_subj = df.groupby(['ver','tgt','fset'])['subj'].count().reset_index()
n_subj.columns = ['ver','tgt','fset','n_subj']
agg = n_subj.groupby(['ver','tgt'])['n_subj'].agg(['min','max','nunique'])
weird = agg[agg['nunique'] > 1]
if len(weird):
    print("  ⚠  Расхождение числа субъектов между fset:")
    print(weird)
    issues.append("Расхождение числа субъектов")
else:
    print(f"  ✅ Везде одинаково. Диапазон: {n_subj['n_subj'].min()}–{n_subj['n_subj'].max()}")

# ─────────────────────── 3. Число окон: согласованность fset ───────────────────────
print(f"\n{'='*60}")
print("[3] Согласованность числа окон между fset (в одном ver+tgt+subj):")
pivot_n = df.pivot_table(index=['ver','tgt','subj'], columns='fset', values='n_windows')
mismatch = pivot_n[pivot_n.nunique(axis=1) > 1]
if len(mismatch):
    print(f"  ⚠  {len(mismatch)} субъект(а) с разным числом окон между fset:")
    print(mismatch.to_string())
    issues.append(f"Расхождение окон между fset: {len(mismatch)} случаев")
else:
    print("  ✅ Окна согласованы между EMG / EMG+NIRS / EMG+NIRS+HRV")

# ─────────────────────── 4. ytrue согласованность между fset ───────────────────────
print(f"\n{'='*60}")
print("[4] Согласованность ytrue между fset (должны быть идентичны):")
ytrue_mismatch = 0
for (ver, tgt, subj), grp in df.groupby(['ver','tgt','subj']):
    starts = grp['ytrue_start'].values
    ends   = grp['ytrue_end'].values
    if starts.max() - starts.min() > 1.0 or ends.max() - ends.min() > 1.0:
        ytrue_mismatch += 1
        if ytrue_mismatch <= 5:
            print(f"  ⚠  {ver} {tgt} subj={subj}: ytrue_start {starts}  ytrue_end {ends}")
if ytrue_mismatch == 0:
    print("  ✅ ytrue идентичен между fset")
else:
    print(f"  ⚠  Итого расхождений: {ytrue_mismatch}")
    issues.append(f"ytrue не совпадает между fset: {ytrue_mismatch}")

# ─────────────────────── 5. Коллапс модели (std ypred) ───────────────────────
print(f"\n{'='*60}")
print("[5] Коллапс модели (std ypred < 5с на субъекта):")
collapsed = df[df['ypred_std'] < 5.0]
if len(collapsed):
    print(f"  ⚠  {len(collapsed)} записей со std < 5с:")
    print(collapsed[['ver','tgt','fset','subj','n_windows','ypred_std','mae']].to_string(index=False))
    issues.append(f"Коллапс модели: {len(collapsed)} записей")
else:
    print("  ✅ Нет коллапса (std > 5с везде)")

# ─────────────────────── 6. Экстремальные MAE по субъектам ───────────────────────
print(f"\n{'='*60}")
print("[6] Субъекты с экстремальным MAE (> 3×медиана для своего ver+tgt+fset):")
extreme_rows = []
for key, grp in df.groupby(['ver','tgt','fset']):
    med = grp['mae'].median()
    bad = grp[grp['mae'] > 3 * med]
    extreme_rows.append(bad)
extreme = pd.concat(extreme_rows)
if len(extreme):
    print(f"  ⚠  {len(extreme)} записей:")
    print(extreme[['ver','tgt','fset','subj','n_windows','mae']].sort_values('mae', ascending=False).head(20).to_string(index=False))
    issues.append(f"Экстремальный MAE у {extreme[['ver','tgt','fset','subj']].drop_duplicates().shape[0]} субъектов")
else:
    print("  ✅ Нет экстремальных выбросов MAE")

# ─────────────────────── 7. Экстремальные ypred значения ───────────────────────
print(f"\n{'='*60}")
print("[7] Экстремальные ypred (>1200с или <-1800с = >20 мин / <-30 мин):")
df['extreme_val'] = (df['ypred_max'] > 1200) | (df['ypred_min'] < -1800)
ext_val = df[df['extreme_val']]
if len(ext_val):
    print(f"  ⚠  {len(ext_val)} записей:")
    print(ext_val[['ver','tgt','fset','subj','ypred_min','ypred_max']].to_string(index=False))
    issues.append(f"Экстремальные ypred: {len(ext_val)}")
else:
    print("  ✅ Все ypred в разумных пределах")

# ─────────────────────── 8. Число окон между версиями (stateless) ───────────────────────
print(f"\n{'='*60}")
print("[8] Число окон по версиям для одного taрget+subj (ожидаем совпадение у stateless):")
stateless = [v for v in VERSIONS if v not in ["v0308"]]  # stateful отдельно
pivot_v = df[df['ver'].isin(stateless)].pivot_table(
    index=['tgt','subj'], columns='ver', values='n_windows', aggfunc='first')
mismatch_v = pivot_v[pivot_v.nunique(axis=1) > 1]
if len(mismatch_v):
    print(f"  ⚠  {len(mismatch_v)} комбинаций tgt+subj с разным числом окон между версиями:")
    print(mismatch_v.to_string())
    issues.append(f"Разное число окон между stateless версиями: {len(mismatch_v)}")
else:
    print("  ✅ Число окон одинаково для одного subj между stateless версиями")

# ─────────────────────── 9. Малое число окон ───────────────────────
print(f"\n{'='*60}")
print("[9] Субъекты с очень малым числом окон (< 10):")
tiny = df[df['n_windows'] < 10]
if len(tiny):
    print(f"  ⚠  {len(tiny)} записей:")
    print(tiny[['ver','tgt','fset','subj','n_windows']].to_string(index=False))
    issues.append(f"Мало окон (<10): {len(tiny)}")
else:
    print("  ✅ Все субъекты имеют ≥ 10 окон")

# ─────────────────────── итог ───────────────────────
print(f"\n{'='*60}")
print(f"ИТОГ: {'⚠  НАЙДЕНЫ ПРОБЛЕМЫ' if issues else '✅ Аномалий не обнаружено'}")
for i, iss in enumerate(issues, 1):
    print(f"  {i}. {iss}")

# Сохраняем полную таблицу субъектов
out = Path("/Users/tascan/Desktop/диссер/results/v03xx_analysis/audit_subjects.csv")
df.to_csv(out, index=False)
print(f"\n  Полная таблица → {out.name}")
