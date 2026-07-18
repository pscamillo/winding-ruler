#!/usr/bin/env python3
"""
ruler_concordance_v1_1.py — pré-braço-(b), etapa de diagnóstico/calibração.

Novidades vs v1:
  - Calibração empírica SEM VAZAMENTO: split por collection (seed fixo),
    fator único k ajustado no train (min quadrados pela origem),
    concordância exata reportada APENAS no test.
  - Scatter por par: percentis do ratio individual (integral/truth) para
    cada truth_dw — é a variância, não a média, que decide a usabilidade.
  - Critério pré-registrado (inalterado, agora aplicado ao test calibrado):
    exata >= 80% => braço (b) autorizado; 60-80% => diagnóstico adicional;
    < 60% => mecanismo reprovado.

Convenções idênticas à v1 (LasagnaNormalSampler.cpp 37c37de; decode 4000;
trilinear; trapezoidal; coords/4 p/ grupo '4').
"""
import argparse
import json
import os
import sys
import csv
import random
from collections import defaultdict

import numpy as np
import zarr
from scipy.ndimage import map_coordinates

ENCODE_SCALE = 1000.0
GRAD_MAG_FACTOR = 0.25
DECODE = ENCODE_SCALE / GRAD_MAG_FACTOR
LASAGNA_SCALE = 4
STEP_VX = 8.0
MAX_TRUTH_DELTA = 6


def load_gradmag_subvolume(dataset, z_begin, z_end, margin_vx=64):
    path = os.path.join(dataset, 'lasagna_inputs', 'las_008_grad_mag.ome.zarr')
    root = zarr.open(path, mode='r')
    arr = root['4']
    z0_full = max(0, z_begin - margin_vx)
    z1_full = z_end + margin_vx
    z0 = z0_full // LASAGNA_SCALE
    z1 = min(arr.shape[0], (z1_full + LASAGNA_SCALE - 1) // LASAGNA_SCALE)
    sub = arr[z0:z1, :, :]
    print(f'grad_mag[4] subvolume: shape={sub.shape} '
          f'(z full-res [{z0*LASAGNA_SCALE}, {z1*LASAGNA_SCALE}))')
    return np.asarray(sub), z0 * LASAGNA_SCALE


def winding_distance(sub, z0_full, a_xyz, b_xyz, step_vx=STEP_VX):
    a = np.asarray(a_xyz, dtype=np.float64)
    b = np.asarray(b_xyz, dtype=np.float64)
    dist = float(np.linalg.norm(b - a))
    if not (dist > 1e-9) or not np.isfinite(dist):
        return 0.0, True
    intervals = max(1, int(np.ceil(dist / step_vx)))
    t = np.linspace(0.0, 1.0, intervals + 1)
    pts = a[None, :] * (1.0 - t)[:, None] + b[None, :] * t[:, None]
    zi = (pts[:, 2] - z0_full) / LASAGNA_SCALE
    yi = pts[:, 1] / LASAGNA_SCALE
    xi = pts[:, 0] / LASAGNA_SCALE
    dz, dy, dx = sub.shape
    inside = (zi >= 0) & (zi <= dz - 1) & (yi >= 0) & (yi <= dy - 1) & \
             (xi >= 0) & (xi <= dx - 1)
    if not inside.all():
        return float('inf'), False
    dens = map_coordinates(sub, np.stack([zi, yi, xi]), order=1,
                           mode='nearest').astype(np.float64) / DECODE
    seg = dist / intervals
    return float(np.sum(0.5 * (dens[:-1] + dens[1:]) * seg)), True


def collect_pairs(dataset, z_begin, z_end):
    d = json.load(open(os.path.join(dataset, 'relative_windings.json')))
    pairs = []
    for cid, coll in d['collections'].items():
        pts = [(pt['p'], float(pt['wind_a']))
               for pt in coll['points'].values()
               if pt.get('wind_a') is not None
               and z_begin <= pt['p'][2] < z_end]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dw = abs(pts[i][1] - pts[j][1])
                if 0 < dw <= MAX_TRUTH_DELTA:
                    pairs.append((cid, pts[i][0], pts[j][0], dw))
    colls = sorted({p[0] for p in pairs})
    print(f'{len(colls)} collections | {len(pairs)} pares')
    return pairs, colls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--step-vx', type=float, default=STEP_VX)
    ap.add_argument('--split-seed', type=int, default=1)
    ap.add_argument('--out', default='/tmp/concordance_calib.csv')
    args = ap.parse_args()

    pairs, colls = collect_pairs(args.dataset, args.z_begin, args.z_end)
    if len(colls) < 4:
        print('Poucas collections para split honesto.'); sys.exit(1)

    sub, z0_full = load_gradmag_subvolume(args.dataset, args.z_begin, args.z_end)

    # integrais de todos os pares (uma passada)
    rows = []
    for cid, a, b, truth in pairs:
        integral, ok = winding_distance(sub, z0_full, a, b, args.step_vx)
        if ok:
            rows.append([cid, truth, integral])
    print(f'pares com cobertura: {len(rows)}')

    # ---- SCATTER por par (variância decide) ----
    by = defaultdict(list)
    for _, truth, integral in rows:
        by[truth].append(integral / truth)
    print(f'\n== SCATTER do ratio individual (integral/truth) ==')
    print(f'{"truth_dw":>8} {"n":>5} {"p10":>7} {"p25":>7} {"p50":>7} '
          f'{"p75":>7} {"p90":>7} {"CV%":>6}')
    for truth in sorted(by):
        r = np.array(by[truth])
        p = np.percentile(r, [10, 25, 50, 75, 90])
        cv = 100.0 * r.std() / r.mean() if r.mean() > 0 else float('nan')
        print(f'{truth:>8.0f} {len(r):>5} {p[0]:>7.3f} {p[1]:>7.3f} '
              f'{p[2]:>7.3f} {p[3]:>7.3f} {p[4]:>7.3f} {cv:>6.1f}')
    print('Leitura: CV <~15% => fator único deve destravar concordância alta.'
          '\n         CV >~30% => nuvem larga, fator único não salva.')

    # ---- CALIBRAÇÃO split por collection ----
    rng = random.Random(args.split_seed)
    shuffled = colls[:]
    rng.shuffle(shuffled)
    train_set = set(shuffled[:len(shuffled) // 2])
    train = [(t, i) for cid, t, i in rows if cid in train_set]
    test = [(cid, t, i) for cid, t, i in rows if cid not in train_set]
    ti = np.array([i for _, i in train]); tt = np.array([t for t, _ in train])
    k = float(np.sum(ti * tt) / np.sum(ti * ti))  # min || k*int - truth ||²
    print(f'\n== CALIBRAÇÃO (split por collection, seed {args.split_seed}) ==')
    print(f'train: {len(train)} pares / {len(train_set)} colls  |  '
          f'test: {len(test)} pares / {len(colls)-len(train_set)} colls')
    print(f'fator k (train): {k:.3f}  (1/k = {1.0/k:.3f})')

    exact = 0
    err = []
    by_t = defaultdict(lambda: [0, 0])
    out_rows = []
    for cid, truth, integral in test:
        pred = int(round(k * integral))
        ex = int(pred == truth)
        exact += ex
        err.append(abs(k * integral - truth))
        by_t[truth][0] += 1; by_t[truth][1] += ex
        out_rows.append([cid, truth, integral, k * integral, pred, ex])
    n = len(test)
    print(f'\nCONCORDÂNCIA EXATA (test, calibrado): {exact}/{n} '
          f'= {100.0*exact/max(n,1):.1f}%')
    print(f'MAE calibrado (test): {np.mean(err):.3f} voltas')
    print(f'{"truth_dw":>8} {"n":>5} {"exact%":>7}')
    for truth in sorted(by_t):
        cnt, ex = by_t[truth]
        print(f'{truth:>8.0f} {cnt:>5} {100.0*ex/cnt:>6.1f}%')

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['collection', 'truth_dw', 'integral_raw',
                    'integral_calib', 'pred_dw', 'exact'])
        w.writerows(out_rows)
    print(f'\nCSV (test): {args.out}')
    print('Critério pré-registrado sobre o TEST calibrado: >=80% autoriza; '
          '60-80% diagnóstico; <60% reprova.')


if __name__ == '__main__':
    main()
