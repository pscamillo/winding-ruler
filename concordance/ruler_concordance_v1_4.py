#!/usr/bin/env python3
"""
ruler_concordance_v1_4.py — pré-braço-(b), mecanismo v4: integral direcional.

Perfis (v1.3) mostraram campo contínuo quase plano => informação está na
integral; variância vem de amplitude local E de inclinação raio×folha.
v1.4 usa os canais de NORMAL (nx, ny) para corrigir a geometria.

Decode das normais (LasagnaNormalSampler.cpp:48-57, commit 37c37de):
    comp = (raw - 128.0) / 127.0 ;  nz = sqrt(max(0, 1 - nx² - ny²)) ; normaliza.
AVISO do próprio C++: sinal do normal é AMBÍGUO por voxel (o sampler usa
tensor de estrutura + hint). Por isso:

Estimadores comparados (mesmo split por collection, mesmo test):
  A integral   : ∫ g dl                     (baseline v1.1, k_A no train)
  B cosweight  : ∫ g·|n̂·d̂| dl              (corrige inclinação; imune a flip)
  C signed     : ∫ g·(n̂·d̂) dl com sinal por consistência ao longo do raio
                 (flip local p/ alinhar com amostra anterior); reporta também
                 ACERTO DE SINAL vs sign(wind_b - wind_a).

Critério pré-registrado: exato >= 80% no test autoriza o braço (b).
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
MAX_TRUTH_DELTA = 6


def load_channel(dataset, name, z_begin, z_end, margin_vx=64):
    path = os.path.join(dataset, 'lasagna_inputs', f'las_008_{name}.ome.zarr')
    root = zarr.open(path, mode='r')
    arr = root['4']
    z0_full = max(0, z_begin - margin_vx)
    z1_full = z_end + margin_vx
    z0 = z0_full // LASAGNA_SCALE
    z1 = min(arr.shape[0], (z1_full + LASAGNA_SCALE - 1) // LASAGNA_SCALE)
    sub = np.asarray(arr[z0:z1, :, :])
    print(f'{name}[4]: shape={sub.shape}')
    return sub, z0 * LASAGNA_SCALE


def sample_all(subs, z0_full, a_xyz, b_xyz, sample_vx):
    """Amostra (grad_mag, nx, ny) trilinear ao longo de a->b.
    Retorna dict de arrays, seg_len, direcao unit d_hat(xyz), ok."""
    a = np.asarray(a_xyz, dtype=np.float64)
    b = np.asarray(b_xyz, dtype=np.float64)
    delta = b - a
    dist = float(np.linalg.norm(delta))
    if not (dist > 1e-9) or not np.isfinite(dist):
        return None, 0.0, None, True
    d_hat = delta / dist
    n = max(2, int(np.ceil(dist / sample_vx)) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts = a[None, :] * (1.0 - t)[:, None] + b[None, :] * t[:, None]
    zi = (pts[:, 2] - z0_full) / LASAGNA_SCALE
    yi = pts[:, 1] / LASAGNA_SCALE
    xi = pts[:, 0] / LASAGNA_SCALE
    dz, dy, dx = subs['grad_mag'].shape
    inside = (zi >= 0) & (zi <= dz - 1) & (yi >= 0) & (yi <= dy - 1) & \
             (xi >= 0) & (xi <= dx - 1)
    if not inside.all():
        return None, 0.0, None, False
    coords = np.stack([zi, yi, xi])
    out = {}
    for name, sub in subs.items():
        out[name] = map_coordinates(sub, coords, order=1, mode='nearest'
                                    ).astype(np.float64)
    return out, dist / (n - 1), d_hat, True


def decode_normals(raw_nx, raw_ny):
    nx = (raw_nx - 128.0) / 127.0
    ny = (raw_ny - 128.0) / 127.0
    nz2 = np.maximum(0.0, 1.0 - nx * nx - ny * ny)
    nz = np.sqrt(nz2)
    v = np.stack([nx, ny, nz], axis=-1)  # [n, 3] em ordem (x, y, z)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return v / norm


def estimators(samples, seg, d_hat):
    g = samples['grad_mag'] / DECODE
    nrm = decode_normals(samples['nx'], samples['ny'])       # [n,3] xyz
    dot = nrm @ d_hat                                        # n̂·d̂ por amostra
    # C: consistência de sinal ao longo do raio — o encoding tem flip
    # arbitrário por voxel (vide tensor+hint no C++); alinhamos cada amostra
    # com a referência acumulada: se dot[i] discorda da referência, flipa.
    sgn = np.ones(len(dot))
    ref = dot[0] if abs(dot[0]) > 1e-6 else 1.0
    for i in range(1, len(dot)):
        if abs(dot[i]) > 1e-6:
            sgn[i] = 1.0 if (dot[i] * ref) >= 0 else -1.0
            ref = dot[i] * sgn[i]
        else:
            sgn[i] = sgn[i - 1]

    def trap(vals):
        return float(np.sum(0.5 * (vals[:-1] + vals[1:]) * seg))

    est_a = trap(g)
    est_b = trap(g * np.abs(dot))
    est_c = trap(g * dot * sgn)
    return est_a, est_b, est_c


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
                dw = pts[j][1] - pts[i][1]  # COM SINAL (a -> b)
                if 0 < abs(dw) <= MAX_TRUTH_DELTA:
                    pairs.append((cid, pts[i][0], pts[j][0], dw))
    colls = sorted({p[0] for p in pairs})
    print(f'{len(colls)} collections | {len(pairs)} pares')
    return pairs, colls


def fit_k(train):
    ti = np.array([abs(i) for _, i in train])
    tt = np.array([t for t, _ in train])
    return float(np.sum(ti * tt) / np.sum(ti * ti))


def report(name, rows, out_csv=None, sign_col=False):
    n = len(rows)
    exact = sum(r[4] for r in rows)
    mae = np.mean([abs(r[3] - r[1]) for r in rows]) if n else float('nan')
    print(f'\n== {name} — TEST ==')
    print(f'EXATA: {exact}/{n} = {100.0*exact/max(n,1):.1f}%   MAE: {mae:.3f}')
    if sign_col:
        sg = [r[5] for r in rows if r[5] is not None]
        if sg:
            print(f'ACERTO DE SINAL: {sum(sg)}/{len(sg)} '
                  f'= {100.0*sum(sg)/len(sg):.1f}%')
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[abs(r[1])][0] += 1; by[abs(r[1])][1] += r[4]
    print(f'{"|dw|":>5} {"n":>5} {"exact%":>7}')
    for truth in sorted(by):
        cnt, ex = by[truth]
        print(f'{truth:>5.0f} {cnt:>5} {100.0*ex/cnt:>6.1f}%')
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['collection', 'truth_dw_signed', 'raw', 'pred_abs',
                        'exact', 'sign_ok'])
            w.writerows(rows)
    return 100.0 * exact / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--sample-vx', type=float, default=2.0)
    ap.add_argument('--split-seed', type=int, default=1)
    ap.add_argument('--out-prefix', default='/tmp/concordance_v14')
    args = ap.parse_args()

    pairs, colls = collect_pairs(args.dataset, args.z_begin, args.z_end)
    subs = {}
    z0 = None
    for name in ('grad_mag', 'nx', 'ny'):
        subs[name], z0 = load_channel(args.dataset, name,
                                      args.z_begin, args.z_end)

    per_pair = []
    for cid, a, b, dw in pairs:
        samples, seg, d_hat, ok = sample_all(subs, z0, a, b, args.sample_vx)
        if not ok or samples is None:
            continue
        ea, eb, ec = estimators(samples, seg, d_hat)
        per_pair.append((cid, dw, ea, eb, ec))
    print(f'pares com cobertura: {len(per_pair)}')

    rng = random.Random(args.split_seed)
    shuffled = colls[:]
    rng.shuffle(shuffled)
    train_set = set(shuffled[:len(shuffled) // 2])

    results = {}
    for label, idx in (('A_integral', 2), ('B_cosweight', 3), ('C_signed', 4)):
        train = [(abs(dw), row[idx]) for row in per_pair
                 for cid, dw in [(row[0], row[1])] if cid in train_set]
        test = [row for row in per_pair if row[0] not in train_set]
        k = fit_k(train)
        rows = []
        for cid, dw, *ests in test:
            raw = ests[idx - 2]
            pred_abs = int(round(k * abs(raw)))
            exact = int(pred_abs == abs(dw))
            sign_ok = None
            if label == 'C_signed' and raw != 0:
                sign_ok = int(np.sign(raw) == np.sign(dw))
            rows.append([cid, dw, raw, pred_abs, exact, sign_ok])
        acc = report(f'{label} (k={k:.3f})', rows,
                     f'{args.out_prefix}_{label}.csv',
                     sign_col=(label == 'C_signed'))
        results[label] = acc

    print('\n=== RESUMO ===')
    for label, acc in results.items():
        print(f'  {label}: {acc:.1f}%')
    print('>= 80% autoriza. B vs A isola o ganho da correção geométrica; '
          'C adiciona o sinal (produto necessário do ruler de todo modo).')


if __name__ == '__main__':
    main()
