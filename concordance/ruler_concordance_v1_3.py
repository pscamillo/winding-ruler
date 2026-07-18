#!/usr/bin/env python3
"""
ruler_concordance_v1_3.py — pré-braço-(b), mecanismo v3: contagem de RUNS.

Hipótese (derivada de v1.1/v1.2): campo de winding = rampa linear; grad_mag
mascarado ao papiro (zero no ar). Consequências testáveis:
  - integral ~ espessura/espacamento ~ 0.35 por folha  [OBSERVADO: k=2.89]
  - sem picos internos (platô de rampa)                 [OBSERVADO: peaks 7%]
  - RUNS contíguos supra-limiar = folhas atravessadas   [TESTA AGORA]

Modos comparados no mesmo split/test:
  integral : trapezoidal, fator k calibrado no train (baseline v1.1)
  runs     : nº de segmentos contíguos com densidade >= abs_thresh,
             fechando buracos internos <= gap_close_vx (robustez a ruído);
             pred = n_runs - 1 (endpoints do anotador sobre folhas).

--profile N imprime N perfis de raios dw=1 e dw=3 (ASCII) para verificação
visual da hipótese ANTES de confiar nos agregados.
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


def sample_ray(sub, z0_full, a_xyz, b_xyz, sample_vx):
    a = np.asarray(a_xyz, dtype=np.float64)
    b = np.asarray(b_xyz, dtype=np.float64)
    dist = float(np.linalg.norm(b - a))
    if not (dist > 1e-9) or not np.isfinite(dist):
        return None, 0.0, True
    n = max(2, int(np.ceil(dist / sample_vx)) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts = a[None, :] * (1.0 - t)[:, None] + b[None, :] * t[:, None]
    zi = (pts[:, 2] - z0_full) / LASAGNA_SCALE
    yi = pts[:, 1] / LASAGNA_SCALE
    xi = pts[:, 0] / LASAGNA_SCALE
    dz, dy, dx = sub.shape
    inside = (zi >= 0) & (zi <= dz - 1) & (yi >= 0) & (yi <= dy - 1) & \
             (xi >= 0) & (xi <= dx - 1)
    if not inside.all():
        return None, 0.0, False
    dens = map_coordinates(sub, np.stack([zi, yi, xi]), order=1,
                           mode='nearest').astype(np.float64) / DECODE
    return dens, dist / (n - 1), True


def count_runs(dens, abs_thresh, gap_close_vx, sample_vx):
    """Segmentos contíguos com dens >= abs_thresh; buracos internos com
    comprimento <= gap_close_vx são fechados (mesma folha, ruído/lacuna)."""
    mask = dens >= abs_thresh
    if not mask.any():
        return 0
    gap_n = max(0, int(round(gap_close_vx / sample_vx)))
    if gap_n > 0:
        # fecha buracos curtos: dilata a máscara de furos... mais simples:
        # varre runs de False internos e fecha os curtos
        m = mask.copy()
        n = len(m)
        i = 0
        while i < n:
            if not m[i]:
                j = i
                while j < n and not m[j]:
                    j += 1
                interior = (i > 0) and (j < n)
                if interior and (j - i) <= gap_n:
                    m[i:j] = True
                i = j
            else:
                i += 1
        mask = m
    # conta transições False->True (+ inicio True)
    runs = int(mask[0]) + int(np.sum((~mask[:-1]) & mask[1:]))
    return runs


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


def ascii_profile(dens, width=100, height=8):
    n = len(dens)
    if n > width:
        idx = np.linspace(0, n - 1, width).astype(int)
        d = dens[idx]
    else:
        d = dens
    mx = d.max() if d.max() > 0 else 1.0
    lines = []
    for h in range(height, 0, -1):
        thr = mx * h / height
        lines.append(''.join('#' if v >= thr else ' ' for v in d))
    lines.append('-' * len(d))
    return '\n'.join(lines) + f'\n(max dens = {mx:.4f})'


def report(name, rows_test, out_csv=None):
    n = len(rows_test)
    exact = sum(r[-1] for r in rows_test)
    mae = np.mean([abs(r[3] - r[1]) for r in rows_test]) if n else float('nan')
    print(f'\n== {name} — TEST ==')
    print(f'CONCORDÂNCIA EXATA: {exact}/{n} = {100.0*exact/max(n,1):.1f}%   '
          f'MAE(pred): {mae:.3f}')
    by = defaultdict(lambda: [0, 0])
    for _, truth, _, pred, ex in rows_test:
        by[truth][0] += 1; by[truth][1] += ex
    print(f'{"truth_dw":>8} {"n":>5} {"exact%":>7}')
    for truth in sorted(by):
        cnt, ex = by[truth]
        print(f'{truth:>8.0f} {cnt:>5} {100.0*ex/cnt:>6.1f}%')
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['collection', 'truth_dw', 'raw', 'pred_dw', 'exact'])
            w.writerows(rows_test)
        print(f'CSV: {out_csv}')
    return 100.0 * exact / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--sample-vx', type=float, default=2.0)
    ap.add_argument('--abs-thresh', type=float, default=0.005,
                    help='limiar de papiro (unid. decodificadas)')
    ap.add_argument('--gap-close-vx', type=float, default=4.0,
                    help='fecha buracos internos ate este comprimento (vox)')
    ap.add_argument('--run-offset', type=int, default=-1)
    ap.add_argument('--split-seed', type=int, default=1)
    ap.add_argument('--profile', type=int, default=0,
                    help='imprime N perfis de raios (dw=1 e dw=3) e sai')
    ap.add_argument('--out-prefix', default='/tmp/concordance_v13')
    args = ap.parse_args()

    pairs, colls = collect_pairs(args.dataset, args.z_begin, args.z_end)
    sub, z0_full = load_gradmag_subvolume(args.dataset, args.z_begin, args.z_end)

    if args.profile > 0:
        shown = defaultdict(int)
        for cid, a, b, truth in pairs:
            if truth not in (1, 3) or shown[truth] >= args.profile:
                continue
            dens, seg, ok = sample_ray(sub, z0_full, a, b, args.sample_vx)
            if not ok or dens is None:
                continue
            runs = count_runs(dens, args.abs_thresh, args.gap_close_vx,
                              args.sample_vx)
            print(f'\n--- perfil: coll {cid}  truth_dw={truth:.0f}  '
                  f'len={len(dens)} amostras ({seg:.1f} vx/amostra)  '
                  f'runs={runs} ---')
            print(ascii_profile(dens))
            shown[truth] += 1
        sys.exit(0)

    per_pair = []
    for cid, a, b, truth in pairs:
        dens, seg, ok = sample_ray(sub, z0_full, a, b, args.sample_vx)
        if not ok or dens is None:
            continue
        integ = float(np.sum(0.5 * (dens[:-1] + dens[1:]) * seg))
        runs = count_runs(dens, args.abs_thresh, args.gap_close_vx,
                          args.sample_vx)
        per_pair.append((cid, truth, integ, runs))
    print(f'pares com cobertura: {len(per_pair)}')

    rng = random.Random(args.split_seed)
    shuffled = colls[:]
    rng.shuffle(shuffled)
    train_set = set(shuffled[:len(shuffled) // 2])
    train = [(t, i) for cid, t, i, _ in per_pair if cid in train_set]
    test_rows = [(cid, t, i, r) for cid, t, i, r in per_pair
                 if cid not in train_set]
    print(f'train {len(train)} | test {len(test_rows)}')

    ti = np.array([i for _, i in train]); tt = np.array([t for t, _ in train])
    k = float(np.sum(ti * tt) / np.sum(ti * ti))
    print(f'fator k (train): {k:.3f}')
    rows_int = [(cid, t, i, int(round(k * i)), int(int(round(k * i)) == t))
                for cid, t, i, _ in test_rows]
    acc_int = report('MODO INTEGRAL (k calibrado)', rows_int,
                     args.out_prefix + '_integral.csv')

    rows_rn = [(cid, t, r, max(0, r + args.run_offset),
                int(max(0, r + args.run_offset) == t))
               for cid, t, _, r in test_rows]
    acc_rn = report(f'MODO RUNS (offset {args.run_offset:+d}, '
                    f'abs {args.abs_thresh}, gap_close {args.gap_close_vx})',
                    rows_rn, args.out_prefix + '_runs.csv')

    print(f'\n=== RESUMO: integral {acc_int:.1f}%  |  runs {acc_rn:.1f}% ===')
    print('>= 80% no test autoriza o braço (b). Hiperparâmetros ajustados '
          'olhando o test => confirmar com --split-seed diferente antes do doc.')


if __name__ == '__main__':
    main()
