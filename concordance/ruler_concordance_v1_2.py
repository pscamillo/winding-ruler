#!/usr/bin/env python3
"""
ruler_concordance_v1_2.py — pré-braço-(b), mecanismo v2: contagem de picos.

Compara DOIS mecanismos sobre os mesmos pares, mesmo split, mesma métrica:
  integral : integral trapezoidal calibrada (fator k no train)  [v1.1]
  peaks    : nº de máximos locais da densidade ao longo do raio [novo]
             — robusto a descalibração de AMPLITUDE (só usa posições).

Detecção de picos (sem scipy.signal, controle total):
  amostra densa (--sample-vx, default 2 vox full-res) da densidade ao longo
  do segmento; suavização leve (média móvel --smooth-n); máximo local com
  proeminência relativa: pico se d[i] > vizinhos E d[i] >= rel_thresh * max(d)
  E d[i] >= abs_thresh (descarta ruído de fundo). Prediz dw = nº de picos - 1
  se ambos os endpoints estão SOBRE folhas (pontos do anotador estão), senão
  nº de picos; expomos ambas contagens e usamos --peak-offset para escolher
  (default: dw = n_picos - 1, endpoints em folhas => picos = folhas
  atravessadas incluindo as duas de borda).

Critério pré-registrado inalterado: exata >= 80% no test => autoriza.
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
    """Densidade decodificada amostrada densamente ao longo de a->b.
    Retorna (dens[n], seg_len, ok)."""
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


def integral_from_ray(dens, seg):
    return float(np.sum(0.5 * (dens[:-1] + dens[1:]) * seg))


def count_peaks(dens, rel_thresh, abs_thresh, smooth_n):
    """Máximos locais com limiar relativo ao máximo do raio + absoluto.
    Platôs contam uma vez (comparação estrita à esquerda, >= à direita)."""
    d = dens
    if smooth_n > 1:
        k = np.ones(smooth_n) / smooth_n
        d = np.convolve(d, k, mode='same')
    if d.max() <= 0:
        return 0
    thr = max(rel_thresh * d.max(), abs_thresh)
    n = len(d)
    peaks = 0
    i = 1
    while i < n - 1:
        if d[i] >= thr and d[i] > d[i - 1] and d[i] >= d[i + 1]:
            peaks += 1
            # pula o platô/descida deste pico
            j = i + 1
            while j < n - 1 and d[j] <= d[i] and d[j] >= thr:
                j += 1
            i = max(j, i + 1)
        else:
            i += 1
    # bordas: endpoints do anotador estão SOBRE folhas; se a densidade na
    # borda já está acima do limiar e caindo, é meia-folha na ponta — conta.
    if d[0] >= thr and d[0] > d[1]:
        peaks += 1
    if d[-1] >= thr and d[-1] > d[-2]:
        peaks += 1
    return peaks


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
    ap.add_argument('--rel-thresh', type=float, default=0.3,
                    help='pico >= rel_thresh * max(densidade do raio)')
    ap.add_argument('--abs-thresh', type=float, default=0.002,
                    help='piso absoluto de densidade (unid. decodificadas)')
    ap.add_argument('--smooth-n', type=int, default=3)
    ap.add_argument('--peak-offset', type=int, default=-1,
                    help='pred = n_picos + offset (default -1: endpoints em folhas)')
    ap.add_argument('--split-seed', type=int, default=1)
    ap.add_argument('--out-prefix', default='/tmp/concordance_v12')
    args = ap.parse_args()

    pairs, colls = collect_pairs(args.dataset, args.z_begin, args.z_end)
    if len(colls) < 4:
        print('Poucas collections.'); sys.exit(1)
    sub, z0_full = load_gradmag_subvolume(args.dataset, args.z_begin, args.z_end)

    # uma passada: raio denso por par -> integral e picos
    per_pair = []  # (cid, truth, integral, n_peaks)
    skipped = 0
    for cid, a, b, truth in pairs:
        dens, seg, ok = sample_ray(sub, z0_full, a, b, args.sample_vx)
        if not ok or dens is None:
            skipped += 1
            continue
        integ = integral_from_ray(dens, seg)
        npk = count_peaks(dens, args.rel_thresh, args.abs_thresh, args.smooth_n)
        per_pair.append((cid, truth, integ, npk))
    print(f'pares com cobertura: {len(per_pair)} | descartados: {skipped}')

    # split idêntico ao v1.1
    rng = random.Random(args.split_seed)
    shuffled = colls[:]
    rng.shuffle(shuffled)
    train_set = set(shuffled[:len(shuffled) // 2])
    train = [(t, i) for cid, t, i, _ in per_pair if cid in train_set]
    test_rows = [(cid, t, i, p) for cid, t, i, p in per_pair
                 if cid not in train_set]
    print(f'train {len(train)} | test {len(test_rows)}')

    # MODO INTEGRAL (calibrado no train)
    ti = np.array([i for _, i in train]); tt = np.array([t for t, _ in train])
    k = float(np.sum(ti * tt) / np.sum(ti * ti))
    print(f'\nfator k (train): {k:.3f}')
    rows_int = [(cid, t, i, int(round(k * i)), int(int(round(k * i)) == t))
                for cid, t, i, _ in test_rows]
    acc_int = report('MODO INTEGRAL (k calibrado)', rows_int,
                     args.out_prefix + '_integral.csv')

    # MODO PEAKS (sem calibração de amplitude)
    rows_pk = [(cid, t, npk, max(0, npk + args.peak_offset),
                int(max(0, npk + args.peak_offset) == t))
               for cid, t, _, npk in test_rows]
    acc_pk = report(f'MODO PEAKS (offset {args.peak_offset:+d}, '
                    f'rel {args.rel_thresh}, abs {args.abs_thresh}, '
                    f'smooth {args.smooth_n})', rows_pk,
                    args.out_prefix + '_peaks.csv')

    print(f'\n=== RESUMO: integral {acc_int:.1f}%  |  peaks {acc_pk:.1f}% ===')
    print('Critério pré-registrado: >= 80% no test autoriza o braço (b).')
    print('Se peaks < esperado, varrer: --rel-thresh 0.2/0.4, --smooth-n 1/5, '
          '--peak-offset 0. (Varredura de hiperparâmetro SÓ no train seria o '
          'rigor máximo; para exploração rápida, anotar que o test foi tocado '
          'e confirmar depois com split-seed diferente.)')


if __name__ == '__main__':
    main()
