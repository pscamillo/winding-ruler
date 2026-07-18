#!/usr/bin/env python3
"""
ruler_concordance_v1_5.py — pré-braço-(b) final: sinal via umbílico +
votação multi-raio.

Estado validado (v1.4, 3 splits): passo unitário 87-95% (média ~91%),
k estável 2,77-2,89, sinal-por-normais morto (53%).

v1.5 adiciona:
  MULTI-RAIO : M raios paralelos ao segmento a→b, deslocados ±offset na
               direção perpendicular in-plane (mesma slice z do anotador);
               predição = round(k * MEDIANA das integrais). Ataca a
               variância local de amplitude que derruba dw>=2.
  SINAL      : via umbilicus.json — r = distância ao eixo na slice z;
               sinal_pred = sign(r_b - r_a) * orientacao. A orientação
               (wind cresce para fora ou para dentro) é AJUSTADA NO TRAIN
               (voto majoritário) e aplicada no test — 1 bit, risco de
               vazamento nulo na prática, reportado explicitamente.

Métricas no TEST (split por collection, como sempre):
  - exato |dw| (mono-raio baseline vs multi-raio)
  - acerto de sinal
  - EXATO COMPLETO (|dw| E sinal) — o número que o ruler entrega de fato.
Critério pré-registrado: >=80% exato completo em dw=1 (o elo da cadeia)
autoriza o MVP do gerador. dw>=2 é secundário (a cadeia não salta).
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


def load_gradmag(dataset, z_begin, z_end, margin_vx=64):
    path = os.path.join(dataset, 'lasagna_inputs', 'las_008_grad_mag.ome.zarr')
    root = zarr.open(path, mode='r')
    arr = root['4']
    z0_full = max(0, z_begin - margin_vx)
    z1_full = z_end + margin_vx
    z0 = z0_full // LASAGNA_SCALE
    z1 = min(arr.shape[0], (z1_full + LASAGNA_SCALE - 1) // LASAGNA_SCALE)
    sub = np.asarray(arr[z0:z1, :, :])
    print(f'grad_mag[4]: shape={sub.shape}')
    return sub, z0 * LASAGNA_SCALE


def load_umbilicus(dataset):
    """umbilicus.json -> função z_full -> (x, y) do eixo (interp linear).
    Formato esperado: lista/dict de pontos com z e (x,y) ou (y,x); detecta
    e imprime o que assumiu — CONFERIR NO STDOUT."""
    d = json.load(open(os.path.join(dataset, 'umbilicus.json')))
    pts = []
    if isinstance(d, dict):
        for key in ('control_points', 'points', 'umbilicus', 'data'):
            if key in d and isinstance(d[key], (list, dict)):
                d = d[key]
                break
    if isinstance(d, dict):
        d = list(d.values())
    for item in d:
        if isinstance(item, dict):
            z = item.get('z'); x = item.get('x'); y = item.get('y')
            p = item.get('p')
            if p is not None and len(p) == 3:
                x, y, z = p[0], p[1], p[2]
            if z is not None and x is not None and y is not None:
                pts.append((float(z), float(x), float(y)))
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            # ambíguo: assumimos [x, y, z] (convenção 'p' do resto do dataset)
            pts.append((float(item[2]), float(item[0]), float(item[1])))
    if len(pts) < 2:
        print('AVISO: umbilicus.json não reconhecido — cole `head -c 400` '
              'do arquivo para eu ajustar o parser.')
        sys.exit(2)
    pts.sort()
    zs = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    print(f'umbilicus: {len(pts)} pontos, z [{zs.min():.0f}, {zs.max():.0f}], '
          f'x médio {xs.mean():.0f}, y médio {ys.mean():.0f} '
          f'(assumido [x,y,z]; conferir plausibilidade: centro ~ metade de 8176)')
    def axis_xy(z):
        return (float(np.interp(z, zs, xs)), float(np.interp(z, zs, ys)))
    return axis_xy


def ray_integral(sub, z0_full, a, b, sample_vx=2.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dist = float(np.linalg.norm(b - a))
    if not (dist > 1e-9) or not np.isfinite(dist):
        return 0.0, True
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
        return float('inf'), False
    dens = map_coordinates(sub, np.stack([zi, yi, xi]), order=1,
                           mode='nearest').astype(np.float64) / DECODE
    seg = dist / (n - 1)
    return float(np.sum(0.5 * (dens[:-1] + dens[1:]) * seg)), True


def multiray_integral(sub, z0_full, a, b, m_rays, max_offset_vx):
    """Mediana de M integrais em raios deslocados perpendicular ao segmento,
    no plano da slice (z constante — anotações são por slice)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    perp = np.array([-d[1], d[0], 0.0])
    nrm = np.linalg.norm(perp)
    if nrm < 1e-9:
        perp = np.array([1.0, 0.0, 0.0])
        nrm = 1.0
    perp /= nrm
    offsets = np.linspace(-max_offset_vx, max_offset_vx, m_rays)
    vals = []
    for off in offsets:
        v, ok = ray_integral(sub, z0_full, a + perp * off, b + perp * off)
        if ok and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float('inf'), False
    return float(np.median(vals)), True


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
                dw = pts[j][1] - pts[i][1]  # com sinal, a->b
                if 0 < abs(dw) <= MAX_TRUTH_DELTA:
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
    ap.add_argument('--m-rays', type=int, default=7)
    ap.add_argument('--max-offset-vx', type=float, default=6.0)
    ap.add_argument('--split-seed', type=int, default=2,
                    help='default 2 (virgem p/ hiperparâmetros novos)')
    ap.add_argument('--out', default='/tmp/concordance_v15.csv')
    args = ap.parse_args()

    pairs, colls = collect_pairs(args.dataset, args.z_begin, args.z_end)
    sub, z0_full = load_gradmag(args.dataset, args.z_begin, args.z_end)
    axis_xy = load_umbilicus(args.dataset)

    per = []
    for cid, a, b, dw in pairs:
        i1, ok1 = ray_integral(sub, z0_full, a, b)
        im, okm = multiray_integral(sub, z0_full, a, b,
                                    args.m_rays, args.max_offset_vx)
        if not (ok1 and okm):
            continue
        ax, ay = axis_xy(0.5 * (a[2] + b[2]))
        ra = np.hypot(a[0] - ax, a[1] - ay)
        rb = np.hypot(b[0] - ax, b[1] - ay)
        radial_sign = 1 if rb > ra else (-1 if rb < ra else 0)
        per.append((cid, dw, i1, im, radial_sign))
    print(f'pares com cobertura: {len(per)}')

    rng = random.Random(args.split_seed)
    shuffled = colls[:]
    rng.shuffle(shuffled)
    train_set = set(shuffled[:len(shuffled) // 2])
    train = [p for p in per if p[0] in train_set]
    test = [p for p in per if p[0] not in train_set]
    print(f'train {len(train)} | test {len(test)} (seed {args.split_seed})')

    # k pela mediana multi-raio (train)
    ti = np.array([abs(p[3]) for p in train])
    tt = np.array([abs(p[1]) for p in train])
    k = float(np.sum(ti * tt) / np.sum(ti * ti))
    # orientação do sinal (train, voto majoritário): sign(dw) == radial_sign?
    votes = [1 if np.sign(p[1]) == p[4] else -1
             for p in train if p[4] != 0]
    orient = 1 if sum(votes) >= 0 else -1
    agree = sum(1 for v in votes if v == orient) / max(len(votes), 1)
    print(f'k(multi-raio, train) = {k:.3f} | orientação = {orient:+d} '
          f'(wind cresce {"para fora" if orient>0 else "para dentro"}; '
          f'concordância no train {100*agree:.1f}%)')

    def evaluate(rows, use_multi):
        stats = defaultdict(lambda: [0, 0, 0, 0])  # n, abs_ok, sign_ok, full_ok
        out = []
        for cid, dw, i1, im, rs in rows:
            raw = im if use_multi else i1
            pred_abs = int(round(k * raw))
            sign_pred = orient * rs
            abs_ok = int(pred_abs == abs(dw))
            sign_ok = int(sign_pred == np.sign(dw)) if rs != 0 else 0
            full_ok = int(abs_ok and sign_ok)
            s = stats[abs(dw)]
            s[0] += 1; s[1] += abs_ok; s[2] += sign_ok; s[3] += full_ok
            out.append([cid, dw, raw, pred_abs * sign_pred, full_ok])
        return stats, out

    for label, use_multi in (('MONO-RAIO', False),
                             (f'MULTI-RAIO (M={args.m_rays}, '
                              f'±{args.max_offset_vx}vx)', True)):
        stats, out = evaluate(test, use_multi)
        tot = [sum(s[i] for s in stats.values()) for i in range(4)]
        print(f'\n== {label} — TEST ==')
        print(f'|dw| exato: {tot[1]}/{tot[0]} = {100*tot[1]/tot[0]:.1f}%  |  '
              f'sinal: {tot[2]}/{tot[0]} = {100*tot[2]/tot[0]:.1f}%  |  '
              f'COMPLETO: {tot[3]}/{tot[0]} = {100*tot[3]/tot[0]:.1f}%')
        print(f'{"|dw|":>5} {"n":>5} {"abs%":>6} {"sinal%":>7} {"FULL%":>6}')
        for t in sorted(stats):
            n_, a_, s_, f_ = stats[t]
            print(f'{t:>5.0f} {n_:>5} {100*a_/n_:>5.1f} {100*s_/n_:>6.1f} '
                  f'{100*f_/n_:>5.1f}')
        if use_multi:
            with open(args.out, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['collection', 'truth_dw_signed', 'raw_median',
                            'pred_dw_signed', 'full_ok'])
                w.writerows(out)
            print(f'CSV: {args.out}')

    print('\nCritério: FULL >= 80% em dw=1 autoriza o MVP do gerador. '
          'Confirmar depois com --split-seed 3.')


if __name__ == '__main__':
    main()
