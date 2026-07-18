#!/usr/bin/env python3
"""
ruler_generate_v2.py — WINDING RULER v2: relações entre pontos que JÁ estão
sobre folhas.

Post-mortem do v1 (registrado no v3.2): emitir pontos em cruzamentos de fase
arbitrária posiciona âncoras FORA das folhas → 1,3% de satisfação in-sample,
eval 45,4% (pior que sem constraints). O concordance validou a MEDIÇÃO de
Δw entre pontos dados; nunca validou POSICIONAMENTO de pontos novos.

v2 elimina a dimensão que falhou:
  - endpoints = TRACK POINTS (2um surf dbm) — amostras que já vivem sobre
    folhas, do próprio pipeline do time (loader reimplementado fiel a
    tracks.py:16-41, dbm+pickle, arrays (N,3) zyx int32→float32);
  - pares candidatos: pontos radialmente alinhados (mesma célula fina de
    (z, θ) em torno do umbílico), consecutivos em raio, gap em [6, 60] vox;
  - o ruler faz SÓ o que foi validado a 91-92%: rotular Δw do par via
    k×integral (k calibrado na w1, janela disjunta);
  - gate de confiança: aceita apenas |kI − round(kI)| <= max_residual
    e 1 <= Δw <= max_dw (a cadeia não salta; dw alto degrada);
  - saída: collections de 2 pontos, wind_a = {1, 1+Δw}, formato VC3D v1.

Determinístico: sem RNG; decimação por stride fixo.
"""
import argparse
import dbm
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import zarr
from scipy.ndimage import map_coordinates

ENCODE_SCALE = 1000.0
GRAD_MAG_FACTOR = 0.25
DECODE = ENCODE_SCALE / GRAD_MAG_FACTOR
LASAGNA_SCALE = 4
SAMPLES_PER_PAIR = 32  # amostragem fixa por par p/ integral em lote


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
    d = json.load(open(os.path.join(dataset, 'umbilicus.json')))
    pts = sorted((float(cp['z']), float(cp['x']), float(cp['y']))
                 for cp in d['control_points'])
    zs = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    print(f'umbilicus: {len(pts)} control points')
    return zs, xs, ys


def load_track_points(dataset, z_lo, z_hi, track_stride, point_stride):
    """Fiel a tracks.py:load_tracks_from_dbm, com decimação determinística.
    Retorna array (M, 3) xyz float64 de pontos sobre folhas."""
    path = os.path.join(dataset, 'tracks', '2um_ds2_ps256_surf_v2.dbm')
    out = []
    n_tracks = 0
    with dbm.open(path, 'r') as db:
        keys = list(db.keys())
        for ki, key in enumerate(keys):
            entries = pickle.loads(db[key])
            for e in entries:
                if len(e) == 0:
                    continue
                zc = e[:, 0]
                if zc.min() < z_lo or zc.max() >= z_hi:
                    continue
                n_tracks += 1
                if n_tracks % track_stride:
                    continue
                pts = e[::point_stride].astype(np.float64)  # zyx
                out.append(pts[:, ::-1])                     # -> xyz
            if ki % 400 == 0:
                print(f'\r  dbm keys {ki}/{len(keys)}', end='')
    print()
    pts = np.concatenate(out) if out else np.zeros((0, 3))
    print(f'tracks na z-roi: {n_tracks} | pontos após stride '
          f'({track_stride}/{point_stride}): {len(pts)}')
    return pts


def batched_integrals(sub, z0_full, A, B, chunk=20000):
    """Integral trapezoidal (SAMPLES_PER_PAIR nós) para pares (A[i], B[i]) xyz.
    Retorna (integrais, valid) — valid=False se alguma amostra saiu do volume."""
    n = len(A)
    dist = np.linalg.norm(B - A, axis=1)
    t = np.linspace(0.0, 1.0, SAMPLES_PER_PAIR)
    ints = np.empty(n)
    valid = np.ones(n, dtype=bool)
    dz, dy, dx = sub.shape
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a = A[s:e]; b = B[s:e]
        pts = a[:, None, :] * (1.0 - t)[None, :, None] + \
              b[:, None, :] * t[None, :, None]          # [m, S, 3] xyz
        zi = (pts[..., 2] - z0_full) / LASAGNA_SCALE
        yi = pts[..., 1] / LASAGNA_SCALE
        xi = pts[..., 0] / LASAGNA_SCALE
        inside = (zi >= 0) & (zi <= dz - 1) & (yi >= 0) & (yi <= dy - 1) & \
                 (xi >= 0) & (xi <= dx - 1)
        ok = inside.all(axis=1)
        coords = np.stack([zi.ravel(), yi.ravel(), xi.ravel()])
        dens = map_coordinates(sub, coords, order=1, mode='nearest')
        dens = dens.reshape(e - s, SAMPLES_PER_PAIR).astype(np.float64) / DECODE
        seg = (dist[s:e] / (SAMPLES_PER_PAIR - 1))[:, None]
        ints[s:e] = np.sum(0.5 * (dens[:, :-1] + dens[:, 1:]) * seg, axis=1)
        valid[s:e] = ok
    return ints, valid


def calibrate_k(dataset, z_begin, z_end):
    sub, z0 = load_gradmag(dataset, z_begin, z_end)
    d = json.load(open(os.path.join(dataset, 'relative_windings.json')))
    A, B, dws = [], [], []
    for coll in d['collections'].values():
        pts = [(pt['p'], float(pt['wind_a']))
               for pt in coll['points'].values()
               if pt.get('wind_a') is not None
               and z_begin <= pt['p'][2] < z_end]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dw = abs(pts[j][1] - pts[i][1])
                if 0 < dw <= 6:
                    A.append(pts[i][0]); B.append(pts[j][0]); dws.append(dw)
    A = np.array(A, dtype=np.float64); B = np.array(B, dtype=np.float64)
    dws = np.array(dws, dtype=np.float64)
    ints, valid = batched_integrals(sub, z0, A, B)
    ints, dws = ints[valid], dws[valid]
    k = float(np.sum(ints * dws) / np.sum(ints * ints))
    print(f'calibração: {len(ints)} pares humanos em z [{z_begin},{z_end}) '
          f'→ k = {k:.3f}')
    del sub
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--calib-z-begin', type=int, default=8000)
    ap.add_argument('--calib-z-end', type=int, default=9000)
    ap.add_argument('--track-stride', type=int, default=6,
                    help='usa 1 a cada N tracks')
    ap.add_argument('--point-stride', type=int, default=6,
                    help='usa 1 a cada N pontos por track')
    ap.add_argument('--z-bin', type=float, default=8.0)
    ap.add_argument('--theta-bins', type=int, default=2880,
                    help='células angulares (2880 = 0,125 grau)')
    ap.add_argument('--min-gap', type=float, default=6.0)
    ap.add_argument('--max-gap', type=float, default=60.0)
    ap.add_argument('--max-dw', type=int, default=3)
    ap.add_argument('--max-residual', type=float, default=0.25)
    ap.add_argument('--max-pairs', type=int, default=150000)
    ap.add_argument('--out', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4/'
        'ruler2_relative_windings.json'))
    args = ap.parse_args()

    if not (args.calib_z_end <= args.z_begin or args.calib_z_begin >= args.z_end):
        print('ERRO: calibração sobrepõe geração.'); sys.exit(1)
    k = calibrate_k(args.dataset, args.calib_z_begin, args.calib_z_end)

    sub, z0_full = load_gradmag(args.dataset, args.z_begin, args.z_end)
    uz, ux, uy = load_umbilicus(args.dataset)
    P = load_track_points(args.dataset, args.z_begin, args.z_end,
                          args.track_stride, args.point_stride)
    if len(P) < 1000:
        print('Poucos pontos de track — afrouxa os strides.'); sys.exit(1)

    # coordenadas polares em torno do umbílico (eixo interpolado por z)
    ax = np.interp(P[:, 2], uz, ux)
    ay = np.interp(P[:, 2], uz, uy)
    r = np.hypot(P[:, 0] - ax, P[:, 1] - ay)
    theta = np.arctan2(P[:, 1] - ay, P[:, 0] - ax)
    tbin = ((theta + np.pi) / (2 * np.pi) * args.theta_bins).astype(np.int64)
    tbin = np.clip(tbin, 0, args.theta_bins - 1)
    zbin = ((P[:, 2] - args.z_begin) / args.z_bin).astype(np.int64)

    # agrupa por célula (zbin, tbin); dentro da célula, ordena por raio e
    # emparelha consecutivos com gap plausível
    cell = zbin * args.theta_bins + tbin
    order = np.lexsort((r, cell))
    cell_s, r_s = cell[order], r[order]
    idxA, idxB = [], []
    same_cell = cell_s[1:] == cell_s[:-1]
    gap = r_s[1:] - r_s[:-1]
    good = same_cell & (gap >= args.min_gap) & (gap <= args.max_gap)
    ia = order[:-1][good]
    ib = order[1:][good]
    print(f'pares candidatos (consecutivos em raio, gap [{args.min_gap},'
          f'{args.max_gap}]): {len(ia)}')

    if len(ia) > args.max_pairs:
        stride = int(np.ceil(len(ia) / args.max_pairs))
        ia, ib = ia[::stride], ib[::stride]
        print(f'decimação determinística 1/{stride} → {len(ia)} pares')

    ints, valid = batched_integrals(sub, z0_full, P[ia], P[ib])
    kI = k * ints
    dw = np.round(kI).astype(int)
    resid = np.abs(kI - dw)
    accept = valid & (dw >= 1) & (dw <= args.max_dw) & \
             (resid <= args.max_residual)
    print(f'aceitos pelo gate (1<=dw<={args.max_dw}, resid<='
          f'{args.max_residual}): {accept.sum()} '
          f'({100.0*accept.sum()/max(len(ia),1):.1f}%)')

    by_dw = defaultdict(int)
    collections = {}
    cid = 0
    for j in np.nonzero(accept)[0]:
        pa, pb = P[ia[j]], P[ib[j]]
        d = int(dw[j])
        by_dw[d] += 1
        collections[str(cid)] = {
            'name': f'ruler2_{cid}',
            'points': {
                '1': {'p': [round(pa[0], 2), round(pa[1], 2), round(pa[2], 2)],
                      'wind_a': 1.0, 'creation_time': 0},
                '2': {'p': [round(pb[0], 2), round(pb[1], 2), round(pb[2], 2)],
                      'wind_a': 1.0 + d, 'creation_time': 0},
            },
            'metadata': {'generator': 'winding_ruler_v2', 'k': round(k, 4),
                         'residual': round(float(resid[j]), 3)},
            'color': [0.9, 0.5, 0.1],
        }
        cid += 1

    out = {'vc_pointcollections_json_version': '1', 'collections': collections}
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'\n=== RULER v2: {len(collections)} collections '
          f'({2*len(collections)} pontos) ===')
    print('distribuição de dw:', dict(sorted(by_dw.items())))
    if accept.sum():
        print(f'residual aceito: mediana {np.median(resid[accept]):.3f}, '
              f'p90 {np.percentile(resid[accept], 90):.3f}')
    print(f'saída: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
