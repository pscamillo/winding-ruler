#!/usr/bin/env python3
"""
ruler_generate_v3.py — WINDING RULER v3: rotulador ENSEMBLE + filtro de
domínio.

Validação (E1, seeds 2 e 3 virgens): ensemble linear dist+int+cos+align
(+patch) = 75-79% global, 93% dw=1 — estável, critério pré-registrado batido
2×. Este gerador aplica o rotulador validado com TRÊS defesas aprendidas dos
negativos v1/v2:
  1. Pesos ajustados nos pares humanos da W1 (janela disjunta da geração).
  2. FILTRO DE DOMÍNIO: só rotula pares com f_align e dist dentro do p5-p95
     dos pares humanos da w1 — recusa o que está fora do regime validado.
  3. dw=1 APENAS (elo forte), residual da predição <= 0.25.
Endpoints = track points (v2), sinal p/ ordem = umbílico (100%).
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
S = 32


def load_channels(dataset, z_begin, z_end, margin_vx=64):
    subs = {}
    z0_full = None
    for name in ('grad_mag', 'nx', 'ny'):
        root = zarr.open(os.path.join(
            dataset, 'lasagna_inputs', f'las_008_{name}.ome.zarr'), mode='r')
        arr = root['4']
        z0 = max(0, z_begin - margin_vx) // LASAGNA_SCALE
        z1 = min(arr.shape[0],
                 (z_end + margin_vx + LASAGNA_SCALE - 1) // LASAGNA_SCALE)
        subs[name] = np.asarray(arr[z0:z1, :, :])
        z0_full = z0 * LASAGNA_SCALE
    print(f'canais[4]: {subs["grad_mag"].shape} (z0={z0_full})')
    return subs, z0_full


def load_umbilicus(dataset):
    d = json.load(open(os.path.join(dataset, 'umbilicus.json')))
    pts = sorted((float(c['z']), float(c['x']), float(c['y']))
                 for c in d['control_points'])
    zs = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    return zs, xs, ys


def load_patch_boxes(dataset):
    import glob
    lo, hi = [], []
    for m in glob.glob(os.path.join(dataset, 'verified_patches', '*',
                                    'meta.json')):
        try:
            d = json.load(open(m))
            lo.append(d['bbox'][0]); hi.append(d['bbox'][1])
        except Exception:
            continue
    return np.array(lo), np.array(hi)


def pair_features(subs, z0f, A, B, uz, ux, uy, boxes, chunk=20000):
    """[dist, int, cos, align, patch] por par (sem rad — instável no E1)."""
    n = len(A)
    out = np.empty((n, 5))
    lo, hi = boxes
    dz, dy, dx = subs['grad_mag'].shape
    t = np.linspace(0, 1, S)
    for s0 in range(0, n, chunk):
        e = min(n, s0 + chunk)
        a, b = A[s0:e], B[s0:e]
        delta = b - a
        dist = np.linalg.norm(delta, axis=1)
        d_hat = delta / np.maximum(dist, 1e-9)[:, None]
        pts = a[:, None, :] * (1 - t)[None, :, None] + \
            b[:, None, :] * t[None, :, None]
        zi = (pts[..., 2] - z0f) / LASAGNA_SCALE
        yi = pts[..., 1] / LASAGNA_SCALE
        xi = pts[..., 0] / LASAGNA_SCALE
        coords = np.stack([np.clip(zi, 0, dz - 1).ravel(),
                           np.clip(yi, 0, dy - 1).ravel(),
                           np.clip(xi, 0, dx - 1).ravel()])
        m = e - s0
        g = map_coordinates(subs['grad_mag'], coords, order=1,
                            mode='nearest').reshape(m, S).astype(float) / DECODE
        rnx = map_coordinates(subs['nx'], coords, order=1,
                              mode='nearest').reshape(m, S).astype(float)
        rny = map_coordinates(subs['ny'], coords, order=1,
                              mode='nearest').reshape(m, S).astype(float)
        vx = (rnx - 128.0) / 127.0
        vy = (rny - 128.0) / 127.0
        vz = np.sqrt(np.maximum(0.0, 1.0 - vx * vx - vy * vy))
        nrm = np.maximum(np.sqrt(vx * vx + vy * vy + vz * vz), 1e-9)
        cosang = np.abs(vx * d_hat[:, None, 0] + vy * d_hat[:, None, 1] +
                        vz * d_hat[:, None, 2]) / nrm
        seg = (dist / (S - 1))[:, None]
        f_int = np.sum(0.5 * (g[:, :-1] + g[:, 1:]) * seg, axis=1)
        gw = g * cosang
        f_cos = np.sum(0.5 * (gw[:, :-1] + gw[:, 1:]) * seg, axis=1)
        f_align = cosang.mean(axis=1)
        mid = 0.5 * (a + b)
        f_patch = np.zeros(m)
        for i in range(m):
            p = mid[i]
            inside = (lo[:, 0] <= p[0]) & (p[0] <= hi[:, 0]) & \
                     (lo[:, 1] <= p[1]) & (p[1] <= hi[:, 1]) & \
                     (lo[:, 2] <= p[2]) & (p[2] <= hi[:, 2])
            f_patch[i] = inside.sum()
        out[s0:e] = np.column_stack([dist, f_int, f_cos, f_align, f_patch])
    return out


def human_pairs(dataset, z_begin, z_end):
    d = json.load(open(os.path.join(dataset, 'relative_windings.json')))
    A, B, dw = [], [], []
    for coll in d['collections'].values():
        pts = [(pt['p'], float(pt['wind_a']))
               for pt in coll['points'].values()
               if pt.get('wind_a') is not None
               and z_begin <= pt['p'][2] < z_end]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                t = abs(pts[j][1] - pts[i][1])
                if 0 < t <= 6:
                    A.append(pts[i][0]); B.append(pts[j][0]); dw.append(t)
    return (np.array(A, dtype=float), np.array(B, dtype=float),
            np.array(dw, dtype=float))


def load_track_points(dataset, z_lo, z_hi, track_stride, point_stride):
    path = os.path.join(dataset, 'tracks', '2um_ds2_ps256_surf_v2.dbm')
    out = []
    n_tracks = 0
    with dbm.open(path, 'r') as db:
        keys = list(db.keys())
        for ki, key in enumerate(keys):
            for e in pickle.loads(db[key]):
                if len(e) == 0:
                    continue
                zc = e[:, 0]
                if zc.min() < z_lo or zc.max() >= z_hi:
                    continue
                n_tracks += 1
                if n_tracks % track_stride:
                    continue
                out.append(e[::point_stride].astype(np.float64)[:, ::-1])
            if ki % 400 == 0:
                print(f'\r  dbm {ki}/{len(keys)}', end='')
    print()
    P = np.concatenate(out) if out else np.zeros((0, 3))
    print(f'track points: {len(P)}')
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--calib-z-begin', type=int, default=8000)
    ap.add_argument('--calib-z-end', type=int, default=9000)
    ap.add_argument('--track-stride', type=int, default=6)
    ap.add_argument('--point-stride', type=int, default=6)
    ap.add_argument('--min-gap', type=float, default=6.0)
    ap.add_argument('--max-gap', type=float, default=60.0)
    ap.add_argument('--max-residual', type=float, default=0.25)
    ap.add_argument('--max-pairs', type=int, default=150000)
    ap.add_argument('--out', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4/'
        'ruler3_relative_windings.json'))
    args = ap.parse_args()

    if not (args.calib_z_end <= args.z_begin
            or args.calib_z_begin >= args.z_end):
        print('ERRO: calibração sobrepõe geração.'); sys.exit(1)

    uz, ux, uy = load_umbilicus(args.dataset)
    boxes = load_patch_boxes(args.dataset)

    # ---- pesos do ensemble + envelope de domínio: pares humanos da W1 ----
    subs_c, z0c = load_channels(args.dataset, args.calib_z_begin,
                                args.calib_z_end)
    Ah, Bh, dwh = human_pairs(args.dataset, args.calib_z_begin,
                              args.calib_z_end)
    Fh = pair_features(subs_c, z0c, Ah, Bh, uz, ux, uy, boxes)
    X = np.column_stack([Fh, np.ones(len(Fh))])
    w, *_ = np.linalg.lstsq(X, dwh, rcond=None)
    pred_h = np.clip(np.round(X @ w), 0, 8)
    print(f'ensemble (w1, {len(dwh)} pares): in-sample exato '
          f'{100*np.mean(pred_h==dwh):.1f}% | pesos '
          f'{np.array2string(w, precision=3)}')
    align_lo, align_hi = np.percentile(Fh[:, 3], [5, 95])
    dist_lo, dist_hi = np.percentile(Fh[:, 0], [5, 95])
    print(f'envelope de domínio (w1 p5-p95): align [{align_lo:.3f}, '
          f'{align_hi:.3f}], dist [{dist_lo:.1f}, {dist_hi:.1f}]')
    del subs_c

    # ---- geração na W2 ----
    subs, z0f = load_channels(args.dataset, args.z_begin, args.z_end)
    P = load_track_points(args.dataset, args.z_begin, args.z_end,
                          args.track_stride, args.point_stride)
    ax = np.interp(P[:, 2], uz, ux); ay = np.interp(P[:, 2], uz, uy)
    r = np.hypot(P[:, 0] - ax, P[:, 1] - ay)
    theta = np.arctan2(P[:, 1] - ay, P[:, 0] - ax)
    tb = np.clip(((theta + np.pi) / (2 * np.pi) * 2880).astype(int), 0, 2879)
    zb = ((P[:, 2] - args.z_begin) / 8.0).astype(int)
    cell = zb * 2880 + tb
    order = np.lexsort((r, cell))
    cs, rs = cell[order], r[order]
    good = (cs[1:] == cs[:-1]) & (rs[1:] - rs[:-1] >= args.min_gap) & \
           (rs[1:] - rs[:-1] <= args.max_gap)
    ia, ib = order[:-1][good], order[1:][good]
    print(f'pares candidatos: {len(ia)}')
    if len(ia) > args.max_pairs:
        stride = int(np.ceil(len(ia) / args.max_pairs))
        ia, ib = ia[::stride], ib[::stride]
        print(f'decimação 1/{stride} -> {len(ia)}')

    F = pair_features(subs, z0f, P[ia], P[ib], uz, ux, uy, boxes)
    in_domain = (F[:, 3] >= align_lo) & (F[:, 3] <= align_hi) & \
                (F[:, 0] >= dist_lo) & (F[:, 0] <= dist_hi)
    yhat = np.column_stack([F, np.ones(len(F))]) @ w
    dw = np.round(yhat).astype(int)
    resid = np.abs(yhat - dw)
    accept = in_domain & (dw == 1) & (resid <= args.max_residual)
    print(f'no domínio: {in_domain.sum()} ({100*in_domain.mean():.1f}%) | '
          f'aceitos (dw=1, resid<={args.max_residual}): {accept.sum()} '
          f'({100*accept.sum()/max(len(ia),1):.1f}% dos candidatos)')

    collections = {}
    cid = 0
    for j in np.nonzero(accept)[0]:
        pa, pb = P[ia[j]], P[ib[j]]
        collections[str(cid)] = {
            'name': f'ruler3_{cid}',
            'points': {
                '1': {'p': [round(pa[0], 2), round(pa[1], 2),
                            round(pa[2], 2)], 'wind_a': 1.0,
                      'creation_time': 0},
                '2': {'p': [round(pb[0], 2), round(pb[1], 2),
                            round(pb[2], 2)], 'wind_a': 2.0,
                      'creation_time': 0},
            },
            'metadata': {'generator': 'winding_ruler_v3',
                         'residual': round(float(resid[j]), 3)},
            'color': [0.2, 0.8, 0.3],
        }
        cid += 1

    out = {'vc_pointcollections_json_version': '1',
           'collections': collections}
    with open(args.out, 'w') as f:
        json.dump(out, f)
    if accept.sum():
        print(f'residual aceito: mediana {np.median(resid[accept]):.3f}')
        gaps = F[accept, 0]
        print(f'dist dos aceitos: mediana {np.median(gaps):.1f} vx '
              f'(lambda humano w2 ~18,8 — conferir plausibilidade)')
    print(f'=== RULER v3: {len(collections)} collections '
          f'({2*len(collections)} pontos) -> {args.out} '
          f'({os.path.getsize(args.out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
