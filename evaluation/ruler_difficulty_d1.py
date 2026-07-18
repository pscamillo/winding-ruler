#!/usr/bin/env python3
"""
ruler_difficulty_d1.py — Teste D1: o mapa de dificuldade do ruler prediz
onde anotação humana é necessária?

Hipótese (pré-registrada, 17/jul): células (z,θ) onde o ruler é AMBÍGUO
(alta taxa de rejeição do gate / residual alto / excesso de dw>=2)
concentram as pcls humanas que o fit SEM anotações não satisfaz
(braço (c), eval train-free).

Critério pré-registrado: AUC >= 0,65 (dificuldade da célula → pcl humana
não-satisfeita pelo (c)) em AMBOS os seeds do (c) ⇒ mapa validado como
ferramenta de direcionamento de anotação (produto da submissão). AUC ~0,5
⇒ ideia morta com dados.

Insumos (já existentes):
  - pares candidatos + residuais: recomputados como no ruler v2 (3 min)
  - satisfied_fitted.json dos evals w2_armC_eval_human e w2_armC_s2_eval_human
  - relative/same windings humanos (posição das pcls)

Sem GPU. Determinístico.
"""
import argparse
import dbm
import glob
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
SAMPLES_PER_PAIR = 32


def load_gradmag(dataset, z_begin, z_end, margin_vx=64):
    path = os.path.join(dataset, 'lasagna_inputs', 'las_008_grad_mag.ome.zarr')
    root = zarr.open(path, mode='r')
    arr = root['4']
    z0f = max(0, z_begin - margin_vx)
    z1f = z_end + margin_vx
    z0 = z0f // LASAGNA_SCALE
    z1 = min(arr.shape[0], (z1f + LASAGNA_SCALE - 1) // LASAGNA_SCALE)
    sub = np.asarray(arr[z0:z1, :, :])
    print(f'grad_mag[4]: {sub.shape}')
    return sub, z0 * LASAGNA_SCALE


def load_umbilicus(dataset):
    d = json.load(open(os.path.join(dataset, 'umbilicus.json')))
    pts = sorted((float(c['z']), float(c['x']), float(c['y']))
                 for c in d['control_points'])
    zs = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    return zs, xs, ys


def load_track_points(dataset, z_lo, z_hi, track_stride=6, point_stride=6):
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


def batched_integrals(sub, z0_full, A, B, chunk=20000):
    n = len(A)
    dist = np.linalg.norm(B - A, axis=1)
    t = np.linspace(0.0, 1.0, SAMPLES_PER_PAIR)
    ints = np.empty(n)
    dz, dy, dx = sub.shape
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        pts = A[s:e, None, :] * (1.0 - t)[None, :, None] + \
              B[s:e, None, :] * t[None, :, None]
        zi = (pts[..., 2] - z0_full) / LASAGNA_SCALE
        yi = pts[..., 1] / LASAGNA_SCALE
        xi = pts[..., 0] / LASAGNA_SCALE
        coords = np.stack([np.clip(zi, 0, dz - 1).ravel(),
                           np.clip(yi, 0, dy - 1).ravel(),
                           np.clip(xi, 0, dx - 1).ravel()])
        dens = map_coordinates(sub, coords, order=1, mode='nearest')
        dens = dens.reshape(e - s, SAMPLES_PER_PAIR).astype(np.float64) / DECODE
        seg = (dist[s:e] / (SAMPLES_PER_PAIR - 1))[:, None]
        ints[s:e] = np.sum(0.5 * (dens[:, :-1] + dens[:, 1:]) * seg, axis=1)
    return ints


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
    ints = batched_integrals(sub, z0, np.array(A), np.array(B))
    dws = np.array(dws, dtype=float)
    k = float(np.sum(ints * dws) / np.sum(ints * ints))
    print(f'k (w1) = {k:.3f}')
    del sub
    return k


def cell_of(xyz, uz, ux, uy, z_begin, z_cell, theta_cells):
    ax = np.interp(xyz[:, 2], uz, ux)
    ay = np.interp(xyz[:, 2], uz, uy)
    theta = np.arctan2(xyz[:, 1] - ay, xyz[:, 0] - ax)
    tb = np.clip(((theta + np.pi) / (2 * np.pi) * theta_cells).astype(int),
                 0, theta_cells - 1)
    zb = ((xyz[:, 2] - z_begin) / z_cell).astype(int)
    return zb * theta_cells + tb


def auc_mannwhitney(scores_pos, scores_neg):
    """AUC via estatística U (sem sklearn). pos = classe 'não satisfeita'."""
    s = np.concatenate([scores_pos, scores_neg])
    r = np.argsort(np.argsort(s)) + 1.0  # ranks 1..n (empates: ok p/ estimativa)
    n1, n0 = len(scores_pos), len(scores_neg)
    if n1 == 0 or n0 == 0:
        return float('nan')
    u = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--runs', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-runs'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--z-cell', type=float, default=250.0)
    ap.add_argument('--theta-cells', type=int, default=72,
                    help='células de 5 graus')
    ap.add_argument('--min-gap', type=float, default=6.0)
    ap.add_argument('--max-gap', type=float, default=60.0)
    ap.add_argument('--max-residual', type=float, default=0.25)
    ap.add_argument('--unsat-frac', type=float, default=0.5,
                    help='pcl "não satisfeita" se fraction < isto')
    args = ap.parse_args()

    k = calibrate_k(args.dataset, 8000, 9000)
    sub, z0f = load_gradmag(args.dataset, args.z_begin, args.z_end)
    uz, ux, uy = load_umbilicus(args.dataset)
    P = load_track_points(args.dataset, args.z_begin, args.z_end)

    # pares candidatos (idêntico ao v2, sem decimação p/ estatística cheia)
    ax = np.interp(P[:, 2], uz, ux); ay = np.interp(P[:, 2], uz, uy)
    r = np.hypot(P[:, 0] - ax, P[:, 1] - ay)
    theta = np.arctan2(P[:, 1] - ay, P[:, 0] - ax)
    tb_fine = np.clip(((theta + np.pi) / (2 * np.pi) * 2880).astype(int),
                      0, 2879)
    zb_fine = ((P[:, 2] - args.z_begin) / 8.0).astype(int)
    cell_fine = zb_fine * 2880 + tb_fine
    order = np.lexsort((r, cell_fine))
    cs, rs = cell_fine[order], r[order]
    good = (cs[1:] == cs[:-1]) & (rs[1:] - rs[:-1] >= args.min_gap) & \
           (rs[1:] - rs[:-1] <= args.max_gap)
    ia, ib = order[:-1][good], order[1:][good]
    print(f'pares candidatos: {len(ia)}')
    ints = batched_integrals(sub, z0f, P[ia], P[ib])
    kI = k * ints
    dw = np.round(kI).astype(int)
    resid = np.abs(kI - dw)
    rejected = ~((dw >= 1) & (dw <= 3) & (resid <= args.max_residual))
    ge2 = (dw >= 2)

    # dificuldade por célula grossa (posição = ponto médio do par)
    mid = 0.5 * (P[ia] + P[ib])
    cells = cell_of(mid, uz, ux, uy, args.z_begin, args.z_cell,
                    args.theta_cells)
    agg = defaultdict(lambda: [0, 0, 0.0, 0])  # n, rej, resid_sum, ge2
    for c, rj, rs_, g2 in zip(cells, rejected, resid, ge2):
        a = agg[c]
        a[0] += 1; a[1] += int(rj); a[2] += float(rs_); a[3] += int(g2)
    diff = {}
    for c, (n, rej, rsum, g2) in agg.items():
        if n >= 8:
            # dificuldade composta: média de 3 indicadores normalizados
            diff[c] = (rej / n + min(rsum / n / 0.25, 1.0) + g2 / n) / 3.0
    print(f'células com estatística (n>=8): {len(diff)}')

    # posição média de cada pcl humana (relative + same) na janela
    pcl_pos = {}
    for fname in ('relative_windings.json', 'same_windings.json'):
        d = json.load(open(os.path.join(args.dataset, fname)))
        for coll in d['collections'].values():
            pts = np.array([pt['p'] for pt in coll['points'].values()
                            if args.z_begin <= pt['p'][2] < args.z_end])
            if len(pts) >= 2:
                key = (fname, coll['name'])
                pcl_pos[key] = pts.mean(axis=0)
    print(f'pcls humanas com posição na janela: {len(pcl_pos)}')

    # evals do braço (c)
    eval_dirs = sorted(
        glob.glob(os.path.join(args.runs, '*w2_armC_eval_human')) +
        glob.glob(os.path.join(args.runs, '*w2_armC_s2_eval_human')))
    if not eval_dirs:
        print('Nenhum eval do (c) encontrado em', args.runs); sys.exit(1)

    for ed in eval_dirs:
        sat = json.load(open(os.path.join(ed, 'satisfied_fitted.json')))
        scores_pos, scores_neg = [], []
        matched = 0
        for e in sat['pcls']:
            sf = os.path.basename(e.get('source_file') or '')
            key = (sf, e.get('name'))
            if key not in pcl_pos:
                continue
            pos = pcl_pos[key][None, :]
            c = cell_of(pos, uz, ux, uy, args.z_begin, args.z_cell,
                        args.theta_cells)[0]
            if c not in diff:
                continue
            matched += 1
            unsat = e['fraction'] < args.unsat_frac
            (scores_pos if unsat else scores_neg).append(diff[c])
        auc = auc_mannwhitney(np.array(scores_pos), np.array(scores_neg))
        tag = os.path.basename(ed).split('734-patch_')[-1]
        print(f'\n== {tag} ==')
        print(f'pcls casadas com célula: {matched} '
              f'(não-satisfeitas: {len(scores_pos)}, '
              f'satisfeitas: {len(scores_neg)})')
        if scores_pos and scores_neg:
            print(f'dificuldade média — não-satisfeitas: '
                  f'{np.mean(scores_pos):.3f} | satisfeitas: '
                  f'{np.mean(scores_neg):.3f}')
        print(f'AUC (dificuldade → não-satisfeita): {auc:.3f}')

    print('\nCritério pré-registrado: AUC >= 0,65 em AMBOS os evals valida '
          'o mapa de dificuldade como direcionador de anotação.')


if __name__ == '__main__':
    main()
