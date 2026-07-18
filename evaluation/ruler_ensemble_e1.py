#!/usr/bin/env python3
"""
ruler_ensemble_e1.py — Teste E1: a COMBINAÇÃO de sinais fracos supera cada
sinal sozinho na predição de Δw? (ideia-3 da revisão externa, com leakage
corrigido: sem lambda-como-feature.)

Features por par humano (A,B) da w2 (todas computáveis pelo ruler em
qualquer par, sem rótulo):
  f_dist  : distância euclidiana
  f_int   : integral crua de grad_mag (convenção do time)
  f_cos   : integral cos-ponderada (normais nx/ny)
  f_align : |cos| médio raio×normal ao longo do segmento
  f_patch : nº de verified patches cujo bbox contém o ponto médio
  f_rad   : raio do ponto médio (dist. ao umbílico)

Modelo: mínimos quadrados linear (transparente, sem ML pesado) em subsets
crescentes de features → pred = round(clip(ŷ, 0, 8)). Split por collection
(seed 2, virgem p/ este teste). Ablation reportada como a revisão sugeriu.

Critério pré-registrado: ensemble completo no TEST >= 75% global E >= 93%
em dw=1 ⇒ gerador v3 com etiquetas-ensemble se justifica. Abaixo ⇒ tabela
vai para o estudo; geração segue bloqueada até lasagna fina.
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
import zarr
from scipy.ndimage import map_coordinates

ENCODE_SCALE = 1000.0
GRAD_MAG_FACTOR = 0.25
DECODE = ENCODE_SCALE / GRAD_MAG_FACTOR
LASAGNA_SCALE = 4
S = 32  # amostras por segmento


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
    print(f'canais[4]: {subs["grad_mag"].shape}')
    return subs, z0_full


def load_umbilicus(dataset):
    d = json.load(open(os.path.join(dataset, 'umbilicus.json')))
    pts = sorted((float(c['z']), float(c['x']), float(c['y']))
                 for c in d['control_points'])
    zs = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    return zs, xs, ys


def pair_features(subs, z0f, A, B, uz, ux, uy, patch_boxes):
    n = len(A)
    delta = B - A
    dist = np.linalg.norm(delta, axis=1)
    d_hat = delta / np.maximum(dist, 1e-9)[:, None]
    t = np.linspace(0, 1, S)
    pts = A[:, None, :] * (1 - t)[None, :, None] + \
        B[:, None, :] * t[None, :, None]
    zi = (pts[..., 2] - z0f) / LASAGNA_SCALE
    yi = pts[..., 1] / LASAGNA_SCALE
    xi = pts[..., 0] / LASAGNA_SCALE
    dz, dy, dx = subs['grad_mag'].shape
    coords = np.stack([np.clip(zi, 0, dz - 1).ravel(),
                       np.clip(yi, 0, dy - 1).ravel(),
                       np.clip(xi, 0, dx - 1).ravel()])
    g = map_coordinates(subs['grad_mag'], coords, order=1,
                        mode='nearest').reshape(n, S).astype(float) / DECODE
    rnx = map_coordinates(subs['nx'], coords, order=1,
                          mode='nearest').reshape(n, S).astype(float)
    rny = map_coordinates(subs['ny'], coords, order=1,
                          mode='nearest').reshape(n, S).astype(float)
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
    mid = 0.5 * (A + B)
    ax = np.interp(mid[:, 2], uz, ux)
    ay = np.interp(mid[:, 2], uz, uy)
    f_rad = np.hypot(mid[:, 0] - ax, mid[:, 1] - ay)
    # f_patch: nº de bboxes de verified patches contendo o ponto médio
    lo, hi = patch_boxes  # (P,3),(P,3) em xyz
    f_patch = np.zeros(n)
    for i in range(n):
        m = mid[i]
        inside = (lo[:, 0] <= m[0]) & (m[0] <= hi[:, 0]) & \
                 (lo[:, 1] <= m[1]) & (m[1] <= hi[:, 1]) & \
                 (lo[:, 2] <= m[2]) & (m[2] <= hi[:, 2])
        f_patch[i] = inside.sum()
    return np.column_stack([dist, f_int, f_cos, f_align, f_patch, f_rad])


def load_patch_boxes(dataset):
    lo, hi = [], []
    for m in glob.glob(os.path.join(dataset, 'verified_patches', '*',
                                    'meta.json')):
        try:
            d = json.load(open(m))
            b = d['bbox']
            lo.append(b[0]); hi.append(b[1])
        except Exception:
            continue
    print(f'patch bboxes: {len(lo)}')
    return np.array(lo), np.array(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--split-seed', type=int, default=2)
    args = ap.parse_args()

    d = json.load(open(os.path.join(args.dataset, 'relative_windings.json')))
    A, B, dw, cids = [], [], [], []
    for cid, coll in d['collections'].items():
        pts = [(pt['p'], float(pt['wind_a']))
               for pt in coll['points'].values()
               if pt.get('wind_a') is not None
               and args.z_begin <= pt['p'][2] < args.z_end]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                t = abs(pts[j][1] - pts[i][1])
                if 0 < t <= 6:
                    A.append(pts[i][0]); B.append(pts[j][0])
                    dw.append(t); cids.append(cid)
    A = np.array(A); B = np.array(B); dw = np.array(dw, dtype=float)
    cids = np.array(cids)
    print(f'{len(dw)} pares | {len(set(cids))} collections')

    subs, z0f = load_channels(args.dataset, args.z_begin, args.z_end)
    uz, ux, uy = load_umbilicus(args.dataset)
    boxes = load_patch_boxes(args.dataset)
    F = pair_features(subs, z0f, A, B, uz, ux, uy, boxes)
    names = ['dist', 'int', 'cos', 'align', 'patch', 'rad']

    colls = sorted(set(cids))
    rng = random.Random(args.split_seed)
    rng.shuffle(colls)
    train_set = set(colls[:len(colls) // 2])
    tr = np.array([c in train_set for c in cids])
    te = ~tr
    print(f'train {tr.sum()} | test {te.sum()} (seed {args.split_seed})')

    def fit_eval(idx):
        X = F[:, idx]
        Xtr = np.column_stack([X[tr], np.ones(tr.sum())])
        Xte = np.column_stack([X[te], np.ones(te.sum())])
        w, *_ = np.linalg.lstsq(Xtr, dw[tr], rcond=None)
        pred = np.clip(np.round(Xte @ w), 0, 8)
        exact = pred == dw[te]
        g = float(exact.mean() * 100)
        m1 = dw[te] == 1
        d1 = float(exact[m1].mean() * 100) if m1.any() else float('nan')
        return g, d1

    subsets = [
        ([0], 'dist'),
        ([1], 'int'),
        ([0, 1], 'dist+int'),
        ([0, 1, 2, 3], 'dist+int+cos+align'),
        ([0, 1, 2, 3, 4], '+patch'),
        ([0, 1, 2, 3, 4, 5], 'TODAS'),
    ]
    print(f'\n{"features":<24} {"global%":>8} {"dw=1%":>7}')
    results = {}
    for idx, label in subsets:
        g, d1 = fit_eval(idx)
        results[label] = (g, d1)
        print(f'{label:<24} {g:>8.1f} {d1:>7.1f}')

    g, d1 = results['TODAS']
    print(f'\nCritério pré-registrado: TODAS >= 75% global E >= 93% dw=1.')
    print(f'Resultado: {g:.1f}% / {d1:.1f}% → '
          f'{"PASSA — v3 ensemble justificado" if g >= 75 and d1 >= 93 else "NÃO PASSA — geração segue bloqueada até lasagna fina"}')
    print('Confirmar com --split-seed 3 antes de qualquer decisão.')


if __name__ == '__main__':
    main()
