#!/usr/bin/env python3
"""
ruler_generate_v1.py — WINDING RULER MVP: gerador determinístico de
relative-winding constraints.

Mecanismo (validado no pré-braço-(b), v1.1-v1.5):
  - densidade = grad_mag[grupo 4] / 4000  (convenção LasagnaNormalSampler)
  - marcha RADIAL a partir do umbílico, por (z, θ) em grade fixa
  - acumula k * integral trapezoidal (passo 2 vox, trilinear)
  - EMITE um ponto a cada cruzamento inteiro do winding acumulado
    → pontos consecutivos têm Δw = +1 por construção (elo validado a 91-92%)
  - sinal/ordem: wind_a cresce PARA FORA (orientação medida: 100%/650 pares)
  - k calibrado nos pares humanos de OUTRA janela (--calib-z) para que nem o
    escalar vaze na avaliação do braço (b)

Qualidade (determinística, sem RNG):
  - cadeia quebra se o espaçamento entre emissões sair de [min,max]_spacing
    (folha implausível → recomeça collection)
  - collections com < min_points são descartadas
  - fora de cobertura do volume → cadeia termina

Saída: JSON point-collection VC3D v1 (schema de point_collection.py),
tratado como RELATIVO pelo fit (winding_is_absolute vem só do nome
abs_winding.json — fit_spiral.py:583).

Uso típico (gerar na w2, calibrar na w1):
  python ruler_generate_v1.py \
      --z-begin 10000 --z-end 11000 \
      --calib-z-begin 8000 --calib-z-end 9000 \
      --out ~/challenges/vesuvius/spiral-dataset/PHercParis4/ruler_relative_windings.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import zarr
from scipy.ndimage import map_coordinates

ENCODE_SCALE = 1000.0
GRAD_MAG_FACTOR = 0.25
DECODE = ENCODE_SCALE / GRAD_MAG_FACTOR
LASAGNA_SCALE = 4


def load_gradmag(dataset, z_begin, z_end, margin_vx=64):
    path = os.path.join(dataset, 'lasagna_inputs', 'las_008_grad_mag.ome.zarr')
    root = zarr.open(path, mode='r')
    arr = root['4']
    z0_full = max(0, z_begin - margin_vx)
    z1_full = z_end + margin_vx
    z0 = z0_full // LASAGNA_SCALE
    z1 = min(arr.shape[0], (z1_full + LASAGNA_SCALE - 1) // LASAGNA_SCALE)
    sub = np.asarray(arr[z0:z1, :, :])
    print(f'grad_mag[4]: shape={sub.shape} '
          f'(z full-res [{z0*LASAGNA_SCALE}, {z1*LASAGNA_SCALE}))')
    return sub, z0 * LASAGNA_SCALE


def load_umbilicus(dataset):
    d = json.load(open(os.path.join(dataset, 'umbilicus.json')))
    pts = [(float(cp['z']), float(cp['x']), float(cp['y']))
           for cp in d['control_points']]
    pts.sort()
    zs = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    print(f'umbilicus: {len(pts)} control points, z [{zs.min():.0f}, {zs.max():.0f}]')
    return lambda z: (float(np.interp(z, zs, xs)), float(np.interp(z, zs, ys)))


def calibrate_k(dataset, sub_cal, z0_cal, z_begin, z_end, sample_vx=2.0):
    """k = argmin ||k*int - |dw||^2 sobre TODOS os pares humanos da janela de
    calibração (disjunta da janela de geração)."""
    d = json.load(open(os.path.join(dataset, 'relative_windings.json')))
    ints, dws = [], []
    for coll in d['collections'].values():
        pts = [(pt['p'], float(pt['wind_a']))
               for pt in coll['points'].values()
               if pt.get('wind_a') is not None
               and z_begin <= pt['p'][2] < z_end]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dw = abs(pts[j][1] - pts[i][1])
                if not (0 < dw <= 6):
                    continue
                v, ok = _integral(sub_cal, z0_cal, pts[i][0], pts[j][0],
                                  sample_vx)
                if ok and np.isfinite(v):
                    ints.append(v); dws.append(dw)
    ints = np.array(ints); dws = np.array(dws)
    k = float(np.sum(ints * dws) / np.sum(ints * ints))
    print(f'calibração: {len(ints)} pares humanos em z [{z_begin},{z_end}) '
          f'→ k = {k:.3f}')
    return k


def _integral(sub, z0_full, a, b, sample_vx):
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


def march_ray(sub, z0_full, ax, ay, z, theta, k,
              r_start, r_max, step_vx,
              min_spacing, max_spacing, min_points):
    """Marcha radial em (z, θ); retorna lista de cadeias; cada cadeia é
    lista de (x, y, z) emitidos em cruzamentos inteiros de k*∫ρ dr."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    n = int((r_max - r_start) / step_vx) + 1
    rs = r_start + np.arange(n) * step_vx
    px = ax + rs * cos_t
    py = ay + rs * sin_t
    zi = np.full(n, (z - z0_full) / LASAGNA_SCALE)
    yi = py / LASAGNA_SCALE
    xi = px / LASAGNA_SCALE
    dz, dy, dx = sub.shape
    inside = (zi >= 0) & (zi <= dz - 1) & (yi >= 0) & (yi <= dy - 1) & \
             (xi >= 0) & (xi <= dx - 1)
    if not inside.any():
        return []
    last = int(np.argmin(inside)) if not inside.all() else n
    if last < 8:
        return []
    dens = map_coordinates(sub, np.stack([zi[:last], yi[:last], xi[:last]]),
                           order=1, mode='nearest').astype(np.float64) / DECODE
    # winding acumulado (trapezoidal incremental)
    incr = 0.5 * (dens[:-1] + dens[1:]) * step_vx * k
    cumw = np.concatenate([[0.0], np.cumsum(incr)])
    chains = []
    chain = []
    next_int = 1.0
    prev_r = None
    for i in range(1, last):
        while cumw[i] >= next_int:
            # interpola r do cruzamento dentro do passo
            f = (next_int - cumw[i - 1]) / max(cumw[i] - cumw[i - 1], 1e-12)
            r_cross = rs[i - 1] + f * step_vx
            if prev_r is not None:
                spacing = r_cross - prev_r
                if spacing < min_spacing or spacing > max_spacing:
                    if len(chain) >= min_points:
                        chains.append(chain)
                    chain = []
            chain.append((ax + r_cross * cos_t, ay + r_cross * sin_t, z))
            prev_r = r_cross
            next_int += 1.0
    if len(chain) >= min_points:
        chains.append(chain)
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--calib-z-begin', type=int, default=8000)
    ap.add_argument('--calib-z-end', type=int, default=9000)
    ap.add_argument('--z-stride', type=int, default=25)
    ap.add_argument('--theta-deg-stride', type=float, default=3.0)
    ap.add_argument('--r-start', type=float, default=100.0)
    ap.add_argument('--r-max', type=float, default=3400.0)
    ap.add_argument('--step-vx', type=float, default=2.0)
    ap.add_argument('--min-spacing', type=float, default=6.0)
    ap.add_argument('--max-spacing', type=float, default=60.0)
    ap.add_argument('--min-points', type=int, default=4)
    ap.add_argument('--max-points-per-chain', type=int, default=40)
    ap.add_argument('--out', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4/'
        'ruler_relative_windings.json'))
    args = ap.parse_args()

    # calibração em janela DISJUNTA
    if not (args.calib_z_end <= args.z_begin or args.calib_z_begin >= args.z_end):
        print('ERRO: janela de calibração sobrepõe a de geração.'); sys.exit(1)
    sub_cal, z0_cal = load_gradmag(args.dataset, args.calib_z_begin,
                                   args.calib_z_end)
    k = calibrate_k(args.dataset, sub_cal, z0_cal,
                    args.calib_z_begin, args.calib_z_end)
    del sub_cal

    sub, z0_full = load_gradmag(args.dataset, args.z_begin, args.z_end)
    axis_xy = load_umbilicus(args.dataset)

    zs = list(range(args.z_begin, args.z_end, args.z_stride))
    thetas = np.deg2rad(np.arange(0.0, 360.0, args.theta_deg_stride))
    print(f'grade: {len(zs)} slices × {len(thetas)} raios '
          f'= {len(zs)*len(thetas)} raios')

    collections = {}
    cid = 0
    spacings_all = []
    for z in zs:
        ax, ay = axis_xy(z)
        for theta in thetas:
            chains = march_ray(sub, z0_full, ax, ay, float(z), float(theta), k,
                               args.r_start, args.r_max, args.step_vx,
                               args.min_spacing, args.max_spacing,
                               args.min_points)
            for chain in chains:
                # quebra cadeias longas para collections manejáveis
                for s in range(0, len(chain), args.max_points_per_chain):
                    part = chain[s:s + args.max_points_per_chain]
                    if len(part) < args.min_points:
                        continue
                    pts = {}
                    for w_idx, (x, y, zz) in enumerate(part, start=1):
                        pts[str(w_idx)] = {
                            'p': [round(x, 3), round(y, 3), round(zz, 3)],
                            'wind_a': float(w_idx),
                            'creation_time': 0,
                        }
                    rr = [np.hypot(p[0] - ax, p[1] - ay) for p in part]
                    spacings_all.extend(np.diff(rr).tolist())
                    collections[str(cid)] = {
                        'name': f'ruler_z{int(z)}_t{np.rad2deg(theta):.0f}_{s}',
                        'points': pts,
                        'metadata': {
                            'generator': 'winding_ruler_v1',
                            'k': round(k, 4),
                        },
                        'color': [0.1, 0.6, 0.9],
                    }
                    cid += 1

    n_pts = sum(len(c['points']) for c in collections.values())
    out = {
        'vc_pointcollections_json_version': '1',
        'collections': collections,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f)
    sp = np.array(spacings_all)
    print(f'\n=== RULER v1: {len(collections)} collections, {n_pts} pontos ===')
    if len(sp):
        print(f'espaçamento entre folhas: mediana {np.median(sp):.1f} vx, '
              f'p10 {np.percentile(sp,10):.1f}, p90 {np.percentile(sp,90):.1f}')
    print(f'saída: {args.out} '
          f'({os.path.getsize(args.out)/1e6:.1f} MB)')
    print('\nSanidade sugerida: comparar mediana de espaçamento com a dos '
          'pares humanos dw=1 (~10-20 vx) e conferir que o fit carrega o '
          'arquivo ("Loaded point collection with N collections").')


if __name__ == '__main__':
    main()
