#!/usr/bin/env python3
"""
ruler_concordance_v1.py — pré-braço-(b) do winding ruler.

Mede a concordância entre o primitivo do ruler (integral trapezoidal da
densidade de winding ao longo de segmento reto — convenção IDÊNTICA ao
LasagnaNormalSampler::windingDistance do VC3D, commit 37c37de) e os rótulos
humanos de relative winding (wind_a por ponto) na janela z [10000, 11000).

Critério pré-registrado (handoff v3): concordância exata >= 80% => braço (b)
autorizado a integrar ao fit. 60-80% => diagnóstico. < 60% => reprova barato.

Convenções herdadas do time (fontes):
  - densidade = grad_mag_uint8 / (grad_mag_encode_scale / grad_mag_factor)
      = raw / (1000.0 / 0.25) = raw / 4000.0        [fit_spiral.py:178-179]
  - integral: segmento reto a->b, trapezoidal, passo 8 vox full-res
      [LasagnaNormalSampler.cpp:1089-1115]
  - amostragem TRILINEAR (Discord 15-16/jul, recomendação Forrest)
  - coordenadas pcl em full-res; zarr grupo '4' => dividir por lasagna_scale=4
  - fora de cobertura => sem resposta (par descartado e contado), nunca chute

Uso:
  python ruler_concordance_v1.py \
      --dataset ~/challenges/vesuvius/spiral-dataset/PHercParis4 \
      --z-begin 10000 --z-end 11000 --out /tmp/concordance_w2.csv
"""
import argparse
import json
import os
import sys
import csv
from collections import defaultdict

import numpy as np
import zarr
from scipy.ndimage import map_coordinates

ENCODE_SCALE = 1000.0   # fit_spiral.py:178
GRAD_MAG_FACTOR = 0.25  # fit_spiral.py:179
DECODE = ENCODE_SCALE / GRAD_MAG_FACTOR  # 4000.0
LASAGNA_SCALE = 4       # fit_spiral.py (lasagna_scale)
STEP_VX = 8.0           # LasagnaNormalSampler default / spacing_integration_steps
MAX_TRUTH_DELTA = 6     # pares com |dw| maior sao raros e segmentos longos demais


def load_gradmag_subvolume(dataset, z_begin, z_end, margin_vx=64):
    """Carrega o subvolume de grad_mag (grupo 4) cobrindo [z_begin, z_end) + margem.
    Retorna (array uint8 [dz, y, x], z0_full) onde z0_full e o z full-res do indice 0.
    """
    path = os.path.join(dataset, 'lasagna_inputs', 'las_008_grad_mag.ome.zarr')
    root = zarr.open(path, mode='r')          # ARMADILHA #1: sempre mode='r'
    arr = root['4']
    z0_full = max(0, z_begin - margin_vx)
    z1_full = z_end + margin_vx
    z0 = z0_full // LASAGNA_SCALE
    z1 = min(arr.shape[0], (z1_full + LASAGNA_SCALE - 1) // LASAGNA_SCALE)
    sub = arr[z0:z1, :, :]                    # uint8, ~1GB por 1000 slices
    print(f'grad_mag[4] subvolume: shape={sub.shape} dtype={sub.dtype} '
          f'(z full-res [{z0*LASAGNA_SCALE}, {z1*LASAGNA_SCALE}))')
    return np.asarray(sub), z0 * LASAGNA_SCALE


def winding_distance(sub, z0_full, a_xyz, b_xyz, step_vx=STEP_VX):
    """Integral trapezoidal da densidade decodificada ao longo do segmento a->b.
    a_xyz/b_xyz em coordenadas full-res [x, y, z] (convenção 'p' dos pcls).
    Retorna (integral, ok). ok=False se qualquer amostra cair fora do subvolume.
    Fiel ao C++: intervals = ceil(dist/step); trapezio por intervalo.
    """
    a = np.asarray(a_xyz, dtype=np.float64)
    b = np.asarray(b_xyz, dtype=np.float64)
    delta = b - a
    dist = float(np.linalg.norm(delta))
    if not (dist > 1e-9) or not np.isfinite(dist):
        return 0.0, True
    intervals = max(1, int(np.ceil(dist / step_vx)))
    # amostras nos n+1 nos do trapezio, de uma vez (vetorizado)
    t = np.linspace(0.0, 1.0, intervals + 1)
    pts = a[None, :] * (1.0 - t)[:, None] + b[None, :] * t[:, None]  # [n+1, xyz]
    # full-res -> indices do zarr grupo 4 na ordem (z, y, x)
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
    integral = float(np.sum(0.5 * (dens[:-1] + dens[1:]) * seg))
    return integral, True


def collect_pairs(dataset, z_begin, z_end):
    """Pares (a, b, delta_truth) dos relative_windings com ambos os pontos na janela.
    Ground truth = |wind_a_i - wind_a_j| (explicito por ponto)."""
    path = os.path.join(dataset, 'relative_windings.json')
    d = json.load(open(path))
    pairs = []
    n_coll = 0
    for cid, coll in d['collections'].items():
        pts = []
        for pid, pt in coll['points'].items():
            p = pt['p']
            w = pt.get('wind_a')
            if w is None:
                continue
            if z_begin <= p[2] < z_end:
                pts.append((p, float(w)))
        if len(pts) < 2:
            continue
        n_coll += 1
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dw = abs(pts[i][1] - pts[j][1])
                if 0 < dw <= MAX_TRUTH_DELTA:
                    pairs.append((cid, pts[i][0], pts[j][0], dw))
    print(f'{n_coll} collections com >=2 pontos na janela; {len(pairs)} pares '
          f'(1 <= |dw| <= {MAX_TRUTH_DELTA})')
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=os.path.expanduser(
        '~/challenges/vesuvius/spiral-dataset/PHercParis4'))
    ap.add_argument('--z-begin', type=int, default=10000)
    ap.add_argument('--z-end', type=int, default=11000)
    ap.add_argument('--step-vx', type=float, default=STEP_VX)
    ap.add_argument('--out', default='/tmp/concordance.csv')
    args = ap.parse_args()

    pairs = collect_pairs(args.dataset, args.z_begin, args.z_end)
    if not pairs:
        print('Nenhum par na janela — nada a medir.'); sys.exit(1)

    sub, z0_full = load_gradmag_subvolume(args.dataset, args.z_begin, args.z_end)

    rows, skipped = [], 0
    for cid, a, b, truth in pairs:
        integral, ok = winding_distance(sub, z0_full, a, b, args.step_vx)
        if not ok:
            skipped += 1
            continue
        pred = int(round(integral))
        rows.append((cid, truth, integral, pred, int(pred == truth)))

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['collection', 'truth_dw', 'integral', 'pred_dw', 'exact'])
        w.writerows(rows)

    n = len(rows)
    exact = sum(r[4] for r in rows)
    print(f'\n=== CONCORDÂNCIA (janela z [{args.z_begin}, {args.z_end}), '
          f'step {args.step_vx} vx) ===')
    print(f'pares avaliados: {n}  |  fora de cobertura (descartados): {skipped}')
    print(f'CONCORDÂNCIA EXATA: {exact}/{n} = {100.0*exact/max(n,1):.1f}%')
    mae = np.mean([abs(r[2] - r[1]) for r in rows]) if rows else float('nan')
    print(f'MAE da integral vs truth: {mae:.3f} voltas')

    # por distancia de voltas (a matriz que diagnostica)
    by = defaultdict(lambda: [0, 0, []])
    for _, truth, integral, pred, ex in rows:
        by[truth][0] += 1
        by[truth][1] += ex
        by[truth][2].append(integral)
    print(f'\n{"truth_dw":>8} {"n":>6} {"exact%":>7} {"mean_int":>9} {"ratio":>6}')
    for truth in sorted(by):
        cnt, ex, ints = by[truth]
        mi = float(np.mean(ints))
        print(f'{truth:>8.0f} {cnt:>6} {100.0*ex/cnt:>6.1f}% {mi:>9.3f} '
              f'{mi/truth:>6.3f}')
    print('\nLeitura: ratio ~1.00 estavel por truth_dw => densidade bem '
          'calibrada. Ratio constante != 1 => erro de escala SISTEMATICO '
          '(corrigivel por fator unico — reportar, nao aplicar). Ratio caindo '
          'com truth_dw => segmentos longos atravessam regioes degeneradas.')
    print(f'\nCSV: {args.out}')
    print('Criterio pre-registrado: exata >= 80% => bracao (b) autorizado; '
          '60-80% => diagnostico; < 60% => reprova.')


if __name__ == '__main__':
    main()
