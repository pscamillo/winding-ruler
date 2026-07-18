#!/usr/bin/env python3
"""
winding_atlas_v1.py — PLANO B: atlas de geometria de enrolamento da coleção.

Mede, por scroll, via streaming do open-data S3 (sem download em disco):
  - wraps por raio (nº de folhas cruzadas do eixo à borda) — comparável ao
    perfil publicado pelo iyando p/ PHerc1218 (~37-46)
  - λ = espaçamento entre folhas (mediana/p25/p75), em voxels do nível E
    convertido ao nível-0 do render

Método: surface prediction m7 (já limiarizada, th0.2 no nome) → máscara
binária → por slice: eixo = CENTROIDE da máscara (umbílico não existe
publicado fora do Paris4; sensibilidade centroide-vs-umbílico será
quantificada no Paris4 como barra de erro do método) → raios radiais →
runs contíguos = folhas → contagem + espaçamentos entre centros de runs.

Streaming: loader HTTP mínimo (urllib + numcodecs.Blosc), lê .zarray e
busca só os chunks dos slabs amostrados (~100-250MB/scroll no nível 2).
Determinístico; rate-limit-safe (sequencial, sem fan-out).

Uso (piloto):
  python winding_atlas_v1.py --scroll PHerc0332 --out /tmp/atlas.csv
Varredura:
  for s in PHerc0125 PHerc0332 ...; do python winding_atlas_v1.py --scroll $s --out ~/atlas.csv; done
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict

import numpy as np
from numcodecs import Blosc

BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def http_get(url, binary=True, ok404=False, retries=5):
    import time
    req = urllib.request.Request(url, headers={"User-Agent": "winding-atlas/1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return data if binary else data.decode()
        except urllib.error.HTTPError as e:
            if e.code == 404 and ok404:
                return None
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt); continue
            raise
        except (ConnectionResetError, TimeoutError, OSError,
                urllib.error.URLError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt); continue
            raise


def list_prefix(prefix, delimiter="/"):
    url = (f"{BASE}/?list-type=2&delimiter={delimiter}&prefix={prefix}"
           f"&max-keys=1000")
    xml = http_get(url, binary=False)
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", xml)
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    return prefixes, keys


def discover_surface_zarr(scroll):
    """Zarr de surface prediction mais recente do scroll (exclui normal-grids)."""
    prefixes, _ = list_prefix(f"{scroll}/representations/predictions/surfaces/")
    zarrs = [p for p in prefixes if p.endswith(".zarr/")
             and "normal-grids" not in p]
    if not zarrs:
        return None
    zarrs.sort()  # timestamps no nome → lexicográfico = cronológico
    return zarrs[-1].rstrip("/")


class RemoteZarrLevel:
    """Acesso por-chunk a um nível de OME-Zarr no S3 (leitura, cache LRU simples)."""

    def __init__(self, zarr_path, level):
        meta = json.loads(http_get(f"{BASE}/{zarr_path}/{level}/.zarray",
                                   binary=False))
        self.path = zarr_path
        self.level = level
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.sep = meta.get("dimension_separator", ".")
        comp = meta.get("compressor")
        self.codec = Blosc.from_config({k: v for k, v in comp.items() if k != "id"}) if comp else None
        self.fill = meta.get("fill_value", 0)
        self._cache = {}
        self.bytes_fetched = 0

    def _chunk(self, cz, cy, cx):
        key = (cz, cy, cx)
        if key in self._cache:
            return self._cache[key]
        name = self.sep.join(str(i) for i in key)
        raw = http_get(f"{BASE}/{self.path}/{self.level}/{name}", ok404=True)
        if raw is None:
            arr = np.full(self.chunks, self.fill, dtype=self.dtype)
        else:
            self.bytes_fetched += len(raw)
            buf = self.codec.decode(raw) if self.codec else raw
            arr = np.frombuffer(buf, dtype=self.dtype).reshape(self.chunks)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = arr
        return arr

    def read_slice(self, z):
        """Slice 2D (y, x) completa em z, montada dos chunks necessários."""
        cz, oz = divmod(z, self.chunks[0])
        ny_c = -(-self.shape[1] // self.chunks[1])
        nx_c = -(-self.shape[2] // self.chunks[2])
        out = np.zeros((self.shape[1], self.shape[2]), dtype=self.dtype)
        for cy in range(ny_c):
            for cx in range(nx_c):
                blk = self._chunk(cz, cy, cx)[oz]
                y0, x0 = cy * self.chunks[1], cx * self.chunks[2]
                y1 = min(y0 + self.chunks[1], self.shape[1])
                x1 = min(x0 + self.chunks[2], self.shape[2])
                out[y0:y1, x0:x1] = blk[: y1 - y0, : x1 - x0]
        return out


def rays_on_slice(mask, thetas, gap_close=1, min_wraps=3):
    """Por raio a partir do centroide: (n_runs, [espaçamentos entre centros])."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 500:
        return []
    cy, cx = ys.mean(), xs.mean()
    H, W = mask.shape
    out = []
    for t in thetas:
        dy, dx = np.sin(t), np.cos(t)
        # r máximo até a borda
        rmax = min((H - 1 - cy) / dy if dy > 0 else (-cy / dy if dy < 0 else 1e9),
                   (W - 1 - cx) / dx if dx > 0 else (-cx / dx if dx < 0 else 1e9))
        rmax = max(0.0, min(rmax, np.hypot(H, W)))
        n = int(rmax)
        if n < 20:
            continue
        rr = np.arange(1, n)
        yy = np.clip((cy + rr * dy).astype(int), 0, H - 1)
        xx = np.clip((cx + rr * dx).astype(int), 0, W - 1)
        m = mask[yy, xx].astype(np.int8)
        # fecha buracos curtos
        if gap_close > 0:
            k = np.ones(2 * gap_close + 1, dtype=np.int8)
            m = (np.convolve(m, k, mode="same") > 0).astype(np.int8)
        d = np.diff(np.concatenate([[0], m, [0]]))
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0]
        if len(starts) < min_wraps:
            continue
        centers = (starts + ends - 1) / 2.0
        out.append((len(starts), np.diff(centers)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--zarr-path", default=None,
                    help="path explícito; default = autodescoberta")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--n-slices", type=int, default=16,
                    help="slices amostradas uniformemente no z útil")
    ap.add_argument("--z-margin-frac", type=float, default=0.12,
                    help="fração de z descartada em cada ponta (tapered tips)")
    ap.add_argument("--thetas", type=int, default=64)
    ap.add_argument("--gap-close", type=int, default=0)
    ap.add_argument("--meta", default="/tmp/meta.json")
    ap.add_argument("--out", default="/tmp/winding_atlas.csv")
    args = ap.parse_args()

    zp = args.zarr_path or discover_surface_zarr(args.scroll)
    if not zp:
        print(f"{args.scroll}: sem surface zarr — pulando"); sys.exit(2)
    print(f"{args.scroll}: {zp} @ nivel {args.level}")
    vol = RemoteZarrLevel(zp, args.level)
    print(f"  shape {vol.shape} chunks {vol.chunks} dtype {vol.dtype}")

    Z = vol.shape[0]
    z0 = int(Z * args.z_margin_frac)
    z1 = Z - z0
    zs = np.linspace(z0, z1 - 1, args.n_slices).astype(int)
    thetas = np.linspace(0, 2 * np.pi, args.thetas, endpoint=False)

    wraps_all, spac_all, rays_valid = [], [], 0
    for z in zs:
        mask = vol.read_slice(int(z)) > 0
        for n_runs, spac in rays_on_slice(mask, thetas, gap_close=args.gap_close):
            rays_valid += 1
            wraps_all.append(n_runs)
            spac_all.extend(spac.tolist())
        print(f"  z={z}: raios acumulados {rays_valid} "
              f"({vol.bytes_fetched/1e6:.0f} MB)")

    if rays_valid < 50:
        print(f"{args.scroll}: raios insuficientes ({rays_valid})"); sys.exit(3)

    wraps = np.array(wraps_all)
    spac = np.array(spac_all)
    # conversao fisica via metadata.json (um do volume-base do render)
    lam_um = None
    try:
        meta = json.load(open(args.meta))
        vid = os.path.basename(zp).split("-")[0]
        um = float(meta["samples"][args.scroll]["volumes"][vid]["properties"]["pixel_size_um"])
        mL = re.search(r"-L(\d+)-", os.path.basename(zp))
        lrender = int(mL.group(1)) if mL else 0
        vox_um = um * (2 ** lrender) * (2 ** args.level)
        lam_um = round(float(np.median(spac)) * vox_um, 1)
        print(f"  um/vox medido: {um} x 2^{lrender}(render) x 2^{args.level}(nivel) = {vox_um:.2f}")
    except Exception as e:
        print(f"  um: indisponivel ({e})")
    lvl_scale = 2 ** args.level
    row = {
        "scroll": args.scroll,
        "zarr": os.path.basename(zp),
        "level": args.level,
        "gap_close": args.gap_close,
        "n_slices": len(zs),
        "n_rays": rays_valid,
        "wraps_p10": round(float(np.percentile(wraps, 10)), 1),
        "wraps_med": round(float(np.median(wraps)), 1),
        "wraps_p90": round(float(np.percentile(wraps, 90)), 1),
        "lambda_med_lvlvox": round(float(np.median(spac)), 2),
        "lambda_p25": round(float(np.percentile(spac, 25)), 2),
        "lambda_p75": round(float(np.percentile(spac, 75)), 2),
        "lambda_med_L0vox": round(float(np.median(spac)) * lvl_scale, 2),
        "lambda_med_um": lam_um,
        "MB_fetched": round(vol.bytes_fetched / 1e6, 1),
    }
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print("\n=== " + " | ".join(f"{k}={v}" for k, v in row.items()))
    print(f"linha adicionada a {args.out}")


if __name__ == "__main__":
    main()
