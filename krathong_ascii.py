#!/usr/bin/env python3
"""
krathong_ascii.py - a Loy Krathong night, rendered entirely in coloured ASCII.

A tiny software renderer. The krathong (banana-leaf float with lotus petals,
marigolds, a candle and incense) is a 3-D point cloud that is projected and
z-buffered into a character grid, then lit by its own flickering candle, the
moon and the golden water. Sky, moon, floating lanterns (khom loi), temple
silhouettes and the river are shaded per character cell. Every motion is a
whole multiple of one 10 s period, so the output loops seamlessly.

    python krathong_ascii.py                 # krathong.mp4   1080x1344 art only
    python krathong_ascii.py --showcase      # krathong.mp4   1080x1920 + code panel
    python krathong_ascii.py --png 4.5       # krathong_4.5.png, one still
    python krathong_ascii.py --term          # live, in a truecolor terminal
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

LOOP = 10.0                      # seconds; every motion repeats exactly
W0 = 2 * math.pi / LOOP
PITCH = 0.42                     # camera looks down at the river
CP, SPH = math.cos(PITCH), math.sin(PITCH)
PERSP = 0.10
F32 = np.float32
FLIP_Y = np.array([1, -1, 1], F32)
CANDLE_TOP = np.array([0.0, 0.86, 0.0], F32)
CANDLE_COL = np.array([1.0, 0.72, 0.38], F32)
MOON_DIR = np.array([0.45, 0.62, -0.64], F32) / np.linalg.norm([0.45, 0.62, -0.64])

RAMPS = {                        # glyphs from faint to dense, per material
    "sky": " ..::",
    "star": " ..+*",
    "moon": " .:oO0@&",
    "glow": "  ..:'+",
    "lantern": " -=#8&",
    "temple": " .:-=+#%",
    "window": " =##",
    "water": " .-~=≈%#",
    "base": " .:-=#",
    "leaf": " .:;!|(){}0@",
    "lotus": " .:=+oO%&@",
    "flower": " .oO0@@",
    "candle": " :|I[]#",
    "incense": " .|||",
    "flame": " ^*#&@@",
    "spark": " .'*+*",
    "firework": " ..'+*#@",
    "dragon": " .:;=+x%#@",
    "dgold": " .-=*#@",
    "whisker": " .'~~",
    "pearl": " .oO0@",
    "smoke": " .:'~",
}
MID = {name: i for i, name in enumerate(RAMPS)}
LV = 24                          # density levels per ramp
FIREWORKS_ON = True
CHARS = sorted({c for r in RAMPS.values() for c in r})
LUT = np.array([[CHARS.index(r[min(len(r) - 1, int(l / LV * len(r)))])
                 for l in range(LV)] for r in RAMPS.values()])


# ----------------------------------------------------------------- helpers ---
def mixc(a, b, t):
    """Blend colours a -> b by a scalar field t."""
    t = np.asarray(t, F32)[..., None]
    return np.asarray(a, F32) * (1 - t) + np.asarray(b, F32) * t


def hash2(a, b, s):
    v = np.sin(a * 12.9898 + b * 78.233 + s * 37.719) * 43758.5453
    return v - np.floor(v)


def lum(c):
    return c @ np.array([0.3, 0.55, 0.15], F32)


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], F32)


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], F32)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], F32)


def canvas():
    """Per-cell (density, colour, material)."""
    return [np.zeros((ROWS, COLS), F32), np.zeros((ROWS, COLS, 3), F32),
            np.zeros((ROWS, COLS), np.int64)]


def put(cv, m, d, col, mid):
    d = np.broadcast_to(np.asarray(d, F32), m.shape)
    col = np.broadcast_to(np.asarray(col, F32), m.shape + (3,))
    cv[0][m], cv[1][m], cv[2][m] = d[m], col[m], MID[mid]


def putmax(cv, d, col, mid, floor=0.04):
    """Overwrite only where this layer is denser (halos, glows)."""
    d = np.broadcast_to(np.asarray(d, F32), cv[0].shape)
    put(cv, (d > cv[0]) & (d > floor), d, col, mid)


def putflat(cv, idx, d, col, mid):
    """Write cells by flat index."""
    cv[0].reshape(-1)[idx] = d
    cv[1].reshape(-1, 3)[idx] = col
    cv[2].reshape(-1)[idx] = MID[mid]


def window(x0, x1, y0, y1):
    """Row/col slices of the cells whose centres can fall inside a pixel box."""
    c0, c1 = max(0, math.floor(x0 / CW - 0.5)), min(COLS, math.ceil(x1 / CW - 0.5) + 1)
    r0, r1 = max(0, math.floor(y0 / CH - 0.5)), min(ROWS, math.ceil(y1 / CH - 0.5) + 1)
    return slice(r0, max(r0, r1)), slice(c0, max(c0, c1))


def sub(cv, rs, cs):
    """A canvas view onto a window; writes go straight through to cv."""
    return [cv[0][rs, cs], cv[1][rs, cs], cv[2][rs, cs]]


def putcells(cv, x, y, d, col, mid, force=False):
    """Splat individual points (sparks, fireflies) at pixel positions."""
    c, r = (np.asarray(x) // CW).astype(int), (np.asarray(y) // CH).astype(int)
    ok = (c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS) & (d > 0.05)
    r, c, d, col = r[ok], c[ok], np.asarray(d)[ok], np.asarray(col, F32)[ok]
    ok = np.ones_like(d, bool) if force else d > cv[0][r, c]
    cv[0][r[ok], c[ok]], cv[1][r[ok], c[ok]], cv[2][r[ok], c[ok]] = d[ok], col[ok], MID[mid]


# ---------------------------------------------------------------- geometry ---
_parts = []


def grid(lu, lv):
    """Parameter grid sized so every char cell gets a couple of samples."""
    nu, nv = max(3, int(lu * SAMP)), max(3, int(lv * SAMP))
    return np.meshgrid(np.linspace(0, 1, nu, dtype=F32),
                       np.linspace(0, 1, nv, dtype=F32), indexing="ij")


def normals(P):
    n = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


def emit(P, N, col, mid, R=np.eye(3, dtype=F32), T=(0, 0, 0), em=False):
    col = np.broadcast_to(np.asarray(col, F32), P.shape).reshape(-1, 3)
    P = P.reshape(-1, 3) @ R.T + np.asarray(T, F32)
    N = N.reshape(-1, 3) @ R.T
    _parts.append((P, N, col, np.full(len(P), MID[mid]), np.full(len(P), em)))


def leaf_petal(r0, y0, a, wd, hh, tilt, fold):
    """Folded banana-leaf petal: centre crease + leaf veins."""
    U, V = grid(hh * 1.05, wd * 2)
    v = V * 2 - 1
    w = wd * (1 - U) ** 0.85
    z = (r0 + tilt * U * hh + 0.10 * U ** 2 * hh
         + fold * (1 - np.abs(v)) * np.sqrt(U + 0.02) * (1 - 0.6 * U))
    P = np.stack([v * w, U * hh, z], -1)
    col = mixc((0.20, 0.50, 0.12), (0.62, 0.82, 0.25), U)
    col = col * (0.82 + 0.18 * (0.5 + 0.5 * np.cos(v * 22)))[..., None]
    col = col * (1 - 0.35 * np.exp(-(v / 0.05) ** 2))[..., None]
    col = mixc(col, (0.85, 0.90, 0.40), 0.25 * np.abs(v) ** 4)
    emit(P, normals(P), col, "leaf", Ry(a), (0, y0, 0))


def lotus_petal(a, op, ln, wd, y0, r0):
    """Lotus petal: pointed oval, cupped, opened by angle op."""
    U, V = grid(ln * 1.1, wd * 2)
    v = V * 2 - 1
    w = wd * np.clip(np.sin(np.pi * U ** 0.75), 0, 1) ** 0.8 + 0.004
    ly = U * ln
    zl = 0.5 * wd * (1 - v * v) * (1 - 0.5 * U)
    y = ly * math.cos(op) - zl * math.sin(op)
    z = r0 + ly * math.sin(op) + zl * math.cos(op)
    P = np.stack([v * w, y, z], -1)
    col = mixc((1.0, 0.30, 0.62), (1.0, 0.82, 0.92), U ** 1.5)
    col = col * (0.88 + 0.12 * np.cos(v * 16))[..., None]
    emit(P, normals(P), col, "lotus", Ry(a), (0, y0, 0))


def cylinder(r, y0, y1, col, mid, x=0.0, z=0.0, lean=(0.0, 0.0), cap=True):
    U, V = grid(y1 - y0, 2 * math.pi * r)
    th = V * 2 * math.pi
    h = U * (y1 - y0)
    P = np.stack([x + r * np.cos(th) + lean[0] * U, y0 + h,
                  z + r * np.sin(th) + lean[1] * U], -1)
    N = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], -1)
    emit(P, N, col, mid)
    if cap:
        U, V = grid(r, 2 * math.pi * r)
        th = V * 2 * math.pi
        P = np.stack([x + lean[0] + r * U * np.cos(th), np.full_like(U, y1),
                      z + lean[1] + r * U * np.sin(th)], -1)
        emit(P, np.broadcast_to(np.array([0, 1, 0], F32), P.shape), col, mid)


def marigold(c, r):
    """Ruffled orange flower ball tucked between the petals."""
    U, V = grid(math.pi * r, 2 * math.pi * r)
    th, ph = U * math.pi, V * 2 * math.pi
    n = np.stack([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)], -1)
    bump = np.sin(th * 9) * np.sin(ph * 11)
    P = np.asarray(c, F32) + r * n * (1 + 0.12 * bump)[..., None]
    col = mixc((1.0, 0.45, 0.04), (1.0, 0.78, 0.15), 0.5 + 0.5 * bump)
    emit(P, n, col, "flower")


def build_krathong():
    """Banana-trunk base, three tiers of folded leaves, a lotus, candle, incense."""
    global GEOM, TIPS
    _parts.clear()
    U, V = grid(0.27, 2 * math.pi * 0.9)                     # trunk-slice base
    th = V * 2 * math.pi
    P = np.stack([0.9 * np.cos(th), -0.15 + 0.27 * U, 0.9 * np.sin(th)], -1)
    N = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], -1)
    emit(P, N, mixc((0.78, 0.82, 0.58), (0.55, 0.70, 0.35),
                    0.5 + 0.5 * np.sin(th * 40)), "base")
    U, V = grid(0.9, 2 * math.pi * 0.9)
    th = V * 2 * math.pi
    P = np.stack([0.9 * U * np.cos(th), np.full_like(U, 0.12), 0.9 * U * np.sin(th)], -1)
    emit(P, np.broadcast_to(np.array([0, 1, 0], F32), P.shape), (0.85, 0.86, 0.66), "base")

    for n, r0, y0, wd, hh, tilt, fold, off in [
            (16, 0.86, -0.04, 0.19, 0.40, 0.60, 0.10, 0.0),
            (16, 0.74, 0.04, 0.16, 0.34, 0.42, 0.09, 0.5),
            (12, 0.62, 0.10, 0.14, 0.26, 0.25, 0.07, 0.25)]:
        for k in range(n):
            leaf_petal(r0, y0, 2 * math.pi * (k + off) / n, wd, hh, tilt, fold)

    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        marigold((0.50 * math.sin(a), 0.20, 0.50 * math.cos(a)), 0.065)

    for n, op, ln, wd, r0, y0, off in [
            (12, 1.10, 0.56, 0.17, 0.32, 0.12, 0.0),
            (10, 0.70, 0.48, 0.15, 0.22, 0.14, 0.5),
            (6, 0.28, 0.32, 0.11, 0.10, 0.16, 0.25)]:
        for k in range(n):
            lotus_petal(2 * math.pi * (k + off) / n, op, ln, wd, y0, r0)

    cylinder(0.06, 0.12, 0.82, (1.0, 0.95, 0.80), "candle")
    cylinder(0.008, 0.82, CANDLE_TOP[1], (0.25, 0.20, 0.15), "candle", cap=False)

    TIPS = []
    for x, z, lean in [(-0.14, -0.08, -0.07), (0.0, -0.16, 0.0), (0.14, -0.08, 0.07)]:
        cylinder(0.013, 0.14, 1.04, (0.55, 0.18, 0.08), "incense", x, z, (lean, 0), cap=False)
        tip = np.array([x + lean, 1.06, z], F32)
        U, V = grid(0.03, 0.08)
        P = tip + np.stack([0.012 * np.cos(V * 6.28), (U - 0.5) * 0.04,
                            0.012 * np.sin(V * 6.28)], -1)
        emit(P, normals(P), (1.0, 0.3, 0.1), "incense", em=True)
        TIPS.append(tip)
    TIPS = np.array(TIPS, F32)
    GEOM = tuple(np.concatenate(x) for x in zip(*_parts))


def body_pose(t):
    """Gentle bob, roll and yaw of the float on the current."""
    a = 0.30 * math.sin(W0 * t) + 0.10 * math.sin(W0 * 2 * t + 1)
    R = Rz(0.05 * math.sin(W0 * 3 * t + 1.1)) @ Rx(0.07 * math.sin(W0 * 2 * t + 0.4)) @ Ry(a)
    T = np.array([0.06 * math.sin(W0 * t + 0.3), 0.025 * math.sin(W0 * 2 * t), 0], F32)
    return R, T


def view(V):
    """World vectors -> camera space (x, up, toward viewer)."""
    return np.stack([V[..., 0], V[..., 1] * CP - V[..., 2] * SPH,
                     V[..., 1] * SPH + V[..., 2] * CP], -1)


def project(P):
    """World points (N,3) -> screen x, y in pixels, plus depth."""
    yv = P[:, 1] * CP - P[:, 2] * SPH
    zv = P[:, 1] * SPH + P[:, 2] * CP
    f = 1.0 / (1.0 - PERSP * zv)
    return KX + KS * P[:, 0] * f, KY - KS * yv * f, zv


def zbuffer(xs, ys, zv):
    """Nearest point per char cell -> (flat cell index, point index).

    Sort-free: a scatter-max finds each cell's nearest depth, then the lowest
    point index among the points at that depth wins (same as a stable sort).
    """
    col, row = np.floor(xs / CW).astype(np.int64), np.floor(ys / CH).astype(np.int64)
    idx = np.flatnonzero((col >= 0) & (col < COLS) & (row >= 0) & (row < ROWS))
    lin, z = row[idx] * COLS + col[idx], zv[idx]
    zb = np.full(ROWS * COLS, -np.inf, z.dtype)
    np.maximum.at(zb, lin, z)
    win = np.flatnonzero(z == zb[lin])
    first = np.full(ROWS * COLS, len(idx), np.int64)
    np.minimum.at(first, lin[win], win)
    cells = np.flatnonzero(first < len(idx))
    return cells, idx[first[cells]]


def krathong_pose(t):
    """Posed points above the waterline, shared by the reflection and main passes."""
    R, T = body_pose(t)
    Pw = GEOM[0] @ R.T + T
    k = np.flatnonzero(Pw[:, 1] > -0.01)                    # below waterline: hidden
    return R, Pw[k], k, CANDLE_TOP @ R.T + T + np.array([0, 0.12, 0], F32)


def render_krathong(cv, t, pose, reflect=False):
    """Z-buffer surface points into the char grid, then light each cell."""
    _, N, Cb, Mt, Em = GEOM
    R, Pw, kidx, light = pose
    if reflect:
        Pw, light = Pw * FLIP_Y, light * FLIP_Y
    xs, ys, zv = project(Pw)
    if reflect:
        xs = xs + 6 * S * np.sin(ys * 0.045 / S - W0 * 4 * t)
    cells, w = zbuffer(xs, ys, zv)
    if reflect:
        ok = cv[2].reshape(-1)[cells] == MID["water"]
        cells, w = cells[ok], w[ok]
    g = kidx[w]                                             # back to geometry indices
    Nw = N[g] @ R.T
    if reflect:
        Nw = Nw * FLIP_Y

    n = view(Nw)
    n *= np.where(n[:, 2] < 0, -1.0, 1.0)[:, None]           # two-sided: face viewer
    d = view(light) - view(Pw[w])
    dist2 = (d ** 2).sum(1)
    ndl = (n * d).sum(1) / np.sqrt(dist2)
    lam = np.clip(ndl, 0, 1) + 0.3 * np.clip(-ndl, 0, 1)    # leaves let light through
    flick = 1 + 0.12 * math.sin(W0 * 7 * t) + 0.08 * math.sin(W0 * 13 * t + 1.0)
    cand = (2.6 * flick * lam / (0.15 + 1.4 * dist2))[:, None] * CANDLE_COL
    moon = 0.30 * np.clip(n @ MOON_DIR, 0, 1)[:, None] * np.array([0.6, 0.7, 1.0], F32)
    amb = (0.30 + 0.08 * n[:, 1] + 0.22 * n[:, 2])[:, None] * np.array([0.60, 0.68, 0.70], F32)
    bounce = 0.45 * np.clip(-n[:, 1], 0.2, 1)[:, None] * np.array([0.9, 0.7, 0.25], F32)
    lit = 1 - np.exp(-1.5 * Cb[g] * (amb + cand + moon + bounce))
    em = Em[g]
    lit[em] = np.array([1.0, 0.42, 0.12], F32) * (0.9 + 0.2 * math.sin(W0 * 3 * t))
    dens = np.clip(0.34 + 0.85 * lum(lit), 0.36, 1.0)
    dens[em] = 1.0
    if reflect:
        rows = cells // COLS
        sh = (0.55 + 0.45 * np.sin(rows * 1.3 - W0 * 6 * t))[:, None]
        lit = lit * 0.40 * sh * np.array([0.75, 0.9, 1.0], F32)
        dens = dens * 0.7
    cv[0].reshape(-1)[cells] = dens
    cv[1].reshape(-1, 3)[cells] = lit
    cv[2].reshape(-1)[cells] = Mt[g]


def anchors(t):
    """Screen position of the wick tip and the three incense tips."""
    R, T = body_pose(t)
    x, y, _ = project(np.vstack([CANDLE_TOP, TIPS]) @ R.T + T)
    return float(x[0]), float(y[0]), np.stack([x[1:], y[1:]], 1)


# ------------------------------------------------------------------ scene ---
def build_scene():
    """Static masks: temple silhouettes + lantern / spark / firefly seeds."""
    global TEMPLE, EDGE, WINDOWS, WIN_H, LANTERNS, SPARKS, MOTES, FIREWORKS
    x, y = X / W, (HOR - Y) / H                  # y = height above the horizon
    m = (y >= 0) & (y < 0.012 + 0.004 * np.sin(x * 40))        # far bank
    cx, hw = 0.20, 0.17                                          # ordination hall
    m |= (np.abs(x - cx) < hw * 0.9) & (y >= 0) & (y < 0.075)
    m |= (y >= 0.07) & (y < 0.07 + 0.08 * (1 - np.abs(x - cx) / hw))
    m |= (y >= 0.10) & (y < 0.10 + 0.09 * (1 - np.abs(x - cx) / (hw * 0.65)))
    m |= (np.abs(x - cx) < 0.004) & (y >= 0.17) & (y < 0.21)
    for ex in (cx - hw * 0.95, cx + hw * 0.95):                  # chofa finials
        m |= (np.abs(x - ex) < 0.006) & (y >= 0.07) & (y < 0.10)
    win = ((y > 0.022) & (y < 0.045) & (np.abs(x - cx) < hw * 0.8)
           & ((x - cx + hw) % 0.04 < 0.017))
    for c, k in [(0.42, 1.0), (0.555, 0.7)]:                     # chedis
        m |= (np.abs(x - c) < 0.05 * k) & (y >= 0) & (y < 0.03 * k)
        m |= (((x - c) / (0.036 * k)) ** 2 + ((y - 0.06 * k) / (0.035 * k)) ** 2 < 1) & (y > 0.02)
        s = (y - 0.08 * k) / (0.22 * k)
        m |= (s >= 0) & (s < 1) & (np.abs(x - c) < 0.013 * k * (1 - s) + 0.002)
    m |= (y >= 0) & (y < 0.10 * (1 - np.abs(x - 0.655) / 0.07))  # small vihara roof
    s = np.clip((y - 0.03) / 0.33, -1, 1)                         # prang
    m |= (np.abs(x - 0.82) < 0.09) & (y >= 0) & (y < 0.03)
    m |= (s >= 0) & (s < 1) & (np.abs(x - 0.82) < 0.065 * (1 - s) ** 0.6 * (1 + 0.08 * np.sin(s * 40)))
    m |= (np.abs(x - 0.82) < 0.003) & (y >= 0.35) & (y < 0.40)
    win |= (np.abs(x - 0.82) < 0.012) & (y > 0.05) & (y < 0.075)
    TEMPLE, WINDOWS = m, win & m
    EDGE = m & ~(np.roll(m, 1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    WIN_H = hash2(RR, CC, 5.0)

    build_static()
    rng = np.random.default_rng(7)
    LANTERNS = [dict(x=rng.uniform(0.04, 0.96), ph=rng.uniform(), sz=rng.uniform(0.55, 1.3),
                     sw=rng.uniform(4, 22), k=int(rng.integers(2, 6))) for _ in range(16)]
    SPARKS = dict(ph=rng.uniform(size=36), k=rng.integers(1, 4, 36),
                  a=rng.uniform(1, 4, 36), dr=rng.uniform(-70, 70, 36), rise=rng.uniform(160, 380, 36))
    MOTES = dict(x=rng.uniform(0.03, 0.97, 50), y=rng.uniform(HOR / H - 0.15, 0.98, 50),
                 p=rng.uniform(0, 6.28, (3, 50)), k=rng.integers(1, 4, (3, 50)))
    FIREWORKS = []
    for i in range(FW_COUNT):                        # evenly spaced shells, varied looks
        style = ["peony", "ring", "willow", "peony", "ring", "peony"][i % 6]
        k = 120 if style == "willow" else 96
        c1, c2 = FW_COLOURS[i % len(FW_COLOURS)]
        r = rng.uniform(170, 240)
        while True:                                  # keep bursts clear of the moon
            x, y = rng.uniform(0.15, 0.85), rng.uniform(0.12, 0.28)
            if math.hypot(x * W - MX, y * H - MY) > r * S + MR * 1.2:
                break
        FIREWORKS.append(dict(
            t0=i * LOOP / FW_COUNT + rng.uniform(-0.25, 0.25), x=x, y=y, r=r,
            c1=c1, c2=c2, style=style,
            ang=rng.uniform(0, 2 * math.pi, k),
            spd=np.full(k, 1.0) + rng.normal(0, 0.03, k) if style == "ring" else rng.uniform(0.35, 1.0, k) ** 0.5,
            ph=rng.uniform(0, 6.28, k), g=110.0 if style == "willow" else 45.0,
            life=3.0 if style == "willow" else FW_LIFE))


def build_static():
    """Everything that depends only on the grid, computed once per size, not per frame."""
    global SKY, TMP, WAT
    m = Y < HOR                                                  # sky, stars, moon
    g = np.clip(Y / HOR, 0, 1)
    dm = np.hypot(X - MX, Y - MY)
    mg = np.exp(-np.maximum(dm - MR, 0) / (MR * 0.9))
    col = mixc((0.020, 0.025, 0.060), (0.16, 0.08, 0.05), g ** 3)
    col = col + (mg * 0.25)[..., None] * np.array([0.75, 0.72, 0.55], F32)
    hv, br = hash2(RR, CC, 1.7), hash2(RR, CC, 3.1)
    star = m & (hv > 0.972) & (Y < HOR - 0.03 * H) & (dm > MR * 1.3)
    scol = mixc((0.75, 0.80, 1.0), (1.0, 0.9, 0.7), hash2(RR, CC, 9.2))
    disk = dm < MR
    limb = np.sqrt(np.clip(1 - (dm / MR) ** 2, 0, 1))
    maria = 0.5 + 0.5 * np.sin(X * 0.045 / S + 1.3) * np.sin(Y * 0.05 / S + 0.4)
    maria = np.clip(maria + 0.3 * np.sin(X * 0.11 / S - Y * 0.07 / S), 0, 1)
    shade = 0.70 + 0.30 * limb - 0.18 * maria
    SKY = dict(i=np.flatnonzero(m), d=(0.3 + 0.35 * mg)[m], col=col[m],
               si=np.flatnonzero(star), hv=hv[star], br=br[star], scol=scol[star],
               mi=np.flatnonzero(disk), md=(0.55 + 0.45 * limb - 0.2 * maria)[disk],
               mcol=(np.array([1.0, 0.96, 0.82], F32) * shade[..., None])[disk])

    TMP = dict(i=np.flatnonzero(TEMPLE), e=np.flatnonzero(EDGE), w=np.flatnonzero(WINDOWS),
               wh=WIN_H[WINDOWS])                                # temple cells only

    r = int(np.argmax(Y[:, 0] >= HOR)) if (Y[:, 0] >= HOR).any() else ROWS
    Xw, Yw, RRw, CCw = X[r:], Y[r:], RR[r:], CC[r:]               # water rows only
    q = np.clip((Yw - HOR) / (H - HOR), 0, 1)
    z = 1.0 / (q + 0.07)
    u = (Xw - KX) / ((60 + q * 700) * S)
    r0 = int(HOR // CH)
    dyr, mr = Yw - KY, Yw - HOR
    rho = np.hypot(Xw - KX, (Yw - KY - 20 * S) * 2.6)
    WAT = dict(
        r=r, X=Xw, CC=CCw, q=q, q46=(0.4 + 0.6 * q), c9=np.clip(1 - q * 9, 0, 1) * 0.7, q12=(0.12 * (1 - q)),
        a1=u * 6 + z * 1.4, a2=u * 11 - z * 2.3, a3=u * 3 + z * 3.9, a4=u * 17 + z * 6.1,
        rm=np.clip(2 * r0 - RRw - 1, 0, ROWS - 1).astype(int), fe=np.exp(-(RRw - r0) / (0.10 * H / CH)),
        wid=24 * S + np.clip(dyr, 0, None) * 0.10, dyrm=(dyr > -40 * S),
        dyre=np.exp(-np.clip(dyr, 0, None) / (320 * S)), dyr12=dyr * 0.12 / S,
        mw=20 * S + mr * 0.10, moe=np.exp(-mr / (420 * S)), mr10=mr * 0.10 / S,
        pool=np.exp(-(((Xw - KX) / (300 * S)) ** 2 + ((Yw - KY - 15 * S) / (120 * S)) ** 2) * 1.4),
        rho55=rho * 0.055 / S, rhoe=np.exp(-rho / (420 * S)))


def sky_layer(cv, t):
    k = SKY
    putflat(cv, k["i"], k["d"], k["col"], "sky")
    br = k["br"]
    tw = 0.55 + 0.45 * np.sin(W0 * (1 + np.floor(br * 4)) * t + k["hv"] * 628)
    putflat(cv, k["si"], 0.45 + 0.5 * br * tw, k["scol"] * (0.30 + 0.70 * tw * br)[..., None], "star")
    putflat(cv, k["mi"], k["md"], k["mcol"], "moon")             # the full moon


def lantern_layer(cv, t):
    """Khom loi: paper lanterns rising, near ones fast and big, far ones fading out."""
    for L in LANTERNS:
        near = (L["sz"] - 0.55) / 0.75
        top = HOR - (0.25 + 0.85 * near) * HOR
        bot = HOR - 0.03 * H
        p = (L["ph"] + t / LOOP) % 1.0
        ly = bot - p * (bot - top)
        lx = L["x"] * W + L["sw"] * S * math.sin(W0 * t + L["ph"] * 6.28)
        fade = min(1, p * 6) * min(1, (1 - p) * (4 + 6 * near))
        flick = 0.85 + 0.15 * math.sin(W0 * L["k"] * t + L["ph"] * 20)
        lw, lh = 17 * S * L["sz"], 22 * S * L["sz"]
        rs, cs = window(lx - 3 * lw, lx + 3 * lw, ly - 3 * lh, ly + 3 * lh)  # glow < floor beyond
        sc = sub(cv, rs, cs)
        if sc[0].size == 0:
            continue
        dx, dy = (X[rs, cs] - lx) / lw, (Y[rs, cs] - ly) / lh
        inside = (np.abs(dy) < 1) & (np.abs(dx) < 0.85 - 0.22 * dy)
        b = (dy + 1) / 2
        col = mixc((0.95, 0.38, 0.08), (1.0, 0.80, 0.38), b) * ((0.55 + 0.45 * b) * fade * flick)[..., None]
        put(sc, inside & (fade > 0.05), (0.45 + 0.55 * b) * fade, col, "lantern")
        g = 0.36 * np.exp(-(dx ** 2 + dy ** 2) / 3.5) * fade * flick
        putmax(sc, g, np.array([1.0, 0.55, 0.18], F32) * g[..., None], "glow")


FW_COUNT, FW_RISE, FW_LIFE = 6, 0.9, 2.3
FW_COLOURS = [((1.0, 0.80, 0.35), (1.0, 0.45, 0.10)), ((1.0, 0.30, 0.30), (1.0, 0.75, 0.55)),
              ((0.40, 1.0, 0.65), (0.20, 0.70, 1.0)), ((1.0, 0.85, 0.55), (0.9, 0.55, 0.15)),
              ((1.0, 0.45, 0.90), (0.65, 0.50, 1.0)), ((0.35, 0.85, 1.0), (0.95, 0.95, 1.0))]


def firework_layer(cv, t):
    """Shells rise from behind the temples, burst, sag under gravity and fade."""
    flash = np.zeros(3, F32)
    for fw in FIREWORKS:
        a = (t - fw["t0"]) % LOOP
        cx, cy, R = fw["x"] * W, fw["y"] * H, fw["r"] * S
        c1, c2 = np.array(fw["c1"], F32), np.array(fw["c2"], F32)
        if a < FW_RISE:                                          # rocket + short trail
            e = np.maximum(a - np.arange(7) * 0.035, 0) / FW_RISE
            e = 1 - (1 - e) ** 2
            x = cx + 6 * S * np.sin(e * 9 + fw["t0"]) * (1 - e)
            y = HOR + (cy - HOR) * e
            b = np.linspace(1, 0.15, 7) * (a > np.arange(7) * 0.035)
            putcells(cv, x, y, np.where(b > 0, 0.35 + 0.6 * b, 0), np.outer(b, [1.0, 0.75, 0.40]), "spark")
            continue
        b = a - FW_RISE
        life = b / fw["life"]
        if life >= 1:
            continue
        f = math.exp(-b * 5)                                     # burst flash
        flash += c1 * f
        rs, cs = window(cx - 5.2 * R, cx + 5.2 * R, cy - 5.2 * R, cy + 5.2 * R)  # flash < 1e-5 beyond
        Xs, Ys = X[rs, cs], Y[rs, cs]
        dd = np.hypot(Xs - cx, Ys - cy) / (R * 1.6)
        sky = Ys < HOR
        cv[1][rs, cs][sky] += (np.exp(-dd ** 2) * f * 0.35)[sky][:, None] * c1
        tw = np.where(life > 0.55, np.sin(W0 * 37 * t + fw["ph"]) > -0.2, True)
        col = c1 * (1 - life) + c2 * life + 0.6 * f
        for j in range(5):                                       # head + streak
            bb = max(b - j * 0.05, 0.0)
            dist = R * fw["spd"] * (1 - math.exp(-2.8 * bb))
            x = cx + np.cos(fw["ang"]) * dist
            y = cy + np.sin(fw["ang"]) * dist * 0.92 + fw["g"] * S * bb * bb
            br = (1 - life) ** 1.2 * (1 - 0.2 * j) * tw * (1 + 1.5 * f)
            putcells(cv, x, y, np.clip(0.32 + 0.6 * br, 0, 1), np.outer(br, col), "firework")
    return flash

DRAGON_ON = True


def stroke(cv, P, width, d, col, mid, force=False):
    """Thick polyline through dense points P (n,2), splatted cell by cell."""
    tg = np.gradient(P, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
    width = np.broadcast_to(np.asarray(width, F32), (len(P),))
    col = np.broadcast_to(np.asarray(col, F32), (len(P), 3))
    offs = np.linspace(-0.5, 0.5, max(1, int(width.max() / (0.45 * min(CW, CH))) + 1))
    if np.ndim(d) or d <= 0.05:                          # general case: one pass at a time
        d = np.broadcast_to(np.asarray(d, F32), (len(P),))
        for o in offs:
            Q = P + nr * (o * width)[:, None]
            putcells(cv, Q[:, 0], Q[:, 1], d, col, mid, force)
        return
    # Constant density: every pass at once, resolved exactly as the pass-by-pass
    # splat would: forced -> the last write wins; otherwise the first pass to touch
    # a cell decides (later passes can't beat an equal density), and within that
    # pass the last point wins.
    Q = (P[None] + nr[None] * (offs[:, None, None] * width[None, :, None])).reshape(-1, 2)
    c, r = (Q[:, 0] // CW).astype(int), (Q[:, 1] // CH).astype(int)
    ok = np.flatnonzero((c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS))
    lin, pid = r[ok] * COLS + c[ok], ok // len(P)
    last = np.full(ROWS * COLS, -1, np.int64)
    if force:
        np.maximum.at(last, lin, np.arange(len(lin)))
    else:
        first = np.full(ROWS * COLS, len(offs), np.int64)
        np.minimum.at(first, lin, pid)
        cand = np.flatnonzero(pid == first[lin])
        np.maximum.at(last, lin[cand], cand)
    cells = np.flatnonzero(last >= 0)
    w = last[cells]
    if not force:
        keep = d > cv[0].reshape(-1)[cells]
        cells, w = cells[keep], w[keep]
    putflat(cv, cells, d, np.tile(col, (len(offs), 1))[ok[w]], mid)


def curve(f, length):
    """Sample a parametric curve f(k), k in [0,1], about every half cell."""
    k = np.linspace(0, 1, int(length / (0.4 * min(CW, CH))) + 2, dtype=F32)
    return np.stack(f(k), 1)


def dragon_layer(cv, t, flash=None):
    """A red Chinese dragon swimming through the sky after its flaming pearl."""
    R0, L, DN = 48 * S, 1.05 * W, 360
    s = np.linspace(0, 1, DN, dtype=F32)
    hx = -0.4 * W + 2.65 * W * (t / LOOP)           # off-screen at both ends of the loop
    xs = hx - s * L
    ys = (HOR - 0.30 * H + 0.06 * H * np.sin(2 * np.pi * xs / (0.62 * W) + W0 * 2 * t)
          + 0.01 * H * math.sin(W0 * 3 * t))
    if xs[-1] - 4 * R0 > W or hx + 6 * R0 < 0:
        return
    tx, ty = -np.gradient(xs), -np.gradient(ys)      # tangent, pointing to the head
    nn = np.hypot(tx, ty)
    tx, ty = tx / nn, ty / nn
    nx, ny = -ty, tx                                  # normal toward the belly
    r = R0 * (0.85 + 0.25 * np.exp(-((s - 0.22) / 0.12) ** 2)) * (1 - 0.82 * s ** 1.7)
    r *= 0.62 + 0.38 * np.clip((s - 0.02) / 0.08, 0, 1)            # slim neck behind the head
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    tint = np.zeros(3, F32) if flash is None else 0.25 * flash
    hy = float(ys[0])
    T0, N0 = np.array([tx[0], ty[0]]), np.array([nx[0], ny[0]])

    def at(i, v):                                     # point on the body surface
        return np.array([xs[i] + nx[i] * r[i] * v, ys[i] + ny[i] * r[i] * v])

    HS = 1.35 * R0                                    # head unit, bigger than the body

    def head(u, w):                                   # head-local (units of HS) -> screen
        return hx + (u * T0[0] + w * N0[0]) * HS, hy + (u * T0[1] + w * N0[1]) * HS

    # legs: thigh swings back, forearm reaches forward, three claws
    for j, (sl, ph) in enumerate([(0.17, 0.0), (0.27, 3.1), (0.56, 1.6), (0.66, 4.7)]):
        i = int(sl * (DN - 1))
        T, N, k = np.array([tx[i], ty[i]]), np.array([nx[i], ny[i]]), r[i] / R0
        sw = 0.6 * math.sin(W0 * 4 * t + ph)
        a1, a2 = 0.45 + sw, 0.55 - 0.4 * sw
        base = at(i, 0.6)
        knee = base + (N * math.cos(a1) - T * math.sin(a1)) * 1.25 * R0 * k
        d2 = N * math.cos(a2) + T * math.sin(a2)
        paw = knee + d2 * 0.95 * R0 * k
        col = np.array([0.80, 0.10, 0.06], F32) + tint
        stroke(cv, np.linspace(base, knee, 12), 0.42 * R0 * k, 0.75, col, "dragon")
        stroke(cv, np.linspace(knee, paw, 10), 0.30 * R0 * k, 0.75, col, "dragon")
        for ca in (-0.7, 0.0, 0.7):
            cd = d2 * math.cos(ca) + T * math.sin(ca)
            stroke(cv, np.linspace(paw, paw + cd * 0.4 * R0 * k, 5), 0.1 * R0, 0.85,
                   (1.0, 0.92, 0.65), "dgold")

    # flowing mane along the neck and a flame tuft on the tail
    for j in range(6):
        i = int((0.015 + 0.016 * j) * (DN - 1))
        T, N = np.array([tx[i], ty[i]]), np.array([nx[i], ny[i]])
        b, ln, ph = at(i, -1.0), (1.7 - 0.1 * j) * R0, j * 0.9
        P = curve(lambda k: tuple(b[q] + (-0.75 * T[q] - 0.65 * N[q]) * k * ln
                                  + N[q] * 0.25 * R0 * k * np.sin(k * 6 - W0 * 6 * t + ph)
                                  for q in (0, 1)), ln)
        kk = np.linspace(0, 1, len(P))
        stroke(cv, P, 0.14 * R0 * (1 - 0.6 * kk), 0.7, mixc((1.0, 0.42, 0.06), (1.0, 0.78, 0.30), kk), "dgold")
    T, N, b = np.array([tx[-1], ty[-1]]), np.array([nx[-1], ny[-1]]), np.array([xs[-1], ys[-1]])
    for j, ang in enumerate(np.linspace(-0.7, 0.7, 6)):
        dv = -T * math.cos(ang) + N * math.sin(ang)
        P = curve(lambda k: tuple(b[q] + dv[q] * k * 1.9 * R0
                                  + N[q] * 0.3 * R0 * k * np.sin(k * 5 - W0 * 7 * t + j) for q in (0, 1)), 1.9 * R0)
        kk = np.linspace(0, 1, len(P))
        stroke(cv, P, 0.30 * R0 * (1 - 0.8 * kk), 0.8, mixc((1.0, 0.35, 0.05), (1.0, 0.85, 0.35), kk), "dgold")

    # scaled body, gold belly plates, saw-tooth dorsal fins
    x0, x1, y0, y1 = xs[-1] - 3 * R0, hx + 3 * R0, ys.min() - 3 * R0, ys.max() + 3 * R0
    rs, cs = window(x0, x1, y0, y1)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    sel = (Xs > x0) & (Xs < x1) & (Ys > y0) & (Ys < y1)
    idx = (RR[rs, cs] * COLS + CC[rs, cs]).astype(np.int64)[sel]
    if len(idx):
        # Only samples within 1.75 * max radius horizontally can be drawn (body q<1,
        # fins q<1.75), and xs is evenly spaced, so search a narrow band, not all DN.
        cx_, cy_ = Xs[sel], Ys[sel]
        ds = L / (DN - 1)
        m = int(math.ceil(1.75 * float(r.max()) / ds)) + 2
        j = np.clip(np.rint((hx - cx_) / ds).astype(np.int64)[:, None] + np.arange(-m, m + 1), 0, DN - 1)
        dx, dy = cx_[:, None] - xs[j], cy_[:, None] - ys[j]
        q = np.hypot(dx, dy) / r[j]
        jj = q.argmin(1)
        ar = np.arange(len(idx))
        i, qi = j[ar, jj], q[ar, jj]
        v = (dx[ar, jj] * nx[i] + dy[ar, jj] * ny[i]) / r[i]
        a = arc[i]
        sh = np.clip(0.45 + 0.55 * np.sqrt(np.clip(1 - v * v, 0, 1)) - 0.2 * v, 0, 1)
        edge = (((a / (0.45 * R0) + 0.5 * np.floor((v + 1) * 2.5)) % 1.0) < 0.3).astype(F32)
        body = qi < 1
        col = mixc((0.42, 0.02, 0.03), (1.0, 0.17, 0.08), sh) * (1 - 0.25 * edge)[:, None] + tint
        putflat(cv, idx[body], (0.62 + 0.3 * sh - 0.12 * edge)[body], col[body], "dragon")
        belly = body & (v > 0.42) & (s[i] < 0.97)
        plate = np.where((a / (0.35 * R0)) % 1.0 < 0.25, 0.6, 1.0)
        bcol = np.array([1.0, 0.74, 0.28], F32) * ((0.5 + 0.5 * sh) * plate)[:, None] + tint
        putflat(cv, idx[belly], 0.72 * plate[belly], bcol[belly], "dgold")
        e, f = qi - 1, (a / (0.9 * R0)) % 1.0
        fin = (qi >= 1) & (v < -0.9) & (e < 0.75 * (1 - f)) & (s[i] > 0.05) & (s[i] < 0.9)
        fcol = np.array([1.0, 0.55, 0.12], F32) * (0.65 + 0.35 * (1 - e))[:, None] + tint
        putflat(cv, idx[fin], 0.62, fcol[fin], "dgold")

    # head: skull, snout, opening jaw, brow, eye, beard
    rs, cs = window(hx - 3.4 * HS, hx + 3.4 * HS, hy - 3.4 * HS, hy + 3.4 * HS)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    hm = np.hypot(Xs - hx, Ys - hy) < 3.4 * HS
    hsel = (RR[rs, cs] * COLS + CC[rs, cs]).astype(np.int64)[hm]
    if len(hsel):
        px, py = Xs[hm] - hx, Ys[hm] - hy
        u = (px * T0[0] + py * T0[1]) / HS
        w = (px * N0[0] + py * N0[1]) / HS
        mo = 0.25 + 0.30 * (0.5 + 0.5 * math.sin(W0 * 4 * t))
        skull = ((u - 0.2) / 1.5) ** 2 + ((w + 0.05) / 1.0) ** 2 < 1
        snout = (u > 0.8) & (u < 2.5) & (w > -0.62 + 0.18 * (u - 0.8)) & (w < 0.25)
        jaw_top = 0.25 + mo * np.clip(u - 0.6, 0, None)
        jaw = (u > 0.5) & (u < 2.2) & (w > jaw_top) & (w < jaw_top + 0.38 * (1 - (u - 0.5) / 2.0))
        mouth = (u > 0.9) & (u < 2.2) & (w >= 0.25) & (w <= jaw_top)
        nose = ((u - 2.35) / 0.3) ** 2 + ((w + 0.35) / 0.22) ** 2 < 1
        brow = (u > 0.15) & (u < 1.2) & (w > -0.95 + 0.1 * u) & (w < -0.62 + 0.05 * u)
        eye = ((u - 0.7) / 0.36) ** 2 + ((w + 0.36) / 0.28) ** 2 < 1
        beard = (u > -0.7) & (u < 0.7) & (w > 0.7) & (w < 0.75 + 0.7 * (1 - ((u + 0.7) / 0.35) % 1.0))
        hs = np.clip(0.95 - 0.3 * (w + 1) / 2, 0, 1)
        red = np.array([1.0, 0.18, 0.08], F32) * hs[:, None] + tint
        for m, d, c, mid in [
                (beard, 0.7, np.array([1.0, 0.70, 0.25], F32) + tint, "dgold"),
                (skull | snout | jaw | nose, 0.6 + 0.3 * hs, red, "dragon"),
                (mouth, 0.45, np.array([0.30, 0.0, 0.02], F32), "dragon"),
                (brow, 0.8, np.array([1.0, 0.80, 0.30], F32) + tint, "dgold"),
                (eye, 1.0, np.array([1.0, 0.97, 0.55], F32), "pearl")]:
            if m.any():
                putflat(cv, hsel[m], np.broadcast_to(d, m.shape)[m],
                        np.broadcast_to(c, m.shape + (3,))[m], mid)
        fu = np.arange(1.0, 2.3, 0.35)                                # teeth on both jaws
        for fw_ in (0.28, None):
            fx_, fy_ = head(fu, 0.28 if fw_ else 0.22 + mo * (fu - 0.6))
            putcells(cv, fx_, fy_, np.full(len(fu), 0.95), np.tile([1.0, 1.0, 0.9], (len(fu), 1)), "dgold", True)

    # antlers and long whiskers in front of the head
    for bu, bw, sp in [(-0.1, -0.85, 1.0), (0.35, -0.9, 0.85)]:
        P = curve(lambda k: head(bu - (0.3 * k + 0.5 * k * k) * sp, bw - (1.3 * k - 0.2 * k * k) * sp), 1.6 * HS)
        kk = np.linspace(0, 1, len(P))
        stroke(cv, P, 0.2 * HS * (1 - 0.7 * kk), 0.75, (1.0, 0.80, 0.40), "dgold", True)
        jb = P[len(P) // 2]
        P2 = curve(lambda k: (jb[0] + (0.4 * T0[0] - N0[0]) * k * 0.6 * HS,
                              jb[1] + (0.4 * T0[1] - N0[1]) * k * 0.6 * HS), 0.6 * HS)
        stroke(cv, P2, 0.12 * HS, 0.85, (0.95, 0.82, 0.50), "dgold", True)
    for sgn, w0 in [(-1, 0.0), (1, 0.15)]:
        P = curve(lambda k: head(2.4 - k * 3.2, w0 + sgn * (0.3 + 1.0 * k)
                                 + 0.3 * k * np.sin(k * 6 - W0 * 8 * t + sgn)), 3.6 * HS)
        stroke(cv, P, 0.09 * R0, 0.7, (1.0, 0.85, 0.45), "whisker")

    # the flaming pearl it chases
    pxc, pyc = head(3.6, -0.3 + 0.3 * math.sin(W0 * 5 * t))
    pr = 4.2 * 0.42 * R0                                       # glow < its floor beyond
    rs, cs = window(pxc - pr, pxc + pr, pyc - pr, pyc + pr)
    cv = sub(cv, rs, cs)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    dd = np.hypot(Xs - pxc, Ys - pyc) / (0.42 * R0)
    ang = np.arctan2(Ys - pyc, Xs - pxc)
    spike = 0.5 + 0.5 * np.sin(ang * 7 + W0 * 6 * t)
    flame = (dd >= 1) & (dd < 1.4 + 0.9 * spike)
    put(cv, flame, 0.55, np.array([1.0, 0.50, 0.10], F32) * (1.2 - 0.4 * (dd - 1))[..., None].clip(0.3, 1), "dgold")
    pearl = dd < 1
    pc = mixc((0.75, 0.85, 1.0), (1.0, 1.0, 1.0), np.clip(1 - dd, 0, 1))
    put(cv, pearl, 0.7 + 0.3 * (1 - dd), pc, "pearl")
    g = 0.45 * np.exp(-((dd / 3.5) ** 2))
    putmax(cv, g, np.array([1.0, 0.75, 0.40], F32) * g[..., None], "glow", 0.12)


def temple_layer(cv, t, fx, fy):
    k = TMP
    Xf, Yf = X.reshape(-1), Y.reshape(-1)
    for idx, base, add, d0, d1 in [
            (k["i"], (0.11, 0.055, 0.025), (0.30, 0.14, 0.04), 0.36, 0.25),
            (k["e"], (0.40, 0.19, 0.06), (0.45, 0.22, 0.06), 0.62, 0.2)]:
        warm = np.exp(-np.hypot(Xf[idx] - fx, Yf[idx] - fy) / (380 * S))[..., None]
        col = np.array(base, F32) + warm * np.array(add, F32)
        putflat(cv, idx, d0 + d1 * warm[..., 0], col, "temple")
    fl = 0.75 + 0.25 * np.sin(W0 * (2 + np.floor(k["wh"] * 4)) * t + k["wh"] * 30)
    putflat(cv, k["w"], 0.95, np.array([1.0, 0.74, 0.30], F32) * fl[..., None], "window")


def water_layer(cv, t, fx, flash=None):
    """Water per cell: summed waves, perspective, and light reflections."""
    k = WAT                                                     # water rows, static fields
    X_ = k["X"]
    h = (np.sin(k["a1"] - W0 * 3 * t) + 0.7 * np.sin(k["a2"] + W0 * 4 * t)
         + 0.5 * np.sin(k["a3"] - W0 * 2 * t + 2.1)
         + 0.3 * np.sin(k["a4"] + W0 * 5 * t))
    wave = 0.5 + 0.5 * np.tanh(h * 0.9)
    dens = 0.10 + 0.30 * wave * k["q46"]
    col = mixc((0.04, 0.28, 0.30), (0.12, 0.55, 0.52), wave)
    col = mixc(col, (0.55, 0.30, 0.10), k["c9"])
    col = col * (0.16 + 0.75 * dens)[..., None]
    if flash is not None:                                       # fireworks light the river
        col = col + flash * k["q12"][..., None]

    rm = k["rm"]                                                # mirror the skyline
    cm = np.clip(k["CC"] + np.round(h * 0.9), 0, COLS - 1).astype(int)
    fade = k["fe"] * (0.55 + 0.45 * wave)
    refl = cv[1][rm, cm] * (0.55 * fade)[..., None]
    col = col + refl
    dens = np.maximum(dens, np.where(cv[2][rm, cm] == MID["sky"], 0, cv[0][rm, cm] * fade * 0.7))

    fr = (np.exp(-((X_ - fx + 5 * h * S) / k["wid"]) ** 2) * k["dyrm"]  # candle streak
          * k["dyre"]
          * (0.5 + 0.5 * np.sin(k["dyr12"] + h * 2.2 - W0 * 6 * t)) ** 1.5)
    fr_f = 0.85 + 0.15 * math.sin(W0 * 7 * t)
    mo = (np.exp(-((X_ - MX + 6 * h * S) / k["mw"]) ** 2) * k["moe"]    # moon path
          * (0.5 + 0.5 * np.sin(k["mr10"] + h * 3 - W0 * 5 * t)) ** 1.5)
    pool = k["pool"] * 0.75 * (0.35 + 0.65 * wave)
    ring = np.clip(np.cos(k["rho55"] - W0 * 5 * t), 0, 1) ** 6 * k["rhoe"]  # ripples
    fields = np.stack([dens, 0.9 * fr * fr_f, 0.8 * mo, pool, 0.6 * ring])
    d = fields.max(0)
    best = fields.argmax(0)
    hl = np.array([[0, 0, 0], [1.0, 0.75, 0.25], [1.0, 0.95, 0.72],
                   [1.0, 0.62, 0.20], [1.0, 0.80, 0.40]], F32)[best]
    colf = np.where((best == 0)[..., None], col, hl * (0.2 + 0.9 * d)[..., None] + 0.5 * refl)
    r = k["r"]
    cv[0][r:], cv[1][r:], cv[2][r:] = d, colf, MID["water"]


def flame_layer(cv, t, fx, fy):
    """Teardrop candle flame: multi-frequency flicker, white core to red tip."""
    fh, fw = 150.0 * S, 32.0 * S
    flick = 1 + 0.10 * math.sin(W0 * 7 * t) + 0.06 * math.sin(W0 * 13 * t + 1.3)
    sway = (9 * math.sin(W0 * 5 * t + 0.7) + 5 * math.sin(W0 * 11 * t)) * S
    rs, cs = window(fx - 130 * S, fx + 130 * S, fy - 1.25 * fh, fy + 60 * S)  # flame + halo
    cv = sub(cv, rs, cs)
    Xs, Ys = X[rs, cs], Y[rs, cs]
    s = (fy - Ys) / (fh * flick)
    sc = np.clip(s, 0, 1)
    xo = sway * sc ** 2 + 3 * S * np.sin(s * 9 - W0 * 9 * t) * sc
    wid = fw * np.clip(np.sin(np.pi * sc ** 0.6), 0, 1) ** 0.8 * (1 + 0.08 * np.sin(W0 * 17 * t + s * 5))
    q = np.abs(Xs - fx - xo) / np.maximum(wid, 1e-3)
    inside = (s > -0.06) & (s < 1.0) & (q < 1.0)
    core = (q < 0.45) & (s < 0.55)
    col = mixc((1.0, 0.40, 0.08), (1.0, 0.82, 0.25), np.clip(1.25 - s - q * 0.6, 0, 1))
    col = np.where(core[..., None], np.array([1.0, 0.97, 0.78], F32), col)
    put(cv, inside, np.where(core, 1.0, 0.55 + 0.4 * (1 - sc)), col * 1.15, "flame")
    dh = np.hypot((Xs - fx - xo * 0.5) / (95 * S), (Ys - (fy - 40 * S)) / (120 * S))
    halo = 0.42 * np.exp(-dh * dh * 2.2) * (hash2(RR[rs, cs], CC[rs, cs], 7.0) > 0.35)
    putmax(cv, halo, np.array([1.0, 0.62, 0.20], F32) * (halo * 1.8)[..., None], "spark", 0.1)


def smoke_layer(cv, t, tips):
    """Thin incense smoke curling upward from each glowing tip."""
    L = 230 * S
    for i, (tx, ty) in enumerate(tips):
        rs, cs = window(tx - 12 * S - CW, tx + 12 * S + CW, ty - L, ty)
        sc = sub(cv, rs, cs)
        Xs, Ys = X[rs, cs], Y[rs, cs]
        hgt = (ty - Ys) / L
        xc = tx + 12 * S * hgt * np.sin(hgt * 6 - W0 * 2 * t + i * 2.0)
        m = (hgt > 0.03) & (hgt < 1) & (np.abs(Xs - xc) < CW * 0.7)
        d = np.where(m, 0.30 * (1 - hgt) * (0.6 + 0.4 * np.sin(hgt * 20 - W0 * 6 * t + i)), 0)
        putmax(sc, d, np.array([0.55, 0.55, 0.62], F32) * (d * 1.6)[..., None], "smoke", 0.05)


def particle_layer(cv, t, fx, fy):
    """Embers rising off the flame, fireflies drifting over the river."""
    sp = SPARKS
    p = (sp["ph"] + sp["k"] * t / LOOP) % 1.0
    x = fx + np.sin(p * math.pi * sp["a"] + sp["ph"] * 20) * 18 * S * p + sp["dr"] * S * p
    y = fy - 30 * S - p * sp["rise"] * S
    b = (1 - p) ** 1.5
    col = np.stack([np.ones_like(b), 0.45 + 0.35 * b, 0.12 + 0.1 * b], 1) * b[:, None]
    putcells(cv, x, y, 0.3 + 0.6 * b, col, "spark")
    mo = MOTES
    x = W * (mo["x"] + 0.02 * np.sin(W0 * mo["k"][0] * t + mo["p"][0]))
    y = H * (mo["y"] + 0.01 * np.sin(W0 * mo["k"][1] * t + mo["p"][1]))
    blink = (0.5 + 0.5 * np.sin(W0 * mo["k"][2] * t + mo["p"][2])) ** 3
    col = np.array([1.0, 0.80, 0.35], F32) * blink[:, None]
    putcells(cv, x, y, 0.25 + 0.6 * blink, col, "spark")


def render_cells(t):
    cv = canvas()
    fx, fy, tips = anchors(t)
    sky_layer(cv, t)
    lantern_layer(cv, t)
    flash = firework_layer(cv, t) if FIREWORKS_ON else None
    temple_layer(cv, t, fx, fy)
    if DRAGON_ON:
        dragon_layer(cv, t, flash)
    water_layer(cv, t, fx, flash)
    pose = krathong_pose(t)
    render_krathong(cv, t, pose, reflect=True)
    render_krathong(cv, t, pose)
    smoke_layer(cv, t, tips)
    flame_layer(cv, t, fx, fy)
    particle_layer(cv, t, fx, fy)
    return cv


# ------------------------------------------------------------------ output ---
FONT_PATHS = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf",
              "C:/Windows/Fonts/consola.ttf"]


def mono_font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def build_atlas():
    """One grayscale tile per glyph, sized to the char cell."""
    global ATLAS
    size = CH
    while size > 6 and mono_font(size).getlength("@") > CW * 1.04:
        size -= 1
    font = mono_font(size)
    tiles = []
    for ch in CHARS:
        im = Image.new("L", (CW, CH))
        ImageDraw.Draw(im).text((CW / 2, CH / 2), ch, font=font, fill=255, anchor="mm")
        tiles.append(np.asarray(im, F32) / 255)
    ATLAS = np.stack(tiles)


def glyph_index(cv):
    lv = np.clip((cv[0] * LV).astype(int), 0, LV - 1)
    return LUT[cv[2], lv]


def to8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def compose_image(cv):
    """Draw every cell's glyph in its colour, then add bloom."""
    tiles = ATLAS[glyph_index(cv)]                                   # R,C,CH,CW
    img = (tiles[..., None] * cv[1][:, :, None, None, :]).transpose(0, 2, 1, 3, 4).reshape(H, W, 3)
    src = cv[1] * np.clip(lum(cv[1]) * 1.6, 0, 1)[..., None]
    glow = Image.fromarray(to8(src)).resize((W, H), Image.BILINEAR).filter(
        ImageFilter.GaussianBlur(2.2 * CW))
    bloom = Image.fromarray(to8(img)).resize((W // 2, H // 2), Image.BILINEAR).filter(
        ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR)
    out = img * 1.15 + np.asarray(glow, F32) / 255 * 0.9 + np.asarray(bloom, F32) / 255 * 0.45
    vy, vx = np.ogrid[-1:1:H * 1j, -1:1:W * 1j]
    out *= (1 - 0.25 * (vx ** 2 + vy ** 2) ** 1.5)[..., None].astype(F32)
    return to8(out)


SYN = re.compile(r"(?P<com>#.*)|(?P<str>\"\"\".*?(?:\"\"\"|$)|\"[^\"]*\"|'[^']*')"
                 r"|(?P<kw>\b(?:def|return|for|in|if|else|elif|import|from|as|while|and|or|not"
                 r"|None|True|False|lambda|with|try|except|finally|global|class|pass|continue)\b)"
                 r"|(?P<num>\b\d+\.?\d*(?:e-?\d+)?\b)|(?P<fn>\b[A-Za-z_]\w*(?=\())"
                 r"|(?P<mod>\b(?:np|math)\b)")
SYN_COL = {"com": (110, 118, 129), "str": (184, 220, 120), "kw": (255, 122, 92),
           "num": (190, 150, 255), "fn": (110, 214, 214), "mod": (120, 180, 255), None: (220, 222, 228)}


class CodePanel:
    """Editor-style panel that scrolls through this file's renderer while it plays."""

    def __init__(self, w, h, frames, fps):
        self.w, self.h, self.frames, self.fps = w, h, frames, fps
        src = open(os.path.abspath(__file__), encoding="utf-8").read().splitlines()
        a = next(i for i, l in enumerate(src) if l.startswith("def leaf_petal"))
        b = next(i for i, l in enumerate(src) if l.startswith("def anchors"))
        self.first, self.lines = a + 1, [self.tokens(l) for l in src[a:b]]
        self.font = mono_font(22)
        self.small = mono_font(18)
        self.lh = 27

    @staticmethod
    def tokens(line):
        out, i = [], 0
        for m in SYN.finditer(line):
            if m.start() > i:
                out.append((line[i:m.start()], SYN_COL[None]))
            out.append((m.group(), SYN_COL[m.lastgroup]))
            i = m.end()
        out.append((line[i:], SYN_COL[None]))
        return out

    def draw(self, frame, t, total_w, total_h):
        img = Image.new("RGB", (self.w, self.h), (22, 23, 28))
        dr = ImageDraw.Draw(img)
        top, bot = 46, self.h - 38
        vis = (bot - top) / self.lh
        cur = frame / max(1, self.frames - 1) * (len(self.lines) - 1)
        start = min(max(0.0, cur - vis * 0.45), max(0.0, len(self.lines) - vis))
        for i in range(int(start), min(len(self.lines), int(start + vis) + 2)):
            y = top + 6 + (i - start) * self.lh
            if i == int(cur):
                dr.rectangle([0, y - 3, self.w, y + self.lh - 4], fill=(70, 52, 30))
                dr.rectangle([0, y - 3, 4, y + self.lh - 4], fill=(255, 160, 60))
            num = str(self.first + i).rjust(4)
            dr.text((8, y), num, font=self.font, fill=(235, 235, 235) if i == int(cur) else (95, 100, 110))
            x = 80
            for txt, c in self.lines[i]:
                dr.text((x, y), txt, font=self.font, fill=c)
                x += self.font.getlength(txt)
        dr.rectangle([0, 0, self.w, top], fill=(30, 31, 38))
        for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            dr.ellipse([18 + k * 28, 15, 34 + k * 28, 31], fill=c)
        dr.text((120, 12), os.path.basename(__file__), font=self.small, fill=(200, 200, 205))
        py = f"Python {sys.version_info.major}.{sys.version_info.minor}"
        dr.text((self.w - 16, 12), py, font=self.small, fill=(140, 145, 155), anchor="ra")
        dr.rectangle([0, bot, self.w, self.h], fill=(30, 31, 38))
        dr.text((14, bot + 9), f"● RUN  frame {frame + 1:03d}/{self.frames}  t={t:5.2f}s",
                font=self.small, fill=(120, 220, 140))
        right = f"{total_w}x{total_h} · {self.fps}fps · UTF-8 · Ln {self.first + int(cur)}"
        dr.text((self.w - 14, bot + 9), right, font=self.small, fill=(140, 145, 155), anchor="ra")
        return np.asarray(img)


def setup(cols, rows, cw, ch):
    global COLS, ROWS, CW, CH, W, H, RR, CC, X, Y, S, HOR, KX, KY, KS, MX, MY, MR, SAMP
    COLS, ROWS, CW, CH = cols, rows, cw, ch
    W, H = cols * cw, rows * ch
    RR, CC = np.mgrid[0:rows, 0:cols].astype(F32)
    X, Y = (CC + 0.5) * CW, (RR + 0.5) * CH
    KS = 0.28 * min(W, 1.05 * H)
    S = KS / 270
    HOR, KX, KY = H * 0.53, W * 0.5, H * 0.74
    MX, MY, MR = W * 0.76, H * 0.135, 0.095 * min(W, 1.05 * H)
    SAMP = 2.4 * KS / min(CW, CH)
    build_krathong()
    build_scene()


def render_video(args):
    setup(args.cols, args.rows, args.cw, args.ch)
    build_atlas()
    frames = int(round(LOOP * args.fps))
    panel = CodePanel(W, 576, frames, args.fps) if args.showcase else None
    ow, oh = W, H + (576 if panel else 0)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{ow}x{oh}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", args.out]
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found - install it (brew install ffmpeg) or use --png / --term")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(frames):
        t = f / args.fps
        img = compose_image(render_cells(t))
        if panel:
            img = np.vstack([img, panel.draw(f, t, ow, oh)])
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
        el = time.time() - t0
        print(f"\rframe {f + 1}/{frames}  {el:5.1f}s  eta {el / (f + 1) * (frames - f - 1):5.1f}s",
              end="", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"\nwrote {args.out}  ({ow}x{oh}, {args.fps} fps, {frames} frames)")


def render_png(args):
    setup(args.cols, args.rows, args.cw, args.ch)
    build_atlas()
    img = compose_image(render_cells(args.png))
    if args.showcase:
        frames = int(round(LOOP * args.fps))
        f = int(args.png * args.fps) % frames
        img = np.vstack([img, CodePanel(W, 576, frames, args.fps).draw(f, args.png, W, H + 576)])
    out = args.out if args.out.endswith(".png") else f"krathong_{args.png:g}.png"
    Image.fromarray(img).save(out)
    print(f"wrote {out}")


def terminal_cells(cv):
    """Glyph index, foreground and glow-background colours (multiples of 8) per cell."""
    src = cv[1] * np.clip(lum(cv[1]) * 1.6, 0, 1)[..., None]
    g = np.pad(src, ((2, 2), (4, 4), (0, 0)), mode="edge")
    hsum = g[:, 0:COLS]                                      # separable 9x5 box blur
    for dx in range(1, 9):
        hsum = hsum + g[:, dx:dx + COLS]
    glow = hsum[0:ROWS]
    for dy in range(1, 5):
        glow = glow + hsum[dy:dy + ROWS]
    glow = glow / 45
    fg = ((np.clip(cv[1] * 1.25 + glow * 0.5, 0, 1) * 31).astype(np.uint8) * 8)
    bg = ((np.clip(glow * 0.45, 0, 1) * 31).astype(np.uint8) * 8)
    return glyph_index(cv).astype(np.uint8), fg, bg


class AnsiScreen:
    """Truecolor output that sends only the cells that changed since the last frame.

    Colour codes are emitted only when they change, a space's foreground is never
    sent (it can't be seen), and short unchanged gaps are rewritten rather than
    jumped over, since a cursor move costs more bytes than a few cells.
    """

    def __init__(self):
        self.prev = None
        self.chars = np.array(CHARS)
        self.space = CHARS.index(" ")

    def frame(self, gi, fg, bg):
        if self.prev is None:
            changed = np.ones(gi.shape, bool)
        else:
            pg, pf, pb = self.prev
            changed = ((gi != pg) | (bg != pb).any(-1)
                       | ((gi != self.space) & (fg != pf).any(-1)))
        self.prev = (gi, fg, bg)
        out, cur_f, cur_b, pos = [], None, None, None
        for r in np.flatnonzero(changed.any(1)).tolist():
            cols = np.flatnonzero(changed[r])
            runs = np.split(cols, np.flatnonzero(np.diff(cols) > 5) + 1)
            ch, fr, br = self.chars[gi[r]].tolist(), fg[r].tolist(), bg[r].tolist()
            for run in runs:
                a, b = int(run[0]), int(run[-1])
                if pos != (r, a):
                    out.append("\x1b[%d;%dH" % (r + 1, a + 1))
                for c in range(a, b + 1):
                    sgr = []
                    if br[c] != cur_b:
                        cur_b = br[c]
                        sgr.append("48;2;%d;%d;%d" % tuple(cur_b))
                    if ch[c] != " " and fr[c] != cur_f:
                        cur_f = fr[c]
                        sgr.append("38;2;%d;%d;%d" % tuple(cur_f))
                    if sgr:
                        out.append("\x1b[" + ";".join(sgr) + "m")
                    out.append(ch[c])
                pos = (r, b + 1)
        return "".join(out)


def to_ansi(cv):
    """One full truecolor frame (no diffing) - handy for tests and piping."""
    return "\x1b[H" + AnsiScreen().frame(*terminal_cells(cv)) + "\x1b[0m"


def play_terminal(args):
    """Live loop. Time snaps to the frame grid, so the 10 s loop repeats exactly: each
    frame is rendered once, and each frame-to-frame diff is encoded once, then both
    are replayed from memory (within a budget) on every later pass of the loop."""
    out = sys.stdout
    out.write("\x1b[?25l\x1b[2J")
    frames = int(round(LOOP * args.fps))
    budget = 100e6                                   # bytes of cached cells + diffs
    size, t0 = None, time.time()
    try:
        while True:
            ts = shutil.get_terminal_size()
            if (ts.columns, ts.lines) != size:
                size = (ts.columns, ts.lines)
                setup(ts.columns, ts.lines - 1, 8, 16)
                screen, cells_cache, diff_cache, used, prev_n = AnsiScreen(), {}, {}, 0, None
                out.write("\x1b[0m\x1b[2J")
            start = time.time()
            n = int((start - t0) * args.fps) % frames
            if prev_n is not None and n < prev_n:      # once per loop: full repaint,
                screen.prev, prev_n = None, None       # so a disturbed screen heals
            cells = cells_cache.get(n)
            if cells is None:
                cells = terminal_cells(render_cells(n / args.fps))
                if used + COLS * ROWS * 7 <= budget:
                    cells_cache[n], used = cells, used + COLS * ROWS * 7
            diff = diff_cache.get((prev_n, n))
            if diff is None:
                diff = screen.frame(*cells)
                if prev_n is not None and used + len(diff) <= budget:
                    diff_cache[(prev_n, n)], used = diff, used + len(diff)
            else:
                screen.prev = cells
            prev_n = n
            out.write(diff)
            out.flush()
            time.sleep(max(0.0, (int((start - t0) * args.fps) + 1) / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\n")


def main():
    ap = argparse.ArgumentParser(description="Loy Krathong in coloured ASCII")
    ap.add_argument("--out", default="krathong.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cols", type=int, default=108)
    ap.add_argument("--rows", type=int, default=84)
    ap.add_argument("--cw", type=int, default=10, help="char cell width in px")
    ap.add_argument("--ch", type=int, default=16, help="char cell height in px")
    ap.add_argument("--showcase", action="store_true", help="add the scrolling code panel (1080x1920)")
    ap.add_argument("--png", type=float, metavar="T", help="render one still at time T seconds")
    ap.add_argument("--term", action="store_true", help="play live in a truecolor terminal")
    ap.add_argument("--no-fireworks", action="store_true", help="a quiet night, no fireworks")
    ap.add_argument("--no-dragon", action="store_true", help="leave the dragon out")
    args = ap.parse_args()
    global FIREWORKS_ON, DRAGON_ON
    FIREWORKS_ON, DRAGON_ON = not args.no_fireworks, not args.no_dragon
    if args.term:
        play_terminal(args)
    elif args.png is not None:
        render_png(args)
    else:
        render_video(args)


if __name__ == "__main__":
    main()
