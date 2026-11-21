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
    "smoke": " .:'~",
}
MID = {name: i for i, name in enumerate(RAMPS)}
LV = 24                          # density levels per ramp
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


def putcells(cv, x, y, d, col, mid):
    """Splat individual points (sparks, fireflies) at pixel positions."""
    c, r = (np.asarray(x) // CW).astype(int), (np.asarray(y) // CH).astype(int)
    ok = (c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS) & (d > 0.05)
    r, c, d, col = r[ok], c[ok], np.asarray(d)[ok], np.asarray(col, F32)[ok]
    ok = d > cv[0][r, c]
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


def zbuffer(xs, ys, zv, keep):
    """Nearest point per char cell -> (flat cell index, point index)."""
    col, row = np.floor(xs / CW).astype(np.int64), np.floor(ys / CH).astype(np.int64)
    idx = np.nonzero(keep & (col >= 0) & (col < COLS) & (row >= 0) & (row < ROWS))[0]
    idx = idx[np.argsort(-zv[idx], kind="stable")]
    cells, first = np.unique(row[idx] * COLS + col[idx], return_index=True)
    return cells, idx[first]


def render_krathong(cv, t, reflect=False):
    """Z-buffer surface points into the char grid, then light each cell."""
    P, N, Cb, Mt, Em = GEOM
    R, T = body_pose(t)
    Pw, Nw = P @ R.T + T, N @ R.T
    keep = Pw[:, 1] > -0.01                                 # below waterline: hidden
    light = CANDLE_TOP @ R.T + T + np.array([0, 0.12, 0], F32)
    if reflect:
        Pw, Nw, light = Pw * FLIP_Y, Nw * FLIP_Y, light * FLIP_Y
    xs, ys, zv = project(Pw)
    if reflect:
        xs = xs + 6 * S * np.sin(ys * 0.045 / S - W0 * 4 * t)
    cells, w = zbuffer(xs, ys, zv, keep)
    if reflect:
        ok = cv[2].reshape(-1)[cells] == MID["water"]
        cells, w = cells[ok], w[ok]

    n = view(Nw[w])
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
    lit = 1 - np.exp(-1.5 * Cb[w] * (amb + cand + moon + bounce))
    em = Em[w]
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
    cv[2].reshape(-1)[cells] = Mt[w]


def anchors(t):
    """Screen position of the wick tip and the three incense tips."""
    R, T = body_pose(t)
    x, y, _ = project(np.vstack([CANDLE_TOP, TIPS]) @ R.T + T)
    return float(x[0]), float(y[0]), np.stack([x[1:], y[1:]], 1)


# ------------------------------------------------------------------ scene ---
def build_scene():
    """Static masks: temple silhouettes + lantern / spark / firefly seeds."""
    global TEMPLE, EDGE, WINDOWS, WIN_H, LANTERNS, SPARKS, MOTES
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

    rng = np.random.default_rng(7)
    LANTERNS = [dict(x=rng.uniform(0.04, 0.96), ph=rng.uniform(), sz=rng.uniform(0.55, 1.3),
                     sw=rng.uniform(4, 22), k=int(rng.integers(2, 6))) for _ in range(16)]
    SPARKS = dict(ph=rng.uniform(size=36), k=rng.integers(1, 4, 36),
                  a=rng.uniform(1, 4, 36), dr=rng.uniform(-70, 70, 36), rise=rng.uniform(160, 380, 36))
    MOTES = dict(x=rng.uniform(0.03, 0.97, 50), y=rng.uniform(HOR / H - 0.15, 0.98, 50),
                 p=rng.uniform(0, 6.28, (3, 50)), k=rng.integers(1, 4, (3, 50)))


def sky_layer(cv, t):
    m = Y < HOR
    g = np.clip(Y / HOR, 0, 1)
    dm = np.hypot(X - MX, Y - MY)
    mg = np.exp(-np.maximum(dm - MR, 0) / (MR * 0.9))
    col = mixc((0.020, 0.025, 0.060), (0.16, 0.08, 0.05), g ** 3)
    col = col + (mg * 0.25)[..., None] * np.array([0.75, 0.72, 0.55], F32)
    put(cv, m, 0.3 + 0.35 * mg, col, "sky")

    hv, br = hash2(RR, CC, 1.7), hash2(RR, CC, 3.1)
    tw = 0.55 + 0.45 * np.sin(W0 * (1 + np.floor(br * 4)) * t + hv * 628)
    star = m & (hv > 0.972) & (Y < HOR - 0.03 * H) & (dm > MR * 1.3)
    scol = mixc((0.75, 0.80, 1.0), (1.0, 0.9, 0.7), hash2(RR, CC, 9.2))
    put(cv, star, 0.45 + 0.5 * br * tw, scol * (0.30 + 0.70 * tw * br)[..., None], "star")

    disk = dm < MR                                               # the full moon
    limb = np.sqrt(np.clip(1 - (dm / MR) ** 2, 0, 1))
    maria = 0.5 + 0.5 * np.sin(X * 0.045 / S + 1.3) * np.sin(Y * 0.05 / S + 0.4)
    maria = np.clip(maria + 0.3 * np.sin(X * 0.11 / S - Y * 0.07 / S), 0, 1)
    shade = 0.70 + 0.30 * limb - 0.18 * maria
    put(cv, disk, 0.55 + 0.45 * limb - 0.2 * maria,
        np.array([1.0, 0.96, 0.82], F32) * shade[..., None], "moon")


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
        dx, dy = (X - lx) / lw, (Y - ly) / lh
        inside = (np.abs(dy) < 1) & (np.abs(dx) < 0.85 - 0.22 * dy)
        b = (dy + 1) / 2
        col = mixc((0.95, 0.38, 0.08), (1.0, 0.80, 0.38), b) * ((0.55 + 0.45 * b) * fade * flick)[..., None]
        put(cv, inside & (fade > 0.05), (0.45 + 0.55 * b) * fade, col, "lantern")
        g = 0.36 * np.exp(-(dx ** 2 + dy ** 2) / 3.5) * fade * flick
        putmax(cv, g, np.array([1.0, 0.55, 0.18], F32) * g[..., None], "glow")


def temple_layer(cv, t, fx, fy):
    warm = np.exp(-np.hypot(X - fx, Y - fy) / (380 * S))[..., None]
    body = np.array([0.11, 0.055, 0.025], F32) + warm * np.array([0.30, 0.14, 0.04], F32)
    put(cv, TEMPLE, 0.36 + 0.25 * warm[..., 0], body, "temple")
    edge = np.array([0.40, 0.19, 0.06], F32) + warm * np.array([0.45, 0.22, 0.06], F32)
    put(cv, EDGE, 0.62 + 0.2 * warm[..., 0], edge, "temple")
    fl = 0.75 + 0.25 * np.sin(W0 * (2 + np.floor(WIN_H * 4)) * t + WIN_H * 30)
    put(cv, WINDOWS, 0.95, np.array([1.0, 0.74, 0.30], F32) * fl[..., None], "window")


def water_layer(cv, t, fx):
    """Water per cell: summed waves, perspective, and light reflections."""
    wm = Y >= HOR
    q = np.clip((Y - HOR) / (H - HOR), 0, 1)
    z = 1.0 / (q + 0.07)
    u = (X - KX) / ((60 + q * 700) * S)
    h = (np.sin(u * 6 + z * 1.4 - W0 * 3 * t) + 0.7 * np.sin(u * 11 - z * 2.3 + W0 * 4 * t)
         + 0.5 * np.sin(u * 3 + z * 3.9 - W0 * 2 * t + 2.1)
         + 0.3 * np.sin(u * 17 + z * 6.1 + W0 * 5 * t))
    wave = 0.5 + 0.5 * np.tanh(h * 0.9)
    dens = 0.10 + 0.30 * wave * (0.4 + 0.6 * q)
    col = mixc((0.04, 0.28, 0.30), (0.12, 0.55, 0.52), wave)
    col = mixc(col, (0.55, 0.30, 0.10), np.clip(1 - q * 9, 0, 1) * 0.7)
    col = col * (0.16 + 0.75 * dens)[..., None]

    r0 = int(HOR // CH)                                         # mirror the skyline
    rm = np.clip(2 * r0 - RR - 1, 0, ROWS - 1).astype(int)
    cm = np.clip(CC + np.round(h * 0.9), 0, COLS - 1).astype(int)
    fade = np.exp(-(RR - r0) / (0.10 * H / CH)) * (0.55 + 0.45 * wave)
    refl = cv[1][rm, cm] * (0.55 * fade)[..., None]
    col = col + refl
    dens = np.maximum(dens, np.where(cv[2][rm, cm] == MID["sky"], 0, cv[0][rm, cm] * fade * 0.7))

    dyr = Y - KY                                                # candle streak
    wid = 24 * S + np.clip(dyr, 0, None) * 0.10
    fr = (np.exp(-((X - fx + 5 * h * S) / wid) ** 2) * (dyr > -40 * S)
          * np.exp(-np.clip(dyr, 0, None) / (320 * S))
          * (0.5 + 0.5 * np.sin(dyr * 0.12 / S + h * 2.2 - W0 * 6 * t)) ** 1.5)
    fr_f = 0.85 + 0.15 * math.sin(W0 * 7 * t)
    mr = Y - HOR                                                # moon path
    mw = 20 * S + mr * 0.10
    mo = (np.exp(-((X - MX + 6 * h * S) / mw) ** 2) * np.exp(-mr / (420 * S))
          * (0.5 + 0.5 * np.sin(mr * 0.10 / S + h * 3 - W0 * 5 * t)) ** 1.5)
    pool = np.exp(-(((X - KX) / (300 * S)) ** 2 + ((Y - KY - 15 * S) / (120 * S)) ** 2) * 1.4)
    pool = pool * 0.75 * (0.35 + 0.65 * wave)
    rho = np.hypot(X - KX, (Y - KY - 20 * S) * 2.6)             # ripples
    ring = np.clip(np.cos(rho * 0.055 / S - W0 * 5 * t), 0, 1) ** 6 * np.exp(-rho / (420 * S))
    fields = np.stack([dens, 0.9 * fr * fr_f, 0.8 * mo, pool, 0.6 * ring])
    d = fields.max(0)
    best = fields.argmax(0)
    hl = np.array([[0, 0, 0], [1.0, 0.75, 0.25], [1.0, 0.95, 0.72],
                   [1.0, 0.62, 0.20], [1.0, 0.80, 0.40]], F32)[best]
    colf = np.where((best == 0)[..., None], col, hl * (0.2 + 0.9 * d)[..., None] + 0.5 * refl)
    put(cv, wm, d, colf, "water")


def flame_layer(cv, t, fx, fy):
    """Teardrop candle flame: multi-frequency flicker, white core to red tip."""
    fh, fw = 150.0 * S, 32.0 * S
    flick = 1 + 0.10 * math.sin(W0 * 7 * t) + 0.06 * math.sin(W0 * 13 * t + 1.3)
    sway = (9 * math.sin(W0 * 5 * t + 0.7) + 5 * math.sin(W0 * 11 * t)) * S
    s = (fy - Y) / (fh * flick)
    sc = np.clip(s, 0, 1)
    xo = sway * sc ** 2 + 3 * S * np.sin(s * 9 - W0 * 9 * t) * sc
    wid = fw * np.clip(np.sin(np.pi * sc ** 0.6), 0, 1) ** 0.8 * (1 + 0.08 * np.sin(W0 * 17 * t + s * 5))
    q = np.abs(X - fx - xo) / np.maximum(wid, 1e-3)
    inside = (s > -0.06) & (s < 1.0) & (q < 1.0)
    core = (q < 0.45) & (s < 0.55)
    col = mixc((1.0, 0.40, 0.08), (1.0, 0.82, 0.25), np.clip(1.25 - s - q * 0.6, 0, 1))
    col = np.where(core[..., None], np.array([1.0, 0.97, 0.78], F32), col)
    put(cv, inside, np.where(core, 1.0, 0.55 + 0.4 * (1 - sc)), col * 1.15, "flame")
    dh = np.hypot((X - fx - xo * 0.5) / (95 * S), (Y - (fy - 40 * S)) / (120 * S))
    halo = 0.42 * np.exp(-dh * dh * 2.2) * (hash2(RR, CC, 7.0) > 0.35)
    putmax(cv, halo, np.array([1.0, 0.62, 0.20], F32) * (halo * 1.8)[..., None], "spark", 0.1)


def smoke_layer(cv, t, tips):
    """Thin incense smoke curling upward from each glowing tip."""
    L = 230 * S
    for i, (tx, ty) in enumerate(tips):
        hgt = (ty - Y) / L
        xc = tx + 12 * S * hgt * np.sin(hgt * 6 - W0 * 2 * t + i * 2.0)
        m = (hgt > 0.03) & (hgt < 1) & (np.abs(X - xc) < CW * 0.7)
        d = np.where(m, 0.30 * (1 - hgt) * (0.6 + 0.4 * np.sin(hgt * 20 - W0 * 6 * t + i)), 0)
        putmax(cv, d, np.array([0.55, 0.55, 0.62], F32) * (d * 1.6)[..., None], "smoke", 0.05)


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
    temple_layer(cv, t, fx, fy)
    water_layer(cv, t, fx)
    render_krathong(cv, t, reflect=True)
    render_krathong(cv, t)
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
        b = next(i for i, l in enumerate(src) if l.startswith("def flame_layer"))
        b = next(i for i in range(b + 1, len(src)) if src[i].startswith("def "))
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


def to_ansi(cv):
    """Truecolor terminal frame: glyph colour in front, blurred glow behind."""
    chars = np.array(CHARS)[glyph_index(cv)]
    src = cv[1] * np.clip(lum(cv[1]) * 1.6, 0, 1)[..., None]
    g = np.pad(src, ((2, 2), (4, 4), (0, 0)), mode="edge")
    glow = sum(g[dy:dy + ROWS, dx:dx + COLS] for dy in range(5) for dx in range(9)) / 45
    fg = (np.clip(cv[1] * 1.25 + glow * 0.5, 0, 1) * 31).astype(int) * 8
    bg = (np.clip(glow * 0.45, 0, 1) * 31).astype(int) * 8
    fg, bg, chars = fg.tolist(), bg.tolist(), chars.tolist()
    lines = []
    for r in range(ROWS):
        parts, last = [], None
        for c in range(COLS):
            key = (*fg[r][c], *bg[r][c])
            if key != last:
                parts.append("\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm" % key)
                last = key
            parts.append(chars[r][c])
        lines.append("".join(parts))
    return "\x1b[H" + "\x1b[0m\n".join(lines) + "\x1b[0m"


def play_terminal(args):
    out = sys.stdout
    out.write("\x1b[?25l\x1b[2J")
    size, t0 = None, time.time()
    try:
        while True:
            ts = shutil.get_terminal_size()
            if (ts.columns, ts.lines) != size:
                size = (ts.columns, ts.lines)
                setup(ts.columns, ts.lines - 1, 8, 16)
                out.write("\x1b[2J")
            start = time.time()
            out.write(to_ansi(render_cells((start - t0) % LOOP)))
            out.flush()
            time.sleep(max(0.0, 1 / args.fps - (time.time() - start)))
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
    args = ap.parse_args()
    if args.term:
        play_terminal(args)
    elif args.png is not None:
        render_png(args)
    else:
        render_video(args)


if __name__ == "__main__":
    main()
