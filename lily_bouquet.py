#!/usr/bin/env python3
"""
lily_bouquet.py - a bouquet of lilies grows itself in coloured ASCII, then sways in a
seamless loop. Built on krathong_ascii.py's engine.

Green shoots rise and arch, long leaves unfurl from the base, buds swell on the stem tips
and the lilies open one after another - six petals each, curling back at the tips, with
freckled throats, stamens and pollen-dusted anthers. A satin ribbon wraps the stems and
ties a bow, then the bouquet breathes and sways while pollen drifts and petals fall.
Everything is a lit 3-D point cloud, z-buffered into the character grid.

    python lily_bouquet.py                # live in the terminal: it grows, then loops forever
    python lily_bouquet.py --video        # lily_bouquet.mp4 (growth + one loop)
    python lily_bouquet.py --png 8        # one still at t = 8 s
"""
import math
from types import SimpleNamespace

import numpy as np

import krathong_ascii as K
from krathong_ascii import F32, hash2, lum, mixc, putcells, putmax, window, sub

K.register_ramps({
    "petal": " .:-=+*#%&@",
    "stem": " .:|!lI",
    "leaf": " .,:;=+*#%",
    "filament": " .'|",
    "anther": " .oO@",
    "ribbon": " .-=+#%@",
    "bokeh": " ..::",
    "pollen": " .'*",
    "bg": " ..",
})

GROW = 7.0                          # the bouquet is complete
LOOP = 10.0                         # then it sways on a 10 s loop
W0 = 2 * math.pi / LOOP
PITCH, PERSP = 0.12, 0.06
CPB, SPB = math.cos(PITCH), math.sin(PITCH)
TIE = np.array([0.0, 1.05, 0.06], F32)


def unit(*v):
    a = np.array(v, F32)
    return a / np.linalg.norm(a)


KEY, FILL, VIEW = unit(-0.55, 0.65, 0.55), unit(0.7, 0.15, 0.6), unit(0, SPB, CPB)
HALF = (KEY + VIEW) / np.linalg.norm(KEY + VIEW)

VARIETIES = {   # base, tip, throat, edge (or None), freckle density
    "red":       ((0.88, 0.08, 0.08), (1.00, 0.32, 0.22), (1.00, 0.62, 0.18), None, 0.10),
    "stargazer": ((0.90, 0.20, 0.52), (1.00, 0.55, 0.75), (0.98, 0.88, 0.60), (1.0, 0.93, 0.96), 0.14),
    "white":     ((0.95, 0.93, 0.88), (1.00, 1.00, 0.97), (0.70, 0.86, 0.35), None, 0.0),
    "yellow":    ((1.00, 0.78, 0.10), (1.00, 0.92, 0.45), (1.00, 0.52, 0.06), None, 0.06),
    "orange":    ((1.00, 0.44, 0.06), (1.00, 0.70, 0.28), (1.00, 0.82, 0.30), None, 0.10),
    "peach":     ((1.00, 0.72, 0.56), (1.00, 0.90, 0.80), (1.00, 0.52, 0.32), None, 0.05),
    "burgundy":  ((0.55, 0.03, 0.12), (0.85, 0.15, 0.25), (0.95, 0.55, 0.25), None, 0.0),
    "magenta":   ((0.80, 0.14, 0.60), (0.95, 0.45, 0.85), (1.00, 0.85, 0.70), (1.0, 0.86, 0.95), 0.08),
}
ORDER = ["white", "red", "stargazer", "yellow", "burgundy", "peach", "orange", "magenta",
         "red", "white", "stargazer", "yellow", "orange", "peach", "magenta"]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def smoother(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


def backout(x):
    x = np.clip(x, 0, 1) - 1
    return 1 + 2.2 * x ** 3 + 1.2 * x ** 2


def perp(a):
    """Two unit vectors perpendicular to each row of a (n,3)."""
    ref = np.where(np.abs(a[:, 1:2]) < 0.9, np.array([[0, 1, 0]], F32), np.array([[1, 0, 0]], F32))
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(a, e1)


# ---------------------------------------------------------------- geometry ---
def build_flowers():
    """Heads on a dome (golden-angle spiral, centre forward), plus closed buds on the rim."""
    global FL, NU, NV, UU, VV, WU, PCOL, PSPOT
    rng = np.random.default_rng(12)
    heads = []
    N = 11
    for i in range(N):
        r = math.sqrt((i + 0.5) / N)
        a = i * 2.39996 + 0.4
        x, y = 1.50 * r * math.cos(a), 2.55 + 1.15 * r * math.sin(a)
        z = 0.65 * (1 - r * r) + 0.05
        axis = unit(x * 0.45, (y - 2.4) * 0.35 + 0.25, 1.0 + 0.3 * (1 - r))
        t0 = 0.15 + 0.45 * rng.random()
        heads.append(dict(pos=(x, y, z), axis=axis, size=1.18 - 0.16 * r + rng.uniform(-0.05, 0.05),
                          var=ORDER[i % len(ORDER)], t0=t0, t_open=t0 + 1.7 + 0.85 + 1.4 * i / N, bud=False))
    for x, y, z, ax, v in [(-0.45, 3.95, -0.1, (-0.3, 1, 0.1), "burgundy"), (0.38, 4.02, -0.1, (0.25, 1, 0.1), "red"),
                           (1.15, 3.62, -0.15, (0.8, 0.8, 0.1), "peach"), (-1.20, 3.55, -0.15, (-0.8, 0.8, 0.1), "orange"),
                           (1.72, 2.65, -0.2, (1, 0.35, 0.2), "burgundy"), (-1.75, 2.45, -0.2, (-1, 0.35, 0.2), "yellow")]:
        heads.append(dict(pos=(x, y, z), axis=unit(*ax), size=0.70, var=v,
                          t0=0.2 + 0.4 * rng.random(), t_open=1e9, bud=True))
    F = len(heads)
    FL = dict(pos=np.array([h["pos"] for h in heads], F32), axis=np.array([h["axis"] for h in heads], F32),
              size=np.array([h["size"] for h in heads], F32), t0=np.array([h["t0"] for h in heads], F32),
              t_open=np.array([h["t_open"] for h in heads], F32), bud=np.array([h["bud"] for h in heads]),
              spin=rng.uniform(0, math.pi / 3, F).astype(F32), ph=rng.uniform(0, 6.28, (F, 4)).astype(F32),
              var=[h["var"] for h in heads])
    FL["e1"], FL["e2"] = perp(FL["axis"])
    FL["t_bud"] = FL["t0"] + 1.7

    NU = int(np.clip(0.50 * SAMP, 8, 72))
    NV = int(np.clip(0.34 * SAMP, 5, 50))
    UU = np.linspace(0, 1, NU, dtype=F32)
    VV = np.linspace(-1, 1, NV, dtype=F32)
    WU = np.clip(np.sin(np.pi * UU ** 0.72), 0, 1) ** 0.9
    U, V = np.meshgrid(UU, VV, indexing="ij")
    PCOL = np.zeros((F, 6, NU, NV, 3), F32)
    PSPOT = np.zeros((F, 6, NU, NV), bool)
    for f, name in enumerate(FL["var"]):
        base, tip, throat, edge, spots = VARIETIES[name]
        c = mixc(throat, base, smooth((U - 0.04) / 0.30))
        c = mixc(c, tip, smooth((U - 0.55) / 0.45) * 0.6)
        if edge is not None:
            c = mixc(c, edge, smooth((np.abs(V) - 0.50) / 0.45) * 0.8)
        rib = (np.abs(V) < 0.11) & (U > 0.08) & (U < 0.85)                  # midrib groove
        c = np.where(rib[..., None], c * 0.82 + np.array(throat, F32) * 0.18, c)
        for k in range(6):
            h = hash2(U * 53 + k * 7.1, V * 29 + f * 3.3, 7.7)
            sp = (h > 1 - spots) & (U > 0.12) & (U < 0.58) & (np.abs(V) < 0.75)
            PSPOT[f, k] = sp
            PCOL[f, k] = np.where(sp[..., None], np.array([0.32, 0.04, 0.07], F32), c)


def head_state(t):
    """Per flower: stem growth, bud swell, opening amount."""
    grow = smooth((t - FL["t0"]) / 1.7)
    swell = smooth((t - FL["t_bud"]) / 0.8)
    e = np.where(FL["bud"], 0.0, smooth((t - FL["t_open"]) / 1.6)).astype(F32)
    return grow, swell, e


def sway(t):
    """Breeze: each head drifts on its own loop-periodic path (whole cycles per loop)."""
    p = FL["ph"]
    hrel = np.clip((FL["pos"][:, 1] - TIE[1]) / 3.0, 0, 1)[:, None]
    s = np.stack([0.050 * np.sin(W0 * t + p[:, 0]) + 0.015 * np.sin(W0 * 3 * t + p[:, 1]),
                  0.008 * np.sin(W0 * 2 * t + p[:, 2]),
                  0.025 * np.sin(W0 * t + p[:, 3])], 1)
    return (s * (0.5 + 0.5 * hrel)).astype(F32)


def build_stems():
    """Each stem: straight from its cut end to the tie, then a curve up to its head."""
    global STEM, BAXIS
    rng = np.random.default_rng(3)
    BAXIS = np.zeros_like(FL["axis"])
    P, N, S, Fi, Wt = [], [], [], [], []
    for f in range(len(FL["pos"])):
        hb = FL["pos"][f] - FL["axis"][f] * 0.05
        tq = TIE + np.array([rng.uniform(-0.04, 0.04), 0, rng.uniform(-0.03, 0.03)], F32)
        p0 = np.array([0.10 * (hb[0] / 1.5) + rng.uniform(-0.05, 0.05), -0.25, rng.uniform(-0.04, 0.06)], F32)
        c = np.array([tq[0] + 0.15 * (hb[0] - tq[0]), tq[1] + 0.62 * (hb[1] - tq[1]), tq[2] + 0.2 * (hb[2] - tq[2])], F32)
        n1 = int(np.linalg.norm(tq - p0) * SAMP * 1.5) + 2
        n2 = int((np.linalg.norm(c - tq) + np.linalg.norm(hb - c)) * SAMP * 1.5) + 2
        k1 = np.linspace(0, 1, n1, endpoint=False, dtype=F32)[:, None]
        k2 = np.linspace(0, 1, n2, dtype=F32)[:, None]
        line = np.vstack([p0 + (tq - p0) * k1, (1 - k2) ** 2 * tq + 2 * (1 - k2) * k2 * c + k2 ** 2 * hb])
        BAXIS[f] = (hb - c) / (np.linalg.norm(hb - c) + 1e-9)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        s = seg / seg[-1]
        w = np.clip((line[:, 1] - TIE[1]) / max(hb[1] - TIE[1], 1e-3), 0, 1) ** 1.6
        for q in (0.0, 2.1, 4.2):
            off = np.array([math.cos(q), 0, math.sin(q)], F32)
            P.append(line + off * 0.017)
            N.append(np.tile(off, (len(line), 1)))
            S.append(s)
            Fi.append(np.full(len(line), f))
            Wt.append(w)
    STEM = dict(P=np.concatenate(P).astype(F32), N=np.concatenate(N).astype(F32), S=np.concatenate(S).astype(F32),
                F=np.concatenate(Fi), W=np.concatenate(Wt).astype(F32))
    STEM["C"] = mixc((0.16, 0.44, 0.16), (0.32, 0.62, 0.24), hash2(STEM["S"] * 97, STEM["F"].astype(F32), 2.0))


def leaf_grid(start, dirn, L, wd, droop, rng, col_a, col_b):
    nu = max(6, int(L * SAMP * 1.3))
    nv = max(3, int(2 * wd * SAMP * 1.3))
    u, v = np.meshgrid(np.linspace(0, 1, nu, dtype=F32), np.linspace(-1, 1, nv, dtype=F32), indexing="ij")
    up = np.array([0, 1, 0], F32)
    side = up * 0.85 + np.cross(dirn, up) * 0.15             # blade turned to face the viewer
    side = side - dirn * (side @ dirn)
    side /= np.linalg.norm(side) + 1e-9
    cen = dirn * (L * u)[..., None] * 0.95 + up * (L * (0.55 * u - droop * u * u))[..., None]
    w = wd * np.clip(np.sin(np.pi * np.clip(u * 1.1, 0, 1) ** 0.55), 0, 1) * (1 - 0.15 * u)
    du = dirn + up[None, None, :] * 0 + up * (0.55 - 2 * droop * u)[..., None] / 0.95
    nrm = np.cross(du, side)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
    P = cen + side * (w * v)[..., None] + nrm * (0.25 * w * np.abs(v))[..., None]
    col = mixc(col_a, col_b, (0.5 + 0.5 * v) * 0.7 + 0.3 * rng.random(u.shape))
    col = np.where((np.abs(v) < 0.14)[..., None], mixc(col, (0.55, 0.80, 0.40), 0.5), col)
    col = mixc(col, (0.62, 0.78, 0.25), smooth((u - 0.8) / 0.2) * 0.4)
    return P.reshape(-1, 3) - start, np.broadcast_to(nrm, P.shape).reshape(-1, 3), col.reshape(-1, 3)


def build_leaves():
    """Long arching leaves fanning from the base, and small leaves along the stems."""
    global LEAF
    rng = np.random.default_rng(8)
    rel, nrm, col, st, fid, wt, tl, dur = [], [], [], [], [], [], [], []
    sides = [(sd, b) for b in (-0.55, -0.2, 0.15, 0.45, 0.75, 1.0, 1.25) for sd in (-1, 1)]
    for i, (sd, b) in enumerate(sides):
        az = math.atan2(math.sin(b), sd * math.cos(b))
        hi = i % 2                                              # two tiers: above and below the tie
        start = TIE + np.array([0.06 * math.cos(az), (0.10 + 0.25 * rng.random()) if hi else (-0.15 - 0.25 * rng.random()),
                                0.06 * math.sin(az)], F32)
        dirn = unit(math.cos(az), 0.50, math.sin(az) * 0.8 + 0.25)
        P, N, C = leaf_grid(start, dirn, rng.uniform(1.0, 1.45), 0.12, rng.uniform(0.30, 0.55), rng,
                            (0.12, 0.42, 0.15), (0.34, 0.72, 0.26))
        n = len(P)
        rel.append(P); nrm.append(N); col.append(C); st.append(np.tile(start, (n, 1)))
        fid.append(np.full(n, -1)); wt.append(np.zeros(n, F32))
        tl.append(np.full(n, 0.45 + 0.07 * i, F32)); dur.append(np.full(n, 1.4, F32))
    for f in rng.choice(len(FL["pos"]), 12, replace=False):    # stem leaves
        m = STEM["F"] == f
        s_at = rng.uniform(0.45, 0.7)
        j = np.flatnonzero(m)[np.argmin(np.abs(STEM["S"][m] - s_at))]
        start = STEM["P"][j]
        out = np.array([start[0], 0, start[2] + 0.3], F32)
        dirn = unit(*(out / (np.linalg.norm(out) + 1e-6) + np.array([0, 0.9, 0], F32)))
        dirn = unit(dirn[0], 0, dirn[2]) if abs(dirn[1]) > 0.99 else dirn
        dh = unit(dirn[0], 0, dirn[2] + 1e-3)
        P, N, C = leaf_grid(start, dh, 0.48, 0.07, 0.30, rng, (0.14, 0.46, 0.16), (0.34, 0.72, 0.26))
        n = len(P)
        rel.append(P); nrm.append(N); col.append(C); st.append(np.tile(start, (n, 1)))
        fid.append(np.full(n, f)); wt.append(np.full(n, STEM["W"][j], F32))
        tl.append(np.full(n, FL["t0"][f] + 1.7 * s_at + 0.1, F32)); dur.append(np.full(n, 0.7, F32))
    LEAF = dict(rel=np.concatenate(rel).astype(F32), N=np.concatenate(nrm).astype(F32), C=np.concatenate(col).astype(F32),
                start=np.concatenate(st).astype(F32), F=np.concatenate(fid), W=np.concatenate(wt),
                tl=np.concatenate(tl), dur=np.concatenate(dur))


def build_ribbon():
    """Satin band around the stems with a bow and two V-cut tails."""
    global RIB
    parts = []
    nt, nh = max(24, int(2 * math.pi * 0.14 * SAMP * 1.4)), max(4, int(0.14 * SAMP * 1.4))
    th, hh = np.meshgrid(np.linspace(0, 2 * np.pi, nt, dtype=F32), np.linspace(-0.07, 0.07, nh, dtype=F32), indexing="ij")
    rr = 0.135 + 0.006 * np.cos(hh * 40)
    P = np.stack([TIE[0] + rr * np.sin(th), TIE[1] + hh, TIE[2] + rr * np.cos(th)], -1)
    N = np.stack([np.sin(th), 0 * th, np.cos(th)], -1)
    ang = (th + np.pi) % (2 * np.pi) - np.pi                  # wraps on from the front
    parts.append((P, N, np.abs(ang) / np.pi, np.zeros_like(th)))
    knot = TIE + np.array([0, 0.02, 0.16], F32)
    for side in (-1, 1):                                       # bow loops
        na, nw = max(16, int(0.8 * SAMP * 1.4)), max(3, int(0.07 * SAMP * 1.4))
        a, wv = np.meshgrid(np.linspace(0, 2 * np.pi, na, dtype=F32), np.linspace(-1, 1, nw, dtype=F32), indexing="ij")
        cl = knot + np.array([side * 0.26, 0.07, 0.02], F32)
        P = cl + np.stack([side * 0.25 * np.cos(a), 0.13 * np.sin(a) + 0.05 * np.cos(a), 0.04 * wv + 0.02 * np.sin(a)], -1)
        N = np.stack([side * np.cos(a) / 0.25, np.sin(a) / 0.13, 0.3 + 0 * a], -1)
        parts.append((P, N, np.full(a.shape, 2.0, F32), np.ones_like(a)))
        ns = max(12, int(0.65 * SAMP * 1.4))
        s, v = np.meshgrid(np.linspace(0, 1, ns, dtype=F32), np.linspace(-1, 1, nw, dtype=F32), indexing="ij")
        cen = knot + np.stack([side * (0.05 + 0.26 * s) + 0.03 * np.sin(6 * s + side), -0.42 * s, 0.04 + 0.02 * np.sin(3 * s)], -1)
        wdir = unit(1.0, 0.25 * side, 0)
        P = cen + wdir * (0.05 * v)[..., None]
        keep = s < 0.88 + 0.12 * np.abs(v)                     # V-cut ends
        N = np.stack([0.3 * np.cos(6 * s), 0 * s, 1 + 0 * s], -1)
        parts.append((P[keep], N[keep], np.full(keep.sum(), 3.0, F32), (1 + s)[keep]))
    U, V = np.meshgrid(np.linspace(0, np.pi, 8), np.linspace(0, 2 * np.pi, 12), indexing="ij")
    n = np.stack([np.sin(U) * np.cos(V), np.cos(U), np.sin(U) * np.sin(V)], -1)
    parts.append((knot + 0.045 * n, n, np.full(U.shape, 2.0, F32), np.ones_like(U)))
    RIB = dict(P=np.concatenate([p[0].reshape(-1, 3) for p in parts]).astype(F32),
               N=np.concatenate([p[1].reshape(-1, 3) for p in parts]).astype(F32),
               kind=np.concatenate([p[2].reshape(-1) for p in parts]).astype(F32),
               s=np.concatenate([p[3].reshape(-1) for p in parts]).astype(F32), knot=knot)


# ------------------------------------------------------------- per frame ---
def head_axis(act, e):
    """A bud points along its stem; as it opens the flower turns to face the viewer."""
    k = smooth(e * 1.25)[:, None]
    a = BAXIS[act] * (1 - k) + FL["axis"][act] * k
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    e1, e2 = perp(a)
    return a, e1, e2


def petal_points(t, act, e, swell, sw):
    """Six petals per active flower, opening by e: (points, normals, colours, glyph boost)."""
    Fa = len(act)
    a, e1, e2 = head_axis(act, e)
    size = FL["size"][act]
    k = np.arange(6)
    outer = (k % 2 == 0)
    phi = FL["spin"][act][:, None] + k[None, :] * (np.pi / 3)
    rdir = np.cos(phi)[..., None] * e1[:, None, :] + np.sin(phi)[..., None] * e2[:, None, :]
    tdir = -np.sin(phi)[..., None] * e1[:, None, :] + np.cos(phi)[..., None] * e2[:, None, :]
    # each petal opens on its own: outer three lead, inner three follow
    lag = np.where(outer, 0.0, 0.22)[None, :] + 0.05 * np.sin(FL["spin"][act][:, None] * 9 + k[None, :] * 2.3)
    ep = smoother((e[:, None] - lag) / (1 - 0.22))
    ee = ep[..., None]
    U2 = UU[None, None, :]
    thO = (1.12 + 0.04 * np.sin(W0 * 2 * t + FL["ph"][act, 0]))[:, None, None]
    rec = np.where(outer, 0.62, 0.50)[None, :, None]
    closed = 0.10 - 0.30 * U2 ** 1.5                           # closed bud: petals meet in a point
    peel = smooth((ee * 1.6 - (1 - U2) * 0.6))                 # the tip parts first, the base follows
    theta = closed + (thO - closed) * (0.55 * ee + 0.45 * peel) + rec * ee * U2 ** 2
    ct, st = np.cos(theta)[..., None], np.sin(theta)[..., None]
    along = ct * a[:, None, None, :] + st * rdir[:, :, None, :]
    L = (size[:, None] * 0.60 * (0.45 + 0.55 * swell[:, None]) * np.where(outer, 1.0, 0.94)[None, :])
    seg = along * (L[..., None, None] / (NU - 1))
    centre = np.cumsum(seg, axis=2) - seg
    nrm = -ct * rdir[:, :, None, :] + st * a[:, None, None, :]
    wd = size[:, None] * 0.16 * np.where(outer, 0.85, 1.1)[None, :] * (0.45 + 0.55 * swell[:, None])
    w = wd[..., None] * WU[None, None, :] * (0.55 + 0.45 * ee)         # a slim closed bud
    cup = 0.38 * w[..., None] * VV ** 2 * (1 - 0.5 * UU)[None, None, :, None] * (1 + 1.6 * (1 - ee[..., None]))
    ruff = 0.012 * size[:, None, None, None] * np.sin(UU[:, None] * 24 + VV[None, :] * 2) * np.abs(VV) ** 3 * ee[..., None]
    base = FL["pos"][act] + sw[act] + a * 0.02 - FL["axis"][act] * 0.0
    P = (base[:, None, None, None, :] + centre[:, :, :, None, :]
         + tdir[:, :, None, None, :] * (w[..., None] * VV)[..., None]
         + nrm[:, :, :, None, :] * (cup + ruff)[..., None])
    N = np.cross(np.gradient(P, axis=2), np.gradient(P, axis=3))
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    C = PCOL[act]
    budc = mixc((0.24, 0.45, 0.18), C.mean(axis=(1, 2, 3)) * 0.70, 0.70)        # (Fa,3) tinted buds
    mixw = (ep ** 0.7)[:, :, None, None, None]                  # colour floods in as each petal parts
    C = budc[:, None, None, None, :] * (1 - mixw) + C * mixw
    return P.reshape(-1, 3), N.reshape(-1, 3), C.reshape(-1, 3)


def stamens(act, e, sw):
    """Six curving filaments with anthers, and a pistil, unfolding from each open flower."""
    out_P, out_N, out_C, out_M = [], [], [], []
    s = np.linspace(0, 1, 14, dtype=F32)
    A, E1, E2 = head_axis(act, e)
    for i, f in enumerate(act):
        ef = float(smooth((e[i] - 0.55) / 0.45))
        if ef <= 0:
            continue
        a, e1, e2, size = A[i], E1[i], E2[i], FL["size"][f]
        base = FL["pos"][f] + sw[f] + a * 0.03
        for j in range(7):
            pist = j == 6
            ph = FL["spin"][f] + j * np.pi / 3 + 0.5
            r = math.cos(ph) * e1 + math.sin(ph) * e2
            L = size * (0.36 if pist else 0.31) * ef
            line = base + a * (L * s)[:, None] + (0 if pist else 1) * r * (0.11 * size * ef * s ** 1.5)[:, None]
            out_P.append(line); out_N.append(np.tile(a, (len(s), 1)))
            out_C.append(np.tile([0.88, 0.92, 0.70], (len(s), 1))); out_M.append(np.full(len(s), K.MID["filament"]))
            tip = line[-1]
            off = np.array([[0, 0, 0], [0.018, 0, 0], [-0.018, 0, 0], [0, 0.014, 0], [0, -0.014, 0], [0, 0, 0.015]], F32) * size
            out_P.append(tip + off); out_N.append(np.tile(a, (6, 1)))
            out_C.append(np.tile([0.72, 0.86, 0.32] if pist else [0.70, 0.30, 0.08], (6, 1)))
            out_M.append(np.full(6, K.MID["anther"]))
    if not out_P:
        return None
    return (np.concatenate(out_P).astype(F32), np.concatenate(out_N).astype(F32),
            np.concatenate(out_C).astype(F32), np.concatenate(out_M))


def project_b(P):
    yv = P[:, 1] * CPB - P[:, 2] * SPB
    zv = P[:, 1] * SPB + P[:, 2] * CPB
    f = 1.0 / (1.0 - PERSP * zv)
    return KX + KS * P[:, 0] * f, KY - KS * yv * f, zv


def bouquet_layer(cv, t):
    grow, swell, e = head_state(t)
    sw = sway(t)
    Ps, Ns, Cs, Ms, Sh = [], [], [], [], []                  # Sh: 0 matte, 1 petal, 2 satin

    vis = STEM["S"] <= grow[STEM["F"]]                         # stems, growing tip first
    if vis.any():
        P = STEM["P"][vis] + sw[STEM["F"][vis]] * STEM["W"][vis, None]
        C = STEM["C"][vis].copy()
        tip = (grow[STEM["F"][vis]] - STEM["S"][vis] < 0.035) & (grow[STEM["F"][vis]] < 1)
        C[tip] = (0.78, 1.0, 0.45)
        Ps.append(P); Ns.append(STEM["N"][vis]); Cs.append(C)
        Ms.append(np.full(len(P), K.MID["stem"])); Sh.append(np.zeros(len(P)))

    g = smooth((t - LEAF["tl"]) / LEAF["dur"])                 # leaves unfurl from their base
    lv = g > 0.02
    if lv.any():
        fsw = np.where(LEAF["F"][lv, None] >= 0, sw[np.maximum(LEAF["F"][lv], 0)], 0) * LEAF["W"][lv, None]
        P = LEAF["start"][lv] + LEAF["rel"][lv] * g[lv, None] + fsw
        Ps.append(P); Ns.append(LEAF["N"][lv]); Cs.append(LEAF["C"][lv])
        Ms.append(np.full(len(P), K.MID["leaf"])); Sh.append(np.zeros(len(P)))

    act = np.flatnonzero(t >= FL["t_bud"])                     # heads: buds, then blooms
    if len(act):
        P, N, C = petal_points(t, act, e[act], swell[act], sw)
        Ps.append(P); Ns.append(N); Cs.append(C)
        Ms.append(np.full(len(P), K.MID["petal"])); Sh.append(np.ones(len(P)))
        stm = stamens(act, e[act], sw)
        if stm is not None:
            Ps.append(stm[0]); Ns.append(stm[1]); Cs.append(stm[2]); Ms.append(stm[3]); Sh.append(np.zeros(len(stm[0])))

    if t >= 5.0:                                               # the ribbon wraps on, then the bow
        band = smooth((t - 5.0) / 0.7)
        pop = float(backout((t - 5.55) / 0.45))
        tail = smooth((t - 5.65) / 0.6)
        k = RIB["kind"]
        m = ((k < 1) & (RIB["s"] <= band)) | ((k == 2) & (pop > 0.01)) | ((k == 3) & (RIB["s"] - 1 <= tail))
        if m.any():
            P = RIB["P"][m].copy()
            bow = k[m] >= 2
            P[bow] = RIB["knot"] + (P[bow] - RIB["knot"]) * max(pop, 0.01)
            n = len(P)
            Ps.append(P); Ns.append(RIB["N"][m]); Cs.append(np.tile([0.98, 0.62, 0.74], (n, 1)))
            Ms.append(np.full(n, K.MID["ribbon"])); Sh.append(np.full(n, 2))

    if not Ps:
        return
    P, N, C, M, SH = (np.concatenate(x) for x in (Ps, Ns, Cs, Ms, Sh))
    xs, ys, zv = project_b(P)
    cells, w = K.zbuffer(xs, ys, zv)
    n = N[w] / (np.linalg.norm(N[w], axis=1, keepdims=True) + 1e-9)
    back = n @ VIEW < 0
    n[back] *= -1
    c, sh = C[w], SH[w]
    key = np.clip(n @ KEY, 0, 1)
    lit = c * (0.20 + 0.70 * key + 0.18 * np.clip(n @ FILL, 0, 1))[:, None]
    lit += (c * 0.30 * np.clip(-(N[w] @ KEY) * np.where(back, -1, 1), 0, 1)[:, None]) * (sh == 1)[:, None]  # light through petals
    spec = np.clip(n @ HALF, 0, 1)
    lit += (np.where(sh == 1, 0.30 * spec ** 22, 0) + np.where(sh == 2, 0.55 * spec ** 14, 0))[:, None]
    lit[back & (sh == 1)] *= np.array([0.78, 0.82, 0.74], F32)
    lit = np.clip(lit, 0, 1.0) ** 0.95
    dens = np.clip(0.38 + 0.75 * lum(lit), 0.35, 1.0)
    dens[sh == 1] = np.maximum(dens[sh == 1], 0.62)
    cv[0].reshape(-1)[cells] = dens
    cv[1].reshape(-1, 3)[cells] = lit
    cv[2].reshape(-1)[cells] = M[w]


# --------------------------------------------------------- atmosphere ---
def build_background():
    global BG, BOKEH, MOTES, FALL
    X, Y, W, H = K.X, K.Y, K.W, K.H
    col = mixc((0.03, 0.02, 0.06), (0.06, 0.03, 0.05), Y / H)
    halo = np.exp(-((X - W / 2) / (0.48 * W)) ** 2 - ((Y - 0.42 * H) / (0.45 * H)) ** 2)
    col = col + halo[..., None] * np.array([0.16, 0.07, 0.12], F32)
    hv = hash2(K.RR, K.CC, 31.0)
    BG = dict(col=col, d=np.where(hv > 0.55, 0.32, 0.02).astype(F32))
    rng = np.random.default_rng(17)
    BOKEH = [dict(x=rng.uniform(0.05, 0.95), y=rng.uniform(0, 1), r=rng.uniform(0.035, 0.08),
                  c=np.array([(1.0, 0.55, 0.62), (1.0, 0.78, 0.45), (0.55, 0.70, 1.0), (0.85, 0.55, 1.0)][i % 4], F32),
                  k=int(rng.integers(1, 3)), ph=rng.uniform(0, 6.28)) for i in range(16)]
    MOTES = dict(cx=rng.normal(0, 0.9, 50), cy=rng.uniform(1.4, 3.8, 50), a=rng.uniform(0.05, 0.25, (2, 50)),
                 k=rng.integers(1, 4, (3, 50)), ph=rng.uniform(0, 6.28, (3, 50)))
    FALL = [dict(x=rng.uniform(0.1, 0.9), p=i / 5, ph=rng.uniform(0, 6.28),
                 c=np.array(VARIETIES[["stargazer", "white", "red", "peach", "yellow"][i]][0], F32)) for i in range(5)]


def background_layer(cv, t):
    cv[0][:], cv[1][:], cv[2][:] = BG["d"], BG["col"], K.MID["bg"]
    for b in BOKEH:                                            # soft out-of-focus lights drifting up
        x = (b["x"] + 0.03 * math.sin(W0 * b["k"] * t + b["ph"])) * K.W
        y = (((b["y"] - b["k"] * t / LOOP) % 1.0) * 1.2 - 0.1) * K.H
        r = b["r"] * min(K.W, K.H)
        rs, cs = window(x - r, x + r, y - r, y + r)
        sc = sub(cv, rs, cs)
        d = np.hypot(K.X[rs, cs] - x, K.Y[rs, cs] - y) / r
        g = 0.20 * np.clip(1 - d ** 6, 0, 1) * (0.55 + 0.45 * math.sin(W0 * 2 * t + b["ph"]))
        putmax(sc, g, b["c"] * (g * 1.6)[..., None], "bokeh", 0.05)


def atmosphere_layer(cv, t):
    m = MOTES                                                  # pollen drifting round the bouquet
    x = m["cx"] + m["a"][0] * np.sin(W0 * m["k"][0] * t + m["ph"][0])
    y = m["cy"] + m["a"][1] * np.sin(W0 * m["k"][1] * t + m["ph"][1])
    xs, ys, _ = project_b(np.stack([x, y, 0.6 + 0 * x], 1).astype(F32))
    tw = (0.5 + 0.5 * np.sin(W0 * m["k"][2] * 3 * t + m["ph"][2])) ** 3
    fade = float(smooth((t - 3.0) / 2.0))
    putcells(cv, xs, ys, 0.3 + 0.6 * tw * fade, np.outer(tw * fade, [1.0, 0.85, 0.45]), "pollen")
    for p in FALL:                                             # a few loose petals tumbling down
        q = (p["p"] + t / LOOP) % 1.0
        x = (p["x"] + 0.06 * math.sin(2 * math.pi * 2 * q + p["ph"])) * K.W
        y = (q * 1.2 - 0.1) * K.H
        tumble = abs(math.cos(2 * math.pi * 3 * q + p["ph"]))
        rx, ry = K.CW * (0.6 + 1.0 * tumble), K.CH * 0.55
        rs, cs = window(x - rx, x + rx, y - ry, y + ry)
        sc = sub(cv, rs, cs)
        m_ = ((K.X[rs, cs] - x) / rx) ** 2 + ((K.Y[rs, cs] - y) / ry) ** 2 < 1
        K.put(sc, m_, 0.8 * fade, p["c"] * (0.55 + 0.45 * tumble) * fade, "petal")
    if t < GROW + 1.0:                                         # a sparkle burst as each lily opens
        done = FL["t_open"] + 1.5
        for f in np.flatnonzero((t > done) & (t < done + 0.9)):
            age = (t - done[f]) / 0.9
            hx, hy, _ = project_b((FL["pos"][f] + FL["axis"][f] * 0.15)[None])
            ang = np.linspace(0, 2 * np.pi, 18, endpoint=False) + f
            r = KS * 0.45 * FL["size"][f] * age ** 0.6
            b = (1 - age) ** 1.5 * np.ones(18)
            tipc = np.array(VARIETIES[FL["var"][f]][1], F32)
            putcells(cv, hx[0] + r * np.cos(ang), hy[0] + 0.8 * r * np.sin(ang), 0.3 + 0.6 * b,
                     np.outer(b, tipc * 0.5 + 0.5), "spark")


# ---------------------------------------------------------------- scene ---
def setup(cols, rows, cw, ch):
    global KS, KX, KY, SAMP
    K.setup_grid(cols, rows, cw, ch)
    KS = min(0.200 * K.H, 0.30 * K.W)
    KX, KY = K.W / 2, 0.975 * K.H - 0.25 * KS
    SAMP = 2.0 * KS / min(K.CW, K.CH)
    build_flowers()
    build_stems()
    build_leaves()
    build_ribbon()
    build_background()


def render(t):
    cv = K.canvas()
    background_layer(cv, t)
    bouquet_layer(cv, t)
    atmosphere_layer(cv, t)
    return cv


LILIES = SimpleNamespace(
    title="A lily bouquet grows itself, then sways", setup=setup, render=render,
    loop=LOOP, intro=GROW, out="lily_bouquet.mp4", panel=(__file__, "def petal_points", "def project_b"))


if __name__ == "__main__":
    K.main(LILIES)
