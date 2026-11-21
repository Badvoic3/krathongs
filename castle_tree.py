#!/usr/bin/env python3
"""
castle_tree.py - a Christmas tree builds itself in front of a fairy-tale castle, then
the castle fireworks show loops forever. Coloured ASCII, on krathong_ascii.py's engine.

The tree is a 3-D point cloud (one fuzzy needle cone, trunk, skirt, garland swags, baubles, fairy
lights and gifts). It rises from the ground, gets dressed, is topped with a star, then
turns slowly while the fireworks play above the castle and the snow keeps falling.

    python castle_tree.py                 # live: the tree builds, then the show loops forever
    python castle_tree.py --video         # castle_tree.mp4  (build + first show + one loop)
    python castle_tree.py --png 30        # one still at t = 30 s
"""
import math
from types import SimpleNamespace

import numpy as np

import krathong_ascii as K
from krathong_ascii import F32, hash2, lum, mixc, putcells, putflat, putmax, window, sub

K.register_ramps({
    "needle": " .,xvV%&$W",
    "snowcap": " .:=##",
    "frost": " .:-+*#@",
    "bauble": " .oO0@",
    "tinsel": " .~*#@",
    "bulb": " .*oO@",
    "gift": " .:=+#%@",
    "gold": " .-=*#@",
    "star": " .+*#@",
    "stone": " .:-=+#%",
    "roof": " .-=^A#",
    "cwin": " =#@",
    "snow": " .,:;-=",
    "flake": " .'*o",
    "path": " .:-=+#",
    "iron": " .:|#",
    "lamp": " .oO@",
    "snowman": " .:;=+*#",
    "coal": " .o@",
    "shrub": " .:;+*#",
    "rail": " .-=#",
    "train": " .:=+#%@",
    "facade": " .:-=+#%",
    "shopwin": " =#@",
    "awning": " -=#",
    "sign": " .:=#",
    "trim": " -=#",
    "person": " .:;+#",
    "balloon": " .oO@",
})

BUILD = 12.5                      # the tree is finished
SHOW = K.LOOP                     # the fireworks show repeats every 10 s
LOOP = 20.0                       # scene loop: one turn of the tree = two shows
W0L = 2 * math.pi / LOOP
PITCH_T, PERSP_T = 0.24, 0.07                 # looking down a little, so the snow cap reads
CPT, SPT = math.cos(PITCH_T), math.sin(PITCH_T)
CONE_Y0, CONE_RB = 0.30, 1.08                  # the needles start above the trunk
TREE_YAW = 0.35                                # the tree stands still, turned a little
TOP, STAR_Y = 2.70, 2.86
LDIR = np.array([-0.55, 0.6, 0.6], F32) / np.linalg.norm([-0.55, 0.6, 0.6])
HALF = (LDIR + np.array([0, 0, 1], F32)) / np.linalg.norm(LDIR + np.array([0, 0, 1], F32))
PALETTE = [(1.0, 0.12, 0.10), (1.0, 0.75, 0.20), (0.20, 0.45, 1.0), (0.85, 0.88, 0.95),
           (1.0, 0.35, 0.70), (0.65, 0.30, 1.0)]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def backout(x):
    """Ease that overshoots a little, then settles: things 'pop' into place."""
    x = np.clip(x, 0, 1) - 1
    return 1 + 2.2 * x ** 3 + 1.2 * x ** 2


def rcone(y):
    """Radius of the needle cone at height y."""
    y = np.asarray(y, F32)
    return np.where((y >= CONE_Y0) & (y <= TOP), CONE_RB * (1 - (y - CONE_Y0) / (TOP - CONE_Y0)), 0).astype(F32)


# ------------------------------------------------------------------ tree ---
def build_tree():
    """Every surface point of the tree, with when and how it appears."""
    global TREE
    rng = np.random.default_rng(5)
    parts = []

    def grid(lu, lv):
        nu, nv = max(3, int(lu * SAMPT)), max(3, int(lv * SAMPT))
        return np.meshgrid(np.linspace(0, 1, nu, dtype=F32), np.linspace(0, 1, nv, dtype=F32), indexing="ij")

    def add(P, N, col, mid, kind, tap=0.0, cen=None, ph=0.0):
        P, N = P.reshape(-1, 3).astype(F32), N.reshape(-1, 3).astype(F32)
        n = len(P)
        N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
        parts.append(dict(P=P, N=N, C=np.broadcast_to(np.asarray(col, F32), (n, 3)).reshape(-1, 3)
                          if np.ndim(col) < 3 else np.asarray(col, F32).reshape(-1, 3),
                          M=np.full(n, K.MID[mid]), kind=np.full(n, kind),
                          tap=np.broadcast_to(np.asarray(tap, F32), (n,)).copy() if np.ndim(tap) == 0
                          else np.asarray(tap, F32).reshape(-1),
                          cen=np.broadcast_to(np.asarray(cen if cen is not None else (0, 0, 0), F32), (n, 3)).copy(),
                          ph=np.broadcast_to(np.asarray(ph, F32), (n,)).copy()))

    # one smooth needle cone with a fuzzy, bumpy surface: two-tone green and white flecks
    h = TOP - CONE_Y0
    U, V = grid(math.hypot(h, CONE_RB) * 1.4, 2 * math.pi * CONE_RB * 1.4)
    th = V * 2 * math.pi
    y = CONE_Y0 + U * h
    bump = (0.5 + 0.5 * np.sin(th * 11 + y * 9)) * (0.5 + 0.5 * np.sin(th * 5 - y * 13 + 1.1))
    r = rcone(y) * (0.97 + 0.08 * bump) + rng.normal(0, 0.018, U.shape) * (1 - U)
    P = np.stack([r * np.cos(th), y, r * np.sin(th)], -1)
    N = np.stack([np.cos(th), np.full_like(th, CONE_RB / h), np.sin(th)], -1)
    noise = rng.uniform(size=U.shape).astype(F32)
    col = np.where((noise > 0.5)[..., None], np.array([0.22, 0.70, 0.28], F32), np.array([0.05, 0.30, 0.11], F32))
    col = col * (0.75 + 0.5 * rng.uniform(size=U.shape))[..., None].astype(F32)
    col = np.where((noise > 0.97)[..., None], np.array([0.88, 0.92, 0.98], F32), col)
    add(P, N, col, "needle", 0)

    U, V = grid(0.32, 2 * math.pi * 0.13)                     # trunk
    th = V * 2 * math.pi
    add(np.stack([0.13 * np.cos(th), U * 0.32, 0.13 * np.sin(th)], -1),
        np.stack([np.cos(th), 0 * th, np.sin(th)], -1), (0.35, 0.20, 0.10), "needle", 0)
    U, V = grid(0.95, 2 * math.pi * 0.95)                     # tree skirt
    th, r = V * 2 * math.pi, 0.14 + U * 0.81
    skirt = np.where((r > 0.86)[..., None], np.array([0.95, 0.95, 0.98], F32), np.array([0.62, 0.06, 0.10], F32))
    add(np.stack([r * np.cos(th), 0.015 + 0 * r, r * np.sin(th)], -1),
        np.stack([0 * th, 1 + 0 * th, 0 * th], -1), skirt, "gift", 0)

    # thick gold garland swags draped round the tree, drawn on level by level from the bottom
    for k in range(5):
        yk = 0.62 + k * 0.40
        nsw = 8 - k
        ns = int(2 * math.pi * float(rcone(np.array([yk]))[0]) * SAMPT * 2) + 24
        a = np.linspace(0, 2 * np.pi, ns, endpoint=False, dtype=F32)
        y = yk - (0.16 - 0.02 * k) * np.abs(np.sin(nsw * a / 2)) + 0.05 * a / (2 * np.pi)
        r = rcone(y) + 0.045
        radial = np.stack([np.cos(a), 0 * a, np.sin(a)], -1)
        centre = radial * r[:, None] + np.stack([0 * a, y, 0 * a], -1)
        for q in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            off = radial * 0.03 * math.cos(q) + np.array([0, 1, 0], F32) * 0.03 * math.sin(q)
            tcol = np.array([1.0, 0.80, 0.32], F32) * (0.75 + 0.25 * np.sin(a * 60 + q))[:, None]
            add(centre + off, off, tcol, "tinsel", 2, tap=7.0 + 0.45 * k + 0.4 * a / (2 * np.pi))

    # baubles pop on, bottom to top, each with its own colour and a specular glint
    for i in range(52):
        y = float(rng.uniform(0.45, 2.35))
        th0 = float(rng.uniform(0, 2 * np.pi))
        rad = 0.075 * (1 - 0.4 * y / 2.5)
        cen = np.array([math.cos(th0), 0, math.sin(th0)], F32) * (float(rcone(y)) * 0.93 + rad * 0.6) + np.array([0, y, 0], F32)
        U, V = grid(math.pi * rad, 2 * math.pi * rad)
        a, b = U * math.pi, V * 2 * math.pi
        n = np.stack([np.sin(a) * np.cos(b), np.cos(a), np.sin(a) * np.sin(b)], -1)
        add(cen + rad * n, n, [(1.0, 0.12, 0.10), (0.88, 0.90, 0.96), (1.0, 0.12, 0.10), (1.0, 0.75, 0.20),
                               (0.20, 0.45, 1.0)][i % 5], "bauble", 1,
            tap=7.6 + 1.5 * (y - 0.45) / 1.9 + rng.uniform(0, 0.8), cen=cen)

    # fairy lights twinkle on from the bottom up
    for i in range(110):
        y = float(rng.uniform(0.35, 2.55))
        th0 = float(rng.uniform(0, 2 * np.pi))
        d = np.array([math.cos(th0), 0, math.sin(th0)], F32)
        warm = rng.uniform() < 0.6
        col = (1.0, 0.82, 0.45) if warm else PALETTE[int(rng.integers(0, len(PALETTE)))]
        add((d * (float(rcone(y)) + 0.03) + np.array([0, y, 0], F32))[None], d[None], col, "bulb", 3,
            tap=9.0 + 1.6 * y / 2.6, ph=th0)

    # gifts: wrapped boxes with a crossed ribbon and a bow, popping in around the base
    for i, (ang, rr, w, d, h, col, rib) in enumerate([
            (0.3, 1.05, 0.38, 0.32, 0.30, PALETTE[0], PALETTE[1]), (1.4, 1.15, 0.30, 0.30, 0.24, PALETTE[2], PALETTE[3]),
            (2.5, 1.00, 0.34, 0.28, 0.36, (0.10, 0.55, 0.25), PALETTE[0]), (3.5, 1.20, 0.28, 0.26, 0.22, PALETTE[5], PALETTE[1]),
            (4.5, 1.05, 0.40, 0.30, 0.28, PALETTE[1], PALETTE[0]), (5.5, 1.15, 0.26, 0.26, 0.30, PALETTE[4], PALETTE[3])]):
        e1 = np.array([math.cos(ang + 0.4), 0, math.sin(ang + 0.4)], F32)
        e2 = np.array([-math.sin(ang + 0.4), 0, math.cos(ang + 0.4)], F32)
        up = np.array([0, 1, 0], F32)
        cen = np.array([rr * math.cos(ang), h / 2, rr * math.sin(ang)], F32)
        faces = [(up, e1, e2, h / 2, w / 2, d / 2, True)]
        for nrm, ta, ext_n, ext_t in [(e1, e2, w / 2, d / 2), (-e1, e2, w / 2, d / 2),
                                      (e2, e1, d / 2, w / 2), (-e2, e1, d / 2, w / 2)]:
            faces.append((nrm, ta, up, ext_n, ext_t, h / 2, False))
        for nrm, ta, tb, en, ea, eb, top in faces:
            U, V = grid(2 * ea, 2 * eb)
            u, v = U * 2 - 1, V * 2 - 1
            P = cen + nrm * en + ta * (u * ea)[..., None] + tb * (v * eb)[..., None]
            band = (np.abs(u) < 0.14) | ((np.abs(v) < 0.14) if top else False)
            cc = np.where(band[..., None], np.array(rib, F32), np.array(col, F32))
            add(P, np.broadcast_to(nrm, P.shape), cc, "gift", 1, tap=10.0 + i * 0.25, cen=cen)
        for sgn in (-1, 1):                                     # bow
            bc = cen + up * (h / 2 + 0.04) + e1 * (0.06 * sgn)
            U, V = grid(math.pi * 0.05, 2 * math.pi * 0.05)
            a, b = U * math.pi, V * 2 * math.pi
            n = np.stack([np.sin(a) * np.cos(b), np.cos(a), np.sin(a) * np.sin(b)], -1)
            add(bc + 0.05 * n, n, rib, "gift", 1, tap=10.0 + i * 0.25, cen=cen)

    TREE = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def project_t(P):
    yv = P[:, 1] * CPT - P[:, 2] * SPT
    zv = P[:, 1] * SPT + P[:, 2] * CPT
    f = 1.0 / (1.0 - PERSP_T * zv)
    return KXT + KST * P[:, 0] * f, KYT - KST * yv * f, zv


def tree_layer(cv, t, flash):
    """Grow, pose, z-buffer and light the tree for time t.

    The build swells the whole tree up from a flat white-capped disc, like a green pile
    of snow being pushed upward; the snow cap melts off as it reaches full height."""
    T = TREE
    bh = TOP * float(smooth((t - 1.0) / 6.0))                  # height of the rising cut
    kd = T["kind"]
    vis = np.where(kd == 0, T["P"][:, 1] <= bh, t >= T["tap"])
    idx = np.flatnonzero(vis)
    P = T["P"][idx].copy()
    capP = np.zeros((0, 3), F32)
    if 0.0 < bh < TOP - 0.02:                                  # heaped snow on the cut face
        r = float(rcone(np.array([bh], F32))[0])
        q = CAP_UV[:, 0] * r
        lump = 0.5 + 0.5 * np.sin(CAP_UV[:, 1] * 7 + q * 9) * np.sin(CAP_UV[:, 1] * 3 - q * 5 + 1.3)
        capP = np.concatenate([                           # two stacked, broken layers of snow
            np.stack([q * np.cos(CAP_UV[:, 1]), bh + 0.03 * lump + 0 * q, q * np.sin(CAP_UV[:, 1])], 1),
            np.stack([0.72 * q * np.cos(CAP_UV[:, 1]), bh + 0.13 + 0.03 * lump, 0.72 * q * np.sin(CAP_UV[:, 1])], 1)]
        ).astype(F32)
        keep_ = np.concatenate([lump > 0.30, lump > 0.42])     # gaps in the snow
        capP = capP[keep_]
    pop = kd[idx] == 1
    if pop.any():
        sc = backout((t - T["tap"][idx][pop]) / 0.35)
        cen = T["cen"][idx][pop]
        P[pop] = cen + (P[pop] - cen) * sc[:, None]
    R = K.Ry(TREE_YAW)
    ex = train_points(t)                                       # track + train don't turn
    if len(capP):
        n_ = len(capP)
        ex = dict(P=np.vstack([capP @ R.T, ex["P"]]), N=np.vstack([np.tile([0, 1, 0], (n_, 1)).astype(F32), ex["N"]]),
                  C=np.vstack([np.tile([0.92, 0.96, 1.0], (n_, 1)).astype(F32), ex["C"]]),
                  K=np.concatenate([np.full(n_, 7), ex["K"]]), M=np.concatenate([np.full(n_, K.MID["frost"]), ex["M"]]))
    nt = len(idx)
    xs, ys, zv = project_t(np.vstack([P @ R.T, ex["P"]]))
    cells, w = K.zbuffer(xs, ys, zv)
    tr = w < nt
    g, e = idx[w[tr]], w[~tr] - nt
    m = len(w)
    n, base = np.empty((m, 3), F32), np.empty((m, 3), F32)
    mat, kk = np.empty(m, np.int64), np.empty(m, np.int64)
    yy, ph = np.full(m, 99.0, F32), np.zeros(m, F32)
    n[tr], base[tr], mat[tr], kk[tr] = T["N"][g] @ R.T, T["C"][g], T["M"][g], kd[g]
    yy[tr], ph[tr] = P[w[tr], 1], T["ph"][g]
    n[~tr], base[~tr], mat[~tr], kk[~tr] = ex["N"][e], ex["C"][e], ex["M"][e], ex["K"][e]
    n *= np.where(n[:, 1] * SPT + n[:, 2] * CPT < 0, -1.0, 1.0)[:, None].astype(F32)
    lam = np.clip(n @ LDIR, 0, 1)
    on = float(np.clip((t - 9.0) / 1.5, 0, 1))                 # fairy lights glow
    lit = base * ((0.22 + 0.85 * lam)[:, None] * np.array([0.75, 0.80, 1.0], F32)
                  + on * 0.25 * np.array([1.0, 0.75, 0.45], F32))
    shiny = mat == K.MID["bauble"]
    if shiny.any():                                            # glints slide as the lights move
        sw = 0.7 * math.sin(W0L * 3 * t)
        hl = np.array([math.sin(sw) * 0.6, 0.55, math.cos(sw)], F32)
        lit[shiny] += (np.clip(n[shiny] @ (hl / np.linalg.norm(hl)), 0, 1) ** 24 * 1.1)[:, None]
    if flash is not None:
        lit += base * flash * 0.5
    if 0.0 < bh < TOP - 0.02:                                  # snow spilling over the rim, lumpy edge
        ang = np.arctan2(n[:, 2], n[:, 0])
        depth = 0.03 + 0.06 * (0.5 + 0.5 * np.sin(ang * 9 + yy * 13)) + 0.03 * np.sin(ang * 23)
        rim = (kk == 0) & (bh - yy < depth)
        lit[rim], mat = np.array([0.80, 0.84, 0.90], F32), np.where(rim, K.MID["snowcap"], mat)
    capc = kk == 7
    if capc.any():
        lit[capc] = base[capc] * (0.62 + 0.18 * np.clip(n[capc] @ LDIR, 0, 1))[:, None]
        mat = np.where(capc, K.MID["snowcap"], mat)
    dens = np.clip(0.30 + 0.85 * lum(lit), 0.32, 1.0)
    needles = (kk == 0) & (mat == K.MID["needle"])
    dens[needles] = np.clip(0.62 + 0.7 * lum(lit[needles]), 0.6, 1.0)   # packed W%&$ needles
    dens[capc | (mat == K.MID["snowcap"])] = 0.97
    if t >= BUILD:                                             # a glittering wave rises up the tree
        yw = TOP * 1.25 * ((t - BUILD) * 4 / LOOP % 1.0) - 0.15
        band = np.exp(-((yy - yw) / 0.10) ** 2) * (kk == 0)
        hit = band * (hash2(cells.astype(F32), 3.0, 5.0) > 0.55) > 0.35
        lit[hit], dens[hit] = np.array([1.0, 1.0, 0.92], F32), 1.0
    bulbs = kk == 3
    if bulbs.any():                                            # lights chase in spirals up the tree
        wave = W0L * 8 * t - 2 * np.pi * (1.3 * yy[bulbs] / TOP + ph[bulbs] / (2 * np.pi))
        tw = np.clip(np.cos(wave), 0, 1) ** 3
        lit[bulbs] = base[bulbs] * (0.35 + 0.75 * tw)[:, None] + (0.35 * tw)[:, None]
        dens[bulbs] = 1.0
    fleck = (kk == 0) & (hash2(cells.astype(F32), 1.0, 2.0) > 0.93)  # white snow flecks in the needles
    lit[fleck] = np.array([0.80, 0.86, 0.95], F32)
    glow = kk == 6                                             # train headlight
    lit[glow], dens[glow] = base[glow], 1.0
    global TREE_CELLS
    TREE_CELLS = cells
    cv[0].reshape(-1)[cells] = dens
    cv[1].reshape(-1, 3)[cells] = np.clip(lit * 1.15, 0, 1.3)
    cv[2].reshape(-1)[cells] = mat


def focus_pass(cv, t):
    """Make the tree the star: everything that isn't the tree (or its star) is pushed back,
    dimmer, less saturated and drawn with lighter glyphs, under a soft spotlight centred on
    the tree. The tree also gets a glow behind it, white while it builds and warm once its
    lights come on."""
    keep = np.zeros(K.ROWS * K.COLS, bool)
    if TREE_CELLS is not None:
        keep[TREE_CELLS] = True
    m = cv[2].reshape(-1)
    keep |= m == K.MID["star"]
    bg = ~keep
    F = FOCUS.reshape(-1)[bg]
    F = np.where(m[bg] == K.MID["firework"], np.maximum(F, 0.72), F)   # the show stays bright-ish
    col = cv[1].reshape(-1, 3)[bg]
    l = lum(col)[:, None]
    col = (l + (col - l) * (0.45 + 0.4 * F)[:, None]) * F[:, None]
    on = float(np.clip((t - 9.0) / 1.5, 0, 1))
    halo = HALO.reshape(-1)[bg]
    col += halo[:, None] * mixc((0.75, 0.85, 1.0), (1.0, 0.72, 0.35), on) * (0.10 + 0.16 * on)
    cv[1].reshape(-1, 3)[bg] = col
    cv[0].reshape(-1)[bg] *= 0.6 + 0.4 * F


def solid_points(SAMP_):
    """Little point-cloud builders for boxes, cylinders and discs (local x fwd, y up, z out)."""
    def grid(lu, lv):
        nu, nv = max(2, int(lu * SAMP_) + 1), max(2, int(lv * SAMP_) + 1)
        return np.meshgrid(np.linspace(0, 1, nu, dtype=F32), np.linspace(0, 1, nv, dtype=F32), indexing="ij")

    def box(lo, hi, col):
        lo, hi = np.asarray(lo, F32), np.asarray(hi, F32)
        out = []
        for ax in range(3):
            a, b = [i for i in range(3) if i != ax]
            U, V = grid(hi[a] - lo[a], hi[b] - lo[b])
            for side in (0, 1):
                P = np.zeros(U.shape + (3,), F32)
                P[..., ax] = hi[ax] if side else lo[ax]
                P[..., a], P[..., b] = lo[a] + U * (hi[a] - lo[a]), lo[b] + V * (hi[b] - lo[b])
                N = np.zeros_like(P)
                N[..., ax] = 1 if side else -1
                out.append((P.reshape(-1, 3), N.reshape(-1, 3), np.tile(np.asarray(col, F32), (P[..., 0].size, 1))))
        return out

    def cyl(axis, c, r, a0, a1, col):
        U, V = grid(a1 - a0, 2 * math.pi * r)
        th = V * 2 * math.pi
        P = np.zeros(U.shape + (3,), F32)
        o = [i for i in range(3) if i != axis]
        P[..., axis] = a0 + U * (a1 - a0)
        P[..., o[0]], P[..., o[1]] = c[0] + r * np.cos(th), c[1] + r * np.sin(th)
        N = np.zeros_like(P)
        N[..., o[0]], N[..., o[1]] = np.cos(th), np.sin(th)
        return [(P.reshape(-1, 3), N.reshape(-1, 3), np.tile(np.asarray(col, F32), (U.size, 1)))]

    def disc(axis, c, r, at, sgn, col):
        U, V = grid(r, 2 * math.pi * r)
        th, rr = V * 2 * math.pi, U * r
        P = np.zeros(U.shape + (3,), F32)
        o = [i for i in range(3) if i != axis]
        P[..., axis] = at
        P[..., o[0]], P[..., o[1]] = c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)
        N = np.zeros_like(P)
        N[..., axis] = sgn
        cc = np.where((U > 0.45)[..., None], np.asarray(col[0], F32), np.asarray(col[1], F32))
        return [(P.reshape(-1, 3), N.reshape(-1, 3), cc.reshape(-1, 3))]

    return box, cyl, disc


def build_train():
    """A toy steam train and its ring of track around the tree."""
    global TRACK, CARS
    box, cyl, disc = solid_points(SAMPT)
    pts = []
    for r in (1.48, 1.62):                                     # rails
        n = int(2 * math.pi * r * SAMPT * 1.2)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=F32)
        pts.append((np.stack([r * np.cos(th), 0.025 + 0 * th, r * np.sin(th)], 1),
                    np.tile([0, 1, 0], (n, 1)).astype(F32), np.tile([0.62, 0.62, 0.70], (n, 1)).astype(F32), 5))
    for th0 in np.arange(0, 2 * np.pi, math.radians(6)):        # sleepers
        f = np.array([-math.sin(th0), 0, math.cos(th0)], F32)
        rd = np.array([math.cos(th0), 0, math.sin(th0)], F32)
        for P, N, C in box((-0.025, 0, 1.42), (0.025, 0.018, 1.68), (0.36, 0.22, 0.12)):
            M = np.stack([f, [0, 1, 0], rd]).astype(F32)
            pts.append((P @ M, N @ M, C, 5))
    TRACK = dict(P=np.concatenate([p[0] for p in pts]), N=np.concatenate([p[1] for p in pts]),
                 C=np.concatenate([p[2] for p in pts]), K=np.concatenate([np.full(len(p[0]), p[3]) for p in pts]),
                 M=np.concatenate([np.full(len(p[0]), K.MID["rail"]) for p in pts]))

    red, black, gold = (0.85, 0.10, 0.08), (0.10, 0.10, 0.12), (1.0, 0.78, 0.30)

    def wheels(xs_):
        out = []
        for x in xs_:
            for zs in (-0.088, 0.088):
                out += disc(2, (x, 0.06), 0.05, zs, np.sign(zs), (black, red))
        return out

    loco = (box((-0.22, 0.05, -0.08), (0.22, 0.10, 0.08), black)
            + cyl(0, (0.17, 0.0), 0.07, -0.02, 0.22, red)
            + cyl(0, (0.17, 0.0), 0.073, 0.05, 0.065, gold) + cyl(0, (0.17, 0.0), 0.073, 0.15, 0.165, gold)
            + box((-0.22, 0.10, -0.085), (-0.02, 0.30, 0.085), red)
            + box((-0.24, 0.30, -0.095), (0.0, 0.32, 0.095), black)
            + box((-0.16, 0.20, 0.084), (-0.08, 0.26, 0.087), (1.0, 0.85, 0.5))   # cab window
            + cyl(1, (0.16, 0.0), 0.03, 0.22, 0.32, black) + cyl(1, (0.16, 0.0), 0.04, 0.32, 0.34, gold)
            + wheels((-0.15, -0.02, 0.12)))
    lamp = [(np.array([[0.235, 0.17, 0.0]], F32), np.array([[1, 0, 0]], F32), np.array([[1.0, 0.92, 0.55]], F32))]
    CARS = [dict(parts=loco, glow=lamp, L=0.46, tap=10.6)]
    for i, (body, cargo, rib) in enumerate([((0.10, 0.55, 0.25), (1.0, 0.20, 0.25), gold),
                                             ((0.20, 0.45, 1.0), (1.0, 0.80, 0.25), red),
                                             ((0.60, 0.30, 1.0), (0.20, 0.75, 0.45), (1.0, 1.0, 1.0))]):
        wag = (box((-0.16, 0.05, -0.08), (0.16, 0.19, 0.08), body)
               + box((-0.165, 0.19, -0.085), (0.165, 0.205, 0.085), gold)
               + box((-0.07, 0.205, -0.06), (0.07, 0.32, 0.06), cargo)
               + box((-0.012, 0.205, -0.062), (0.012, 0.325, 0.062), rib)
               + wheels((-0.10, 0.10)))
        CARS.append(dict(parts=wag, glow=[], L=0.32, tap=10.75 + 0.15 * i))
    for c in CARS:
        c["P"] = np.concatenate([p[0] for p in c["parts"]])
        c["N"] = np.concatenate([p[1] for p in c["parts"]])
        c["C"] = np.concatenate([p[2] for p in c["parts"]])
        c["K"] = np.full(len(c["P"]), 5)
        if c["glow"]:
            c["P"] = np.vstack([c["P"], c["glow"][0][0]])
            c["N"] = np.vstack([c["N"], c["glow"][0][1]])
            c["C"] = np.vstack([c["C"], c["glow"][0][2]])
            c["K"] = np.append(c["K"], 6)


TRACK_R = 1.55


def car_frames(t):
    """World placement of each car: the train runs two laps per scene loop."""
    th = math.pi / 2 - 2 * W0L * t
    out, gap = [], 0.0
    for i, c in enumerate(CARS):
        if i:
            gap += (CARS[i - 1]["L"] / 2 + c["L"] / 2 + 0.06) / TRACK_R
        a = th + gap
        fwd = np.array([math.sin(a), 0, -math.cos(a)], F32)
        rad = np.array([math.cos(a), 0, math.sin(a)], F32)
        out.append((c, rad * TRACK_R, np.stack([fwd, np.array([0, 1, 0], F32), rad])))
    return out


def train_points(t):
    Ps, Ns, Cs, Ks = [TRACK["P"]], [TRACK["N"]], [TRACK["C"]], [TRACK["K"]]
    for c, pos, M in car_frames(t):
        if t < c["tap"]:
            continue
        sc = float(backout((t - c["tap"]) / 0.35))
        P = c["P"].copy()
        P[:, 1] = 0.15 + (P[:, 1] - 0.15) * sc
        P[:, [0, 2]] *= sc
        Ps.append(pos + P @ M)
        Ns.append(c["N"] @ M)
        Cs.append(c["C"])
        Ks.append(c["K"])
    n = sum(len(p) for p in Ps)
    return dict(P=np.concatenate(Ps), N=np.concatenate(Ns), C=np.concatenate(Cs), K=np.concatenate(Ks),
                M=np.concatenate([TRACK["M"], np.full(n - len(TRACK["P"]), K.MID["train"])]))


def smoke_puffs(cv, t):
    """Puffs from the locomotive's chimney, rising and fading behind it."""
    if t < 10.8:
        return
    c, pos, M = car_frames(t)[0]
    stack = pos + np.array([0.16, 0.36, 0.0], F32) @ M
    x0, y0, _ = project_t(stack[None])
    i = np.arange(10)
    p = (i / 10 + 4 * t / LOOP) % 1.0
    back = np.array([-0.16, 0.0, 0.0], F32) @ M                 # drift behind the train
    bx, by, _ = project_t((stack + back)[None])
    dx, dy = float(bx[0] - x0[0]), float(by[0] - y0[0])
    x = x0[0] + dx * p * 2.0 + 10 * K.S * np.sin(p * 9 + i)
    y = y0[0] + dy * p * 2.0 - 70 * K.S * p
    b = (1 - p) ** 1.3
    putcells(cv, x, y, 0.25 + 0.5 * b, np.outer(b, [0.80, 0.82, 0.88]), "smoke")


def build_plaza():
    """Static 2-D plaza pieces: cobbled path, lit shrubs, lamp posts, snowmen."""
    global SHRUBS, LAMPS, SNOWMEN
    X, Y, W, H = K.X, K.Y, K.W, K.H
    gm = Y >= HOR
    q = np.clip((Y - HOR) / (H - HOR), 0, 1)
    hw = 0.04 * CSU + q * (0.30 * W - 0.04 * CSU)
    path = gm & (np.abs(X - W / 2) < hw)
    row = 6 / (q + 0.12)
    cpos = ((X - W / 2) / np.maximum(hw, 1) + 1) * 3.5 + 0.5 * (np.floor(row) % 2)
    mortar = ((cpos % 1) < 0.13) | ((row % 1) < 0.13)
    hv = hash2(K.RR, K.CC, 12.0)
    stone = mixc((0.42, 0.38, 0.44), (0.58, 0.52, 0.56), hv) * (0.35 + 0.4 * q)[..., None]
    stone = np.where(mortar[..., None], stone * 0.45, stone)
    stone = np.where((hv > 0.72)[..., None] & ~mortar[..., None], np.array([0.62, 0.66, 0.80], F32) * (0.5 + 0.4 * q)[..., None], stone)
    mound = np.exp(-(((X - 0) / (0.22 * W)) ** 2 + ((Y - H) / (0.07 * H)) ** 2)) +         np.exp(-(((X - W) / (0.22 * W)) ** 2 + ((Y - H) / (0.07 * H)) ** 2))
    k = GROUND
    pi = path[gm]
    k["col"] = k["col"] + (mound[gm] * 0.35)[:, None] * np.array([0.85, 0.9, 1.0], F32)
    k["col"][pi] = stone[gm][pi]
    k["d"] = np.where(pi, np.where(mortar[gm], 0.3, 0.55), k["d"] + 0.3 * mound[gm])
    k["mat"] = np.where(pi, K.MID["path"], k["mat"])

    # lamp posts, with warm pools of light on the snow
    LAMPS, parts, pool = [], [], np.zeros(X.shape, F32)
    off = max(1.25 * KST, 0.20 * W)
    for lx in (KXT - off, KXT + off):
        ly, LH = KYT - 0.07 * H, 0.24 * H
        pw = max(0.75 * K.CW, 0.008 * W)
        hwd, hh = max(3.0 * K.CW, 0.032 * W), max(3.5 * K.CH, 0.06 * H)
        top = ly - LH
        pole = (np.abs(X - lx) < pw) & (Y >= top) & (Y < ly)
        plinth = (np.abs(X - lx) < 2.5 * pw) & (Y >= ly - 0.03 * H) & (Y < ly)
        head = (np.abs(X - lx) < hwd) & (Y >= top - hh) & (Y < top)
        glass = head & (np.abs(X - lx) < hwd - 0.8 * K.CW) & (Y >= top - hh + 0.6 * K.CH) & (Y < top - 0.4 * K.CH)
        cap = (Y >= top - hh - 0.025 * H) & (Y < top - hh) & (np.abs(X - lx) < 1.2 * hwd * (1 - (top - hh - Y) / (0.025 * H)))
        rw = max(1.5 * K.CW, 0.025 * W)
        dw = np.hypot(X - lx, (Y - (ly - 0.55 * LH)) * 1.2)
        wreath = (dw < rw) & (dw > 0.5 * rw)
        berry = wreath & (hash2(K.RR, K.CC, 3.3) > 0.7)
        bow = (np.abs(X - lx) < 0.6 * rw) & (np.abs(Y - (ly - 0.55 * LH + rw)) < 0.5 * K.CH)
        for m, col, d, mid in [(pole | plinth | (head & ~glass) | cap, (0.42, 0.44, 0.52), 0.75, "iron"),
                               (wreath, (0.10, 0.45, 0.18), 0.7, "shrub"), (berry | bow, (0.95, 0.12, 0.12), 0.8, "coal"),
                               (glass, (1.0, 0.82, 0.45), 1.0, "lamp")]:
            parts.append((np.flatnonzero(m), d, np.asarray(col, F32), K.MID[mid]))
        pool += np.exp(-(((X - lx) / (0.10 * W)) ** 2 + ((Y - ly) / (0.03 * H)) ** 2))
        LAMPS.append((lx, top - hh / 2))
    k["pool"] = pool[gm]
    LAMPS = dict(at=LAMPS, parts=parts)

    # a row of lit shrubs along the castle wall
    rng = np.random.default_rng(21)
    sh = np.zeros(X.shape, bool)
    for bx in np.arange(0.06, 0.95, 0.07) * W:
        if abs(bx - W / 2) < 0.07 * W:
            continue
        bh, bw = 0.05 * H * rng.uniform(0.8, 1.15), 0.026 * W
        sh |= (Y >= HOR - bh) & (Y < HOR) & (np.abs(X - bx) < bw * (1 - (HOR - Y) / bh))
    lights = sh & (hash2(K.RR, K.CC, 5.5) > 0.78)
    shade = 0.6 + 0.4 * hash2(K.RR, K.CC, 6.6)
    SHRUBS = dict(i=np.flatnonzero(sh & ~lights), col=(np.array([0.05, 0.28, 0.12], F32) * shade[..., None])[sh & ~lights],
                  li=np.flatnonzero(lights), lh=hash2(K.RR, K.CC, 7.7)[lights])

    # two snowmen in the foreground
    parts = []
    off = min(max(1.85 * KST, 0.40 * W), 0.42 * W)
    for sx, face in ((KXT - off, 1), (KXT + off, -1)):
        su, sy = 0.052 * H, 0.985 * H
        r1, r2, r3 = su, 0.72 * su, 0.52 * su
        c1, c2 = sy - r1, sy - 2 * r1 - 0.8 * r2
        c3 = c2 - r2 - 0.8 * r3
        body = np.zeros(X.shape, bool)
        shade = np.zeros(X.shape, F32)
        for cy, r in ((c1, r1), (c2, r2), (c3, r3)):
            dx, dy = (X - sx) / r, (Y - cy) / r
            m = dx * dx + dy * dy < 1
            nz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
            shade = np.where(m, 0.55 + 0.45 * np.clip(-0.5 * dx - 0.5 * dy + 0.7 * nz, 0, 1), shade)
            body |= m
        dots = np.zeros(X.shape, bool)
        for px, py in [(0, c2 - 0.4 * r2), (0, c2), (0, c2 + 0.4 * r2),
                       (-0.3 * r3, c3 - 0.2 * r3), (0.3 * r3, c3 - 0.2 * r3)]:
            dots |= (np.abs(X - sx - px) < 0.5 * K.CW) & (np.abs(Y - py) < 0.5 * K.CH)
        nose = (face * (X - sx) > 0) & (face * (X - sx) < 1.1 * r3) & (np.abs(Y - c3) < 0.5 * K.CH)
        hat = ((np.abs(X - sx) < 0.75 * r3) & (np.abs(Y - (c3 - r3)) < 0.5 * K.CH)) |               ((np.abs(X - sx) < 0.45 * r3) & (Y < c3 - r3) & (Y > c3 - 2.0 * r3))
        band = (np.abs(X - sx) < 0.46 * r3) & (np.abs(Y - (c3 - 1.25 * r3)) < 0.5 * K.CH)
        scarf = ((np.abs(X - sx) < 0.75 * r3) & (np.abs(Y - (c2 - r2 + 0.1 * r3)) < 0.55 * K.CH)) |                 ((np.abs(X - sx + face * 0.45 * r3) < 0.55 * K.CW) & (Y > c2 - r2) & (Y < c2 - 0.1 * r2))
        snowc = np.array([0.90, 0.94, 1.0], F32) * shade[..., None]
        for m, col, d, mid in [(body & ~(dots | nose | scarf), snowc, 0.55, "snowman"),
                               (dots, (0.05, 0.05, 0.06), 0.8, "coal"), (nose, (1.0, 0.50, 0.10), 0.8, "coal"),
                               (hat & ~band, (0.07, 0.07, 0.09), 0.8, "iron"), (band | scarf, (0.90, 0.10, 0.12), 0.8, "coal")]:
            i = np.flatnonzero(m)
            c = np.broadcast_to(np.asarray(col, F32), X.shape + (3,)).reshape(-1, 3)[i]
            parts.append((i, d, c, K.MID[mid]))
        for sgn in (-1, 1):                                      # stick arms
            ax0, ay0 = sx + sgn * 0.9 * r2, c2 - 0.2 * r2
            P = np.stack([ax0 + sgn * np.linspace(0, 1.6 * r2, 12), ay0 - np.linspace(0, 0.9 * r2, 12)], 1)
            parts.append(("stroke", P, sgn))
    SNOWMEN = parts


def plaza_back(cv, t, flash):
    fl = np.zeros(3, F32) if flash is None else flash
    putflat(cv, SHRUBS["i"], 0.6, SHRUBS["col"] + fl * 0.15, "shrub")
    h = SHRUBS["lh"]
    tw = 0.5 + 0.5 * np.sin(W0L * (4 + np.floor(h * 5)) * t + h * 40)
    cols = np.array(PALETTE, F32)[(h * 97).astype(int) % len(PALETTE)]
    putflat(cv, SHRUBS["li"], 1.0, cols * (0.4 + 0.7 * tw)[:, None], "bulb")
    flick = 0.9 + 0.1 * math.sin(W0L * 11 * t)
    for i, d, col, mid in LAMPS["parts"]:
        cv[0].reshape(-1)[i] = d
        cv[1].reshape(-1, 3)[i] = col * (flick if mid == K.MID["lamp"] else 1.0) + fl * 0.2
        cv[2].reshape(-1)[i] = mid
    for lx, ly in LAMPS["at"]:
        r = 0.10 * K.H
        rs, cs = window(lx - 2.5 * r, lx + 2.5 * r, ly - 2.5 * r, ly + 2.5 * r)
        sc = sub(cv, rs, cs)
        g = 0.55 * flick * np.exp(-((K.X[rs, cs] - lx) ** 2 + (K.Y[rs, cs] - ly) ** 2) / r ** 2)
        putmax(sc, g, np.array([1.0, 0.78, 0.40], F32) * g[..., None], "glow", 0.12)


def plaza_front(cv, t, flash):
    fl = np.zeros(3, F32) if flash is None else flash
    for part in SNOWMEN:
        if isinstance(part[0], str):
            K.stroke(cv, part[1], 0.5 * K.CW, 0.7, (0.40, 0.24, 0.12), "iron", True)
            continue
        i, d, col, mid = part
        cv[0].reshape(-1)[i] = d
        cv[1].reshape(-1, 3)[i] = col + fl * 0.25
        cv[2].reshape(-1)[i] = mid


def raster(cv, prims, ox, oy, u, flip, tint):
    """Rasterise a little 2-D sprite (ellipses, shaded balls, rects, triangles) at (ox, oy),
    in units of u pixels, mirrored when flip = -1."""
    xs = [v for p in prims for v in p[1][0::2]]
    ys = [v for p in prims for v in p[1][1::2]]
    pad = 1.5
    x0, x1 = min(xs) - pad, max(xs) + pad
    rs, cs = window(ox + min(x0 * flip, x1 * flip) * u, ox + max(x0 * flip, x1 * flip) * u,
                    oy + (min(ys) - pad) * u, oy + (max(ys) + pad) * u)
    sc = sub(cv, rs, cs)
    if sc[0].size == 0:
        return
    lx, ly = (K.X[rs, cs] - ox) / u * flip, (K.Y[rs, cs] - oy) / u
    mx, my = 0.5 * K.CW / u, 0.5 * K.CH / u                    # never thinner than a cell
    for kind, g, col, d, mid in prims:
        col = np.asarray(col, F32)
        if kind in ("ell", "ball"):
            cx, cy, rx, ry = g
            ex, ey = (lx - cx) / max(rx, mx), (ly - cy) / max(ry, my)
            m = ex * ex + ey * ey < 1
            if kind == "ball":
                nz = np.sqrt(np.clip(1 - ex * ex - ey * ey, 0, 1))
                col = col * (0.55 + 0.45 * np.clip(-0.5 * ex * flip - 0.5 * ey + 0.7 * nz, 0, 1))[..., None]
        elif kind == "rect":
            ax, bx, ay, by = g
            m = (lx >= ax - mx) & (lx <= bx + mx) & (ly >= ay - my) & (ly <= by + my) if bx - ax < 2 * mx \
                else (lx >= ax) & (lx <= bx) & (ly >= ay) & (ly <= by)
        else:                                                    # triangle
            (ax, ay, bx, by, qx, qy) = g
            d1 = (lx - bx) * (ay - by) - (ax - bx) * (ly - by)
            d2 = (lx - qx) * (by - qy) - (bx - qx) * (ly - qy)
            d3 = (lx - ax) * (qy - ay) - (qx - ax) * (ly - ay)
            m = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
        K.put(sc, m, d, col + tint, mid)


def walker(t):
    """The snowman who waddles around the tree: angle, screen spot, size, depth (-1 back .. 1 front)."""
    phi = W0L * t + 0.8
    dz = math.sin(phi)
    x = KXT + min(max(1.95 * KST, 0.34 * K.W), 0.40 * K.W) * math.cos(phi)
    y = KYT - 0.055 * K.H + 0.06 * K.H * dz
    return phi, x, y, 0.040 * K.H * (0.8 + 0.2 * dz), dz


def walking_snowman(cv, t, flash, front):
    phi, x, y, su, dz = walker(t)
    if (dz > 0) != front:
        return
    face = -1 if math.sin(phi) > 0 else 1                       # which way he's walking
    step = W0L * 24 * t
    lean, bob = 0.18 * math.sin(step), 0.12 * abs(math.sin(step))
    snow = (0.90, 0.94, 1.0)
    r1, r2, r3 = 1.0, 0.72, 0.52
    c1, c2, c3 = -r1 - bob, -2 * r1 - 0.8 * r2 - bob, -2 * r1 - 1.6 * r2 - 0.8 * r3 - bob
    l2, l3 = lean * 0.5, lean
    prims = [("ball", (0, c1, r1, r1), snow, 0.55, "snowman"), ("ball", (l2, c2, r2, r2), snow, 0.55, "snowman"),
             ("ball", (l3, c3, r3, r3), snow, 0.55, "snowman"),
             ("ell", (l2, c2 - 0.35, 0.09, 0.09), (0.05, 0.05, 0.06), 0.8, "coal"),
             ("ell", (l2, c2 + 0.1, 0.09, 0.09), (0.05, 0.05, 0.06), 0.8, "coal"),
             ("ell", (l3 + 0.18, c3 - 0.12, 0.07, 0.07), (0.05, 0.05, 0.06), 0.8, "coal"),
             ("tri", (l3 + 0.3, c3 - 0.03, l3 + 0.3, c3 + 0.12, l3 + 1.05, c3 + 0.05), (1.0, 0.5, 0.1), 0.8, "coal"),
             ("rect", (l2 - 0.62, l2 + 0.62, c2 - r2 - 0.05, c2 - r2 + 0.18), (0.20, 0.45, 1.0), 0.8, "coal"),
             ("rect", (l2 - 0.55, l2 - 0.35, c2 - r2, c2 - r2 + 0.75), (0.20, 0.45, 1.0), 0.8, "coal"),
             ("rect", (l3 - 0.62, l3 + 0.62, c3 - r3 - 0.06, c3 - r3 + 0.04), (0.07, 0.07, 0.09), 0.8, "iron"),
             ("rect", (l3 - 0.38, l3 + 0.38, c3 - r3 - 0.85, c3 - r3 - 0.06), (0.07, 0.07, 0.09), 0.8, "iron"),
             ("rect", (l3 - 0.38, l3 + 0.38, c3 - r3 - 0.25, c3 - r3 - 0.12), (0.95, 0.15, 0.15), 0.8, "coal")]
    tint = np.zeros(3, F32) if flash is None else flash * 0.25
    raster(cv, prims, x, y, su, face, tint)
    for sgn in (-1, 1):                                          # arms swing as he walks
        sw = 0.5 * math.sin(step + (0 if sgn > 0 else math.pi))
        ax0, ay0 = x + face * sgn * 0.75 * su + face * l2 * su, y + (c2 - 0.1) * su
        k = np.linspace(0, 1, 10)
        P = np.stack([ax0 + face * sgn * k * 1.2 * su, ay0 - k * (0.5 + sw) * su], 1)
        K.stroke(cv, P, 0.4 * K.CW, 0.7, (0.40, 0.24, 0.12), "iron", True)


def sleigh_path(t):
    """Santa circles the castle once per loop: across the sky in front, round the back."""
    psi = W0L * t + 2.2
    x = K.W / 2 + 0.62 * K.W * math.cos(psi)
    y = HOR - 0.40 * CSU - 0.085 * K.H * math.sin(psi) + 0.015 * K.H * math.sin(3 * psi)  # high in front, low behind
    return psi, x, y, math.sin(psi)


def santa_sleigh(cv, t, flash, front):
    """Santa's sleigh: a curled red sleigh with gold filigree and runners, a sack spilling
    gifts, Santa holding the reins (fur-trimmed coat, belt and buckle, beard, floppy hat),
    and three reindeer at full gallop with antlers, harness collars and jingle bells,
    trailing stardust."""
    psi, x, y, dz = sleigh_path(t)
    if (dz > 0) != front:
        return
    face = -1 if math.sin(psi) > 0 else 1                        # flying left in front, right behind
    u = 0.036 * K.H * (0.80 + 0.25 * dz)
    tint = np.zeros(3, F32) if flash is None else flash * 0.3
    k = np.arange(1, 34)                                         # stardust trail
    tx = [sleigh_path(t - 0.022 * i)[1:3] for i in k]
    tw = np.sin(W0L * 31 * t + k * 2.1) > -0.3
    br = (1 - k / 34) ** 0.8 * tw
    jit = np.sin(k * 7.3 + W0L * 5 * t) * 0.6 * u
    putcells(cv, np.array([p[0] for p in tx]), np.array([p[1] for p in tx]) + 0.2 * u + jit,
             0.35 + 0.6 * br, np.outer(br, [1.0, 0.88, 0.50]), "spark")
    red, white, gold, skin = (0.92, 0.10, 0.10), (0.97, 0.97, 1.0), (1.0, 0.80, 0.30), (1.0, 0.80, 0.66)
    brown, tan, dark = (0.92, 0.58, 0.30), (1.0, 0.86, 0.66), (0.55, 0.32, 0.16)
    sack, coal = (0.55, 0.36, 0.20), (0.08, 0.08, 0.10)
    prims = [
        # sack spilling gifts
        ("ell", (0.6, -2.5, 0.95, 1.05), sack, 0.7, "gift"),
        ("rect", (0.0, 0.65, -3.75, -3.15), (0.15, 0.60, 0.30), 0.8, "gift"),
        ("rect", (0.27, 0.38, -3.75, -3.15), red, 0.85, "gift"),
        ("rect", (0.75, 1.35, -3.55, -3.05), (0.25, 0.45, 1.0), 0.8, "gift"),
        ("rect", (0.75, 1.35, -3.33, -3.25), gold, 0.85, "gold"),
        ("ell", (0.6, -3.45, 0.18, 0.12), gold, 0.9, "gold"),
        # sleigh body with a scroll curl at the front
        ("rect", (-0.3, 0.9, -2.3, 0.0), red, 0.8, "train"),
        ("rect", (0.0, 3.4, -1.3, 0.0), red, 0.8, "train"),
        ("ell", (3.55, -0.95, 0.62, 0.62), red, 0.8, "train"),
        ("ell", (3.55, -0.95, 0.30, 0.30), gold, 0.9, "gold"),
        ("rect", (-0.3, 3.4, -1.38, -1.18), gold, 0.9, "gold"),
        ("rect", (-0.38, -0.2, -2.35, 0.0), gold, 0.9, "gold"),
    ] + [("ell", (xx, -0.62, 0.13, 0.13), gold, 0.9, "gold") for xx in (0.5, 1.2, 1.9, 2.6)] + [
        # runners and struts
        ("rect", (-0.6, 3.9, 0.42, 0.58), gold, 0.9, "gold"),
        ("ell", (4.05, 0.12, 0.36, 0.36), gold, 0.9, "gold"),
        ("ell", (4.05, 0.12, 0.16, 0.16), (0.10, 0.10, 0.20), 0.4, "gold"),
        ("rect", (0.5, 0.65, 0.0, 0.45), gold, 0.9, "gold"),
        ("rect", (2.6, 2.75, 0.0, 0.45), gold, 0.9, "gold"),
        # Santa
        ("ell", (2.05, -2.05, 0.85, 0.92), red, 0.8, "train"),
        ("rect", (1.98, 2.18, -2.9, -1.2), white, 0.85, "snowman"),
        ("rect", (1.2, 2.9, -1.78, -1.52), coal, 0.8, "coal"),
        ("rect", (1.85, 2.3, -1.82, -1.48), gold, 0.95, "gold"),
        ("tri", (2.3, -2.65, 2.55, -2.25, 3.5, -2.05), red, 0.8, "train"),       # arm out to the reins
        ("tri", (2.3, -2.65, 3.5, -2.05, 3.45, -1.8), red, 0.8, "train"),
        ("ell", (3.55, -1.95, 0.22, 0.22), white, 0.85, "snowman"),               # glove
        ("ell", (2.45, -3.25, 0.46, 0.46), skin, 0.75, "snowman"),               # face
        ("ell", (2.38, -2.78, 0.58, 0.48), white, 0.85, "snowman"),              # beard
        ("ell", (2.7, -3.08, 0.28, 0.11), white, 0.9, "snowman"),                # moustache
        ("ell", (2.68, -3.22, 0.12, 0.09), (1.0, 0.55, 0.55), 0.8, "snowman"),   # rosy cheek
        ("ell", (2.62, -3.42, 0.07, 0.07), coal, 0.9, "coal"),                   # eye
        ("tri", (1.95, -3.55, 2.9, -3.55, 1.45, -4.55), red, 0.8, "train"),      # floppy hat
        ("rect", (1.9, 2.95, -3.72, -3.45), white, 0.9, "snowman"),
        ("ell", (1.42, -4.55, 0.24, 0.24), white, 0.9, "snowman"),
    ]
    gallop = W0L * 30 * t
    collars = []
    for i in range(3):                                           # reindeer at full gallop
        bx = 5.4 + i * 2.7
        by = -1.15 + 0.22 * math.sin(gallop + i * 1.3) - 0.25 * (i % 2)
        st = 0.5 + 0.5 * math.sin(gallop + i)                    # stretch of the stride
        prims += [
            ("ell", (bx, by, 1.05, 0.45), brown, 0.85, "train"),
            ("ell", (bx + 0.05, by + 0.18, 0.75, 0.2), tan, 0.75, "train"),
            ("tri", (bx + 0.55, by - 0.25, bx + 0.95, by - 0.35, bx + 1.25, by - 1.05), brown, 0.75, "train"),
            ("ell", (bx + 1.35, by - 1.08, 0.33, 0.26), brown, 0.75, "train"),
            ("ell", (bx + 1.72, by - 0.98, 0.21, 0.15), tan, 0.75, "train"),
            ("ell", (bx + 1.9, by - 1.0, 0.07, 0.07), coal, 0.9, "coal"),
            ("ell", (bx + 1.42, by - 1.16, 0.06, 0.06), coal, 0.9, "coal"),
            ("tri", (bx + 1.1, by - 1.2, bx + 1.25, by - 1.25, bx + 0.85, by - 1.45), brown, 0.75, "train"),
            ("ell", (bx - 1.02, by - 0.28, 0.2, 0.14), white, 0.85, "snowman"),
            ("rect", (bx + 0.72, bx + 1.0, by - 0.62, by - 0.2), red, 0.85, "train"),     # harness collar
        ]
        for ax, sp in ((1.22, 1.0), (1.45, 0.8)):                # branching antlers
            prims += [
                ("tri", (bx + ax - 0.05, by - 1.25, bx + ax + 0.05, by - 1.25, bx + ax - 0.25 * sp, by - 2.15 * sp), tan, 0.85, "gold"),
                ("tri", (bx + ax - 0.12 * sp, by - 1.65 * sp, bx + ax - 0.07 * sp, by - 1.62 * sp, bx + ax + 0.25 * sp, by - 2.0 * sp), tan, 0.85, "gold"),
                ("tri", (bx + ax - 0.18 * sp, by - 1.9 * sp, bx + ax - 0.13 * sp, by - 1.88 * sp, bx + ax - 0.6 * sp, by - 2.2 * sp), tan, 0.85, "gold"),
            ]
        for lx_, fwd in ((0.65, 1), (0.40, 1), (-0.50, -1), (-0.75, -1)):   # legs flung fore and aft
            ex_ = lx_ + fwd * (0.55 + 0.45 * st)
            ey_ = 1.0 - 0.35 * st
            prims += [("tri", (bx + lx_ - 0.1, by + 0.2, bx + lx_ + 0.1, by + 0.2, bx + ex_, by + ey_), dark, 0.75, "train"),
                      ("ell", (bx + ex_, by + ey_, 0.09, 0.09), coal, 0.9, "coal")]
        for j in range(3):                                       # jingle bells on the collar
            prims.append(("ell", (bx + 0.78 + 0.1 * j, by - 0.16 + 0.05 * math.sin(gallop * 2 + j), 0.08, 0.08), gold, 0.95, "gold"))
        collars.append((bx + 0.86, by - 0.4))
    gx, gr = x + face * 6.5 * u, 8.5 * u                         # warm glow around the team
    rs, cs = window(gx - 2 * gr, gx + 2 * gr, y - 1.2 * gr, y + 0.8 * gr)
    sc = sub(cv, rs, cs)
    g = 0.38 * np.exp(-(((K.X[rs, cs] - gx) / gr) ** 2 + ((K.Y[rs, cs] - y + 1.4 * u) / (0.5 * gr)) ** 2))
    putmax(sc, g, np.array([0.95, 0.78, 0.55], F32) * g[..., None], "glow", 0.1)
    raster(cv, prims, x, y, u, face, tint)
    hx, hy = 3.55, -1.95                                         # golden reins from Santa's hand
    for cx_, cy_ in collars:
        kk = np.linspace(0, 1, 24)
        P = np.stack([x + face * (hx + (cx_ - hx) * kk) * u, y + (hy + (cy_ - hy) * kk + 0.35 * np.sin(kk * math.pi)) * u], 1)
        K.stroke(cv, P, 0.25 * K.CW, 0.55, (1.0, 0.78, 0.30), "gold", True)


PASTELS = [(0.85, 0.38, 0.50), (0.30, 0.68, 0.52), (0.92, 0.72, 0.32), (0.38, 0.52, 0.92),
           (0.66, 0.46, 0.88), (0.92, 0.52, 0.34)]
AWNINGS = [((0.85, 0.12, 0.15), (0.95, 0.95, 0.95)), ((0.10, 0.50, 0.30), (0.95, 0.95, 0.95)),
           ((0.20, 0.35, 0.85), (1.0, 0.85, 0.40)), ((0.70, 0.20, 0.55), (1.0, 0.90, 0.80))]


def street(f, side):
    """Screen x of the street edge and the building base y, at perspective depth f (1 = near)."""
    return K.W / 2 + side * STREET_D * f, HOR + (1.02 * K.H - HOR) * f


def build_mainstreet():
    """Two rows of Victorian shopfronts in perspective, leading the eye to the castle."""
    global SHOPS, STRINGS, CROWD
    X, Y = K.X, K.Y
    parts = {k: [[], [], [], []] for k in ("wall", "trim", "win", "shop", "awning", "sign", "door", "bulb")}
    bulb_phase = []
    for side in (-1, 1):
        f = np.abs(X - K.W / 2) / STREET_D
        onside = (np.sign(X - K.W / 2) == side) & (f > 0.22) & (f < 1.4)
        uw = 1.0 / np.maximum(f, 1e-3)                           # distance down the street
        bid = np.floor((uw - 0.70) / 0.45).astype(int)
        a = ((uw - 0.70) / 0.45) % 1.0                           # across one building, 0..1
        a = a if side < 0 else 1 - a
        hb = 0.80 + 0.35 * hash2(bid.astype(F32), side + 3.0, 1.1)
        pedi = hash2(bid.astype(F32), side + 3.0, 2.2) > 0.5
        base = HOR + (1.02 * K.H - HOR) * f
        hgt = 0.46 * K.H * f * hb
        v = (base - Y) / np.maximum(hgt, 1)
        roof = pedi & (v >= 1) & (v - 1 < 0.20 * (1 - np.abs(2 * a - 1)))
        wall = onside & (((v >= 0) & (v <= 1)) | roof)
        col = np.array(PASTELS, F32)[(bid * 7 + (side > 0) * 3) % len(PASTELS)]
        glow = 0.30 + 0.35 * np.clip(1 - v, 0, 1)
        sel = {}
        edge = roof & (v - 1 > 0.20 * (1 - np.abs(2 * a - 1)) - 0.045)
        sel["shop"] = wall & (v >= 0.05) & (v < 0.24) & (a > 0.12) & (a < 0.88)
        sel["door"] = sel["shop"] & (a > 0.45) & (a < 0.55)
        sel["awning"] = wall & (v >= 0.24) & (v < 0.32) & (a > 0.06) & (a < 0.94)
        sel["win"] = wall & ((((v > 0.45) & (v < 0.60)) | ((v > 0.70) & (v < 0.84))) & ((a * 2) % 1 > 0.30) & ((a * 2) % 1 < 0.70))
        sel["trim"] = wall & (((v > 0.90) & (v <= 1.0)) | edge | (a < 0.05) | (a > 0.95))
        sel["bulb"] = wall & ((np.abs(v - 0.985) < 0.02) | edge) & ((a * 12) % 1 < 0.5)
        taken = np.zeros(X.shape, bool)
        for k in ("bulb", "trim", "door", "shop", "awning", "win"):
            m = sel[k] & ~taken
            taken |= m
            i = np.flatnonzero(m)
            if k == "shop":
                c = mixc((1.0, 0.80, 0.45), (1.0, 0.62, 0.70), hash2(bid.astype(F32), 7.0, 3.3))[m]
                d = np.full(len(i), 0.95)
            elif k == "door":
                c, d = np.tile([0.22, 0.12, 0.08], (len(i), 1)), np.full(len(i), 0.5)
            elif k == "awning":
                stripes = np.array([AWNINGS[j][0] for j in range(4)], F32)[bid % 4]
                alt = np.array([AWNINGS[j][1] for j in range(4)], F32)[bid % 4]
                c = np.where(((a * 9).astype(int) % 2 == 0)[..., None], stripes, alt)[m] * 0.8
                d = np.full(len(i), 0.7)
            elif k == "sign":
                letters = hash2(K.RR, K.CC, 9.9) > 0.45
                c = np.where(letters[..., None], np.array([1.0, 0.82, 0.35], F32), np.array([0.15, 0.10, 0.12], F32))[m]
                d = np.where(letters, 0.8, 0.4)[m]
            elif k == "win":
                lit = hash2(bid.astype(F32), np.floor(v * 3), 4.2 + side) > 0.25
                c = np.where(lit[..., None], np.array([1.0, 0.74, 0.38], F32), np.array([0.12, 0.12, 0.22], F32))[m]
                d = np.where(lit, 0.9, 0.45)[m]
            elif k == "trim":
                c, d = np.tile([0.90, 0.88, 0.85], (len(i), 1)) * (0.40 + 0.40 * glow[m])[:, None], np.full(len(i), 0.65)
            else:
                c, d = np.tile([1.0, 0.85, 0.55], (len(i), 1)), np.full(len(i), 1.0)
                bulb_phase.append((uw * 6 + v * 4)[m])
            for lst, val in zip(parts[k], (i, c, d, None)):
                lst.append(val)
        rest = wall & ~taken
        i = np.flatnonzero(rest)
        parts["wall"][0].append(i)
        parts["wall"][1].append((col * (0.30 + 0.50 * glow)[..., None])[rest])
        parts["wall"][2].append((0.55 + 0.3 * glow)[rest])
    mats = dict(wall="facade", trim="trim", win="cwin", shop="shopwin", awning="awning", sign="sign",
                door="facade", bulb="bulb")
    SHOPS = {k: (np.concatenate(v[0]), np.concatenate(v[1]).astype(F32), np.concatenate(v[2]).astype(F32), K.MID[mats[k]])
             for k, v in parts.items() if v[0]}
    SHOPS["bulb_ph"] = np.concatenate(bulb_phase)

    STRINGS = []                                                 # strands swagged across the street
    for uw in (1.05, 1.6, 2.3, 3.3):
        f = 1 / uw
        xl, yb = street(f, -1)
        yt = yb - 0.46 * K.H * f * 0.85
        n = int(2 * STREET_D * f / (0.9 * K.CW)) + 2
        x = np.linspace(xl, K.W - xl, n)
        y = yt + 0.07 * K.H * f * (1 - ((x - K.W / 2) / (STREET_D * f)) ** 2)
        STRINGS.append((x, y, np.arange(n)))

    rng = np.random.default_rng(31)                              # the crowd on the sidewalks
    CROWD = []
    for side in (-1, 1):
        for uw in np.sort(rng.uniform(1.0, 3.6, 9))[::-1]:
            f = 1 / uw
            ex, yb = street(f, side)
            x = ex - side * (0.035 + 0.05 * rng.uniform()) * K.W * f
            CROWD.append(dict(x=x, y=yb - 0.01 * K.H * f, ph=0.16 * K.H * f * rng.uniform(0.85, 1.1),
                              coat=PASTELS[int(rng.integers(0, 6))], hat=rng.uniform() < 0.35,
                              balloons=int(rng.integers(1, 4)) if rng.uniform() < 0.35 else 0,
                              cols=[PALETTE[int(rng.integers(0, len(PALETTE)))] for _ in range(3)],
                              w=float(rng.uniform(0, 6.28)), f=f))
    CROWD.sort(key=lambda p: p["f"])


def mainstreet_layer(cv, t, flash):
    fl = np.zeros(3, F32) if flash is None else flash
    for k in ("wall", "trim", "door", "awning", "win", "shop"):
        if k in SHOPS:
            i, c, d, mid = SHOPS[k]
            cv[0].reshape(-1)[i] = d
            cv[1].reshape(-1, 3)[i] = c + fl * (0.25 if k in ("wall", "trim") else 0.1)
            cv[2].reshape(-1)[i] = mid
    i, c, d, mid = SHOPS["bulb"]                                 # chasing marquee bulbs
    tw = 0.5 + 0.5 * np.sin(W0L * 16 * t - SHOPS["bulb_ph"] * 3)
    cv[0].reshape(-1)[i] = 1.0
    cv[1].reshape(-1, 3)[i] = c * (0.35 + 0.75 * tw)[:, None]
    cv[2].reshape(-1)[i] = mid
    for x, y, n in STRINGS:                                      # string lights across the street
        cols = np.array(PALETTE, F32)[n % len(PALETTE)]
        tw = 0.55 + 0.45 * np.sin(W0L * 10 * t + n * 1.9)
        putcells(cv, x, y, np.full(len(x), 0.95), cols * tw[:, None], "bulb", True)


def crowd_layer(cv, t, flash):
    """People along the sidewalks, swaying; some in Santa hats, some with balloons."""
    tint = np.zeros(3, F32) if flash is None else flash * 0.25
    for p in CROWD:
        u = p["ph"] / 6.0                                         # sprite unit: a person is ~6 units tall
        sway = 0.12 * math.sin(W0L * 3 * t + p["w"])
        coat = np.array(p["coat"], F32) * 0.45
        prims = [("ell", (0 + sway * 0.3, -1.6, 0.85, 1.6), coat, 0.6, "person"),
                 ("ell", (sway, -3.6, 0.55, 0.6), (0.85, 0.68, 0.58), 0.6, "person")]
        if p["hat"]:
            prims += [("tri", (sway - 0.55, -3.95, sway + 0.55, -3.95, sway + 0.7, -4.9), (0.90, 0.12, 0.12), 0.7, "train"),
                      ("rect", (sway - 0.6, sway + 0.6, -4.05, -3.85), (0.95, 0.95, 0.98), 0.7, "snowman"),
                      ("ell", (sway + 0.7, -4.9, 0.18, 0.18), (0.95, 0.95, 0.98), 0.7, "snowman")]
        for j in range(p["balloons"]):
            bx = 0.9 + 0.6 * (j - 1) + 0.15 * math.sin(W0L * 4 * t + p["w"] + j)
            by = -7.2 - 0.5 * (j % 2) + 0.15 * math.sin(W0L * 5 * t + j)
            prims += [("ell", (bx, by, 0.55, 0.65), p["cols"][j], 0.85, "balloon"),
                      ("rect", (0.6 + 0.3 * (j - 1), 0.62 + 0.3 * (j - 1), by + 0.6, -2.2), (0.8, 0.8, 0.8), 0.35, "person")]
        raster(cv, prims, p["x"], p["y"], u, 1, tint)


def star_layer(cv, t):
    """The tree-top star drops in, bounces, then glows and sparkles."""
    if t < 11.2:
        return
    p = min(1.0, (t - 11.2) / 0.9)
    yw = STAR_Y + (1 - float(backout(p))) * 0.9
    x0, y0, _ = project_t(np.array([[0, yw, 0]], F32))
    sx, sy = float(x0[0]), float(y0[0])
    rs = 0.17 * KST * (0.4 + 0.6 * min(1.0, p * 1.5))
    rot = 0.15 * math.sin(W0L * t)
    rs_, cs_ = window(sx - 3 * rs, sx + 3 * rs, sy - 3 * rs, sy + 3 * rs)
    sc = sub(cv, rs_, cs_)
    dx, dy = K.X[rs_, cs_] - sx, K.Y[rs_, cs_] - sy
    d = np.hypot(dx, dy)
    a = (np.arctan2(dx, -dy) - rot) % (2 * np.pi / 5)
    a = np.minimum(a, 2 * np.pi / 5 - a)                       # fold into one star arm
    A = np.array([1.0, 0.0])
    B = 0.42 * np.array([math.cos(math.pi / 5), math.sin(math.pi / 5)])
    e = B - A
    rb = (A[0] * e[1] - A[1] * e[0]) / (np.cos(a) * e[1] - np.sin(a) * e[0])
    inside = d < rb * rs
    core = d < 0.35 * rs
    col = np.where(core[..., None], np.array([1.0, 0.98, 0.85], F32), np.array([1.0, 0.82, 0.30], F32))
    K.put(sc, inside, np.where(core, 1.0, 0.85), col, "star")
    g = 0.5 * np.exp(-(d / (1.6 * rs)) ** 2)
    putmax(sc, g, np.array([1.0, 0.85, 0.45], F32) * g[..., None], "glow", 0.12)
    k = np.linspace(-3.2, 3.2, 48) * rs                        # slowly turning, pulsing rays
    xs, ys, brs = [], [], []
    for j in range(4):
        a = W0L * t + j * math.pi / 4
        pulse = 0.55 + 0.45 * math.sin(W0L * 10 * t + j * 1.6)
        L = 1.0 if j % 2 == 0 else 0.6
        xs.append(sx + k * L * math.cos(a))
        ys.append(sy + k * L * math.sin(a) * 0.8)
        brs.append((1 - np.abs(k) / (3.2 * rs)) * pulse)
    br = np.concatenate(brs)
    putcells(cv, np.concatenate(xs), np.concatenate(ys), 0.3 + 0.6 * br, np.outer(br, [1.0, 0.9, 0.6]), "spark")


# ------------------------------------------------------------ background ---
def build_castle():
    """A fairy-tale castle in the classic park style: lavender stone base with drum towers
    and a gothic gate, salmon-pink towers, slate-blue fish-scale roofs, gold finials, and a
    tall slender main spire with a clock. Original design, drawn from simple shapes."""
    global CAS
    CS = 0.85 * CSU
    u, v = (K.X - K.W / 2) / CS, (HOR - K.Y) / CS
    lab = np.zeros(u.shape, np.int8)       # 1 stone 2 roof 3 gold 4 window 5 gate 6 pink 7 glass
    shade = np.zeros(u.shape, F32)
    cell = 0.5 * min(K.CW, K.CH) / CS                           # half a character, in castle units

    def paint(m, l, sh=1.0):
        lab[m] = l
        shade[m] = np.broadcast_to(np.asarray(sh, F32), m.shape)[m]

    def drum(cx, hw, v0, v1, l):                                # round tower, lit from the left
        x = (u - cx) / hw
        m = (np.abs(x) < 1) & (v >= v0) & (v < v1)
        paint(m, l, 0.5 + 0.5 * np.sqrt(np.clip(1 - x * x, 0, 1)) - 0.18 * x)
        return m

    def crenels(cx, hw, v0, h=0.018, step=0.022):
        paint((np.abs(u - cx) < hw * 1.08) & (v >= v0) & (v < v0 + h) & (((u - cx) / step) % 2 < 1), 1, 0.8)

    def cone(cx, hw, v0, v1, gold_tip=True, golden=False):      # scalloped slate roof
        x = (u - cx) / hw
        m = (v >= v0) & (v < v1) & (np.abs(x) < 1 - (v - v0) / (v1 - v0))
        scale = ((u / 0.014 + np.floor(v / 0.014) * 0.5) % 1 < 0.5)
        paint(m, 3 if golden else 2, (0.55 + 0.45 * (1 - (x + 1) / 2)) * np.where(scale, 1.0, 0.78))
        paint((np.abs(x) < 1.05) & (v >= v0 - 0.006) & (v < v0 + 0.006), 3, 0.9)      # gold eave
        if gold_tip:
            paint((np.abs(u - cx) < max(0.003, cell)) & (v >= v1) & (v < v1 + 0.04), 3, 1.0)
            paint(((u - cx) ** 2 + ((v - v1 - 0.045) * 1.2) ** 2) < 0.006 ** 2 + cell ** 2, 3, 1.0)

    def windows(cx, hw, v0, v1, step=0.07, w=0.25):
        for wv in np.arange(v0, v1, step):
            paint((np.abs(u - cx) < hw * w) & (v >= wv) & (v < wv + 0.03) &
                  ((v < wv + 0.022) | (np.abs(u - cx) < hw * w * (1 - (v - wv - 0.022) / 0.008))), 4)

    # stone curtain wall with crenellations and block courses
    wall = (np.abs(u) < 0.40) & (v >= 0) & (v < 0.10)
    paint(wall, 1, 0.70 + 0.12 * ((np.floor(v / 0.013) % 2 == 0) ^ ((u / 0.03 + 0.5 * np.floor(v / 0.013)) % 1 < 0.08)))
    crenels(0.0, 0.40, 0.10)
    # back: pink keep with a big fish-scale hip roof and a teal-blue front gable
    paint((np.abs(u + 0.01) < 0.16) & (v >= 0.10) & (v < 0.30), 6, 0.85 - 0.25 * (u + 0.01) / 0.16)
    hip = (v >= 0.30) & (v < 0.42) & (np.abs(u + 0.01) < 0.17 - (v - 0.30) * 0.9)
    paint(hip, 2, (0.65 + 0.3 * (1 - (u + 0.18) / 0.36)) * np.where(((u / 0.016 + np.floor(v / 0.012) * 0.5) % 1 < 0.5), 1.0, 0.75))
    paint((np.abs(u + 0.01) < 0.17 - (0.42 - 0.30) * 0.9) & (v >= 0.415) & (v < 0.43), 3, 1.0)       # gold crest
    for wx in (-0.13, -0.09, 0.05, 0.10):
        windows(wx, 0.03, 0.14, 0.28, 0.07, 0.5)
    gable = (v >= 0.28) & (v < 0.47) & (np.abs(u + 0.075) < 0.05 * (1 - (v - 0.28) / 0.19))
    paint(gable, 2, 0.95)
    paint((np.abs(u + 0.075) < 0.05) & (v >= 0.275) & (v < 0.285), 3)
    paint(((u + 0.075) ** 2 + ((v - 0.225) * 1.0) ** 2 < 0.032 ** 2), 7)                      # rose window
    # tall main spire, offset right, with a gallery, a clock and a needle roof
    drum(0.03, 0.036, 0.10, 0.62, 6)
    windows(0.03, 0.036, 0.34, 0.48, 0.07, 0.3)
    paint((np.abs(u - 0.03) < 0.048) & (v >= 0.48) & (v < 0.50), 3, 0.9)                         # gallery
    paint((np.abs(u - 0.03) < 0.044) & (v >= 0.50) & (v < 0.515) & ((u / 0.008) % 2 < 1), 3, 0.8)
    paint(((u - 0.03) ** 2 + (v - 0.575) ** 2 < 0.02 ** 2), 7)                                   # clock
    cone(0.03, 0.04, 0.62, 0.84)
    drum(0.072, 0.014, 0.40, 0.66, 6)                                                            # stair turret
    cone(0.072, 0.018, 0.66, 0.74)
    # secondary towers: pink drums with slate cones, one slender gold spire
    for cx, hw, top, rh in [(0.13, 0.024, 0.46, 0.13), (-0.175, 0.022, 0.38, 0.11), (-0.235, 0.018, 0.28, 0.09),
                            (0.26, 0.022, 0.24, 0.09)]:
        drum(cx, hw, 0.10, top, 6)
        windows(cx, hw, 0.15, top - 0.04, 0.08, 0.35)
        cone(cx, hw * 1.25, top, top + rh)
    drum(0.20, 0.012, 0.10, 0.36, 6)
    cone(0.20, 0.016, 0.36, 0.56, golden=True)
    # front: crenellated drum towers flanking a pointed gothic gate
    for cx, hw, top, rh in [(-0.10, 0.045, 0.16, 0.10), (0.10, 0.045, 0.16, 0.10), (-0.33, 0.042, 0.13, 0.09),
                            (0.31, 0.04, 0.13, 0.09)]:
        drum(cx, hw, 0.0, top, 1)
        crenels(cx, hw, top)
        windows(cx, hw, 0.05, top - 0.03, 0.06, 0.25)
        drum(cx, hw * 0.75, top + 0.018, top + 0.05, 6)
        cone(cx, hw * 0.95, top + 0.05, top + 0.05 + rh)
    arch = (np.abs(u) < 0.055) & (v >= 0) & ((v < 0.06) | (v < 0.06 + 0.045 * (1 - np.abs(u) / 0.055)))
    paint(arch & ~((np.abs(u) < 0.045) & ((v < 0.055) | (v < 0.055 + 0.038 * (1 - np.abs(u) / 0.045)))), 3, 0.7)
    paint(arch & (np.abs(u) < 0.045) & ((v < 0.055) | (v < 0.055 + 0.038 * (1 - np.abs(u) / 0.045))), 5,
          np.where(((u / 0.012) % 1 < 0.25) | ((v / 0.014) % 1 < 0.25), 0.6, 1.0))
    i = np.flatnonzero(lab)
    L = lab.reshape(-1)[i]
    base = np.zeros((len(i), 3), F32)
    glass = np.array([(1.0, 0.25, 0.30), (0.25, 0.45, 1.0), (1.0, 0.80, 0.30), (0.30, 0.85, 0.50)], F32)
    for l, c in [(1, (0.66, 0.62, 0.82)), (2, (0.16, 0.27, 0.68)), (3, (1.0, 0.80, 0.36)),
                 (4, (1.0, 0.78, 0.42)), (5, (0.42, 0.22, 0.10)), (6, (0.98, 0.58, 0.62))]:
        base[L == l] = c
    hv = hash2(K.RR, K.CC, 4.0).reshape(-1)[i]
    base[L == 7] = glass[(hv[L == 7] * 97).astype(int) % 4]
    CAS = dict(i=i, L=L, sh=shade.reshape(-1)[i], v=v.reshape(-1)[i], base=base, hv=hv,
               mat=np.select([L == 2, L == 3, (L == 4) | (L == 7)], [K.MID["roof"], K.MID["gold"], K.MID["cwin"]],
                             K.MID["stone"]))


def castle_layer(cv, t, flash):
    """Night lighting like a projection show: blue / purple / pink washes rolling over the
    stone, warm pink towers, glinting gold, flickering windows, and the show's flashes."""
    c = CAS
    sh, L, v = c["sh"], c["L"], c["v"]
    wash = mixc((0.45, 0.55, 1.0), (0.80, 0.50, 1.0), 0.5 + 0.5 * math.sin(W0L * t))
    wash = mixc(wash, (1.0, 0.60, 0.85), 0.25 + 0.25 * math.sin(W0L * 2 * t + 1))
    roll = 0.75 + 0.25 * np.sin(v * 25 - W0L * 6 * t)                     # light rolling upward
    grad = 0.55 + 0.45 * np.clip(1 - v / 0.45, 0, 1)
    fl = np.zeros(3, F32) if flash is None else flash
    col = c["base"] * (wash * (0.22 + 0.62 * sh * grad * roll)[:, None])                     # stone
    pink = L == 6
    col[pink] = c["base"][pink] * (mixc(wash, (1.0, 0.75, 0.65), 0.6) * (0.25 + 0.70 * sh[pink] * grad[pink])[:, None])
    roof = L == 2
    col[roof] = c["base"][roof] * ((0.40 + 0.75 * sh[roof]) * (0.7 + 0.3 * grad[roof]))[:, None] + wash * 0.06
    gold = L == 3
    col[gold] = c["base"][gold] * (0.7 + 0.3 * np.sin(W0L * 6 * t + c["hv"][gold] * 20))[:, None] * sh[gold][:, None]
    win = (L == 4) | (L == 7)
    col[win] = c["base"][win] * (0.75 + 0.25 * np.sin(W0L * (3 + np.floor(c["hv"][win] * 4)) * t + c["hv"][win] * 30))[:, None]
    gate = L == 5
    col[gate] = c["base"][gate] * sh[gate][:, None] * 1.3
    col += fl * (0.45 * sh)[:, None]
    dens = np.select([roof, gold | win, gate], [0.55 + 0.35 * sh, 0.9, 0.55], 0.42 + 0.40 * sh * grad)
    cv[0].reshape(-1)[c["i"]] = dens
    cv[1].reshape(-1, 3)[c["i"]] = col
    cv[2].reshape(-1)[c["i"]] = c["mat"]


def build_static():
    global SKY, GROUND, FLAKES
    m = K.Y < HOR
    g = np.clip(K.Y / HOR, 0, 1)
    col = mixc((0.010, 0.015, 0.05), (0.08, 0.06, 0.18), g ** 2)
    hv, br = hash2(K.RR, K.CC, 2.3), hash2(K.RR, K.CC, 6.1)
    star = m & (hv > 0.982) & (K.Y < HOR - 0.08 * K.H)
    SKY = dict(i=np.flatnonzero(m), d=np.full(m.sum(), 0.3, F32), col=col[m],
               si=np.flatnonzero(star), hv=hv[star], br=br[star])
    gm = K.Y >= HOR
    q = np.clip((K.Y - HOR) / (K.H - HOR), 0, 1)
    gcol = mixc((0.30, 0.34, 0.50), (0.62, 0.68, 0.85), q) * (0.35 + 0.35 * q)[..., None]
    glow = np.exp(-(K.Y - HOR) / (0.08 * K.H)) * (np.abs(K.X - K.W / 2) < 0.4 * CSU)
    tree = np.exp(-(((K.X - KXT) / (1.6 * KST)) ** 2 + ((K.Y - KYT) / (0.45 * KST)) ** 2))
    GROUND = dict(i=np.flatnonzero(gm), q=q[gm], col=gcol[gm], glow=glow[gm], tree=tree[gm],
                  hv=hash2(K.RR, K.CC, 8.8)[gm], d=(0.25 + 0.35 * q)[gm], mat=np.full(gm.sum(), K.MID["snow"]),
                  pool=np.zeros(gm.sum(), F32))
    rng = np.random.default_rng(3)
    n = 240
    FLAKES = dict(x=rng.uniform(0, 1, n), y=rng.uniform(0, 1, n), z=rng.integers(0, 3, n),
                  m=rng.integers(1, 4, n), ph=rng.uniform(0, 6.28, n), amp=rng.uniform(4, 16, n))


def sky_layer(cv, t):
    k = SKY
    putflat(cv, k["i"], k["d"], k["col"], "sky")
    tw = 0.55 + 0.45 * np.sin(W0L * (2 + np.floor(k["br"] * 4)) * t + k["hv"] * 628)
    putflat(cv, k["si"], 0.45 + 0.5 * k["br"] * tw,
            np.array([0.85, 0.88, 1.0], F32) * (0.3 + 0.7 * tw * k["br"])[:, None], "star")


def ground_layer(cv, t, flash):
    """Snowy plaza: glints, the castle's coloured glow, the tree lights' warm pool."""
    k = GROUND
    on = float(np.clip((t - 9.0) / 1.5, 0, 1))
    wash = mixc((1.0, 0.62, 0.85), (0.70, 0.62, 1.0), 0.5 + 0.5 * math.sin(W0L * t))
    warm = np.array([1.0, 0.75, 0.4], F32)
    col = (k["col"] + (k["glow"] * 0.25)[:, None] * wash + (k["tree"] * on * 0.35)[:, None] * warm
           + (k["pool"] * 0.45 * (0.9 + 0.1 * math.sin(W0L * 11 * t)))[:, None] * warm)
    glint = (k["hv"] > 0.93) & (np.sin(W0L * 7 * t + k["hv"] * 900) > 0.6)
    col = np.where(glint[:, None], np.array([0.95, 0.97, 1.0], F32), col)
    if flash is not None:
        col = col + flash * (0.18 * (1 - k["q"]))[:, None]
    cv[0].reshape(-1)[k["i"]] = np.where(glint, 0.9, k["d"])
    cv[1].reshape(-1, 3)[k["i"]] = col
    cv[2].reshape(-1)[k["i"]] = k["mat"]


def snow_layer(cv, t, near):
    """Falling snow in three depths; every flake falls a whole number of screens per loop."""
    f = FLAKES
    sel = (f["z"] == 2) if near else (f["z"] < 2)
    z = f["z"][sel]
    y = ((f["y"][sel] + (z + 1) * t / LOOP) % 1.0) * K.H * 1.04 - 0.02 * K.H
    x = f["x"][sel] * K.W + f["amp"][sel] * K.S * np.sin(W0L * f["m"][sel] * t + f["ph"][sel])
    b = np.array([0.45, 0.7, 1.0], F32)[z]
    putcells(cv, x, y, np.array([0.3, 0.62, 0.85], F32)[z], np.outer(b, [0.85, 0.9, 1.0]), "flake")


# ----------------------------------------------------------------- scene ---
def setup(cols, rows, cw, ch):
    global HOR, CSU, KXT, KYT, KST, SAMPT, STREET_D
    K.setup_grid(cols, rows, cw, ch)
    HOR = 0.62 * K.H
    CSU = min(0.95 * K.W, 1.15 * K.H)                           # castle unit (px)
    KXT, KYT = 0.5 * K.W, 0.93 * K.H
    KST = min(0.21 * K.H, 0.31 * K.W)                           # tree scale (px per unit)
    SAMPT = 2.2 * KST / min(K.CW, K.CH)
    STREET_D = 0.40 * K.W                                       # street half-width at the near edge
    K.HOR, K.MX, K.MY, K.MR = HOR, -1e9, -1e9, 0.0              # rockets rise from the castle
    K.build_show()
    build_tree()
    build_train()
    global CAP_UV
    na = max(24, int(2 * math.pi * 1.1 * SAMPT))
    rr, aa = np.meshgrid(np.linspace(0, 1, max(6, int(1.1 * SAMPT))), np.linspace(0, 2 * np.pi, na, endpoint=False))
    CAP_UV = np.stack([np.sqrt(rr.ravel()), aa.ravel()], 1).astype(F32)
    build_castle()
    build_static()
    build_plaza()
    build_mainstreet()
    global FOCUS, HALO, TREE_CELLS
    cy = KYT - 1.35 * KST                                       # centre of the tree, on screen
    e = ((K.X - KXT) / (1.05 * KST)) ** 2 + ((K.Y - cy) / (1.75 * KST)) ** 2
    FOCUS = (0.38 + 0.47 * np.exp(-0.5 * e)).astype(F32)
    HALO = np.exp(-0.9 * e).astype(F32)
    TREE_CELLS = None


def render(t):
    global TREE_CELLS
    TREE_CELLS = None
    cv = K.canvas()
    sky_layer(cv, t)
    flash = None
    if K.FIREWORKS_ON and t >= BUILD:
        flash = K.firework_layer(cv, (t - BUILD) % SHOW, fresh=t < BUILD + SHOW)
    santa_sleigh(cv, t, flash, front=False)                    # behind the castle
    castle_layer(cv, t, flash)
    ground_layer(cv, t, flash)
    mainstreet_layer(cv, t, flash)
    crowd_layer(cv, t, flash)
    plaza_back(cv, t, flash)
    snow_layer(cv, t, near=False)
    walking_snowman(cv, t, flash, front=False)                  # behind the tree
    tree_layer(cv, t, flash)
    star_layer(cv, t)
    smoke_puffs(cv, t)
    walking_snowman(cv, t, flash, front=True)
    plaza_front(cv, t, flash)
    focus_pass(cv, t)
    santa_sleigh(cv, t, flash, front=True)                     # in front: kept bright
    snow_layer(cv, t, near=True)
    return cv


CASTLE_TREE = SimpleNamespace(
    title="A Christmas tree builds in front of a fairy-tale castle, then the fireworks loop",
    setup=setup, render=render, loop=LOOP, intro=BUILD + SHOW, out="castle_tree.mp4",
    panel=(__file__, "def build_tree", "def project_t"))


if __name__ == "__main__":
    K.main(CASTLE_TREE)
