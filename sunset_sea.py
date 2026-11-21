#!/usr/bin/env python3
"""
sunset_sea.py - a sunset over the sea, with the castle fireworks show on a 10 s loop.
Coloured ASCII, on krathong_ascii.py's engine.

A sun half-sunk into the horizon under a dusk gradient, drifting cloud banks lit from
below, a headland with a lighthouse sweeping its beam, sailboats, gulls, and a sea that
mirrors the whole sky - sun glitter, clouds and every firework burst.

    python sunset_sea.py                  # sunset_sea.mp4
    python sunset_sea.py --term           # live in the terminal
    python sunset_sea.py --png 6          # one still at t = 6 s
"""
import math
from types import SimpleNamespace

import numpy as np

import krathong_ascii as K
from krathong_ascii import F32, hash2, lum, mixc, putcells, putflat, putmax, window, sub

K.register_ramps({
    "dusk": " .:-=+*",
    "sun": " .oO0@@",
    "cloud": " .:=+%#",
    "sea": " .-~=≈%#",
    "glint": " .*#@",
    "hill": " .:-=#",
    "boat": " .:|#",
    "beam": " .:'",
    "bird": " v^",
    "wing_l": " -\\",
    "wing_r": " -/",
    "lamp": " .oO@",
})

LOOP = K.LOOP                       # 10 s: one fireworks show
W0 = K.W0
SKY_STOPS = [(0.00, (0.05, 0.04, 0.16)), (0.35, (0.22, 0.08, 0.30)), (0.62, (0.62, 0.18, 0.32)),
             (0.82, (0.98, 0.42, 0.22)), (1.00, (1.00, 0.70, 0.32))]
CLOUDS = [  # centre (fraction of the horizon height), thickness (fraction of H), colours dark->lit, drift
    (0.30, 0.040, (0.20, 0.08, 0.26), (0.85, 0.35, 0.55), 1),
    (0.58, 0.055, (0.28, 0.10, 0.25), (1.00, 0.50, 0.40), -1),
    (0.84, 0.028, (0.40, 0.14, 0.22), (1.00, 0.68, 0.40), 1)]


def gradient(v):
    v = np.clip(v, 0, 1)
    col = np.zeros(v.shape + (3,), F32)
    for (a, ca), (b, cb) in zip(SKY_STOPS[:-1], SKY_STOPS[1:]):
        m = (v >= a) & (v <= b)
        col[m] = mixc(ca, cb, ((v - a) / (b - a))[m])
    return col


def build_static():
    """Sky, sun, headland, lighthouse and the sea's fixed fields: computed once per size."""
    global SKY, SUNC, HILL, SEA
    X, Y, W, H = K.X, K.Y, K.W, K.H
    m = Y < HOR
    dist = np.hypot(X - SX, Y - SY)
    col = gradient(Y / HOR)
    glow = 0.35 * np.exp(-dist / (2.5 * SR)) + 0.15 * np.exp(-dist / (1.2 * SR)) + 0.18 * np.exp(-(HOR - Y) / (0.05 * H))
    col = col + glow[..., None] * np.array([1.0, 0.70, 0.38], F32)
    hv, br = hash2(K.RR, K.CC, 2.9), hash2(K.RR, K.CC, 4.4)
    star = m & (hv > 0.975) & (Y < 0.30 * HOR)
    SKY = dict(i=np.flatnonzero(m), col=col[m], d=np.clip(0.25 + 0.55 * lum(col), 0, 0.95)[m],
               si=np.flatnonzero(star), hv=hv[star], br=br[star], fade=(1 - Y / (0.30 * HOR))[star])
    disk = (dist < SR) & m
    limb = np.sqrt(np.clip(1 - (dist / SR) ** 2, 0, 1))
    scol = mixc((1.0, 0.38, 0.12), (1.0, 0.82, 0.45), limb ** 0.7)
    band = ((HOR - Y) / SR < 0.55) & (((HOR - Y) / (0.09 * SR)) % 1 < 0.35)  # heat-haze slices
    SUNC = dict(i=np.flatnonzero(disk & ~band), col=scol[disk & ~band], d=(0.75 + 0.25 * limb)[disk & ~band])

    x = X / W                                                    # headland + far island
    hh = 0.075 * H * np.clip(np.sin(np.clip(x / 0.34, 0, 1) * math.pi), 0, 1) ** 0.7 * (1 + 0.15 * np.sin(x * 40))
    hh = np.where(x < 0.17, np.maximum(hh, 0.065 * H), hh)
    isl = 0.022 * H * np.clip(np.sin(np.clip((x - 0.62) / 0.30, 0, 1) * math.pi), 0, 1) ** 1.5
    hm = (Y < HOR) & ((HOR - Y < hh) | (HOR - Y < isl))
    top = hm & ~np.roll(hm, 1, 0)
    hcol = np.where(top[..., None], np.array([0.95, 0.45, 0.20], F32), np.array([0.10, 0.05, 0.14], F32))
    lx, base = 0.15 * W, HOR - 0.065 * H                         # lighthouse on the headland
    lh = 0.14 * H
    tw = np.maximum(0.9 * K.CW, 0.018 * W * (1 - 0.3 * (base - Y) / lh))
    tower = (Y < base) & (Y > base - lh) & (np.abs(X - lx) < tw)
    stripe = (((base - Y) / (0.02 * H)) % 2 < 1)
    lamp_m = (Y <= base - lh) & (Y > base - lh - 0.02 * H) & (np.abs(X - lx) < 1.3 * tw)
    roof = (Y <= base - lh - 0.02 * H) & (Y > base - lh - 0.035 * H) & \
        (np.abs(X - lx) < 1.4 * tw * (1 - (base - lh - 0.02 * H - Y) / (0.015 * H)))
    tcol = np.where(stripe[..., None], np.array([0.75, 0.15, 0.15], F32), np.array([0.92, 0.85, 0.88], F32))
    HILL = dict(i=np.flatnonzero(hm), col=hcol[hm], d=np.where(top, 0.7, 0.45)[hm],
                ti=np.flatnonzero(tower), tcol=tcol[tower] * 0.55,
                li=np.flatnonzero(lamp_m), ri=np.flatnonzero(roof), lamp=(lx, base - lh - 0.01 * H))

    r = int(np.argmax(Y[:, 0] >= HOR)) if (Y[:, 0] >= HOR).any() else K.ROWS
    Xw, Yw, RRw, CCw = X[r:], Y[r:], K.RR[r:], K.CC[r:]
    q = np.clip((Yw - HOR) / (H - HOR), 0, 1)
    z = 1.0 / (q + 0.07)
    u = (Xw - SX) / ((60 + q * 700) * K.S)
    r0 = int(HOR // K.CH)
    SEA = dict(r=r, X=Xw, Y=Yw, CC=CCw, q=q,
               a1=u * 6 + z * 1.4, a2=u * 11 - z * 2.3, a3=u * 3 + z * 3.9, a4=u * 17 + z * 6.1,
               rm=np.clip(2 * r0 - RRw - 1, 0, K.ROWS - 1).astype(int),
               fres=0.65 * np.exp(-q * 2.2) + 0.18, gw=SR * 0.45 + q * 0.16 * W,
               base=(0.6 + 0.4 * (1 - q)), hv=hash2(RRw, CCw, 13.1), hp=hash2(RRw, CCw, 17.3) * 60)


def sky_layer(cv, t):
    k = SKY
    putflat(cv, k["i"], k["d"], k["col"], "dusk")
    tw = 0.55 + 0.45 * np.sin(W0 * (1 + np.floor(k["br"] * 4)) * t + k["hv"] * 628)
    putflat(cv, k["si"], 0.45 + 0.5 * k["br"] * tw,
            np.array([0.9, 0.9, 1.0], F32) * (0.25 + 0.6 * tw * k["br"] * k["fade"])[:, None], "star")


def cloud_layer(cv, t):
    """Banks of cloud that drift sideways (whole wavelengths per loop) and glow from below."""
    for yc, th, dark, lit, drift in CLOUDS:
        yc, th = yc * HOR, th * K.H
        rs, cs = window(0, K.W, yc - 2.5 * th, yc + 2.5 * th)
        sc = sub(cv, rs, cs)
        Xs, Ys = K.X[rs, cs], K.Y[rs, cs]
        p = 2 * math.pi * Xs / K.W
        n = (0.5 + 0.25 * np.sin(3 * p + drift * W0 * t + yc) + 0.15 * np.sin(7 * p - drift * 2 * W0 * t + 1.7)
             + 0.12 * np.sin(13 * p + drift * 3 * W0 * t + 0.4) + 0.08 * np.sin(23 * p - drift * W0 * t + Ys * 0.05 / K.S))
        prof = np.exp(-((Ys - yc) / th) ** 2)
        dcl = np.clip((n * prof - 0.36) * 3.0, 0, 1)
        near_sun = np.exp(-np.abs(Xs - SX) / (0.45 * K.W))
        below = np.clip((Ys - yc) / th * 0.5 + 0.5, 0, 1)
        ccol = mixc(dark, lit, np.clip(0.25 + 0.45 * near_sun + 0.4 * below, 0, 1))
        m = dcl > 0.08
        a = (0.85 * dcl)[..., None]
        sc[1][m] = (sc[1] * (1 - a) + ccol * a)[m]
        sc[0][m] = np.maximum(sc[0], 0.35 + 0.5 * dcl)[m]
        sc[2][m] = K.MID["cloud"]


def lighthouse_layer(cv, t):
    """Headland and lighthouse; its beam sweeps round once per loop."""
    k = HILL
    putflat(cv, k["i"], k["d"], k["col"], "hill")
    putflat(cv, k["ti"], 0.7, k["tcol"] * 1.4, "boat")
    putflat(cv, k["ri"], 0.6, np.array([0.12, 0.06, 0.10], F32), "boat")
    lx, ly = k["lamp"]
    phi = W0 * t
    s, c = math.sin(phi), math.cos(phi)
    face = 0.35 + 0.65 * max(c, 0) ** 4
    putflat(cv, k["li"], 1.0, np.array([1.0, 0.92, 0.65], F32) * (0.6 + 0.6 * face), "lamp")
    if abs(s) > 0.05:                                            # the beam, foreshortened as it turns
        L = 0.7 * K.W * abs(s)
        x0, x1 = (lx, lx + L) if s > 0 else (lx - L, lx)
        rs, cs = window(x0, x1, ly - 0.06 * K.H, ly + 0.06 * K.H)
        sc = sub(cv, rs, cs)
        along = (K.X[rs, cs] - lx) * math.copysign(1, s)
        ang = np.abs(K.Y[rs, cs] - ly) / np.maximum(along, 1)
        I = (along > 0) * np.exp(-(ang / 0.05) ** 2) * np.exp(-along / (L + 1)) * abs(s) ** 0.7 * 0.6
        sky = (K.Y[rs, cs] < HOR) & np.isin(sc[2], [K.MID["dusk"], K.MID["cloud"], K.MID["star"]])
        putmax(sc, np.where(sky, I, 0), np.array([1.0, 0.92, 0.70], F32) * I[..., None], "beam", 0.06)
    g = 0.5 * face
    r = 0.05 * K.H
    rs, cs = window(lx - 2 * r, lx + 2 * r, ly - 2 * r, ly + 2 * r)
    sc = sub(cv, rs, cs)
    gg = g * np.exp(-((K.X[rs, cs] - lx) ** 2 + (K.Y[rs, cs] - ly) ** 2) / r ** 2)
    putmax(sc, gg, np.array([1.0, 0.9, 0.6], F32) * gg[..., None], "glow", 0.1)


def boat_layer(cv, t):
    """Two sailboats on the horizon, bobbing gently."""
    for bx, sz, ph in ((0.74, 1.0, 0.0), (0.40, 0.5, 1.7)):
        bx = bx * K.W
        u = 0.03 * K.H * sz
        by = HOR + 0.15 * u + 0.08 * u * math.sin(W0 * 2 * t + ph)
        rs, cs = window(bx - 2 * u, bx + 2 * u, by - 3.2 * u, by + 0.5 * u)
        sc = sub(cv, rs, cs)
        lx, ly = (K.X[rs, cs] - bx) / u, (K.Y[rs, cs] - by) / u
        mx, my = 0.5 * K.CW / u, 0.5 * K.CH / u
        hull = (ly > -0.45) & (ly < 0) & (np.abs(lx) < 1.5 - 0.6 * (ly + 0.45) / 0.45)
        mast = (np.abs(lx - 0.1) < max(0.05, mx)) & (ly < -0.4) & (ly > -3.0)
        main = (lx > 0.15) & (ly < -0.55) & (ly > -2.9) & (lx - 0.15 < 1.1 * (ly + 2.9) / 2.35)
        jib = (lx < 0.05) & (ly < -0.6) & (ly > -2.6) & (0.05 - lx < 0.9 * (ly + 2.6) / 2.0)
        sail_rim = (main | jib) & (lx > 0.3)
        K.put(sc, hull | mast, 0.7, (0.06, 0.03, 0.08), "boat")
        K.put(sc, main | jib, 0.55, (0.16, 0.08, 0.14), "boat")
        K.put(sc, sail_rim & ~np.roll(main | jib, -1, 1), 0.7, (1.0, 0.55, 0.30), "boat")


def sun_layer(cv, t):
    k = SUNC
    putflat(cv, k["i"], k["d"], k["col"] * (0.95 + 0.05 * math.sin(W0 * 3 * t)), "sun")


def sea_layer(cv, t, flash):
    """Waves in perspective that mirror the sky (sun, clouds, fireworks) plus the sun's glitter path."""
    k = SEA
    h = (np.sin(k["a1"] - W0 * 3 * t) + 0.7 * np.sin(k["a2"] + W0 * 4 * t)
         + 0.5 * np.sin(k["a3"] - W0 * 2 * t + 2.1) + 0.3 * np.sin(k["a4"] + W0 * 5 * t))
    wave = 0.5 + 0.5 * np.tanh(h * 0.9)
    col = mixc((0.05, 0.04, 0.15), (0.20, 0.11, 0.30), wave) * k["base"][..., None]
    cm = np.clip(k["CC"] + np.round(h * 0.9), 0, K.COLS - 1).astype(int)
    refl = cv[1][k["rm"], cm] * (k["fres"] * (0.55 + 0.45 * wave))[..., None]
    col = col + refl
    dens = np.maximum(0.18 + 0.35 * wave * (0.4 + 0.6 * k["q"]), np.clip(lum(refl) * 0.9, 0, 0.9))
    path = np.exp(-((k["X"] - SX + 6 * h * K.S) / k["gw"]) ** 2)      # sun's glitter path
    col = col + (path * wave * 0.35)[..., None] * np.array([1.0, 0.60, 0.30], F32)
    tw = np.sin(W0 * 13 * t + k["hp"])
    glint = path * (k["hv"] > 1 - 0.45 * path) * np.clip(tw, 0, 1)      # scattered sparkles
    gm = glint > 0.2
    col = np.where(gm[..., None], np.array([1.0, 0.85, 0.50], F32) * (0.6 + 0.6 * glint)[..., None], col)
    dens = np.where(gm, 0.55 + 0.45 * glint, dens)
    if flash is not None:
        col = col + flash * (0.10 * (1 - k["q"]))[..., None]
    r = k["r"]
    cv[0][r:], cv[1][r:] = dens, col
    cv[2][r:] = np.where(gm, K.MID["glint"], K.MID["sea"])


def gull_layer(cv, t):
    """Gulls gliding across, flapping now and then."""
    for i, (x0, y0, d, f) in enumerate(BIRDS):
        x = (((x0 + d * t / LOOP) % 1.0) * 1.3 - 0.15) * K.W
        y = y0 * HOR + 6 * K.S * math.sin(W0 * 2 * t + i)
        up = math.sin(W0 * f * t + i * 1.3) > 0
        putcells(cv, np.array([x]), np.array([y]), np.array([0.5]), np.array([[0.10, 0.05, 0.12]], F32), "bird")
        for dx, mid in ((-K.CW, "wing_l"), (K.CW, "wing_r")):
            putcells(cv, np.array([x + dx]), np.array([y - (0.3 * K.CH if up else 0)]),
                     np.array([0.95 if up else 0.5]), np.array([[0.10, 0.05, 0.12]], F32), mid, True)


def setup(cols, rows, cw, ch):
    global HOR, SR, SX, SY, BIRDS
    K.setup_grid(cols, rows, cw, ch)
    HOR = 0.56 * K.H
    SR = 0.10 * min(K.W, 1.05 * K.H)
    SX, SY = 0.5 * K.W, HOR + 0.15 * SR
    K.HOR, K.MX, K.MY, K.MR = HOR, SX, SY, SR                    # shells launch from the sea
    K.build_show()
    rng = np.random.default_rng(9)
    BIRDS = [(rng.uniform(), rng.uniform(0.35, 0.75), rng.choice([-1, 1]), int(rng.integers(5, 9))) for _ in range(6)]
    build_static()


def render(t):
    cv = K.canvas()
    sky_layer(cv, t)
    cloud_layer(cv, t)
    sun_layer(cv, t)
    flash = K.firework_layer(cv, t % LOOP) if K.FIREWORKS_ON else None
    lighthouse_layer(cv, t)
    boat_layer(cv, t)
    sea_layer(cv, t, flash)
    gull_layer(cv, t)
    return cv


SUNSET = SimpleNamespace(
    title="A sunset over the sea with the fireworks show", setup=setup, render=render,
    loop=LOOP, intro=0.0, out="sunset_sea.mp4", panel=(__file__, "def cloud_layer", "def setup"))


if __name__ == "__main__":
    K.main(SUNSET)
