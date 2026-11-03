"""
Seestar HaRGB-Combine  --  Seestar S30 Pro  (Siril >= 1.4 Python-Script)
=========================================================================

Kombiniert IRCUT-Stacks (normales OSC-RGB) mit LP-Stacks (Dual-Band-Filter
Ha/OIII) zu einem HaRGB- bzw. HaOIII-RGB-Bild.

Ablauf
------
 1. Stacks laden - beliebig viele pro Filter (z.B. mehrere Naechte), wahlweise
    automatisch per Ordner-Scan (Filter wird am Dateinamen erkannt: _IRCUT_/_LP_).
 2. Alle Stacks per Sternregistrierung (Siril `register`) auf den IRCUT-Stack
    mit den meisten Frames ausrichten (mit Rueckfall-Strategien).
 3. Mehrere Stacks desselben Filters normieren und rauschgewichtet mitteln.
 4. Optional: auf den gemeinsamen Bildbereich zuschneiden, Gradienten entfernen.
 5. Ha = Rot des LP-Stacks, OIII = Mittel aus Gruen+Blau.
 6. Sternschaerfe (FWHM) von RGB und Ha/OIII angleichen, Kontinuum-Abgleich
    ueber die Sterne, Ha-/OIII-Linienemission (Ueberschuss) bestimmen.
 7. Ueberschuss glaetten, Rauschschwelle anwenden, Sterne ausmaskieren
    (Bereich unter den Sternen wird aus der Umgebung aufgefuellt).
 8. Ha in Rot, OIII in Gruen/Blau einrechnen, optional Ha-Luminanz-Boost.
 9. Ergebnis als 32-bit FITS speichern (linear) und in Siril laden.

Modi
----
  excess (Standard)  R' = R + k_Ha * E_Ha      G',B' += k_OIII * E_OIII
          E = geglaettete, rauschbereinigte Linienemission ueber dem Kontinuum.
          Sterne und Hintergrund bleiben unveraendert.
  blend   R' = (1-k_Ha) * R + k_Ha * Ha_lin    (G,B analog mit OIII)
          Mischt das (rauschaermere) Schmalband ein. k = "auto" waehlt die
          rauschoptimale Gewichtung.

  Ha-Luminanz (k_Lum): Helligkeit wird zusaetzlich Richtung Ha gezogen
  (Details/SNR des Ha), Farben bleiben erhalten, Sterne geschuetzt.

Aufruf
------
  * Siril: Skripte-Menue -> Seestar_HaRGB-Combine  (Dialog mit Vorschau)
  * Siril-Konsole / .ssf:
      pyscript "C:/pfad/zu/Seestar_HaRGB-Combine.py" -dir "C:/daten/NGC7000"
      pyscript "C:/pfad/zu/Seestar_HaRGB-Combine.py" -ircut "a.fit" -lp "b.fit" "c.fit"
    Optionen: -out DATEI  -mode excess|blend  -k 0.5|auto  -koiii 0  -klum 0
              -noise 2  -smooth 2  -nopsf  -nostarprot  -bge  -crop  -keep

Ausgabe (neben der Ausgabedatei):
  <name>.fit            Ergebnis (linear)
  <name>_Ha.fit         ausgerichteter Ha-Kanal, linear an R angepasst
  <name>_OIII.fit       ausgerichteter OIII-Kanal, linear an (G+B)/2 angepasst
  <name>_HaExcess.fit   reine Ha-Linienemission (kontinuumsbereinigt)
"""

import os
import re
import sys
import glob
import shutil
import argparse
import threading
import traceback

import sirilpy as s

s.ensure_installed("numpy", "astropy")

import numpy as np                       # noqa: E402
from astropy.io import fits              # noqa: E402

VERSION = "2.0.0"
SEQ = "hapair"
WORKDIR_NAME = "_ha_combine_work"
FITS_EXT = (".fit", ".fits", ".fts")
DEFAULT_OUT = "HaRGB_combined.fit"

DEFAULTS = dict(mode="excess", k_ha=0.5, k_oiii=0.0, k_lum=0.0, noise_t=2.0,
                smooth=2, star_protect=True, psf_match=True, bge=False,
                crop=False, keep=False)


class Params:
    def __init__(self, **kw):
        for k, v in DEFAULTS.items():
            setattr(self, k, kw.get(k, v))

    def prep_key(self):
        """Optionen, die eine neue Vorbereitung erfordern."""
        return (self.psf_match, self.bge, self.crop)


# ===========================================================================
# FITS I/O
# ===========================================================================
def read_fits(path):
    """Liest ein FITS-Bild als float32 (C, H, W) in [0..1] + Header."""
    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
        if hdu is None:
            raise ValueError(f"Keine Bilddaten gefunden in: {path}")
        data = hdu.data
        header = hdu.header.copy()

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim == 3 and data.shape[0] not in (1, 3) and data.shape[-1] in (1, 3):
        data = np.moveaxis(data, -1, 0)
    if data.ndim != 3 or data.shape[0] not in (1, 3):
        raise ValueError(f"Unerwartete Bildform {data.shape} in: {path}")

    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / 65535.0
    else:
        data = data.astype(np.float32)
        mx = float(np.nanmax(data))
        if mx > 1.5:
            data /= 65535.0 if mx <= 65535.0 * 1.01 else mx
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(data, 0.0, 1.0, out=data)
    return data, header


def read_header(path):
    try:
        return fits.getheader(path)
    except Exception:
        return None


def write_fits(path, data, header=None, history=None):
    data = np.ascontiguousarray(np.clip(data, 0.0, 1.0).astype(np.float32))
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    hdr = fits.Header()
    if header is not None:
        skip = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
                "EXTEND", "BZERO", "BSCALE", "DATAMIN", "DATAMAX"}
        for card in header.cards:
            if card.keyword in skip or card.keyword in ("", "COMMENT", "HISTORY"):
                continue
            try:
                hdr.append(card)
            except Exception:
                pass
    for line in history or []:
        hdr.add_history(line)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def to_rgb(img):
    return img if img.shape[0] == 3 else np.repeat(img, 3, axis=0)


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except Exception:
        shutil.copyfile(src, dst)


# ===========================================================================
# Stack-Erkennung (Punkt 6)
# ===========================================================================
RE_IRCUT = re.compile(r"(?<![a-z0-9])(ircut|ir-cut|uvir|uv-ir|uvircut)(?![a-z0-9])", re.I)
RE_LP = re.compile(r"(?<![a-z0-9])(lp|lpf|dual|dualband|duoband|l-enhance|lenhance)"
                   r"(?![a-z0-9])", re.I)


def classify_stack(path):
    """'ircut', 'lp' oder None - zuerst am Dateinamen, sonst am FILTER-Header."""
    name = os.path.basename(path)
    if RE_LP.search(name):
        return "lp"
    if RE_IRCUT.search(name):
        return "ircut"
    hdr = read_header(path)
    if hdr is not None:
        flt = str(hdr.get("FILTER", "")).strip()
        if RE_LP.search(flt):
            return "lp"
        if RE_IRCUT.search(flt):
            return "ircut"
    return None


def frame_count(path):
    """Anzahl gestackter Frames aus Dateiname oder Header (0 = unbekannt)."""
    name = os.path.basename(path)
    m = re.search(r"Stacked_(\d+)_", name, re.I) or re.search(r"_(\d+)f(?=[_.])", name)
    if m:
        return int(m.group(1))
    hdr = read_header(path)
    if hdr is not None:
        for key in ("STACKCNT", "NCOMBINE", "NB_FRAMES"):
            if key in hdr:
                try:
                    return int(hdr[key])
                except Exception:
                    pass
    return 0


def detect_stacks(folder):
    """Sucht IRCUT- und LP-Stacks in einem Ordner (nicht rekursiv)."""
    ircut, lp, unknown = [], [], []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        n = os.path.basename(p).lower()
        if not n.endswith(FITS_EXT) or "hargb" in n or n.startswith(("r_", "bge_")):
            continue
        c = classify_stack(p)
        (ircut if c == "ircut" else lp if c == "lp" else unknown).append(p)
    key = lambda p: -frame_count(p)  # noqa: E731
    return sorted(ircut, key=key), sorted(lp, key=key), unknown


# ===========================================================================
# Bildverarbeitung
# ===========================================================================
def box_mean(a, r):
    """Box-Mittelwert (Radius r) ueber Integralbild, Raender gespiegelt (edge)."""
    if r <= 0:
        return a.astype(np.float32, copy=True)
    k = 2 * r + 1
    p = np.pad(a.astype(np.float64), ((r + 1, r), (r + 1, r)), mode="edge")
    np.cumsum(p, axis=0, out=p)
    np.cumsum(p, axis=1, out=p)
    out = p[k:, k:] - p[:-k, k:] - p[k:, :-k] + p[:-k, :-k]
    out /= k * k
    return out.astype(np.float32)


def gauss_blur(a, sigma):
    """Gauss-Naeherung durch 3 Box-Durchlaeufe, Varianz exakt interpoliert."""
    if sigma < 0.25:
        return a.astype(np.float32, copy=True)

    def box3(img, r):
        for _ in range(3):
            img = box_mean(img, r)
        return img

    v = sigma * sigma                      # 3 Boxen mit Radius r: Varianz r(r+1)
    r0 = int(np.floor((-1 + np.sqrt(1 + 4 * v)) / 2))
    r1 = r0 + 1
    v0, v1 = r0 * (r0 + 1), r1 * (r1 + 1)
    t = (v - v0) / (v1 - v0)
    b0 = a.astype(np.float32) if r0 == 0 else box3(a, r0)
    if t < 0.02:
        return b0
    b1 = box3(a, r1)
    return (1 - t) * b0 + t * b1


def max_filter(a, r):
    out = a.copy()
    for axis in (0, 1):
        src = out.copy()
        for d in range(1, r + 1):
            sl_a = [slice(None)] * 2
            sl_b = [slice(None)] * 2
            sl_a[axis], sl_b[axis] = slice(d, None), slice(None, -d)
            np.maximum(out[tuple(sl_a)], src[tuple(sl_b)], out=out[tuple(sl_a)])
            np.maximum(out[tuple(sl_b)], src[tuple(sl_a)], out=out[tuple(sl_b)])
    return out


def mad_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def center_slice(shape, size=2048):
    H, W = shape
    h, w = min(H, size) // 2, min(W, size) // 2
    return slice(H // 2 - h, H // 2 + h), slice(W // 2 - w, W // 2 + w)


def noise_sigma(img, valid):
    """Pixelrauschen (MAD eines Hochpasses) auf einem zentralen Ausschnitt."""
    sl = center_slice(img.shape)
    c = img[sl]
    hp = c - box_mean(c, 2)
    v = valid[sl]
    return mad_sigma(hp[v][::3]) * 1.09 if v.any() else 0.0


def smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def linear_fit(src, ref, valid, low=1e-6, high=0.92, max_n=2_000_000):
    """Least-Squares  src*a + b ~ ref  (wie Siril linear_match)."""
    m = valid & (ref > low) & (ref < high) & (src > low) & (src < high)
    idx = np.flatnonzero(m)
    if idx.size < 1000:
        raise RuntimeError("Zu wenige gemeinsame Pixel fuer die lineare Anpassung "
                           "- passen die Stacks zusammen?")
    if idx.size > max_n:
        idx = idx[:: idx.size // max_n + 1]
    x = src.ravel()[idx].astype(np.float64)
    y = ref.ravel()[idx].astype(np.float64)
    a, b = np.polyfit(x, y, 1)
    if not np.isfinite(a) or a <= 0:
        mx, my = np.median(x), np.median(y)
        sx = np.median(np.abs(x - mx)) or 1e-6
        sy = np.median(np.abs(y - my)) or 1e-6
        a = sy / sx
        b = my - a * mx
    return float(a), float(b)


def common_crop(valid, min_frac=0.999):
    """Groesstes Rechteck, dessen Raender vollstaendig gueltig sind (Punkt 8)."""
    y0, y1, x0, x1 = 0, valid.shape[0], 0, valid.shape[1]
    while y1 - y0 > 16 and x1 - x0 > 16:
        sub = valid[y0:y1, x0:x1]
        edges = (sub[0].mean(), sub[-1].mean(), sub[:, 0].mean(), sub[:, -1].mean())
        i = int(np.argmin(edges))
        if edges[i] >= min_frac:
            break
        if i == 0:
            y0 += 1
        elif i == 1:
            y1 -= 1
        elif i == 2:
            x0 += 1
        else:
            x1 -= 1
    return y0, y1, x0, x1


def measure_fwhm(chans, det, valid, sat=0.85, max_stars=300):
    """Median-FWHM (Pixel) je Kanal an denselben Sternen (Punkt 2)."""
    sl = center_slice(det.shape, 3000)
    d, v = det[sl], valid[sl]
    hp = box_mean(d, 1) - box_mean(d, 10)
    sig = mad_sigma(hp[v][::5])
    if sig <= 0:
        return None
    peaks = v & (hp > 10 * sig) & (hp >= max_filter(hp, 4))
    ys, xs = np.nonzero(peaks)
    R = 12
    ok = (ys >= R) & (ys < d.shape[0] - R) & (xs >= R) & (xs < d.shape[1] - R)
    ys, xs = ys[ok], xs[ok]
    order = np.argsort(-hp[ys, xs])
    crops = {k: c[sl] for k, c in chans.items()}
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    disk = (yy ** 2 + xx ** 2) <= (R - 1) ** 2
    border = ~((np.abs(yy) < R) & (np.abs(xx) < R))
    res = {k: [] for k in chans}
    used = 0
    for i in order:
        y, x = ys[i], xs[i]
        stamps = {k: c[y - R:y + R + 1, x - R:x + R + 1].astype(np.float64)
                  for k, c in crops.items()}
        if any(st.max() >= sat for st in stamps.values()):
            continue
        vals = {}
        for k, st in stamps.items():
            p = st - np.median(st[border])
            peak = p[R - 1:R + 2, R - 1:R + 2].max()
            if peak <= 0:
                break
            area = np.count_nonzero((p > 0.5 * peak) & disk)
            vals[k] = 2.0 * np.sqrt(area / np.pi)
        else:
            for k, f in vals.items():
                res[k].append(f)
            used += 1
            if used >= max_stars:
                break
    if used < 15:
        return None
    out = {k: float(np.median(v_)) for k, v_ in res.items()}
    out["n"] = used
    return out


def psf_match(a, b, fa, fb):
    """Weichzeichnet das schaerfere Bild auf die FWHM des anderen."""
    sa, sb = fa / 2.3548, fb / 2.3548
    if abs(sa - sb) < 0.15:
        return a, b, 0.0, 0.0
    if sa < sb:
        sig = float(np.sqrt(sb * sb - sa * sa))
        return gauss_blur(a, sig), b, sig, 0.0
    sig = float(np.sqrt(sa * sa - sb * sb))
    return a, gauss_blur(b, sig), 0.0, sig


def star_mask(r, valid, fwhm):
    """Weiche Sternmaske 0..1 aus dem RGB-Rotkanal."""
    rs = max(1, int(round(fwhm / 4)))
    rb = max(6, int(round(2.5 * fwhm)))
    hp = box_mean(r, rs) - box_mean(r, rb)
    sig = mad_sigma(hp[valid][::11])
    m = (hp > 5 * sig) | (r > 0.9)
    rd = int(np.ceil(0.75 * fwhm)) + 1
    m = box_mean(m.astype(np.float32), rd) > 0.01
    soft = box_mean(m.astype(np.float32), max(1, rd // 2))
    soft[~valid] = 0.0
    return soft, rd


def continuum_scale(nb, bb, valid, fwhm, sat=0.9):
    """Skalierung Schmalband -> Breitband ueber die Sterne (Kontinuum)."""
    rs = max(1, int(round(fwhm / 4)))
    rb = max(6, int(round(2.5 * fwhm)))
    hp_b = box_mean(bb, rs) - box_mean(bb, rb)
    hp_n = box_mean(nb, rs) - box_mean(nb, rb)
    base = valid & (bb < sat) & (nb < sat)
    if base.sum() < 10000:
        return None
    thr_b = np.percentile(hp_b[base][::7], 99.5)
    thr_n = np.percentile(hp_n[base][::7], 99.5)
    m = base & (hp_b > max(thr_b, 1e-5)) & (hp_n > max(thr_n, 1e-5))
    n = int(m.sum())
    if n < 50:
        return None
    scale = float(np.median(hp_b[m] / hp_n[m]))
    if not np.isfinite(scale) or scale <= 0:
        return None
    offset = float(np.median(bb[valid]) - scale * np.median(nb[valid]))
    return scale, offset, n


def excess_map(diff, star, valid, rs, sigma, t, rin):
    """Linienemission: glaetten, Rauschschwelle, Sterne auffuellen (Punkt 1+2)."""
    e = box_mean(diff, rs) if rs > 0 else diff.astype(np.float32, copy=True)
    if t > 0 and sigma > 0:
        w = smoothstep(0.5 * t, 1.5 * t, e / sigma)
        e = np.maximum(e, 0.0) * w
    else:
        e = np.maximum(e, 0.0)
    if star is not None:
        keep = 1.0 - star
        num, den = box_mean(e * keep, rin), box_mean(keep, rin)
        fill = num / np.maximum(den, 1e-3)
        bad = den < 0.05
        if bad.any():
            num2, den2 = box_mean(e * keep, 3 * rin), box_mean(keep, 3 * rin)
            fill = np.where(bad, num2 / np.maximum(den2, 1e-3), fill)
        e = e * keep + fill * star
    e[~valid] = 0.0
    return e.astype(np.float32)


def stretch_params(rgb, valid):
    """Unlinked Autostretch-Parameter (wie Siril) aus dem IRCUT-Bild."""
    prm = []
    for c in range(rgb.shape[0]):
        x = rgb[c][valid][::5] if valid.any() else rgb[c].ravel()
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) * 1.4826
        lo = max(med - 2.8 * mad, 0.0)
        m = (med - lo) / max(1 - lo, 1e-6)
        tgt = 0.25
        mt = m * (tgt - 1) / ((2 * tgt - 1) * m - tgt) if m > 0 else 0.5
        prm.append((lo, mt))
    return prm


def apply_stretch(rgb, prm):
    out = np.empty_like(rgb)
    for c, (lo, mt) in enumerate(prm):
        x = np.clip((rgb[c] - lo) / max(1 - lo, 1e-6), 0, 1)
        out[c] = ((mt - 1) * x) / ((2 * mt - 1) * x - mt)
    return np.clip(out, 0, 1)


def bin_image(a, f, mode="mean"):
    if f <= 1:
        return a
    H, W = a.shape[-2:]
    h, w = H // f, W // f
    a = a[..., :h * f, :w * f]
    shp = a.shape[:-2] + (h, f, w, f)
    if mode == "min":
        return a.reshape(shp).min(axis=(-1, -3))
    if mode == "max":
        return a.reshape(shp).max(axis=(-1, -3))
    return a.reshape(shp).mean(axis=(-1, -3), dtype=np.float32)


class GroupStack:
    """Mehrere Stacks eines Filters: normieren + rauschgewichtet mitteln (Punkt 9)."""

    def __init__(self):
        self.ref = self.refv = self.acc = self.wsum = None
        self.n = 0

    def _weights(self, img, v):
        return [1.0 / max(noise_sigma(img[c], v), 1e-7) ** 2 for c in range(img.shape[0])]

    def _accumulate(self, img, v):
        for c, w in enumerate(self._weights(img, v)):
            self.acc[c] += w * img[c] * v
            self.wsum[c] += w * v

    def add(self, img):
        v = np.all(img > 0, axis=0)
        self.n += 1
        if self.ref is None:
            self.ref, self.refv = img, v
            return
        if self.acc is None:
            self.acc = np.zeros_like(self.ref)
            self.wsum = np.zeros_like(self.ref)
            self._accumulate(self.ref, self.refv)
        norm = np.empty_like(img)
        for c in range(img.shape[0]):
            a, b = linear_fit(img[c], self.ref[c], v & self.refv)
            norm[c] = a * img[c] + b
        self._accumulate(norm, v)

    def result(self):
        if self.acc is None:
            return self.ref, self.refv
        cover = self.wsum[0] > 0
        out = self.acc / np.maximum(self.wsum, 1e-30)
        out[:, ~cover] = 0.0
        return out.astype(np.float32), cover


def q(path):
    return '"' + path.replace("\\", "/") + '"'


# ===========================================================================
# Vorbereitete Daten
# ===========================================================================
class Prepared:
    """Ausgerichtete Daten + alle Analyse-Ergebnisse; Basis fuer Vorschau/Export."""

    def __init__(self):
        self.arrays = {}
        self.coef = {}
        self.info = []
        self._sigma = {}
        self._bins = {}
        self.fwhm = None
        self.inpaint_r = 6
        self.auto_k = {}
        self.hdr = None
        self.inputs = ([], [])

    @property
    def shape(self):
        return self.arrays["ha"].shape

    def sigma(self, key, r):
        """Rauschen des (geglaetteten) Differenzbilds - immer in voller Aufloesung."""
        if (key, r) not in self._sigma:
            diff = self.arrays.get(key)
            if diff is None:
                return 0.0
            sl = center_slice(diff.shape)
            c = diff[sl]
            e = box_mean(c, r) if r > 0 else c
            hp = e - box_mean(e, 4 * r + 4)
            m = self.arrays["valid"][sl] & (self.arrays["star"][sl] < 0.1)
            self._sigma[(key, r)] = mad_sigma(hp[m][::3]) if m.any() else 0.0
        return self._sigma[(key, r)]

    def binned(self, f):
        if f not in self._bins:
            A = self.arrays
            B = {}
            for k, v in A.items():
                if v is None:
                    B[k] = None
                elif k == "valid":
                    B[k] = bin_image(v, f, "min")
                elif k == "star":
                    B[k] = bin_image(v, f, "max")
                else:
                    B[k] = bin_image(v, f)
            self._bins = {f: B}
            self._bins[f]["_stretch"] = stretch_params(B["rgb"], B["valid"])
        return self._bins[f]

    def region(self, y0, y1, x0, x1):
        return {k: (None if v is None else v[..., y0:y1, x0:x1])
                for k, v in self.arrays.items()}


def render(P, A, prm, f=1, want_excess=False):
    """Berechnet das kombinierte Bild fuer die Arrays A (voll, Ausschnitt oder
    gebinnt mit Faktor f)."""
    rgb, valid = A["rgb"], A["valid"]
    out = rgb.copy()
    star = A["star"] if prm.star_protect else None
    rs = int(round(prm.smooth / f))
    rin = max(2, int(round(P.inpaint_r / f)))
    e_ha = None

    if prm.mode == "excess":
        if prm.k_ha > 0 or want_excess:
            e_ha = excess_map(A["diff_ha"], star, valid, rs,
                              P.sigma("diff_ha", prm.smooth), prm.noise_t, rin)
            out[0] += prm.k_ha * e_ha
        if prm.k_oiii > 0 and A.get("diff_o") is not None:
            e_o = excess_map(A["diff_o"], star, valid, rs,
                             P.sigma("diff_o", prm.smooth), prm.noise_t, rin)
            out[1] += prm.k_oiii * e_o
            out[2] += prm.k_oiii * e_o
    elif prm.mode == "blend":
        a, b = P.coef["ha_r"]
        k = min(max(prm.k_ha, 0.0), 1.0)
        out[0] = (1 - k) * rgb[0] + k * (a * A["ha"] + b)
        if prm.k_oiii > 0:
            k = min(max(prm.k_oiii, 0.0), 1.0)
            for c, key in ((1, "o_g"), (2, "o_b")):
                a, b = P.coef[key]
                out[c] = (1 - k) * rgb[c] + k * (a * A["oiii"] + b)
        if want_excess:
            e_ha = excess_map(A["diff_ha"], star, valid, rs,
                              P.sigma("diff_ha", prm.smooth), prm.noise_t, rin)
    else:
        raise ValueError(f"Unbekannter Modus: {prm.mode}")

    if prm.k_lum > 0:                                    # Punkt 4
        # Nur das Signal UEBER dem Himmelshintergrund skalieren - sonst
        # wuerde der Hintergrund-Sockel mitskaliert und die Farben verblassen.
        a, b = P.coef["ha_l"]
        ped = np.asarray(P.coef["ped"], dtype=np.float32)[:, None, None]
        delta = prm.k_lum * (a * A["ha"] + b - rgb.mean(axis=0))
        if star is not None:
            delta *= (1.0 - star)
        sig = out - ped
        Ls = sig.mean(axis=0)
        ratio = np.clip((Ls + delta) / np.maximum(Ls, P.coef["l_floor"]), 0.0, 4.0)
        out = ped + sig * ratio[np.newaxis]

    out[:, ~valid] = rgb[:, ~valid]
    np.clip(out, 0.0, 1.0, out=out)
    return out, e_ha


# ===========================================================================
# Ablauf
# ===========================================================================
class Combiner:
    def __init__(self, siril):
        self.siril = siril

    # --- Siril-Hilfen -------------------------------------------------------
    def log(self, msg, color=s.LogColor.DEFAULT):
        print(msg)
        try:
            self.siril.log(msg, color)
        except Exception:
            pass

    def progress(self, msg, frac):
        try:
            self.siril.update_progress(msg, frac)
        except Exception:
            pass

    def subsky(self, workdir, name, data):
        """Gradientenentfernung (Polynom Grad 1) ueber Siril."""
        path = os.path.join(workdir, name + ".fit")
        write_fits(path, data)
        self.siril.cmd("cd", q(workdir))
        self.siril.cmd("load", name)
        self.siril.cmd("subsky", "1", "-samples=20", "-tolerance=1.0")
        self.siril.cmd("save", name)
        return read_fits(path)[0]

    def register_to_ref(self, workdir, idx, path, ref_rgb, keep=False):
        """Registriert einen Stack auf ref.fit, mit Rueckfall-Strategien (Punkt 7)."""
        pdir = os.path.join(workdir, f"pair{idx:02d}")
        os.makedirs(pdir, exist_ok=True)
        link_or_copy(os.path.join(workdir, "ref.fit"), os.path.join(pdir, "a_ref.fit"))
        img, _ = read_fits(path)
        write_fits(os.path.join(pdir, "b_img.fit"), to_rgb(img))
        del img
        self.siril.cmd("cd", q(pdir))
        self.siril.cmd("convert", SEQ)
        self.siril.cmd("setref", SEQ, "1")
        attempts = [("-layer=0", "-transf=homography"),
                    ("-layer=1", "-transf=homography"),
                    ("-layer=0", "-transf=similarity"),
                    ("-layer=1", "-transf=shift")]
        r1 = os.path.join(pdir, f"r_{SEQ}_00001.fit")
        r2 = os.path.join(pdir, f"r_{SEQ}_00002.fit")
        for n, args in enumerate(attempts):
            for f_ in glob.glob(os.path.join(pdir, "r_*")):
                os.remove(f_)
            try:
                self.siril.cmd("register", SEQ, *args, "-interp=lanczos4",
                               "-maxstars=2000", "-nostarlist")
            except Exception as e:
                self.log(f"  Registrierung {' '.join(args)} fehlgeschlagen: {e}",
                         s.LogColor.SALMON)
                continue
            if os.path.isfile(r1) and os.path.isfile(r2):
                if n > 0:
                    self.log(f"  Registrierung gelungen mit {' '.join(args)}",
                             s.LogColor.SALMON)
                break
        else:
            raise RuntimeError(f"Registrierung fehlgeschlagen fuer {os.path.basename(path)} "
                               "(zu wenige gemeinsame Sterne?)")
        reg1, _ = read_fits(r1)
        reg2, _ = read_fits(r2)
        if reg1.shape == ref_rgb.shape == reg2.shape:
            d1 = float(np.mean(np.abs(reg1[:, ::7, ::7] - ref_rgb[:, ::7, ::7])))
            d2 = float(np.mean(np.abs(reg2[:, ::7, ::7] - ref_rgb[:, ::7, ::7])))
            if d2 < d1:
                reg2 = reg1
        self.siril.cmd("cd", q(workdir))
        if not keep:
            shutil.rmtree(pdir, ignore_errors=True)
        return reg2

    # --- Vorbereitung -------------------------------------------------------
    def prepare(self, ircut_files, lp_files, out_dir, prm):
        if not ircut_files or not lp_files:
            raise ValueError("Mindestens ein IRCUT- und ein LP-Stack noetig.")
        P = Prepared()
        P.inputs = (list(ircut_files), list(lp_files))
        P.prep_key = prm.prep_key()
        workdir = os.path.join(out_dir, WORKDIR_NAME)
        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir)
        try:
            # --- 1. Referenz ----------------------------------------------------
            ircut_files = sorted(ircut_files, key=lambda p: -frame_count(p))
            for kind, files in (("IRCUT", ircut_files), ("LP", lp_files)):
                for p in files:
                    n = frame_count(p)
                    self.log(f"{kind}: {os.path.basename(p)}"
                             + (f"  ({n} Frames)" if n else ""))
            self.progress("Lade Referenz ...", 0.02)
            ref, hdr = read_fits(ircut_files[0])
            ref = to_rgb(ref)
            write_fits(os.path.join(workdir, "ref.fit"), ref, hdr)
            P.hdr = hdr

            # --- 2./3. Registrieren + Gruppen mitteln ---------------------------
            g_ircut, g_lp = GroupStack(), GroupStack()
            g_ircut.add(ref)
            jobs = [("ircut", p) for p in ircut_files[1:]] + [("lp", p) for p in lp_files]
            for i, (kind, p) in enumerate(jobs):
                self.progress(f"Registriere {os.path.basename(p)} ...",
                              0.05 + 0.45 * i / len(jobs))
                self.log(f"Registriere {os.path.basename(p)} ...", s.LogColor.BLUE)
                reg = self.register_to_ref(workdir, i + 1, p, ref, prm.keep)
                (g_ircut if kind == "ircut" else g_lp).add(reg)
                del reg
            del ref
            rgb, cov_rgb = g_ircut.result()
            lp, cov_lp = g_lp.result()
            del g_ircut, g_lp
            if len(ircut_files) > 1:
                self.log(f"{len(ircut_files)} IRCUT-Stacks rauschgewichtet gemittelt")
            if len(lp_files) > 1:
                self.log(f"{len(lp_files)} LP-Stacks rauschgewichtet gemittelt")
            valid = cov_rgb & cov_lp & np.all(rgb > 0, axis=0) & np.all(lp > 0, axis=0)
            self.log(f"Ueberlappung IRCUT/LP: {100.0 * valid.mean():.1f} %")

            # --- 4. Zuschneiden ---------------------------------------------------
            if prm.crop:
                y0, y1, x0, x1 = common_crop(valid)
                rgb = np.ascontiguousarray(rgb[:, y0:y1, x0:x1])
                lp = np.ascontiguousarray(lp[:, y0:y1, x0:x1])
                valid = np.ascontiguousarray(valid[y0:y1, x0:x1])
                for key, off in (("CRPIX1", x0), ("CRPIX2", y0)):
                    if key in P.hdr:
                        P.hdr[key] = float(P.hdr[key]) - off
                self.log(f"Zugeschnitten auf {x1 - x0}x{y1 - y0} px "
                         f"(x {x0}..{x1}, y {y0}..{y1})")

            ha = np.ascontiguousarray(lp[0])
            oiii = 0.5 * (lp[1] + lp[2])
            del lp

            # --- Gradientenentfernung ------------------------------------------
            if prm.bge:
                self.progress("Gradientenentfernung ...", 0.55)
                self.log("Gradientenentfernung (subsky Grad 1) ...")
                for c in range(3):
                    rgb[c][~valid] = np.median(rgb[c][valid])
                rgb = self.subsky(workdir, "bge_rgb", rgb)
                ha[~valid] = np.median(ha[valid])
                ha = self.subsky(workdir, "bge_ha", ha)[0]
                oiii[~valid] = np.median(oiii[valid])
                oiii = self.subsky(workdir, "bge_oiii", oiii)[0]
                rgb[:, ~valid] = 0.0
                ha[~valid] = 0.0
                oiii[~valid] = 0.0

            # --- 5./6. Analyse ------------------------------------------------
            self.progress("Analysiere Sterne ...", 0.65)
            r = rgb[0]
            gb = 0.5 * (rgb[1] + rgb[2])
            fw = measure_fwhm({"R": r, "Ha": ha, "GB": gb, "OIII": oiii}, r, valid)
            P.fwhm = fw
            if fw:
                self.log(f"FWHM ({fw['n']} Sterne): R {fw['R']:.2f}  Ha {fw['Ha']:.2f}  "
                         f"G/B {fw['GB']:.2f}  OIII {fw['OIII']:.2f} px")
                fmax = max(fw["R"], fw["Ha"], fw["GB"], fw["OIII"])
            else:
                self.log("Zu wenige Sterne fuer FWHM-Messung - kein PSF-Angleich.",
                         s.LogColor.SALMON)
                fmax = 4.0
            star, rd = star_mask(r, valid, fmax)
            P.inpaint_r = 3 * rd
            self.log(f"Sternmaske: {100.0 * (star > 0.5).mean():.1f} % der Bildflaeche")

            def line_diff(nb, bb, f_nb, f_bb, label):
                if prm.psf_match and fw:
                    nb_m, bb_m, s_nb, s_bb = psf_match(nb, bb, f_nb, f_bb)
                    if s_nb or s_bb:
                        which = label if s_nb else "RGB"
                        self.log(f"PSF-Angleich {label}: {which} weichgezeichnet "
                                 f"(sigma {max(s_nb, s_bb):.2f} px)")
                else:
                    nb_m, bb_m = nb, bb
                cs = continuum_scale(nb_m, bb_m, valid, fmax)
                if cs is None:
                    a, b = linear_fit(nb, bb, valid)
                    self.log(f"{label}: zu wenige Sterne fuer Kontinuum-Abgleich - "
                             "lineare Anpassung", s.LogColor.SALMON)
                else:
                    a, b, n = cs
                    self.log(f"{label}: Kontinuum-Abgleich ({n} Sternpixel) "
                             f"scale={a:.4f} off={b:+.5f}")
                diff = (a * nb_m + b) - bb_m
                del nb_m, bb_m
                return zero_point(diff), (a, b)

            def zero_point(diff):
                """Nullpunkt = dunkelste 5 % (geglaettet) -> dort keine Emission."""
                sm = box_mean(diff, 16)
                diff -= float(np.percentile(sm[valid & (star < 0.5)][::5], 5))
                diff[~valid] = 0.0
                return diff.astype(np.float32)

            self.progress("Bestimme Ha-Linienemission ...", 0.75)
            diff_ha, P.coef["ha_cont"] = line_diff(
                ha, r, fw and fw["Ha"], fw and fw["R"], "Ha")
            self.progress("Bestimme OIII-Linienemission ...", 0.85)
            diff_o, P.coef["o_cont"] = line_diff(
                oiii, gb, fw and fw["OIII"], fw and fw["GB"], "OIII")

            # Ha-Uebersprechen im OIII: Die Bayer-Gruen/Blau-Filter lassen etwas
            # Ha (656 nm) durch. Annahme: in den hellsten Ha-Regionen gibt es
            # Stellen ohne OIII -> kleinstes OIII/Ha-Verhaeltnis = Uebersprechen.
            fb = 8
            sh = bin_image(diff_ha, fb)
            so = bin_image(diff_o, fb)
            vb = bin_image(valid & (star < 0.5), fb, "min")
            strong = vb & (sh > max(np.percentile(sh[vb], 90), 1e-6))
            kappa = 0.0
            if strong.sum() > 50:
                kappa = max(0.0, float(np.percentile(so[strong] / sh[strong], 5)))
            if kappa > 0:
                diff_o -= kappa * diff_ha
                diff_o = zero_point(diff_o)
            P.coef["o_leak"] = kappa
            self.log(f"Ha-Uebersprechen im OIII korrigiert: kappa={kappa:.3f}")
            del sh, so, vb, strong

            # lineare Anpassungen (blend, Ausgabedateien, Luminanz)
            P.coef["ha_r"] = linear_fit(ha, r, valid)
            P.coef["o_g"] = linear_fit(oiii, rgb[1], valid)
            P.coef["o_b"] = linear_fit(oiii, rgb[2], valid)
            P.coef["o_gb"] = linear_fit(oiii, gb, valid)
            P.coef["ha_l"] = linear_fit(ha, rgb.mean(axis=0), valid)
            del gb
            # Hintergrund-Sockel je Kanal (fuer den Luminanz-Boost)
            vb4 = bin_image(valid, 4, "min")
            P.coef["ped"] = [float(np.percentile(bin_image(rgb[c], 4)[vb4], 1))
                             for c in range(3)]
            P.coef["l_floor"] = 3.0 * noise_sigma(rgb.mean(axis=0), valid)

            # rauschoptimale Mischgewichte fuer 'blend' (Punkt 9)
            s_r = noise_sigma(r, valid)
            s_h = P.coef["ha_r"][0] * noise_sigma(ha, valid)
            s_g = noise_sigma(rgb[1], valid)
            s_o = P.coef["o_g"][0] * noise_sigma(oiii, valid)
            P.auto_k = {"ha": s_r ** 2 / max(s_r ** 2 + s_h ** 2, 1e-30),
                        "oiii": s_g ** 2 / max(s_g ** 2 + s_o ** 2, 1e-30)}
            self.log(f"Rauschen: R {s_r:.2e}  Ha(an R) {s_h:.2e}  G {s_g:.2e}  "
                     f"OIII(an G) {s_o:.2e}")
            self.log(f"Auto-k fuer 'blend': Ha {P.auto_k['ha']:.2f}  "
                     f"OIII {P.auto_k['oiii']:.2f}")

            P.arrays = dict(rgb=rgb, ha=ha, oiii=oiii, valid=valid, star=star,
                            diff_ha=diff_ha, diff_o=diff_o)
            self.progress("Vorbereitung abgeschlossen", 1.0)
            return P
        finally:
            try:
                self.siril.cmd("cd", q(out_dir))
            except Exception:
                pass
            if not prm.keep:
                shutil.rmtree(workdir, ignore_errors=True)

    # --- Export ---------------------------------------------------------------
    def save(self, P, prm, out_path):
        out_path = os.path.abspath(out_path)
        if not out_path.lower().endswith(FITS_EXT):
            out_path += ".fit"
        base = os.path.splitext(out_path)[0]
        self.progress("Berechne Ergebnis ...", 0.3)
        result, e_ha = render(P, P.arrays, prm, 1, want_excess=True)
        A = P.arrays
        k_txt = (f"mode={prm.mode} k_ha={prm.k_ha} k_oiii={prm.k_oiii} k_lum={prm.k_lum} "
                 f"noise={prm.noise_t} smooth={prm.smooth}")
        hist = [f"Seestar HaRGB-Combine v{VERSION}",
                k_txt,
                f"starprot={prm.star_protect} psf={prm.psf_match} bge={prm.bge} "
                f"crop={prm.crop}",
                "IRCUT: " + ", ".join(os.path.basename(p) for p in P.inputs[0])[:60],
                "LP: " + ", ".join(os.path.basename(p) for p in P.inputs[1])[:60]]
        self.progress("Speichere ...", 0.7)
        write_fits(out_path, result, P.hdr, hist)
        a, b = P.coef["ha_r"]
        write_fits(base + "_Ha.fit", np.where(A["valid"], a * A["ha"] + b, 0.0),
                   P.hdr, ["Ha aus LP-Stack, linear an R angepasst"])
        a, b = P.coef["o_gb"]
        write_fits(base + "_OIII.fit", np.where(A["valid"], a * A["oiii"] + b, 0.0),
                   P.hdr, ["OIII aus LP-Stack, linear an (G+B)/2 angepasst"])
        write_fits(base + "_HaExcess.fit", e_ha, P.hdr,
                   ["Ha-Linienemission ueber Kontinuum (k=1)"])
        for p in (out_path, base + "_Ha.fit", base + "_OIII.fit", base + "_HaExcess.fit"):
            self.log(f"Gespeichert: {p}", s.LogColor.GREEN)
        self.siril.cmd("cd", q(os.path.dirname(out_path)))
        self.siril.cmd("load", q(out_path))
        self.progress("Fertig", 1.0)
        self.log("Fertig. Das Ergebnis ist linear - jetzt z.B. SPCC und Stretch.",
                 s.LogColor.GREEN)
        try:
            self.siril.reset_progress()
        except Exception:
            pass
        return out_path


# ===========================================================================
# GUI (Punkt 5)
# ===========================================================================
def run_gui(combiner, initial_dir):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title(f"Seestar HaRGB-Combine  v{VERSION}")
    try:
        from sirilpy import tksiril
        tksiril.elevate(root)
    except Exception:
        pass

    st = {"P": None, "busy": False, "crop_c": None, "img_ov": None, "img_cr": None,
          "f": 1, "pending": None}
    ftypes = [("FITS", "*.fit *.fits *.fts"), ("Alle Dateien", "*.*")]

    V = dict(mode=tk.StringVar(value=DEFAULTS["mode"]),
             k_ha=tk.DoubleVar(value=DEFAULTS["k_ha"]),
             k_oiii=tk.DoubleVar(value=DEFAULTS["k_oiii"]),
             k_lum=tk.DoubleVar(value=DEFAULTS["k_lum"]),
             noise_t=tk.DoubleVar(value=DEFAULTS["noise_t"]),
             smooth=tk.IntVar(value=DEFAULTS["smooth"]),
             star_protect=tk.BooleanVar(value=DEFAULTS["star_protect"]),
             psf_match=tk.BooleanVar(value=DEFAULTS["psf_match"]),
             bge=tk.BooleanVar(value=DEFAULTS["bge"]),
             crop=tk.BooleanVar(value=DEFAULTS["crop"]),
             keep=tk.BooleanVar(value=DEFAULTS["keep"]))
    v_out = tk.StringVar()
    v_before = tk.BooleanVar(value=False)
    v_status = tk.StringVar(value="Stacks waehlen oder Ordner scannen.")

    def params():
        return Params(**{k: (round(v.get(), 3) if isinstance(v, tk.DoubleVar) else v.get())
                         for k, v in V.items()})

    # ---------------- Layout ----------------
    left = ttk.Frame(root, padding=8)
    left.grid(row=0, column=0, sticky="ns")
    right = ttk.Frame(root, padding=8)
    right.grid(row=0, column=1, sticky="nsew")

    lists = {}

    def file_list(row, label, key):
        ttk.Label(left, text=label).grid(row=row, column=0, sticky="nw", pady=2)
        fr = ttk.Frame(left)
        fr.grid(row=row, column=1, columnspan=2, sticky="we")
        lb = tk.Listbox(fr, height=3, width=55, selectmode="extended", exportselection=False)
        lb.pack(side="left", fill="x")
        bf = ttk.Frame(fr)
        bf.pack(side="left", padx=3)

        def add():
            ps = filedialog.askopenfilenames(title=label, initialdir=initial_dir,
                                             filetypes=ftypes, parent=root)
            for p in ps:
                if p not in lb.get(0, "end"):
                    lb.insert("end", p)
            inputs_changed()

        def rem():
            for i in reversed(lb.curselection()):
                lb.delete(i)
            inputs_changed()
        ttk.Button(bf, text="+", width=3, command=add).pack()
        ttk.Button(bf, text="-", width=3, command=rem).pack()
        lists[key] = lb

    file_list(0, "IRCUT-Stacks:", "ircut")
    file_list(1, "LP-Stacks:", "lp")

    def scan():
        d = filedialog.askdirectory(title="Ordner mit Seestar-Stacks",
                                    initialdir=initial_dir, parent=root)
        if not d:
            return
        ir, lp, unk = detect_stacks(d)
        for key, ps in (("ircut", ir), ("lp", lp)):
            lists[key].delete(0, "end")
            for p in ps:
                lists[key].insert("end", p.replace("\\", "/"))
        msg = f"{len(ir)} IRCUT- und {len(lp)} LP-Stacks gefunden."
        if unk:
            msg += f" {len(unk)} Datei(en) nicht zuordenbar."
        v_status.set(msg + " Ggf. unpassende Eintraege entfernen (z.B. Einzel- und "
                     "Gesamt-Stacks nicht mischen).")
        v_out.set(os.path.join(d, DEFAULT_OUT).replace("\\", "/"))
        inputs_changed()

    ttk.Button(left, text="Ordner scannen ...", command=scan).grid(
        row=2, column=1, sticky="w", pady=(2, 6))

    ttk.Label(left, text="Ausgabe:").grid(row=3, column=0, sticky="w")
    of = ttk.Frame(left)
    of.grid(row=3, column=1, columnspan=2, sticky="we")
    ttk.Entry(of, textvariable=v_out, width=55).pack(side="left")

    def pick_out():
        p = filedialog.asksaveasfilename(title="Ausgabedatei", parent=root,
                                         initialdir=initial_dir, defaultextension=".fit",
                                         filetypes=ftypes, initialfile=DEFAULT_OUT)
        if p:
            v_out.set(p)
    ttk.Button(of, text="...", width=3, command=pick_out).pack(side="left", padx=3)

    lf1 = ttk.LabelFrame(left, text="Vorbereitung (Aenderung = neu berechnen)", padding=6)
    lf1.grid(row=4, column=0, columnspan=3, sticky="we", pady=(10, 4))
    for i, (key, txt) in enumerate((
            ("psf_match", "Sternschaerfe angleichen (PSF)"),
            ("bge", "Gradientenentfernung (subsky Grad 1)"),
            ("crop", "Auf gemeinsamen Bildbereich zuschneiden"))):
        ttk.Checkbutton(lf1, text=txt, variable=V[key],
                        command=lambda: inputs_changed(False)).grid(row=i, column=0, sticky="w")

    lf2 = ttk.LabelFrame(left, text="Kombination (Vorschau live)", padding=6)
    lf2.grid(row=5, column=0, columnspan=3, sticky="we", pady=4)
    ttk.Label(lf2, text="Modus:").grid(row=0, column=0, sticky="w")
    mode_box = ttk.Combobox(lf2, textvariable=V["mode"], state="readonly", width=10,
                            values=["excess", "blend"])
    mode_box.grid(row=0, column=1, sticky="w")
    lbl_mode = ttk.Label(lf2, foreground="gray")
    lbl_mode.grid(row=1, column=0, columnspan=3, sticky="w")

    sliders = {}

    def slider(row, key, txt, lo, hi, fmt="{:.2f}"):
        ttk.Label(lf2, text=txt).grid(row=row, column=0, sticky="w")
        sc = ttk.Scale(lf2, from_=lo, to=hi, variable=V[key], length=260)
        sc.grid(row=row, column=1, sticky="w")
        lb = ttk.Label(lf2, width=6)
        lb.grid(row=row, column=2, sticky="w")

        def upd(*_):
            val = V[key].get()
            if isinstance(V[key], tk.IntVar):
                val = int(round(val))
            lb.config(text=fmt.format(val))
        V[key].trace_add("write", upd)
        upd()
        sc.bind("<ButtonRelease-1>", lambda e: schedule_preview())
        sliders[key] = sc

    slider(2, "k_ha", "Ha-Staerke", 0.0, 2.0)
    slider(3, "k_oiii", "OIII-Staerke", 0.0, 2.0)
    slider(4, "k_lum", "Ha-Luminanz", 0.0, 1.0)
    slider(5, "noise_t", "Rauschschwelle (sigma)", 0.0, 5.0, "{:.1f}")
    slider(6, "smooth", "Glaettung (px)", 0, 8, "{:d}")
    ttk.Checkbutton(lf2, text="Sterne schuetzen", variable=V["star_protect"],
                    command=lambda: schedule_preview()).grid(row=7, column=0, sticky="w")

    def auto_k():
        P = st["P"]
        if P is None:
            messagebox.showinfo("Auto-k", "Erst 'Vorschau' ausfuehren.", parent=root)
            return
        V["k_ha"].set(round(P.auto_k["ha"], 2))
        if V["k_oiii"].get() > 0:
            V["k_oiii"].set(round(P.auto_k["oiii"], 2))
        schedule_preview()
    btn_auto = ttk.Button(lf2, text="Auto-k (blend)", command=auto_k)
    btn_auto.grid(row=7, column=1, sticky="e")

    mode_help = {"excess": "Linienemission ueber dem Kontinuum addieren (empfohlen)",
                 "blend": "Schmalband einmischen (k 0..1, Auto-k = rauschoptimal)"}

    def on_mode(*_):
        m = V["mode"].get()
        lbl_mode.config(text=mode_help[m])
        for key in ("k_ha", "k_oiii"):
            sliders[key].config(to=2.0 if m == "excess" else 1.0)
            if m == "blend" and V[key].get() > 1.0:
                V[key].set(1.0)
        schedule_preview()
    mode_box.bind("<<ComboboxSelected>>", on_mode)
    lbl_mode.config(text=mode_help[V["mode"].get()])

    ttk.Checkbutton(left, text="Arbeitsordner behalten (Debug)", variable=V["keep"]).grid(
        row=6, column=1, sticky="w")

    bfr = ttk.Frame(left)
    bfr.grid(row=7, column=0, columnspan=3, sticky="e", pady=(8, 0))
    btn_prev = ttk.Button(bfr, text="Vorschau", command=lambda: do_preview())
    btn_save = ttk.Button(bfr, text="Speichern", command=lambda: do_save())
    btn_close = ttk.Button(bfr, text="Schliessen", command=root.destroy)
    btn_close.pack(side="right", padx=3)
    btn_save.pack(side="right", padx=3)
    btn_prev.pack(side="right", padx=3)
    ttk.Label(left, textvariable=v_status, wraplength=480, foreground="#2060a0").grid(
        row=8, column=0, columnspan=3, sticky="w", pady=(8, 0))

    OV = 640
    CR = 360
    cv_ov = tk.Canvas(right, width=OV, height=OV, bg="#202020", highlightthickness=0)
    cv_ov.grid(row=0, column=0, rowspan=2, sticky="n")
    cv_cr = tk.Canvas(right, width=CR, height=CR, bg="#202020", highlightthickness=0)
    cv_cr.grid(row=0, column=1, sticky="n", padx=(8, 0))
    rf = ttk.Frame(right)
    rf.grid(row=1, column=1, sticky="nw", padx=(8, 0), pady=4)
    ttk.Checkbutton(rf, text="Vorher (nur IRCUT)", variable=v_before,
                    command=lambda: schedule_preview(0)).pack(anchor="w")
    ttk.Label(rf, text="Klick in die Uebersicht waehlt\nden 1:1-Ausschnitt.",
              foreground="gray").pack(anchor="w", pady=4)

    # ---------------- Logik ----------------
    def inputs_changed(reset_inputs=True):
        st["P"] = None if reset_inputs or st["P"] is None else st["P"]
        if st["P"] is not None and st["P"].prep_key != params().prep_key():
            st["P"] = None
        if st["P"] is None:
            v_status.set(v_status.get().split(" | ")[0] + " | Vorbereitung noetig "
                         "(Vorschau oder Speichern).")

    def set_busy(b):
        st["busy"] = b
        for w in (btn_prev, btn_save):
            w.config(state="disabled" if b else "normal")

    def run_bg(fn, done):
        set_busy(True)
        res = {}

        def work():
            try:
                res["v"] = fn()
            except Exception as e:
                res["e"] = e
                res["tb"] = traceback.format_exc()

        th = threading.Thread(target=work, daemon=True)
        th.start()

        def poll():
            if th.is_alive():
                root.after(150, poll)
                return
            set_busy(False)
            if "e" in res:
                combiner.log(res["tb"], s.LogColor.RED)
                v_status.set(f"Fehler: {res['e']}")
                messagebox.showerror("Fehler", str(res["e"]), parent=root)
            else:
                done(res.get("v"))
        poll()

    def files():
        return list(lists["ircut"].get(0, "end")), list(lists["lp"].get(0, "end"))

    def out_path():
        o = v_out.get().strip()
        if not o:
            ir = files()[0]
            o = os.path.join(os.path.dirname(ir[0]) if ir else initial_dir, DEFAULT_OUT)
            v_out.set(o.replace("\\", "/"))
        return o

    def ensure_prepared(then):
        ir, lp = files()
        if not ir or not lp:
            messagebox.showerror("Fehler", "Mindestens einen IRCUT- und einen LP-Stack "
                                 "angeben.", parent=root)
            return
        P = st["P"]
        if P is not None and P.inputs == (ir, lp) and P.prep_key == params().prep_key():
            then()
            return
        prm = params()
        od = os.path.dirname(os.path.abspath(out_path()))
        os.makedirs(od, exist_ok=True)
        v_status.set("Vorbereitung laeuft (Registrierung, Analyse) ...")

        def done(P):
            st["P"] = P
            v_status.set("Vorbereitet. Regler aendern -> Vorschau aktualisiert sich.")
            then()
        run_bg(lambda: combiner.prepare(ir, lp, od, prm), done)

    def do_preview():
        ensure_prepared(lambda: schedule_preview(0))

    def to_photo(rgb):
        a = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        a = np.ascontiguousarray(np.flipud(np.moveaxis(a, 0, -1)))
        h, w = a.shape[:2]
        return tk.PhotoImage(data=b"P6 %d %d 255\n" % (w, h) + a.tobytes(), format="PPM")

    def schedule_preview(delay=250):
        if st["pending"]:
            root.after_cancel(st["pending"])
            st["pending"] = None
        if st["P"] is not None and not st["busy"]:
            st["pending"] = root.after(delay, update_preview)

    def update_preview():
        st["pending"] = None
        P = st["P"]
        if P is None:
            return
        prm = params()
        H, W = P.shape
        f = max(1, int(np.ceil(max(H, W) / OV)))
        st["f"] = f
        B = P.binned(f)
        sp = B["_stretch"]
        if v_before.get():
            ov = B["rgb"]
        else:
            ov, _ = render(P, B, prm, f)
        st["img_ov"] = to_photo(apply_stretch(ov, sp))
        cv_ov.delete("all")
        cv_ov.create_image(0, 0, anchor="nw", image=st["img_ov"])
        hb = B["rgb"].shape[1]
        # 1:1-Ausschnitt
        if st["crop_c"] is None:
            st["crop_c"] = (H // 2, W // 2)
        cy, cx = st["crop_c"]
        half = CR // 2
        cy = min(max(cy, half), H - half)
        cx = min(max(cx, half), W - half)
        pad = 4 * (P.inpaint_r + prm.smooth) + 8
        y0, y1 = max(cy - half - pad, 0), min(cy + half + pad, H)
        x0, x1 = max(cx - half - pad, 0), min(cx + half + pad, W)
        A = P.region(y0, y1, x0, x1)
        cr = A["rgb"] if v_before.get() else render(P, A, prm, 1)[0]
        cr = cr[:, cy - half - y0:cy + half - y0, cx - half - x0:cx + half - x0]
        st["img_cr"] = to_photo(apply_stretch(cr, sp))
        cv_cr.delete("all")
        cv_cr.create_image(0, 0, anchor="nw", image=st["img_cr"])
        # Rahmen in der Uebersicht (Anzeige ist vertikal gespiegelt)
        rx0, rx1 = (cx - half) / f, (cx + half) / f
        ry0, ry1 = hb - (cy + half) / f, hb - (cy - half) / f
        cv_ov.create_rectangle(rx0, ry0, rx1, ry1, outline="#40ff40")

    def on_click(ev):
        P = st["P"]
        if P is None:
            return
        f = st["f"]
        hb = P.binned(f)["rgb"].shape[1]
        st["crop_c"] = (int((hb - 1 - ev.y) * f), int(ev.x * f))
        schedule_preview(0)
    cv_ov.bind("<Button-1>", on_click)

    def do_save():
        o = out_path()

        def go():
            prm = params()
            v_status.set("Speichere ...")
            run_bg(lambda: combiner.save(st["P"], prm, o),
                   lambda p: v_status.set(f"Gespeichert: {p} (in Siril geladen)"))
        ensure_prepared(go)

    root.mainloop()


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Seestar HaRGB-Combine")
    ap.add_argument("-ircut", nargs="+", help="IRCUT-Stack(s)")
    ap.add_argument("-lp", nargs="+", help="LP-Stack(s)")
    ap.add_argument("-dir", help="Ordner scannen (IRCUT/LP am Dateinamen erkennen)")
    ap.add_argument("-out", help=f"Ausgabedatei (Standard: {DEFAULT_OUT})")
    ap.add_argument("-mode", choices=["excess", "blend"], default=DEFAULTS["mode"])
    ap.add_argument("-k", default=str(DEFAULTS["k_ha"]), help="Ha-Staerke oder 'auto'")
    ap.add_argument("-koiii", default=str(DEFAULTS["k_oiii"]), help="OIII-Staerke oder 'auto'")
    ap.add_argument("-klum", type=float, default=DEFAULTS["k_lum"], help="Ha-Luminanz 0..1")
    ap.add_argument("-noise", type=float, default=DEFAULTS["noise_t"],
                    help="Rauschschwelle in sigma (0 = aus)")
    ap.add_argument("-smooth", type=int, default=DEFAULTS["smooth"], help="Glaettung (px)")
    ap.add_argument("-nopsf", action="store_true", help="kein PSF-Angleich")
    ap.add_argument("-nostarprot", action="store_true", help="kein Sternschutz")
    ap.add_argument("-bge", action="store_true", help="Gradientenentfernung")
    ap.add_argument("-crop", action="store_true", help="auf gemeinsamen Bereich zuschneiden")
    ap.add_argument("-keep", action="store_true", help="Arbeitsordner behalten")
    args = ap.parse_args()

    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception as e:
        print(f"Keine Verbindung zu Siril: {e}")
        sys.exit(1)

    cb = Combiner(siril)
    cb.log(f"Seestar HaRGB-Combine v{VERSION}", s.LogColor.BLUE)
    try:
        ircut, lp = args.ircut or [], args.lp or []
        if args.dir:
            d_ir, d_lp, _ = detect_stacks(args.dir)
            ircut = ircut or d_ir
            lp = lp or d_lp
        if ircut and lp:
            base_dir = args.dir or os.path.dirname(os.path.abspath(ircut[0]))
            out = os.path.abspath(args.out or os.path.join(base_dir, DEFAULT_OUT))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            prm = Params(mode=args.mode, k_lum=args.klum, noise_t=args.noise,
                         smooth=args.smooth, psf_match=not args.nopsf,
                         star_protect=not args.nostarprot, bge=args.bge,
                         crop=args.crop, keep=args.keep)
            P = cb.prepare([os.path.abspath(p) for p in ircut],
                           [os.path.abspath(p) for p in lp], os.path.dirname(out), prm)

            def kval(v, key, cap):
                if str(v).lower() == "auto":
                    return round(P.auto_k[key], 3) if prm.mode == "blend" else cap
                return float(v)
            prm.k_ha = kval(args.k, "ha", DEFAULTS["k_ha"])
            prm.k_oiii = kval(args.koiii, "oiii", 0.5)
            cb.log(f"Modus {prm.mode}: k_Ha={prm.k_ha} k_OIII={prm.k_oiii} k_Lum={prm.k_lum}")
            cb.save(P, prm, out)
        else:
            if args.dir or args.ircut or args.lp:
                cb.log("Keine passenden IRCUT- und LP-Stacks gefunden.", s.LogColor.RED)
                sys.exit(1)
            if siril.is_cli():
                cb.log("Im CLI-Modus bitte -dir oder -ircut und -lp angeben.",
                       s.LogColor.RED)
                sys.exit(1)
            try:
                wd = siril.get_siril_wd()
            except Exception:
                wd = os.path.expanduser("~")
            run_gui(cb, wd)
    except SystemExit:
        raise
    except Exception as e:
        cb.log(f"Fehler: {e}", s.LogColor.RED)
        cb.log(traceback.format_exc(), s.LogColor.RED)
        sys.exit(1)
    finally:
        try:
            siril.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
