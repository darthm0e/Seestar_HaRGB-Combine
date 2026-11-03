"""
Seestar HaRGB-Combine  --  Seestar S30 Pro  (Siril >= 1.4 Python script)
========================================================================

Combines IRCUT stacks (regular OSC RGB) with LP stacks (dual-band Ha/OIII
filter) into an HaRGB or HaOIII-RGB image.

Languages: English (default) and German. The language follows the
system/Siril language and can be changed in the dialog or with -lang en|de.

Pipeline
--------
 1. Load stacks - any number per filter (e.g. several nights), optionally
    found automatically by folder scan (filter detected from the file name:
    _IRCUT_ / _LP_, or the FILTER header keyword).
 2. Register all stacks onto the IRCUT stack with the most frames using star
    matching (Siril `register`), with fallback strategies.
 3. Normalize multiple stacks of the same filter and average them, weighted
    by noise.
 4. Optional: crop to the common area, remove gradients.
 5. Ha = red of the LP stack, OIII = mean of green + blue.
 6. Match star sharpness (FWHM) of RGB and Ha/OIII, match the continuum on the
    stars, determine the Ha/OIII line emission (excess).
 7. Smooth the excess, apply a noise threshold, mask the stars (the area under
    the stars is filled in from the surroundings).
 8. Add Ha to red, OIII to green/blue, optional Ha luminance boost.
 9. Save the result as 32-bit FITS (linear) and load it in Siril.

Modes
-----
  excess (default)  R' = R + k_Ha * E_Ha      G',B' += k_OIII * E_OIII
          E = smoothed, noise-cleaned line emission above the continuum.
          Stars and background stay unchanged.
  blend   R' = (1-k_Ha) * R + k_Ha * Ha_lin    (G,B likewise with OIII)
          Blends in the (lower-noise) narrowband. k = "auto" picks the
          noise-optimal weight.

  Ha luminance (k_Lum): brightness is additionally pulled towards Ha
  (detail/SNR of the Ha), colours are kept, stars are protected.

Usage
-----
  * Siril: Scripts menu -> Seestar_HaRGB-Combine  (dialog with preview)
  * Siril console / .ssf:
      pyscript "C:/path/to/Seestar_HaRGB-Combine.py" -dir "C:/data/NGC7000"
      pyscript "C:/path/to/Seestar_HaRGB-Combine.py" -ircut "a.fit" -lp "b.fit" "c.fit"
    Options: -out FILE  -mode excess|blend  -k 0.5|auto  -koiii 0  -klum 0
             -noise 2  -smooth 2  -nopsf  -nostarprot  -bge  -crop  -keep
             -lang en|de

Output (next to the output file):
  <name>.fit            result (linear)
  <name>_Ha.fit         registered Ha channel, linearly fitted to R
  <name>_OIII.fit       registered OIII channel, linearly fitted to (G+B)/2
  <name>_HaExcess.fit   pure Ha line emission (continuum-subtracted)
"""

import os
import re
import sys
import glob
import json
import locale
import shutil
import argparse
import threading
import traceback

# Capture the system language BEFORE importing sirilpy - sirilpy overwrites
# LANG/LC_ALL with en_US.UTF-8 on import.
_SYS_LANG_HINTS = [os.environ.get(k) for k in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG")]
try:
    _SYS_LANG_HINTS.append(locale.getlocale()[0])
except Exception:
    pass

import sirilpy as s  # noqa: E402

s.ensure_installed("numpy", "astropy")

import numpy as np                       # noqa: E402
from astropy.io import fits              # noqa: E402

VERSION = "2.1.0"
SEQ = "hapair"
WORKDIR_NAME = "_ha_combine_work"
FITS_EXT = (".fit", ".fits", ".fts")
DEFAULT_OUT = "HaRGB_combined.fit"

DEFAULTS = dict(mode="excess", k_ha=0.5, k_oiii=0.0, k_lum=0.0, noise_t=2.0,
                smooth=2, star_protect=True, psf_match=True, bge=False,
                crop=False, keep=False)


# ===========================================================================
# Translations
# ===========================================================================
LANGS = {"en": "English", "de": "Deutsch"}
LANG = "en"

TEXTS = {
    # --- log / errors ---
    "frames": ("{kind}: {name}  ({n} frames)", "{kind}: {name}  ({n} Frames)"),
    "loading_ref": ("Loading reference ...", "Lade Referenz ..."),
    "registering": ("Registering {name} ...", "Registriere {name} ..."),
    "reg_try_failed": ("  Registration {args} failed: {err}",
                       "  Registrierung {args} fehlgeschlagen: {err}"),
    "reg_ok_with": ("  Registration succeeded with {args}",
                    "  Registrierung gelungen mit {args}"),
    "reg_failed": ("Registration failed for {name} (too few common stars?)",
                   "Registrierung fehlgeschlagen für {name} (zu wenige gemeinsame Sterne?)"),
    "need_both": ("At least one IRCUT and one LP stack required.",
                  "Mindestens ein IRCUT- und ein LP-Stack nötig."),
    "n_averaged": ("{n} {kind} stacks averaged (noise-weighted)",
                   "{n} {kind}-Stacks rauschgewichtet gemittelt"),
    "overlap": ("Overlap IRCUT/LP: {p:.1f} %", "Überlappung IRCUT/LP: {p:.1f} %"),
    "cropped": ("Cropped to {w}x{h} px (x {x0}..{x1}, y {y0}..{y1})",
                "Zugeschnitten auf {w}x{h} px (x {x0}..{x1}, y {y0}..{y1})"),
    "bge_prog": ("Background extraction ...", "Gradientenentfernung ..."),
    "bge_log": ("Background extraction (subsky degree 1) ...",
                "Gradientenentfernung (subsky Grad 1) ..."),
    "analyze_stars": ("Analysing stars ...", "Analysiere Sterne ..."),
    "fwhm": ("FWHM ({n} stars): R {r:.2f}  Ha {ha:.2f}  G/B {gb:.2f}  OIII {o:.2f} px",
             "FWHM ({n} Sterne): R {r:.2f}  Ha {ha:.2f}  G/B {gb:.2f}  OIII {o:.2f} px"),
    "fwhm_none": ("Too few stars for FWHM measurement - no PSF matching.",
                  "Zu wenige Sterne für FWHM-Messung - kein PSF-Angleich."),
    "starmask": ("Star mask: {p:.1f} % of the image area",
                 "Sternmaske: {p:.1f} % der Bildfläche"),
    "psf": ("PSF matching {label}: {which} blurred (sigma {s:.2f} px)",
            "PSF-Angleich {label}: {which} weichgezeichnet (sigma {s:.2f} px)"),
    "cont_fallback": ("{label}: too few stars for continuum matching - using linear fit",
                      "{label}: zu wenige Sterne für Kontinuum-Abgleich - lineare Anpassung"),
    "cont": ("{label}: continuum matching ({n} star pixels) scale={a:.4f} off={b:+.5f}",
             "{label}: Kontinuum-Abgleich ({n} Sternpixel) scale={a:.4f} off={b:+.5f}"),
    "line_ha": ("Determining Ha line emission ...", "Bestimme Ha-Linienemission ..."),
    "line_o": ("Determining OIII line emission ...", "Bestimme OIII-Linienemission ..."),
    "leak": ("Ha crosstalk in OIII corrected: kappa={k:.3f}",
             "Ha-Übersprechen im OIII korrigiert: kappa={k:.3f}"),
    "noise": ("Noise: R {r:.2e}  Ha(to R) {h:.2e}  G {g:.2e}  OIII(to G) {o:.2e}",
              "Rauschen: R {r:.2e}  Ha(an R) {h:.2e}  G {g:.2e}  OIII(an G) {o:.2e}"),
    "autok": ("Auto-k for 'blend': Ha {h:.2f}  OIII {o:.2f}",
              "Auto-k für 'blend': Ha {h:.2f}  OIII {o:.2f}"),
    "prep_done": ("Preparation complete", "Vorbereitung abgeschlossen"),
    "computing": ("Computing result ...", "Berechne Ergebnis ..."),
    "saving": ("Saving ...", "Speichere ..."),
    "saved": ("Saved: {p}", "Gespeichert: {p}"),
    "done": ("Done. The result is linear - next e.g. SPCC and stretch.",
             "Fertig. Das Ergebnis ist linear - jetzt z.B. SPCC und Stretch."),
    "mode_line": ("Mode {mode}: k_Ha={kh} k_OIII={ko} k_Lum={kl}",
                  "Modus {mode}: k_Ha={kh} k_OIII={ko} k_Lum={kl}"),
    "no_stacks": ("No matching IRCUT and LP stacks found.",
                  "Keine passenden IRCUT- und LP-Stacks gefunden."),
    "cli_need": ("In CLI mode please specify -dir or -ircut and -lp.",
                 "Im CLI-Modus bitte -dir oder -ircut und -lp angeben."),
    "error": ("Error: {e}", "Fehler: {e}"),
    "no_conn": ("No connection to Siril: {e}", "Keine Verbindung zu Siril: {e}"),
    "too_few_px": ("Too few common pixels for the linear fit - do the stacks match?",
                   "Zu wenige gemeinsame Pixel für die lineare Anpassung - "
                   "passen die Stacks zusammen?"),
    "no_data": ("No image data found in: {p}", "Keine Bilddaten gefunden in: {p}"),
    "bad_shape": ("Unexpected image shape {s} in: {p}", "Unerwartete Bildform {s} in: {p}"),
    "bad_mode": ("Unknown mode: {m}", "Unbekannter Modus: {m}"),
    # --- GUI ---
    "ircut_stacks": ("IRCUT stacks:", "IRCUT-Stacks:"),
    "lp_stacks": ("LP stacks:", "LP-Stacks:"),
    "scan": ("Scan folder ...", "Ordner scannen ..."),
    "scan_title": ("Folder with Seestar stacks", "Ordner mit Seestar-Stacks"),
    "scan_found": ("Found {i} IRCUT and {l} LP stacks.",
                   "{i} IRCUT- und {l} LP-Stacks gefunden."),
    "scan_unknown": (" {n} file(s) could not be assigned.",
                     " {n} Datei(en) nicht zuordenbar."),
    "scan_hint": (" Remove unsuitable entries if needed (do not mix per-session "
                  "and combined stacks).",
                  " Ggf. unpassende Einträge entfernen (Einzel- und Gesamt-Stacks "
                  "nicht mischen)."),
    "output": ("Output:", "Ausgabe:"),
    "out_title": ("Output file", "Ausgabedatei"),
    "all_files": ("All files", "Alle Dateien"),
    "grp_prep": ("Preparation (change = recompute)", "Vorbereitung (Änderung = neu berechnen)"),
    "opt_psf": ("Match star sharpness (PSF)", "Sternschärfe angleichen (PSF)"),
    "opt_bge": ("Background extraction (subsky degree 1)",
                "Gradientenentfernung (subsky Grad 1)"),
    "opt_crop": ("Crop to common area", "Auf gemeinsamen Bildbereich zuschneiden"),
    "grp_comb": ("Combination (live preview)", "Kombination (Vorschau live)"),
    "mode": ("Mode:", "Modus:"),
    "sl_kha": ("Ha strength", "Ha-Stärke"),
    "sl_ko": ("OIII strength", "OIII-Stärke"),
    "sl_kl": ("Ha luminance", "Ha-Luminanz"),
    "sl_noise": ("Noise threshold (sigma)", "Rauschschwelle (sigma)"),
    "sl_smooth": ("Smoothing (px)", "Glättung (px)"),
    "star_prot": ("Protect stars", "Sterne schützen"),
    "autok_btn": ("Auto-k (blend)", "Auto-k (blend)"),
    "autok_first": ("Run 'Preview' first.", "Erst 'Vorschau' ausführen."),
    "mode_excess": ("Add line emission above the continuum (recommended)",
                    "Linienemission über dem Kontinuum addieren (empfohlen)"),
    "mode_blend": ("Blend in narrowband (k 0..1, Auto-k = noise-optimal)",
                   "Schmalband einmischen (k 0..1, Auto-k = rauschoptimal)"),
    "keep": ("Keep working folder (debug)", "Arbeitsordner behalten (Debug)"),
    "preview": ("Preview", "Vorschau"),
    "save": ("Save", "Speichern"),
    "close": ("Close", "Schließen"),
    "before": ("Before (IRCUT only)", "Vorher (nur IRCUT)"),
    "click_hint": ("Click into the overview to select\nthe 1:1 crop.",
                   "Klick in die Übersicht wählt\nden 1:1-Ausschnitt."),
    "language": ("Language:", "Sprache:"),
    "st_start": ("Select stacks or scan a folder.", "Stacks wählen oder Ordner scannen."),
    "st_needprep": ("Preparation required (Preview or Save).",
                    "Vorbereitung nötig (Vorschau oder Speichern)."),
    "st_preparing": ("Preparing (registration, analysis) ...",
                     "Vorbereitung läuft (Registrierung, Analyse) ..."),
    "st_prepared": ("Prepared. Move the sliders - the preview updates.",
                    "Vorbereitet. Regler ändern - die Vorschau aktualisiert sich."),
    "st_saved": ("Saved: {p} (loaded in Siril)", "Gespeichert: {p} (in Siril geladen)"),
    "err_title": ("Error", "Fehler"),
}


def tr(key, **kw):
    en, de = TEXTS[key]
    txt = de if LANG == "de" else en
    return txt.format(**kw) if kw else txt


def settings_path():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "Seestar_HaRGB-Combine.json")


def load_settings():
    try:
        with open(settings_path(), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_settings(**kw):
    d = load_settings()
    d.update(kw)
    try:
        os.makedirs(os.path.dirname(settings_path()), exist_ok=True)
        with open(settings_path(), "w", encoding="utf-8") as fh:
            json.dump(d, fh)
    except Exception:
        pass


def _lang_code(value):
    return "de" if value.lower().startswith(("de", "german")) else "en"


def siril_ui_lang():
    """Language set in Siril's preferences ('lang=' in config.*.ini), or None."""
    dirs = [os.path.join(os.environ.get("LOCALAPPDATA", ""), "siril"),
            os.path.join(os.path.expanduser("~"), ".config", "siril"),
            os.path.join(os.path.expanduser("~"), "Library", "Application Support",
                         "org.siril.Siril", "siril")]
    cfgs = [p for d in dirs if d for p in glob.glob(os.path.join(d, "config*.ini"))]
    for cfg in sorted(cfgs, key=os.path.getmtime, reverse=True):
        try:
            with open(cfg, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("lang="):
                        val = line.split("=", 1)[1].strip()
                        return val or None
        except Exception:
            pass
    return None


def detect_lang(cli=None):
    """-lang argument > saved choice > Siril language > OS language > English."""
    if cli in LANGS:
        return cli
    saved = load_settings().get("lang")
    if saved in LANGS:
        return saved
    sl = siril_ui_lang()
    if sl:
        return _lang_code(sl)
    if sys.platform == "win32":
        try:
            import ctypes
            lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            return "de" if (lid & 0x3FF) == 0x07 else "en"
        except Exception:
            pass
    for c in _SYS_LANG_HINTS:
        if c:
            return _lang_code(c)
    return "en"


class Params:
    def __init__(self, **kw):
        for k, v in DEFAULTS.items():
            setattr(self, k, kw.get(k, v))

    def prep_key(self):
        """Options that require a new preparation."""
        return (self.psf_match, self.bge, self.crop)


# ===========================================================================
# FITS I/O
# ===========================================================================
def read_fits(path):
    """Reads a FITS image as float32 (C, H, W) in [0..1] plus header."""
    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
        if hdu is None:
            raise ValueError(tr("no_data", p=path))
        data = hdu.data
        header = hdu.header.copy()

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim == 3 and data.shape[0] not in (1, 3) and data.shape[-1] in (1, 3):
        data = np.moveaxis(data, -1, 0)
    if data.ndim != 3 or data.shape[0] not in (1, 3):
        raise ValueError(tr("bad_shape", s=data.shape, p=path))

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
    for line in history or []:          # FITS headers are ASCII -> always English
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
# Stack detection
# ===========================================================================
RE_IRCUT = re.compile(r"(?<![a-z0-9])(ircut|ir-cut|uvir|uv-ir|uvircut)(?![a-z0-9])", re.I)
RE_LP = re.compile(r"(?<![a-z0-9])(lp|lpf|dual|dualband|duoband|l-enhance|lenhance)"
                   r"(?![a-z0-9])", re.I)


def classify_stack(path):
    """'ircut', 'lp' or None - from the file name first, else the FILTER header."""
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
    """Number of stacked frames from file name or header (0 = unknown)."""
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
    """Finds IRCUT and LP stacks in a folder (not recursive)."""
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
# Image processing
# ===========================================================================
def box_mean(a, r):
    """Box mean (radius r) via integral image, edge padding."""
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
    """Gaussian approximation by 3 box passes, variance interpolated exactly."""
    if sigma < 0.25:
        return a.astype(np.float32, copy=True)

    def box3(img, r):
        for _ in range(3):
            img = box_mean(img, r)
        return img

    v = sigma * sigma                      # 3 boxes of radius r: variance r(r+1)
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
    """Pixel noise (MAD of a high-pass) on a central crop."""
    sl = center_slice(img.shape)
    c = img[sl]
    hp = c - box_mean(c, 2)
    v = valid[sl]
    return mad_sigma(hp[v][::3]) * 1.09 if v.any() else 0.0


def smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def linear_fit(src, ref, valid, low=1e-6, high=0.92, max_n=2_000_000):
    """Least squares  src*a + b ~ ref  (like Siril linear_match)."""
    m = valid & (ref > low) & (ref < high) & (src > low) & (src < high)
    idx = np.flatnonzero(m)
    if idx.size < 1000:
        raise RuntimeError(tr("too_few_px"))
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
    """Largest rectangle whose borders are fully valid."""
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
    """Median FWHM (pixels) per channel, measured on the same stars."""
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
    """Blurs the sharper image to the FWHM of the other one."""
    sa, sb = fa / 2.3548, fb / 2.3548
    if abs(sa - sb) < 0.15:
        return a, b, 0.0, 0.0
    if sa < sb:
        sig = float(np.sqrt(sb * sb - sa * sa))
        return gauss_blur(a, sig), b, sig, 0.0
    sig = float(np.sqrt(sa * sa - sb * sb))
    return a, gauss_blur(b, sig), 0.0, sig


def star_mask(r, valid, fwhm):
    """Soft star mask 0..1 from the RGB red channel."""
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
    """Scale narrowband -> broadband using the stars (continuum)."""
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
    """Line emission: smooth, noise threshold, fill in star areas."""
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
    """Unlinked autostretch parameters (like Siril) from the IRCUT image."""
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
    """Several stacks of one filter: normalize + noise-weighted mean."""

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
# Prepared data
# ===========================================================================
class Prepared:
    """Registered data + all analysis results; basis for preview/export."""

    def __init__(self):
        self.arrays = {}
        self.coef = {}
        self._sigma = {}
        self._bins = {}
        self.fwhm = None
        self.inpaint_r = 6
        self.auto_k = {}
        self.hdr = None
        self.inputs = ([], [])
        self.prep_key = None

    @property
    def shape(self):
        return self.arrays["ha"].shape

    def sigma(self, key, r):
        """Noise of the (smoothed) difference image - always at full resolution."""
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
            B = {}
            for k, v in self.arrays.items():
                if v is None:
                    B[k] = None
                elif k == "valid":
                    B[k] = bin_image(v, f, "min")
                elif k == "star":
                    B[k] = bin_image(v, f, "max")
                else:
                    B[k] = bin_image(v, f)
            B["_stretch"] = stretch_params(B["rgb"], B["valid"])
            self._bins = {f: B}
        return self._bins[f]

    def region(self, y0, y1, x0, x1):
        return {k: (None if v is None else v[..., y0:y1, x0:x1])
                for k, v in self.arrays.items()}


def render(P, A, prm, f=1, want_excess=False):
    """Computes the combined image for arrays A (full, crop, or binned by f)."""
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
        raise ValueError(tr("bad_mode", m=prm.mode))

    if prm.k_lum > 0:
        # Scale only the signal ABOVE the sky background - scaling the
        # background pedestal as well would wash out the colours.
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
# Processing
# ===========================================================================
class Combiner:
    def __init__(self, siril):
        self.siril = siril

    # --- Siril helpers -------------------------------------------------------
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
        """Background extraction (polynomial degree 1) using Siril."""
        path = os.path.join(workdir, name + ".fit")
        write_fits(path, data)
        self.siril.cmd("cd", q(workdir))
        self.siril.cmd("load", name)
        self.siril.cmd("subsky", "1", "-samples=20", "-tolerance=1.0")
        self.siril.cmd("save", name)
        return read_fits(path)[0]

    def register_to_ref(self, workdir, idx, path, ref_rgb, keep=False):
        """Registers one stack onto ref.fit, with fallback strategies."""
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
                self.log(tr("reg_try_failed", args=" ".join(args), err=e),
                         s.LogColor.SALMON)
                continue
            if os.path.isfile(r1) and os.path.isfile(r2):
                if n > 0:
                    self.log(tr("reg_ok_with", args=" ".join(args)), s.LogColor.SALMON)
                break
        else:
            raise RuntimeError(tr("reg_failed", name=os.path.basename(path)))
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

    # --- Preparation -----------------------------------------------------------
    def prepare(self, ircut_files, lp_files, out_dir, prm):
        if not ircut_files or not lp_files:
            raise ValueError(tr("need_both"))
        P = Prepared()
        P.inputs = (list(ircut_files), list(lp_files))
        P.prep_key = prm.prep_key()
        workdir = os.path.join(out_dir, WORKDIR_NAME)
        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir)
        try:
            # --- 1. Reference ---------------------------------------------------
            ircut_files = sorted(ircut_files, key=lambda p: -frame_count(p))
            for kind, files in (("IRCUT", ircut_files), ("LP", lp_files)):
                for p in files:
                    n = frame_count(p)
                    name = os.path.basename(p)
                    self.log(tr("frames", kind=kind, name=name, n=n) if n
                             else f"{kind}: {name}")
            self.progress(tr("loading_ref"), 0.02)
            ref, hdr = read_fits(ircut_files[0])
            ref = to_rgb(ref)
            write_fits(os.path.join(workdir, "ref.fit"), ref, hdr)
            P.hdr = hdr

            # --- 2./3. Register + average groups ---------------------------------
            g_ircut, g_lp = GroupStack(), GroupStack()
            g_ircut.add(ref)
            jobs = [("ircut", p) for p in ircut_files[1:]] + [("lp", p) for p in lp_files]
            for i, (kind, p) in enumerate(jobs):
                msg = tr("registering", name=os.path.basename(p))
                self.progress(msg, 0.05 + 0.45 * i / len(jobs))
                self.log(msg, s.LogColor.BLUE)
                reg = self.register_to_ref(workdir, i + 1, p, ref, prm.keep)
                (g_ircut if kind == "ircut" else g_lp).add(reg)
                del reg
            del ref
            rgb, cov_rgb = g_ircut.result()
            lp, cov_lp = g_lp.result()
            del g_ircut, g_lp
            if len(ircut_files) > 1:
                self.log(tr("n_averaged", n=len(ircut_files), kind="IRCUT"))
            if len(lp_files) > 1:
                self.log(tr("n_averaged", n=len(lp_files), kind="LP"))
            valid = cov_rgb & cov_lp & np.all(rgb > 0, axis=0) & np.all(lp > 0, axis=0)
            self.log(tr("overlap", p=100.0 * valid.mean()))

            # --- 4. Crop -----------------------------------------------------------
            if prm.crop:
                y0, y1, x0, x1 = common_crop(valid)
                rgb = np.ascontiguousarray(rgb[:, y0:y1, x0:x1])
                lp = np.ascontiguousarray(lp[:, y0:y1, x0:x1])
                valid = np.ascontiguousarray(valid[y0:y1, x0:x1])
                for key, off in (("CRPIX1", x0), ("CRPIX2", y0)):
                    if key in P.hdr:
                        P.hdr[key] = float(P.hdr[key]) - off
                self.log(tr("cropped", w=x1 - x0, h=y1 - y0, x0=x0, x1=x1, y0=y0, y1=y1))

            ha = np.ascontiguousarray(lp[0])
            oiii = 0.5 * (lp[1] + lp[2])
            del lp

            # --- Background extraction ----------------------------------------------
            if prm.bge:
                self.progress(tr("bge_prog"), 0.55)
                self.log(tr("bge_log"))
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

            # --- 5./6. Analysis ---------------------------------------------------
            self.progress(tr("analyze_stars"), 0.65)
            r = rgb[0]
            gb = 0.5 * (rgb[1] + rgb[2])
            fw = measure_fwhm({"R": r, "Ha": ha, "GB": gb, "OIII": oiii}, r, valid)
            P.fwhm = fw
            if fw:
                self.log(tr("fwhm", n=fw["n"], r=fw["R"], ha=fw["Ha"], gb=fw["GB"],
                            o=fw["OIII"]))
                fmax = max(fw["R"], fw["Ha"], fw["GB"], fw["OIII"])
            else:
                self.log(tr("fwhm_none"), s.LogColor.SALMON)
                fmax = 4.0
            star, rd = star_mask(r, valid, fmax)
            P.inpaint_r = 3 * rd
            self.log(tr("starmask", p=100.0 * (star > 0.5).mean()))

            def zero_point(diff):
                """Zero point = darkest 5 % (smoothed) -> no emission there."""
                sm = box_mean(diff, 16)
                diff -= float(np.percentile(sm[valid & (star < 0.5)][::5], 5))
                diff[~valid] = 0.0
                return diff.astype(np.float32)

            def line_diff(nb, bb, f_nb, f_bb, label):
                if prm.psf_match and fw:
                    nb_m, bb_m, s_nb, s_bb = psf_match(nb, bb, f_nb, f_bb)
                    if s_nb or s_bb:
                        self.log(tr("psf", label=label, which=label if s_nb else "RGB",
                                    s=max(s_nb, s_bb)))
                else:
                    nb_m, bb_m = nb, bb
                cs = continuum_scale(nb_m, bb_m, valid, fmax)
                if cs is None:
                    a, b = linear_fit(nb, bb, valid)
                    self.log(tr("cont_fallback", label=label), s.LogColor.SALMON)
                else:
                    a, b, n = cs
                    self.log(tr("cont", label=label, n=n, a=a, b=b))
                diff = (a * nb_m + b) - bb_m
                del nb_m, bb_m
                return zero_point(diff), (a, b)

            self.progress(tr("line_ha"), 0.75)
            diff_ha, P.coef["ha_cont"] = line_diff(
                ha, r, fw and fw["Ha"], fw and fw["R"], "Ha")
            self.progress(tr("line_o"), 0.85)
            diff_o, P.coef["o_cont"] = line_diff(
                oiii, gb, fw and fw["OIII"], fw and fw["GB"], "OIII")

            # Ha crosstalk in OIII: the Bayer green/blue filters pass some Ha
            # (656 nm). Assumption: the brightest Ha regions contain places
            # without OIII -> smallest OIII/Ha ratio = crosstalk.
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
            self.log(tr("leak", k=kappa))
            del sh, so, vb, strong

            # linear fits (blend, output files, luminance)
            P.coef["ha_r"] = linear_fit(ha, r, valid)
            P.coef["o_g"] = linear_fit(oiii, rgb[1], valid)
            P.coef["o_b"] = linear_fit(oiii, rgb[2], valid)
            P.coef["o_gb"] = linear_fit(oiii, gb, valid)
            P.coef["ha_l"] = linear_fit(ha, rgb.mean(axis=0), valid)
            del gb
            # background pedestal per channel (for the luminance boost)
            vb4 = bin_image(valid, 4, "min")
            P.coef["ped"] = [float(np.percentile(bin_image(rgb[c], 4)[vb4], 1))
                             for c in range(3)]
            P.coef["l_floor"] = 3.0 * noise_sigma(rgb.mean(axis=0), valid)

            # noise-optimal blend weights
            s_r = noise_sigma(r, valid)
            s_h = P.coef["ha_r"][0] * noise_sigma(ha, valid)
            s_g = noise_sigma(rgb[1], valid)
            s_o = P.coef["o_g"][0] * noise_sigma(oiii, valid)
            P.auto_k = {"ha": s_r ** 2 / max(s_r ** 2 + s_h ** 2, 1e-30),
                        "oiii": s_g ** 2 / max(s_g ** 2 + s_o ** 2, 1e-30)}
            self.log(tr("noise", r=s_r, h=s_h, g=s_g, o=s_o))
            self.log(tr("autok", h=P.auto_k["ha"], o=P.auto_k["oiii"]))

            P.arrays = dict(rgb=rgb, ha=ha, oiii=oiii, valid=valid, star=star,
                            diff_ha=diff_ha, diff_o=diff_o)
            self.progress(tr("prep_done"), 1.0)
            return P
        finally:
            try:
                self.siril.cmd("cd", q(out_dir))
            except Exception:
                pass
            if not prm.keep:
                shutil.rmtree(workdir, ignore_errors=True)

    # --- Export ------------------------------------------------------------------
    def save(self, P, prm, out_path):
        out_path = os.path.abspath(out_path)
        if not out_path.lower().endswith(FITS_EXT):
            out_path += ".fit"
        base = os.path.splitext(out_path)[0]
        self.progress(tr("computing"), 0.3)
        result, e_ha = render(P, P.arrays, prm, 1, want_excess=True)
        A = P.arrays
        hist = [f"Seestar HaRGB-Combine v{VERSION}",
                f"mode={prm.mode} k_ha={prm.k_ha} k_oiii={prm.k_oiii} k_lum={prm.k_lum} "
                f"noise={prm.noise_t} smooth={prm.smooth}",
                f"starprot={prm.star_protect} psf={prm.psf_match} bge={prm.bge} "
                f"crop={prm.crop}",
                "IRCUT: " + ", ".join(os.path.basename(p) for p in P.inputs[0])[:60],
                "LP: " + ", ".join(os.path.basename(p) for p in P.inputs[1])[:60]]
        self.progress(tr("saving"), 0.7)
        write_fits(out_path, result, P.hdr, hist)
        a, b = P.coef["ha_r"]
        write_fits(base + "_Ha.fit", np.where(A["valid"], a * A["ha"] + b, 0.0),
                   P.hdr, ["Ha from LP stack, linearly fitted to R"])
        a, b = P.coef["o_gb"]
        write_fits(base + "_OIII.fit", np.where(A["valid"], a * A["oiii"] + b, 0.0),
                   P.hdr, ["OIII from LP stack, linearly fitted to (G+B)/2"])
        write_fits(base + "_HaExcess.fit", e_ha, P.hdr,
                   ["Ha line emission above continuum (k=1)"])
        for p in (out_path, base + "_Ha.fit", base + "_OIII.fit", base + "_HaExcess.fit"):
            self.log(tr("saved", p=p), s.LogColor.GREEN)
        self.siril.cmd("cd", q(os.path.dirname(out_path)))
        self.siril.cmd("load", q(out_path))
        self.progress(tr("done"), 1.0)
        self.log(tr("done"), s.LogColor.GREEN)
        try:
            self.siril.reset_progress()
        except Exception:
            pass
        return out_path


# ===========================================================================
# GUI
# ===========================================================================
def run_gui(combiner, initial_dir, state):
    """Shows the dialog. Returns True if it should be rebuilt (language change)."""
    global LANG
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title(f"Seestar HaRGB-Combine  v{VERSION}")
    try:
        from sirilpy import tksiril
        tksiril.elevate(root)
    except Exception:
        pass

    st = {"P": state.get("P"), "busy": False, "crop_c": state.get("crop_c"),
          "img_ov": None, "img_cr": None, "f": 1, "pending": None, "restart": False}
    ftypes = [("FITS", "*.fit *.fits *.fts"), (tr("all_files"), "*.*")]

    p0 = state.get("params") or Params()
    V = dict(mode=tk.StringVar(value=p0.mode),
             k_ha=tk.DoubleVar(value=p0.k_ha),
             k_oiii=tk.DoubleVar(value=p0.k_oiii),
             k_lum=tk.DoubleVar(value=p0.k_lum),
             noise_t=tk.DoubleVar(value=p0.noise_t),
             smooth=tk.IntVar(value=p0.smooth),
             star_protect=tk.BooleanVar(value=p0.star_protect),
             psf_match=tk.BooleanVar(value=p0.psf_match),
             bge=tk.BooleanVar(value=p0.bge),
             crop=tk.BooleanVar(value=p0.crop),
             keep=tk.BooleanVar(value=p0.keep))
    v_out = tk.StringVar(value=state.get("out", ""))
    v_before = tk.BooleanVar(value=False)
    v_status = tk.StringVar(value=tr("st_prepared") if st["P"] else tr("st_start"))
    v_lang = tk.StringVar(value=LANGS[LANG])

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
        for p in state.get(key, []):
            lb.insert("end", p)
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

    file_list(0, tr("ircut_stacks"), "ircut")
    file_list(1, tr("lp_stacks"), "lp")

    def scan():
        d = filedialog.askdirectory(title=tr("scan_title"), initialdir=initial_dir,
                                    parent=root)
        if not d:
            return
        ir, lp, unk = detect_stacks(d)
        for key, ps in (("ircut", ir), ("lp", lp)):
            lists[key].delete(0, "end")
            for p in ps:
                lists[key].insert("end", p.replace("\\", "/"))
        msg = tr("scan_found", i=len(ir), l=len(lp))
        if unk:
            msg += tr("scan_unknown", n=len(unk))
        v_out.set(os.path.join(d, DEFAULT_OUT).replace("\\", "/"))
        inputs_changed()
        v_status.set(msg + tr("scan_hint"))

    top = ttk.Frame(left)
    top.grid(row=2, column=1, columnspan=2, sticky="we", pady=(2, 6))
    ttk.Button(top, text=tr("scan"), command=scan).pack(side="left")
    lang_box = ttk.Combobox(top, textvariable=v_lang, state="readonly", width=9,
                            values=list(LANGS.values()))
    lang_box.pack(side="right")
    ttk.Label(top, text=tr("language")).pack(side="right", padx=4)

    ttk.Label(left, text=tr("output")).grid(row=3, column=0, sticky="w")
    of = ttk.Frame(left)
    of.grid(row=3, column=1, columnspan=2, sticky="we")
    ttk.Entry(of, textvariable=v_out, width=55).pack(side="left")

    def pick_out():
        p = filedialog.asksaveasfilename(title=tr("out_title"), parent=root,
                                         initialdir=initial_dir, defaultextension=".fit",
                                         filetypes=ftypes, initialfile=DEFAULT_OUT)
        if p:
            v_out.set(p)
    ttk.Button(of, text="...", width=3, command=pick_out).pack(side="left", padx=3)

    lf1 = ttk.LabelFrame(left, text=tr("grp_prep"), padding=6)
    lf1.grid(row=4, column=0, columnspan=3, sticky="we", pady=(10, 4))
    for i, (key, txt) in enumerate((("psf_match", "opt_psf"), ("bge", "opt_bge"),
                                     ("crop", "opt_crop"))):
        ttk.Checkbutton(lf1, text=tr(txt), variable=V[key],
                        command=lambda: inputs_changed(False)).grid(row=i, column=0, sticky="w")

    lf2 = ttk.LabelFrame(left, text=tr("grp_comb"), padding=6)
    lf2.grid(row=5, column=0, columnspan=3, sticky="we", pady=4)
    ttk.Label(lf2, text=tr("mode")).grid(row=0, column=0, sticky="w")
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

    kmax = 2.0 if p0.mode == "excess" else 1.0
    slider(2, "k_ha", tr("sl_kha"), 0.0, kmax)
    slider(3, "k_oiii", tr("sl_ko"), 0.0, kmax)
    slider(4, "k_lum", tr("sl_kl"), 0.0, 1.0)
    slider(5, "noise_t", tr("sl_noise"), 0.0, 5.0, "{:.1f}")
    slider(6, "smooth", tr("sl_smooth"), 0, 8, "{:d}")
    ttk.Checkbutton(lf2, text=tr("star_prot"), variable=V["star_protect"],
                    command=lambda: schedule_preview()).grid(row=7, column=0, sticky="w")

    def auto_k():
        P = st["P"]
        if P is None:
            messagebox.showinfo("Auto-k", tr("autok_first"), parent=root)
            return
        V["k_ha"].set(round(P.auto_k["ha"], 2))
        if V["k_oiii"].get() > 0:
            V["k_oiii"].set(round(P.auto_k["oiii"], 2))
        schedule_preview()
    ttk.Button(lf2, text=tr("autok_btn"), command=auto_k).grid(row=7, column=1, sticky="e")

    def on_mode(*_):
        m = V["mode"].get()
        lbl_mode.config(text=tr("mode_" + m))
        for key in ("k_ha", "k_oiii"):
            sliders[key].config(to=2.0 if m == "excess" else 1.0)
            if m == "blend" and V[key].get() > 1.0:
                V[key].set(1.0)
        schedule_preview()
    mode_box.bind("<<ComboboxSelected>>", on_mode)
    lbl_mode.config(text=tr("mode_" + V["mode"].get()))

    ttk.Checkbutton(left, text=tr("keep"), variable=V["keep"]).grid(
        row=6, column=1, sticky="w")

    bfr = ttk.Frame(left)
    bfr.grid(row=7, column=0, columnspan=3, sticky="e", pady=(8, 0))
    btn_prev = ttk.Button(bfr, text=tr("preview"), command=lambda: do_preview())
    btn_save = ttk.Button(bfr, text=tr("save"), command=lambda: do_save())
    ttk.Button(bfr, text=tr("close"), command=root.destroy).pack(side="right", padx=3)
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
    ttk.Checkbutton(rf, text=tr("before"), variable=v_before,
                    command=lambda: schedule_preview(0)).pack(anchor="w")
    ttk.Label(rf, text=tr("click_hint"), foreground="gray").pack(anchor="w", pady=4)

    # ---------------- Logic ----------------
    def inputs_changed(reset_inputs=True):
        if reset_inputs:
            st["P"] = None
        elif st["P"] is not None and st["P"].prep_key != params().prep_key():
            st["P"] = None
        if st["P"] is None:
            v_status.set(tr("st_needprep"))

    def set_busy(b):
        st["busy"] = b
        for w in (btn_prev, btn_save):
            w.config(state="disabled" if b else "normal")
        lang_box.config(state="disabled" if b else "readonly")

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
                v_status.set(tr("error", e=res["e"]))
                messagebox.showerror(tr("err_title"), str(res["e"]), parent=root)
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
            messagebox.showerror(tr("err_title"), tr("need_both"), parent=root)
            return
        P = st["P"]
        if P is not None and P.inputs == (ir, lp) and P.prep_key == params().prep_key():
            then()
            return
        prm = params()
        od = os.path.dirname(os.path.abspath(out_path()))
        os.makedirs(od, exist_ok=True)
        v_status.set(tr("st_preparing"))

        def done(P):
            st["P"] = P
            v_status.set(tr("st_prepared"))
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
        ov = B["rgb"] if v_before.get() else render(P, B, prm, f)[0]
        st["img_ov"] = to_photo(apply_stretch(ov, sp))
        cv_ov.delete("all")
        cv_ov.create_image(0, 0, anchor="nw", image=st["img_ov"])
        hb = B["rgb"].shape[1]
        # 1:1 crop
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
        # frame in the overview (display is flipped vertically)
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
            v_status.set(tr("saving"))
            run_bg(lambda: combiner.save(st["P"], prm, o),
                   lambda p: v_status.set(tr("st_saved", p=p)))
        ensure_prepared(go)

    def on_lang(*_):
        global LANG
        new = next(k for k, v in LANGS.items() if v == v_lang.get())
        if new == LANG or st["busy"]:
            return
        LANG = new
        save_settings(lang=new)
        ir, lp = files()
        state.update(P=st["P"], ircut=ir, lp=lp, out=v_out.get(), params=params(),
                     crop_c=st["crop_c"])
        st["restart"] = True
        root.destroy()
    lang_box.bind("<<ComboboxSelected>>", on_lang)

    if st["P"] is not None:
        root.after(100, lambda: schedule_preview(0))
    root.mainloop()
    return st["restart"]


# ===========================================================================
# main
# ===========================================================================
def main():
    global LANG
    ap = argparse.ArgumentParser(description="Seestar HaRGB-Combine")
    ap.add_argument("-ircut", nargs="+", help="IRCUT stack(s)")
    ap.add_argument("-lp", nargs="+", help="LP stack(s)")
    ap.add_argument("-dir", help="scan folder (detect IRCUT/LP from file names)")
    ap.add_argument("-out", help=f"output file (default: {DEFAULT_OUT})")
    ap.add_argument("-mode", choices=["excess", "blend"], default=DEFAULTS["mode"])
    ap.add_argument("-k", default=str(DEFAULTS["k_ha"]), help="Ha strength or 'auto'")
    ap.add_argument("-koiii", default=str(DEFAULTS["k_oiii"]), help="OIII strength or 'auto'")
    ap.add_argument("-klum", type=float, default=DEFAULTS["k_lum"], help="Ha luminance 0..1")
    ap.add_argument("-noise", type=float, default=DEFAULTS["noise_t"],
                    help="noise threshold in sigma (0 = off)")
    ap.add_argument("-smooth", type=int, default=DEFAULTS["smooth"], help="smoothing (px)")
    ap.add_argument("-nopsf", action="store_true", help="no PSF matching")
    ap.add_argument("-nostarprot", action="store_true", help="no star protection")
    ap.add_argument("-bge", action="store_true", help="background extraction")
    ap.add_argument("-crop", action="store_true", help="crop to common area")
    ap.add_argument("-keep", action="store_true", help="keep working folder")
    ap.add_argument("-lang", choices=list(LANGS), help="language (en, de)")
    args = ap.parse_args()
    LANG = detect_lang(args.lang)

    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception as e:
        print(tr("no_conn", e=e))
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
            cb.log(tr("mode_line", mode=prm.mode, kh=prm.k_ha, ko=prm.k_oiii, kl=prm.k_lum))
            cb.save(P, prm, out)
        else:
            if args.dir or args.ircut or args.lp:
                cb.log(tr("no_stacks"), s.LogColor.RED)
                sys.exit(1)
            if siril.is_cli():
                cb.log(tr("cli_need"), s.LogColor.RED)
                sys.exit(1)
            try:
                wd = siril.get_siril_wd()
            except Exception:
                wd = os.path.expanduser("~")
            state = {}
            while run_gui(cb, wd, state):
                pass
    except SystemExit:
        raise
    except Exception as e:
        cb.log(tr("error", e=e), s.LogColor.RED)
        cb.log(traceback.format_exc(), s.LogColor.RED)
        sys.exit(1)
    finally:
        try:
            siril.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
