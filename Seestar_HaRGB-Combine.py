"""
H-Alpha Extract & Combine  --  Seestar S30 Pro  (Siril >= 1.4 Python-Script)
=============================================================================

Kombiniert einen IRCUT-Stack (normales OSC-RGB) mit einem LP-Stack
(Dual-Band-Filter Ha/OIII) zu einem HaRGB-Bild.

Ablauf
------
1. Beide Stacks laden (FITS, RGB oder mono), nach float32 [0..1] wandeln.
2. Den LP-Stack per Sternregistrierung (Siril `register`) exakt auf den
   IRCUT-Stack ausrichten (IRCUT = Referenz, dessen Bildausschnitt bleibt).
3. H-Alpha = Rot-Kanal des LP-Stacks, OIII = Mittel aus Gruen+Blau.
4. Optional: Gradientenentfernung (Siril `subsky`) auf Ha und RGB.
5. Ha wird auf den IRCUT-Rotkanal skaliert:
     - excess / max:  Kontinuum-Abgleich ueber die Sterne (Sterne in Ha und R
                      gleich hell -> Linienemission bleibt als Ueberschuss)
     - blend:         globale lineare Anpassung (wie Siril linear_match)
6. Ha wird in den Rotkanal eingerechnet (Modus waehlbar, s.u.).
7. Ergebnis wird als 32-bit FITS gespeichert (linear, noch nicht gestretcht)
   und in Siril geladen.

Kombinations-Modi
-----------------
  excess  (Standard)  R' = R + k * max(Ha_c - R, 0)
          Nur die Ha-Linienemission UEBER dem Rot-Kontinuum wird addiert.
          Sterne und Hintergrund bleiben weitgehend unveraendert, die
          Nebelregionen werden verstaerkt.  k ~ 0.2 .. 1.5 (Standard 0.5)
  blend   R' = (1-k) * R + k * Ha_lin      (k = 0..1, k=1 -> HaGB, Std. 0.6)
          Bringt v.a. das bessere SNR / die Details des Ha in den Rotkanal.
  max     R' = R + k * (max(R, Ha_c) - R)  ("Aufhellen", k = 0..1, Std. 1.0)

Aufruf
------
  * In Siril: Skripte-Menue -> Script auswaehlen  -> Dialog oeffnet sich.
  * Siril-Konsole / .ssf:
        pyscript "C:/pfad/zu/Seestar_HaRGB-Combine.py" -ircut "C:/pfad/ircut.fit" -lp "C:/pfad/lp.fit"
             [-out "C:/pfad/ergebnis.fit"] [-mode excess|blend|max] [-k 0.5]
             [-bge] [-keep]

Ausgabe (neben der Ausgabedatei):
  <name>.fit            HaRGB-Ergebnis (linear)
  <name>_Ha.fit         ausgerichteter, normierter Ha-Kanal (mono)
  <name>_OIII.fit       ausgerichteter OIII-Kanal (mono), z.B. fuer HOO
"""

import os
import sys
import shutil
import argparse
import traceback

import sirilpy as s

s.ensure_installed("numpy", "astropy")

import numpy as np                       # noqa: E402
from astropy.io import fits              # noqa: E402

VERSION = "1.0.0"
DEFAULT_K = {"excess": 0.5, "blend": 0.6, "max": 1.0}
MAX_K = {"excess": 2.0, "blend": 1.0, "max": 1.0}
SEQ = "hapair"            # Name der temporaeren Registrier-Sequenz
WORKDIR_NAME = "_ha_combine_work"


# ---------------------------------------------------------------------------
# Hilfsfunktionen: FITS I/O
# ---------------------------------------------------------------------------
def read_fits(path):
    """Liest ein FITS-Bild als float32-Array (C, H, W) in [0..1] + Header."""
    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None and h.data.ndim >= 2), None)
        if hdu is None:
            raise ValueError(f"Keine Bilddaten gefunden in: {path}")
        data = hdu.data
        header = hdu.header.copy()

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim == 3 and data.shape[0] not in (1, 3) and data.shape[-1] in (1, 3):
        data = np.moveaxis(data, -1, 0)          # (H, W, C) -> (C, H, W)
    if data.ndim != 3 or data.shape[0] not in (1, 3):
        raise ValueError(f"Unerwartete Bildform {data.shape} in: {path}")

    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / 65535.0
    else:
        data = data.astype(np.float32)
        mx = float(np.nanmax(data))
        if mx > 1.5:                              # Float-FITS im 16-bit-Bereich
            data /= 65535.0 if mx <= 65535.0 * 1.01 else mx
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0), header


def write_fits(path, data, header=None, history=None):
    """Schreibt float32-Daten (C,H,W oder H,W) als FITS."""
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
    if history:
        for line in history:
            hdr.add_history(line)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def to_rgb(img):
    return img if img.shape[0] == 3 else np.repeat(img, 3, axis=0)


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Mathematik
# ---------------------------------------------------------------------------
def _subsample(mask, max_n=2_000_000):
    idx = np.flatnonzero(mask)
    if idx.size > max_n:
        idx = idx[:: idx.size // max_n + 1]
    return idx


def linear_fit(src, ref, valid, low=1e-6, high=0.92):
    """Least-Squares-Fit  src*a + b ~ ref  (wie Siril linear_match),
    nur ueber gueltige, nicht gesaettigte Pixel."""
    m = valid & (ref > low) & (ref < high) & (src > low) & (src < high)
    idx = _subsample(m)
    if idx.size < 1000:
        raise RuntimeError("Zu wenige gemeinsame Pixel fuer die lineare Anpassung "
                           "- passen die beiden Stacks zusammen?")
    x = src.ravel()[idx].astype(np.float64)
    y = ref.ravel()[idx].astype(np.float64)
    a, b = np.polyfit(x, y, 1)
    if not np.isfinite(a) or a <= 0:
        # Fallback: Median/MAD-Abgleich
        mx, my = np.median(x), np.median(y)
        sx = np.median(np.abs(x - mx)) or 1e-6
        sy = np.median(np.abs(y - my)) or 1e-6
        a = sy / sx
        b = my - a * mx
    return float(a), float(b)


def box_mean(a, r):
    """Schneller Box-Filter (Mittelwert, Radius r) ueber Integralbild."""
    k = 2 * r + 1
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    s_ = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s_ / (k * k)).astype(np.float32)


def continuum_scale(ha, r, valid, sat=0.9):
    """Skalierung Ha -> R ueber die Sterne (Kontinuum).

    Sterne sind (nahezu) reines Kontinuum: nach dieser Skalierung sind sie in
    Ha und R gleich hell, waehrend die Ha-Linienemission im Nebel deutlich
    ueber R liegt - genau dieser Ueberschuss wird spaeter addiert.
    Gibt (scale, offset, n_pixel) zurueck oder None, falls zu wenige Sterne.
    """
    hp_r = box_mean(r, 1) - box_mean(r, 12)       # kleine Strukturen = Sterne
    hp_h = box_mean(ha, 1) - box_mean(ha, 12)
    base = valid & (r < sat) & (ha < sat)
    if base.sum() < 10000:
        return None
    thr_r = np.percentile(hp_r[base][::7], 99.5)
    thr_h = np.percentile(hp_h[base][::7], 99.5)
    m = base & (hp_r > max(thr_r, 1e-5)) & (hp_h > max(thr_h, 1e-5))
    n = int(m.sum())
    if n < 50:
        return None
    scale = float(np.median(hp_r[m] / hp_h[m]))
    if not np.isfinite(scale) or scale <= 0:
        return None
    offset = float(np.median(r[valid]) - scale * np.median(ha[valid]))
    return scale, offset, n


def combine(rgb, ha, mode, k, valid):
    r = rgb[0]
    out = rgb.copy()
    if mode == "excess":
        diff = ha - r
        # Rauschen im Hintergrund nicht als "Ha-Ueberschuss" werten:
        med = float(np.median(diff[valid])) if valid.any() else 0.0
        excess = np.clip(diff - med, 0.0, None)
        out[0] = r + k * excess
    elif mode == "blend":
        k = min(max(k, 0.0), 1.0)
        out[0] = (1.0 - k) * r + k * ha
    elif mode == "max":
        k = min(max(k, 0.0), 1.0)
        out[0] = r + k * (np.maximum(r, ha) - r)
    else:
        raise ValueError(f"Unbekannter Modus: {mode}")
    out[0] = np.where(valid, out[0], r)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Siril-Interaktion
# ---------------------------------------------------------------------------
def q(path):
    """Pfad fuer Siril-Befehle quoten (Leerzeichen!)."""
    return '"' + path.replace("\\", "/") + '"'


class Combiner:
    def __init__(self, siril):
        self.siril = siril

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

    def subsky(self, workdir, name):
        """Gradientenentfernung mit Siril (Polynom Grad 1) auf workdir/name."""
        self.siril.cmd("load", name)
        self.siril.cmd("subsky", "1", "-samples=20", "-tolerance=1.0")
        self.siril.cmd("save", name)

    def run(self, ircut_path, lp_path, out_path, mode="excess", k=0.5,
            bge=False, keep=False):
        ircut_path = os.path.abspath(ircut_path)
        lp_path = os.path.abspath(lp_path)
        out_path = os.path.abspath(out_path)
        if not out_path.lower().endswith((".fit", ".fits", ".fts")):
            out_path += ".fit"
        out_dir = os.path.dirname(out_path)
        base = os.path.splitext(out_path)[0]
        os.makedirs(out_dir, exist_ok=True)

        old_wd = None
        try:
            old_wd = self.siril.get_siril_wd()
        except Exception:
            pass

        workdir = os.path.join(out_dir, WORKDIR_NAME)
        if os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir)

        try:
            # --- 1. Laden ------------------------------------------------------
            self.progress("Lade Stacks ...", 0.05)
            self.log(f"IRCUT-Stack: {ircut_path}")
            self.log(f"LP-Stack:    {lp_path}")
            ircut, ircut_hdr = read_fits(ircut_path)
            lp, _ = read_fits(lp_path)
            if ircut.shape[0] != 3:
                self.log("Warnung: IRCUT-Stack ist mono - Ergebnis wird RGB "
                         "mit identischen Kanaelen.", s.LogColor.SALMON)
            # Fuer die Registrierung muessen beide Bilder 3 Kanaele haben
            write_fits(os.path.join(workdir, "a_ircut.fit"), to_rgb(ircut), ircut_hdr)
            write_fits(os.path.join(workdir, "b_lp.fit"), to_rgb(lp))

            # --- 2. Registrierung -------------------------------------------
            self.progress("Registriere LP auf IRCUT ...", 0.15)
            self.log("Registriere LP-Stack auf IRCUT-Stack (Sterne, Rotkanal) ...",
                     s.LogColor.BLUE)
            self.siril.cmd("cd", q(workdir))
            self.siril.cmd("convert", SEQ)
            self.siril.cmd("setref", SEQ, "1")
            self.siril.cmd("register", SEQ, "-layer=0", "-transf=homography",
                           "-interp=lanczos4", "-maxstars=2000", "-nostarlist")

            r1 = os.path.join(workdir, f"r_{SEQ}_00001.fit")
            r2 = os.path.join(workdir, f"r_{SEQ}_00002.fit")
            if not (os.path.isfile(r1) and os.path.isfile(r2)):
                raise RuntimeError("Registrierung fehlgeschlagen: zu wenige "
                                   "gemeinsame Sterne gefunden? (siehe Siril-Log)")

            reg1, _ = read_fits(r1)
            reg2, _ = read_fits(r2)
            # Sicherstellen, dass r_..._00001 wirklich die IRCUT-Referenz ist
            ref_rgb = to_rgb(ircut)
            if reg1.shape == ref_rgb.shape and reg2.shape == ref_rgb.shape:
                d1 = float(np.mean(np.abs(reg1[:, ::7, ::7] - ref_rgb[:, ::7, ::7])))
                d2 = float(np.mean(np.abs(reg2[:, ::7, ::7] - ref_rgb[:, ::7, ::7])))
                if d2 < d1:
                    reg1, reg2 = reg2, reg1
            rgb, lp_reg = reg1, reg2

            # --- 3. Kanaele extrahieren -----------------------------------------
            ha = lp_reg[0].copy()
            oiii = 0.5 * (lp_reg[1] + lp_reg[2])
            # Pixel ausserhalb des LP-Bildfelds (Registrierungsraender) ausschliessen
            valid = np.all(lp_reg > 0.0, axis=0) & np.all(rgb > 0.0, axis=0)
            cover = 100.0 * valid.mean()
            self.log(f"Ueberlappung IRCUT/LP: {cover:.1f} %")
            if cover < 30:
                self.log("Warnung: geringe Ueberlappung der beiden Stacks!",
                         s.LogColor.SALMON)

            # --- 4. Optionale Gradientenentfernung ------------------------------
            if bge:
                self.progress("Gradientenentfernung ...", 0.55)
                self.log("Gradientenentfernung (subsky, Grad 1) auf Ha und RGB ...")
                write_fits(os.path.join(workdir, "bge_ha.fit"), ha)
                write_fits(os.path.join(workdir, "bge_oiii.fit"), oiii)
                write_fits(os.path.join(workdir, "bge_rgb.fit"), rgb, ircut_hdr)
                for n in ("bge_ha", "bge_oiii", "bge_rgb"):
                    self.subsky(workdir, n)
                ha = read_fits(os.path.join(workdir, "bge_ha.fit"))[0][0]
                oiii = read_fits(os.path.join(workdir, "bge_oiii.fit"))[0][0]
                rgb = read_fits(os.path.join(workdir, "bge_rgb.fit"))[0]
                ha = np.where(valid, ha, 0.0)
                oiii = np.where(valid, oiii, 0.0)

            # --- 5. Ha auf Rotkanal normieren ---------------------------------
            self.progress("Normiere Ha auf IRCUT-Rot ...", 0.75)
            # a) globale lineare Anpassung (fuer 'blend' und die Ha-Ausgabedatei)
            a, b = linear_fit(ha, rgb[0], valid)
            ha_n = np.where(valid, np.clip(a * ha + b, 0.0, 1.0), 0.0)
            self.log(f"Lineare Anpassung: Ha' = {a:.4f} * Ha + {b:+.5f}")
            # b) Kontinuum-(Sterne-)Abgleich fuer 'excess' und 'max'
            ha_comb, norm_info = ha_n, f"linfit a={a:.5f} b={b:+.6f}"
            if mode in ("excess", "max"):
                cs = continuum_scale(ha, rgb[0], valid)
                if cs is None:
                    self.log("Zu wenige Sterne fuer den Kontinuum-Abgleich - "
                             "verwende lineare Anpassung.", s.LogColor.SALMON)
                else:
                    sc, off, n = cs
                    ha_comb = np.where(valid, np.clip(sc * ha + off, 0.0, 1.0), 0.0)
                    norm_info = f"continuum scale={sc:.5f} off={off:+.6f}"
                    self.log(f"Kontinuum-Abgleich ueber {n} Sternpixel: "
                             f"Ha_c = {sc:.4f} * Ha + {off:+.5f}")
            ao, bo = linear_fit(oiii, 0.5 * (rgb[1] + rgb[2]), valid)
            oiii_n = np.where(valid, np.clip(ao * oiii + bo, 0.0, 1.0), 0.0)

            # --- 6. Kombination -------------------------------------------------
            self.progress("Kombiniere ...", 0.85)
            result = combine(rgb, ha_comb, mode, k, valid)

            # --- 7. Speichern ---------------------------------------------------
            hist = [f"HaRGB combine v{VERSION}: mode={mode} k={k} bge={bge}",
                    f"Ha norm: {norm_info}",
                    f"IRCUT: {os.path.basename(ircut_path)}",
                    f"LP: {os.path.basename(lp_path)}"]
            write_fits(out_path, result, ircut_hdr, hist)
            write_fits(base + "_Ha.fit", ha_n, ircut_hdr,
                       [f"Ha aus LP-Stack {os.path.basename(lp_path)}"])
            write_fits(base + "_OIII.fit", oiii_n, ircut_hdr,
                       [f"OIII aus LP-Stack {os.path.basename(lp_path)}"])
            self.log(f"Gespeichert: {out_path}", s.LogColor.GREEN)
            self.log(f"Gespeichert: {base}_Ha.fit", s.LogColor.GREEN)
            self.log(f"Gespeichert: {base}_OIII.fit", s.LogColor.GREEN)

            self.siril.cmd("cd", q(out_dir))
            self.siril.cmd("load", q(out_path))
            self.progress("Fertig", 1.0)
            self.log("Fertig. Das Ergebnis ist linear - jetzt z.B. SPCC/"
                     "Farbkalibrierung und Stretch anwenden.", s.LogColor.GREEN)
            return out_path
        finally:
            # Siril aus dem Arbeitsordner herausholen, bevor er geloescht wird
            try:
                if self.siril.get_siril_wd() and \
                        os.path.abspath(self.siril.get_siril_wd()) == os.path.abspath(workdir):
                    self.siril.cmd("cd", q(old_wd if old_wd and os.path.isdir(old_wd)
                                           else out_dir))
            except Exception:
                pass
            if not keep:
                shutil.rmtree(workdir, ignore_errors=True)
            try:
                self.siril.reset_progress()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GUI (tkinter)
# ---------------------------------------------------------------------------
def run_gui(combiner, initial_dir):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title(f"H-Alpha Extract & Combine  v{VERSION}  (Seestar S30 Pro)")
    root.resizable(False, False)
    try:
        from sirilpy import tksiril
        tksiril.elevate(root)
    except Exception:
        pass

    v_ircut = tk.StringVar()
    v_lp = tk.StringVar()
    v_out = tk.StringVar()
    v_mode = tk.StringVar(value="excess")
    v_k = tk.DoubleVar(value=DEFAULT_K["excess"])
    v_bge = tk.BooleanVar(value=False)
    v_keep = tk.BooleanVar(value=False)

    ftypes = [("FITS", "*.fit *.fits *.fts"), ("Alle Dateien", "*.*")]

    def pick(var, title):
        p = filedialog.askopenfilename(title=title, initialdir=initial_dir,
                                       filetypes=ftypes, parent=root)
        if p:
            var.set(p)
            if not v_out.get() and v_ircut.get():
                d = os.path.dirname(v_ircut.get())
                v_out.set(os.path.join(d, "HaRGB_combined.fit").replace("\\", "/"))

    def pick_out():
        p = filedialog.asksaveasfilename(title="Ausgabedatei", parent=root,
                                         initialdir=initial_dir,
                                         defaultextension=".fit", filetypes=ftypes,
                                         initialfile="HaRGB_combined.fit")
        if p:
            v_out.set(p)

    frm = ttk.Frame(root, padding=12)
    frm.grid(sticky="nsew")

    rows = [("IRCUT-Stack (RGB):", v_ircut, lambda: pick(v_ircut, "IRCUT-Stack waehlen")),
            ("LP-Stack (Ha/OIII):", v_lp, lambda: pick(v_lp, "LP-Stack waehlen")),
            ("Ausgabe:", v_out, pick_out)]
    for i, (lbl, var, cmd) in enumerate(rows):
        ttk.Label(frm, text=lbl).grid(row=i, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=var, width=60).grid(row=i, column=1, padx=5)
        ttk.Button(frm, text="...", width=4, command=cmd).grid(row=i, column=2)

    ttk.Label(frm, text="Modus:").grid(row=3, column=0, sticky="w", pady=(10, 3))
    mode_box = ttk.Combobox(frm, textvariable=v_mode, state="readonly", width=12,
                            values=["excess", "blend", "max"])
    mode_box.grid(row=3, column=1, sticky="w", padx=5, pady=(10, 3))

    help_txt = {
        "excess": "Ha-Linienemission zu Rot addieren (empfohlen). Staerke 0.2 - 1.5",
        "blend": "Rot mit Ha mischen (SNR/Details). 0 = nur RGB, 1 = HaGB",
        "max": "Aufhellen: max(R, Ha). 0 - 1",
    }
    lbl_help = ttk.Label(frm, text=help_txt["excess"], foreground="gray")
    lbl_help.grid(row=4, column=1, columnspan=2, sticky="w", padx=5)

    ttk.Label(frm, text="Staerke k:").grid(row=5, column=0, sticky="w", pady=3)
    kfrm = ttk.Frame(frm)
    kfrm.grid(row=5, column=1, columnspan=2, sticky="w", padx=5)
    scale = ttk.Scale(kfrm, from_=0.0, to=MAX_K["excess"], variable=v_k, length=300)
    scale.pack(side="left")
    lbl_k = ttk.Label(kfrm, text=f"{v_k.get():.2f}", width=6)
    lbl_k.pack(side="left", padx=6)
    v_k.trace_add("write", lambda *_: lbl_k.config(text=f"{v_k.get():.2f}"))

    def on_mode(*_):
        m = v_mode.get()
        lbl_help.config(text=help_txt[m])
        scale.config(to=MAX_K[m])
        v_k.set(DEFAULT_K[m])
    mode_box.bind("<<ComboboxSelected>>", on_mode)

    ttk.Checkbutton(frm, text="Gradientenentfernung (subsky Grad 1) vor dem Kombinieren",
                    variable=v_bge).grid(row=6, column=1, columnspan=2, sticky="w", padx=5)
    ttk.Checkbutton(frm, text="Arbeitsordner behalten (Debug)",
                    variable=v_keep).grid(row=7, column=1, columnspan=2, sticky="w", padx=5)

    def go():
        if not (os.path.isfile(v_ircut.get()) and os.path.isfile(v_lp.get())):
            messagebox.showerror("Fehler", "Bitte IRCUT- und LP-Stack waehlen.", parent=root)
            return
        if not v_out.get():
            messagebox.showerror("Fehler", "Bitte Ausgabedatei angeben.", parent=root)
            return
        btn.config(state="disabled")
        root.update()
        try:
            combiner.run(v_ircut.get(), v_lp.get(), v_out.get(), v_mode.get(),
                         round(v_k.get(), 3), v_bge.get(), v_keep.get())
            messagebox.showinfo("Fertig", "HaRGB-Kombination abgeschlossen.\n"
                                "Das Ergebnis ist in Siril geladen.", parent=root)
            root.destroy()
        except Exception as e:
            combiner.log(traceback.format_exc(), s.LogColor.RED)
            messagebox.showerror("Fehler", str(e), parent=root)
            btn.config(state="normal")

    bfrm = ttk.Frame(frm)
    bfrm.grid(row=8, column=0, columnspan=3, pady=(12, 0), sticky="e")
    ttk.Button(bfrm, text="Abbrechen", command=root.destroy).pack(side="right", padx=4)
    btn = ttk.Button(bfrm, text="Kombinieren", command=go)
    btn.pack(side="right")

    root.mainloop()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ha aus LP-Stack extrahieren und "
                                                 "mit IRCUT-Stack kombinieren")
    parser.add_argument("-ircut", help="IRCUT-Stack (RGB FITS)")
    parser.add_argument("-lp", help="LP-Stack (Dual-Band FITS)")
    parser.add_argument("-out", help="Ausgabedatei (Standard: HaRGB_combined.fit "
                                     "neben dem IRCUT-Stack)")
    parser.add_argument("-mode", choices=["excess", "blend", "max"], default="excess")
    parser.add_argument("-k", type=float, default=None,
                        help="Staerke (Standard: excess 0.5, blend 0.6, max 1.0)")
    parser.add_argument("-bge", action="store_true", help="Gradientenentfernung")
    parser.add_argument("-keep", action="store_true", help="Arbeitsordner behalten")
    args = parser.parse_args()

    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception as e:
        print(f"Keine Verbindung zu Siril: {e}")
        sys.exit(1)

    combiner = Combiner(siril)
    combiner.log(f"H-Alpha Extract & Combine v{VERSION}", s.LogColor.BLUE)
    try:
        if args.ircut and args.lp:
            out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ircut)),
                                           "HaRGB_combined.fit")
            k = args.k if args.k is not None else DEFAULT_K[args.mode]
            combiner.run(args.ircut, args.lp, out, args.mode, k,
                         args.bge, args.keep)
        else:
            if siril.is_cli():
                combiner.log("Im CLI-Modus bitte -ircut und -lp angeben.",
                             s.LogColor.RED)
                sys.exit(1)
            try:
                wd = siril.get_siril_wd()
            except Exception:
                wd = os.path.expanduser("~")
            run_gui(combiner, wd)
    except Exception as e:
        combiner.log(f"Fehler: {e}", s.LogColor.RED)
        combiner.log(traceback.format_exc(), s.LogColor.RED)
        sys.exit(1)
    finally:
        try:
            siril.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
