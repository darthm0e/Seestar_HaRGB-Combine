# Seestar HaRGB-Combine (Siril script)

*English · [Deutsch](#deutsch)*

Siril Python script (Siril ≥ 1.4) that combines the **IRCUT stacks** (regular RGB) and the
**LP stacks** (dual-band Ha/OIII) of a **Seestar S30 Pro** into an **HaRGB or HaOIII-RGB image**
– with live preview.

## Features

- **Automatic detection** of the stacks in a folder (`_IRCUT_` / `_LP_` in the Seestar file name, otherwise the FILTER header)
- **Several stacks per filter** (e.g. several nights): registered, normalized and averaged with noise weighting
- **Registration** by star matching (Siril `register`) with fallback strategies (other channel, similarity, shift)
- **Continuum matching on the stars** → true Ha/OIII line emission instead of a simple brightness match
- **PSF matching**: stars of different sharpness (e.g. different nights) cause no rings or colour fringes
- **Star protection**: star areas are filled in from their surroundings – no red star cores
- **Noise threshold + smoothing** of the emission → no red noise veil in the background
- **OIII** (from green/blue of the LP stack) into green/blue, including correction of the Ha crosstalk of the Bayer filters
- **Ha luminance**: transfers detail/SNR of the Ha to the brightness without shifting colours
- **Auto-k** for blend mode (noise-optimal weighting)
- Optional: **crop** to the common area, **background extraction** (`subsky`)
- **GUI with preview** (overview + 1:1 crop, before/after)
- **English and German**: follows the Siril/system language, switchable in the dialog or with `-lang`

## Modes

| Mode | Formula | Effect |
|------|---------|--------|
| `excess` (default) | `R' = R + k_Ha · E_Ha`, `G',B' += k_OIII · E_OIII` | Only line emission above the continuum is added – stars and background stay unchanged |
| `blend` | `R' = (1−k) · R + k · Ha_lin` | Blends in the lower-noise narrowband, `k = auto` → noise-optimal |

In addition in both modes: `k_Lum` (Ha luminance, 0–1).

## Output

| File | Content |
|------|---------|
| `HaRGB_combined.fit` | result (linear, 32-bit) |
| `HaRGB_combined_Ha.fit` | registered Ha channel, linearly fitted to R |
| `HaRGB_combined_OIII.fit` | registered OIII channel (e.g. for HOO) |
| `HaRGB_combined_HaExcess.fit` | pure Ha line emission (continuum-subtracted, star-free) |

## Installation

1. Clone the repository or download `Seestar_HaRGB-Combine.py`.
2. In Siril: *Preferences → Scripts* → add the folder containing the script, restart Siril if needed.
3. Requires `numpy` and `astropy` (installed automatically if missing).

## Usage

**GUI:** *Scripts → Seestar_HaRGB-Combine* → *Scan folder* (or add stacks with `+`)
→ *Preview* → adjust the sliders (the preview updates live, clicking into the overview selects the
1:1 crop) → *Save*.

> Tip: do not mix per-session stacks with the combined stack made from them – the data would be counted twice.

**Console / `.ssf`:**

```
pyscript "C:/path/to/Seestar_HaRGB-Combine.py" -dir "C:/data/NGC7000" -crop
pyscript "C:/path/to/Seestar_HaRGB-Combine.py" -ircut "a.fit" -lp "b.fit" "c.fit" -mode blend -k auto
```

| Option | Meaning | Default |
|--------|---------|---------|
| `-dir` | scan folder (detect IRCUT/LP automatically) | – |
| `-ircut` / `-lp` | specify stack(s) explicitly | – |
| `-out` | output file | `HaRGB_combined.fit` |
| `-mode` | `excess` or `blend` | `excess` |
| `-k` | Ha strength or `auto` | 0.5 |
| `-koiii` | OIII strength or `auto` | 0 (off) |
| `-klum` | Ha luminance 0–1 | 0 (off) |
| `-noise` | noise threshold in σ (0 = off) | 2 |
| `-smooth` | smoothing of the emission in px | 2 |
| `-nopsf` | no PSF matching | – |
| `-nostarprot` | no star protection | – |
| `-bge` | background extraction | – |
| `-crop` | crop to common area | – |
| `-keep` | keep the working folder `_ha_combine_work` | – |
| `-lang` | language `en` or `de` | Siril/system language |

The result is **linear** – continue as usual with colour calibration (e.g. SPCC) and stretching.

### Language

Order: `-lang` option → choice saved in the dialog → Siril's language setting → system language → English.
The dialog choice is stored in `%APPDATA%\Seestar_HaRGB-Combine.json` (Linux/macOS: `~/.config/`).

---

<a name="deutsch"></a>
## Deutsch

Siril-Python-Script (Siril ≥ 1.4), das die **IRCUT-Stacks** (normales RGB) und die **LP-Stacks**
(Dual-Band Ha/OIII) eines **Seestar S30 Pro** zu einem **HaRGB- bzw. HaOIII-RGB-Bild** kombiniert –
mit Live-Vorschau. Die Oberfläche ist auf Deutsch, wenn Siril bzw. das System auf Deutsch eingestellt ist
(umschaltbar im Dialog oder mit `-lang de`).

**Kurzanleitung:**

1. In Siril unter *Einstellungen → Skripte* den Ordner mit dem Script hinzufügen.
2. *Skripte → Seestar_HaRGB-Combine* → *Ordner scannen* → *Vorschau* → Regler einstellen → *Speichern*.
3. Ergebnis ist linear – danach Farbkalibrierung (z.B. SPCC) und Stretch.

**Modi:** `excess` (Standard) addiert nur die Ha-/OIII-Linienemission über dem Kontinuum – Sterne und
Hintergrund bleiben unverändert. `blend` mischt das rauschärmere Schmalband ein (`-k auto` = rauschoptimal).
`Ha-Luminanz` überträgt zusätzlich die Details des Ha auf die Helligkeit.

Alle Funktionen und Optionen sind oben im englischen Teil beschrieben; die Kommandozeilen-Optionen sind identisch.

> Tipp: Einzel-Session-Stacks und den daraus erzeugten Gesamt-Stack nicht mischen – sonst zählen die Daten doppelt.
