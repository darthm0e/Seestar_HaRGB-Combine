# Seestar HaRGB-Combine (Siril-Script)

Siril-Python-Script (Siril ≥ 1.4), das bei einem **Seestar S30 Pro** den
**IRCUT-Stack** (normales RGB) mit dem **LP-Stack** (Dual-Band Ha/OIII) zu einem
**HaRGB-Bild** kombiniert.

## Funktionsweise

1. Beide Stacks laden (FITS, 16/32 bit, RGB oder mono).
2. LP-Stack per Sternregistrierung (Siril `register`) auf den IRCUT-Stack ausrichten.
3. **Ha** = Rotkanal des LP-Stacks, **OIII** = Mittel aus Grün + Blau.
4. Optional: Gradientenentfernung (`subsky`, Polynom Grad 1).
5. Ha in den Rotkanal einrechnen und Ergebnis linear als 32-bit FITS speichern.

### Modi

| Modus | Formel | Standard-k | Wirkung |
|-------|--------|-----------|---------|
| `excess` | `R' = R + k · max(Ha_c − R, 0)` | 0.5 | Nur Ha-Linienemission über dem Kontinuum wird addiert (Ha über Sterne skaliert) – Sterne & Hintergrund bleiben unverändert |
| `blend` | `R' = (1−k) · R + k · Ha_lin` | 0.6 | Mischt Ha in Rot (besseres SNR / Details), k=1 → HaGB |
| `max` | `R' = R + k · (max(R, Ha_c) − R)` | 1.0 | Aufhellen |

### Ausgabe

- `HaRGB_combined.fit` – HaRGB-Ergebnis (linear)
- `HaRGB_combined_Ha.fit` – ausgerichteter Ha-Kanal (mono)
- `HaRGB_combined_OIII.fit` – ausgerichteter OIII-Kanal (mono, z.B. für HOO)

## Installation

1. Repository klonen oder `Seestar_HaRGB-Combine.py` herunterladen.
2. In Siril: *Einstellungen → Skripte* → Ordner mit dem Script hinzufügen, Siril ggf. neu starten.
3. Benötigt `numpy` und `astropy` (werden bei Bedarf automatisch in Siril's Python-venv installiert).

## Verwendung

**GUI:** Menü *Skripte → Seestar_HaRGB-Combine* → Stacks wählen → *Kombinieren*.

**Konsole / `.ssf`:**

```
pyscript "C:/pfad/zu/Seestar_HaRGB-Combine.py" -ircut "C:/daten/IRCUT.fit" -lp "C:/daten/LP.fit" [-out "C:/daten/ergebnis.fit"] [-mode excess|blend|max] [-k 0.5] [-bge] [-keep]
```

| Option | Bedeutung |
|--------|-----------|
| `-ircut` | IRCUT-Stack (RGB) |
| `-lp` | LP-Stack (Dual-Band) |
| `-out` | Ausgabedatei (Standard: `HaRGB_combined.fit` neben dem IRCUT-Stack) |
| `-mode` | `excess` (Standard), `blend`, `max` |
| `-k` | Stärke |
| `-bge` | Gradientenentfernung vor dem Kombinieren |
| `-keep` | Temporären Arbeitsordner `_ha_combine_work` behalten |

Das Ergebnis ist **linear** – danach wie gewohnt Farbkalibrierung (z.B. SPCC) und Stretch.
