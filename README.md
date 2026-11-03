# Seestar HaRGB-Combine (Siril-Script)

Siril-Python-Script (Siril ≥ 1.4), das bei einem **Seestar S30 Pro** die
**IRCUT-Stacks** (normales RGB) mit den **LP-Stacks** (Dual-Band Ha/OIII) zu einem
**HaRGB- bzw. HaOIII-RGB-Bild** kombiniert – mit Live-Vorschau.

## Funktionen

- **Automatische Erkennung** der Stacks in einem Ordner (`_IRCUT_` / `_LP_` im Seestar-Dateinamen, sonst FILTER-Header)
- **Mehrere Stacks pro Filter** (z.B. mehrere Nächte): werden registriert, normiert und rauschgewichtet gemittelt
- **Registrierung** per Sternabgleich (Siril `register`), mit Rückfall-Strategien (anderer Kanal, Similarity, Shift)
- **Kontinuum-Abgleich über die Sterne** → echte Ha-/OIII-Linienemission statt nur Helligkeitsangleich
- **PSF-Angleich**: unterschiedlich scharfe Sterne (z.B. verschiedene Nächte) erzeugen keine Ringe/Farbsäume
- **Sternschutz**: Sternbereiche werden aus der Umgebung aufgefüllt – keine roten Sternkerne
- **Rauschschwelle + Glättung** der Emission → kein roter Rausch-Schleier im Hintergrund
- **OIII** (aus Grün/Blau des LP-Stacks) in Grün/Blau, inkl. Korrektur des Ha-Übersprechens der Bayer-Filter
- **Ha-Luminanz**: überträgt Details/SNR des Ha auf die Helligkeit, ohne Farben zu verfälschen
- **Auto-k** für den Blend-Modus (rauschoptimale Gewichtung)
- Optional: **Zuschnitt** auf den gemeinsamen Bildbereich, **Gradientenentfernung** (`subsky`)
- **GUI mit Vorschau** (Übersicht + 1:1-Ausschnitt, Vorher/Nachher)

## Modi

| Modus | Formel | Wirkung |
|-------|--------|---------|
| `excess` (Standard) | `R' = R + k_Ha · E_Ha`, `G',B' += k_OIII · E_OIII` | Nur Linienemission über dem Kontinuum wird addiert – Sterne & Hintergrund bleiben unverändert |
| `blend` | `R' = (1−k) · R + k · Ha_lin` | Mischt das rauschärmere Schmalband ein, `k = auto` → rauschoptimal |

Zusätzlich in beiden Modi: `k_Lum` (Ha-Luminanz, 0–1).

## Ausgabe

| Datei | Inhalt |
|-------|--------|
| `HaRGB_combined.fit` | Ergebnis (linear, 32-bit) |
| `HaRGB_combined_Ha.fit` | ausgerichteter Ha-Kanal, linear an R angepasst |
| `HaRGB_combined_OIII.fit` | ausgerichteter OIII-Kanal (z.B. für HOO) |
| `HaRGB_combined_HaExcess.fit` | reine Ha-Linienemission (kontinuumsbereinigt, sternfrei) |

## Installation

1. Repository klonen oder `Seestar_HaRGB-Combine.py` herunterladen.
2. In Siril: *Einstellungen → Skripte* → Ordner mit dem Script hinzufügen, Siril ggf. neu starten.
3. Benötigt `numpy` und `astropy` (werden bei Bedarf automatisch installiert).

## Verwendung

**GUI:** *Skripte → Seestar_HaRGB-Combine* → *Ordner scannen* (oder Stacks per `+` hinzufügen)
→ *Vorschau* → Regler einstellen (Vorschau aktualisiert sich live, Klick in die Übersicht wählt den
1:1-Ausschnitt) → *Speichern*.

> Tipp: Nicht Einzel-Session-Stacks und den daraus erzeugten Gesamt-Stack mischen – sonst zählen die Daten doppelt.

**Konsole / `.ssf`:**

```
pyscript "C:/pfad/zu/Seestar_HaRGB-Combine.py" -dir "C:/daten/NGC7000" -crop
pyscript "C:/pfad/zu/Seestar_HaRGB-Combine.py" -ircut "a.fit" -lp "b.fit" "c.fit" -mode blend -k auto
```

| Option | Bedeutung | Standard |
|--------|-----------|----------|
| `-dir` | Ordner scannen (IRCUT/LP automatisch) | – |
| `-ircut` / `-lp` | Stack(s) explizit angeben | – |
| `-out` | Ausgabedatei | `HaRGB_combined.fit` |
| `-mode` | `excess` oder `blend` | `excess` |
| `-k` | Ha-Stärke oder `auto` | 0.5 |
| `-koiii` | OIII-Stärke oder `auto` | 0 (aus) |
| `-klum` | Ha-Luminanz 0–1 | 0 (aus) |
| `-noise` | Rauschschwelle in σ (0 = aus) | 2 |
| `-smooth` | Glättung der Emission in px | 2 |
| `-nopsf` | kein PSF-Angleich | – |
| `-nostarprot` | kein Sternschutz | – |
| `-bge` | Gradientenentfernung | – |
| `-crop` | auf gemeinsamen Bereich zuschneiden | – |
| `-keep` | Arbeitsordner `_ha_combine_work` behalten | – |

Das Ergebnis ist **linear** – danach wie gewohnt Farbkalibrierung (z.B. SPCC) und Stretch.
