# Confound-Beispiele für das CPM-Paper – Stand 26.10.2026

Übergabenotiz aus einem Claude-Chat. Ziel: öffentliche Datensätze finden, an denen sich
illustrieren lässt, dass CPM ohne Confound-Korrektur irreführende Ergebnisse liefert.

## Skripte (in `examples/`)

- `abide_confound_cpm.py` – ABIDE I (PCP, CC200, `fetch_abide_pcp`, quality_checked), n = 871.
  Target: Autismus vs. Kontrolle. Confounds: Geschlecht, Alter, mittlere FD, Site (One-Hot).
- `development_confound_cpm.py` – Development fMRI (Richardson 2018, `fetch_development_fmri`,
  nur Kinder), n = 122, MSDL-Atlas. Target: Alter. Confounds: mittlere FD, Geschlecht.

Beide rechnen zwei cccpm-Läufe:
1. naiv (`selection_input="raw"`, `model_input="raw"`, keine Covariates)
2. korrigiert (`selection_input="residualized"`, `model_input="residualized"`), liefert zusätzlich
   die Modelle `covariates` (nur Confounds), `full` und `increment`.

Einstellungen: RepeatedKFold 5×5, PThreshold 0.01, 100 Permutationen (fürs Paper 1000).
Daten liegen unter `~/nilearn_data/`.

## Ergebnisse (100 Permutationen, GPU)

| Datensatz | Connectome naiv | nur Confounds | Connectome korrigiert | Confounds + Connectome |
|---|---|---|---|---|
| ABIDE (ROC-AUC) | 0,685 (p=.01) | 0,550 (p=.06) | 0,676 (p=.01) | 0,680 (p=.01) |
| Development (r) | 0,528 (p=.01) | −0,015 (p=.55) | 0,543 (p=.01) | 0,535 (p=.01) |

Deskriptiv: ABIDE 87 % vs. 81 % männlich, FD 0,126 vs. 0,094. Development: Alter–FD r = −0,11.

**Interpretation:** Keines der beiden Beispiele zeigt den gewünschten Effekt (naiv gut, nur Confounds
ähnlich gut, korrigiert bricht ein). Die Confounds sagen das Target kaum vorher, und das
Connectome-Signal überlebt die Korrektur unverändert. Beide eignen sich als Gegenbeispiel:
„Confound-Korrektur kostet nichts, wenn kein Confounding vorliegt.“
Konsequenz für das Paper: These präzisieren. Problematisch wird es, wenn ein Confound sowohl mit
dem Target als auch mit dem Connectome stark zusammenhängt; das muss man prüfen, nicht voraussetzen.

Hinweise zur Einordnung: AUC-Zufallsniveau ist 0,5 (0,55 ≈ Cohen's d 0,18; 0,69 ≈ d 0,70).
Leicht negative CV-r-Werte unter H0 sind ein bekanntes CV-Artefakt.

## Nächste Schritte (Kandidaten für ein Positivbeispiel)

1. **ABIDE, Target Alter, Confound Site** – Sites haben sehr unterschiedliche Altersbereiche,
   Scanner-Effekte stecken im Connectome → starker Confound-Effekt erwartet. (Favorit)
2. Development mit `age_group="both"` (Kinder + Erwachsene, größerer Motion-Unterschied).
3. ABIDE mit `quality_checked=False` (mehr Bewegungsvarianz).
4. Notfalls semi-synthetisch: echte Connectomes, Target teilweise aus FD/Site konstruiert.
Ergänzend: mlconfound Partial-Confounder-Test (Spisak 2022). Achtung: GAM-Backend (pygam) ist mit
aktuellem scipy kaputt (`'csr_matrix' has no attribute 'A'`) → `cond_dist_method="linear"`.

## Gefundene cccpm-Bugs (v0.7.0)

1. **`preprocessing.to_device`**: `torch.tensor(array)` scheitert an numpy-Arrays mit negativen
   Strides (entstehen z.B. bei DataFrame-Spaltenauswahl). Fix:
   `torch.tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)`.
2. **3D-Input**: Doku sagt, `run()` akzeptiert (n, nodes, nodes); `generate_data_insights` läuft aber
   vor `check_data` und baut `pd.DataFrame(X)` → `Must pass 2-d input`. Fix: Insights nach
   `check_data` aufrufen.
3. **Pairplot in `generate_data_insights`** (`pairplot_flexible`): n×n Seaborn-Subplots aller
   Covariates, 300 dpi. Bei ~20 Site-Dummies > 45 min. Fix: ab z.B. 8 Covariates überspringen oder
   Dummies zusammenfassen. Workaround in den Skripten:
   `cccpm.reporting.data_insights.pairplot_flexible = lambda df, path: None`.
4. **MAGMA auf Turing-GPUs**: torch 2.14+cu130 – MAGMA hat keinen sm_75-Code, gebündeltes
   `torch.linalg.solve` (Permutationen in `LinearCPM._solve_ols`) → `no kernel image is available`.
   Fix: bei CUDA `torch.backends.cuda.preferred_linalg_library("cusolver")` setzen.
5. Nebenbei: Wenn CUDA nicht verfügbar ist, fällt cccpm still auf CPU zurück (nur Log-Warnung).

## Umgebung (NixOS, Laptop mit Quadro T2000)

`shell.nix` für pip-Wheels (numpy/torch) und CUDA:
```nix
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [ stdenv.cc.cc.lib zlib ])
                    + ":/run/opengl-driver/lib";
}
```
PyCharm erbt die Variable nur, wenn es aus der nix-shell gestartet wird (Alternative: `nix-ld`).
Treiber 595.84 (CUDA 13.2) ist ok.