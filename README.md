# Astro-TSP

Projekt porownuje trzy metody rozwiazywania TSP dla stalej macierzy kosztow zbudowanej na bazie rzeczywistych wspolrzednych cial niebieskich:

- `branch_and_bound`
- `ilp`
- `aco`

Koszt krawedzi jest stala odlegloscia euklidesowa 3D liczona dla jednej epoki (brak modelu czasozaleznego i brak kosztu delta-v).
Pipeline dziala statystycznie: dla kazdego `n` generuje wiele instancji i powtorzen ACO, a kazde wywolanie solvera ma timeout.

## Struktura

- `src/astrotsp/data` - pobieranie i cache danych JPL Horizons
- `src/astrotsp/models` - kontrakty danych i model kosztu
- `src/astrotsp/solvers` - implementacje solverow
- `src/astrotsp/experiments` - runner benchmarkow
- `src/astrotsp/reporting` - CSV, wykresy i podsumowanie
- `config/benchmark.json` - scenariusze testowe
- `results/` - artefakty uruchomien
- `tests/` - testy jednostkowe i integracyjne

## Quick start

1. Utworz i aktywuj srodowisko:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
2. Zainstaluj projekt:
  - `pip install -e .[dev]`
3. Uruchom benchmark:
  - `run-benchmark --config config/benchmark.json`

4. Wygeneruj wykresy ponownie na podstawie CSV (bez ponownego liczenia solverow):
  - `generate-plots --mode generate-plots --raw-csv results/benchmark_raw.csv --summary-csv results/benchmark_summary.csv --output-dir results`

5. Uruchom benchmark na dokladnie tych samych instancjach z katalogu (bez losowania nowych):
  - `python -m astrotsp.cli --mode run-benchmark-from-catalog --config config/benchmark.json --catalog-csv results/instances_catalog.csv`

Po uruchomieniu wyniki znajdziesz w `results/`:

- `benchmark_raw.csv`
- `benchmark_summary.csv`
- `instances_catalog.csv`
- `gap_stability_vs_nodes.png`
- `time_vs_nodes_mean.png`
- `time_vs_nodes_max.png`
- `memory_vs_nodes_mean.png`
- `memory_vs_nodes_max.png`
- `exact_validation_delta_vs_nodes.png`
- `<instance_id>/<solver>_best_3d.png`
- `<instance_id>/<solver>_best_2d.png`
- `summary.txt`

`benchmark_raw.csv` zawiera m.in.:
- `n_nodes`, `instance_id`, `repetition_id`
- `solver`, `status`, `total_cost`, `elapsed_seconds`, `memory_usage_mb`
- `gap_pct`, `exact_match`, `consistency_error`
- `selected_body_ids` (ID obiektow z Horizons dla danej instancji)

`instances_catalog.csv` zawiera mapowanie instancji potrzebne do odtworzenia danych:
- `instance_id`, `n_nodes`, `epoch`, `selected_body_ids`

## Reproducibility

- Dla `aco` ustawione sa ziarna losowe (`seeds`) w konfiguracji.
- Wszystkie metody pracuja na tej samej stalej macierzy kosztow.
- Konfiguracja eksperymentu jest jawnie opisana w `config/benchmark.json`.

## Testy

Uruchom:

- `pytest`

