----------------------------------------------------
ENGLISH VERSION
----------------------------------------------------

PROJECT CONTENTS
----------------------------------------------------
- `nk_virus_model.py`: a script implementing an agent-based model of epithelial tissue on a 2D hexagonal grid, featuring moving NK (Natural Killer) cells. The system models viral infection of epithelial cells, viral spread to neighboring cells, infection progression, and cell death. System dynamics are simulated using the Gillespie algorithm for a continuous-time Markov process. To optimize node selection, possible event types are represented using Fenwick trees that are updated locally after each event. 
- `run_experiments.py`: a script that executes two paired simulations using identical initial parameters loaded from `params.json`. One simulation runs with NK cells present, while the other runs without them. A summary plot is then generated to compare the course of the infection in both scenarios. The comparison is based on cell population counts, the distribution of infection states, the number of dead cells over time, and the number of active infection states over time. 
- `params.json`: a configuration file containing simulation parameters. 
- `README` file.


REQUIREMENTS
----------------------------------------------------
The project was developed for Python 3.12 (64-bit) but should be compatible with versions 3.9+. The following libraries are required for proper operation:
- numpy
- scipy
- matplotlib
- numba. 

EXECUTION
----------------------------------------------------
Simply run the `run_experiments.py` script from the directory containing `nk_virus_model.py` and `params.json`:

python run_experiments.py

By default, the script loads simulation parameters from the `params.json` file. The meanings of the individual simulation parameters are described in the model script. The first execution of the script will take longer than subsequent runs due to the need to compile functions marked with the `@njit(cache=True)` decorator.

RESULTS
----------------------------------------------------
After running `run_experiments.py`, the following files will appear in the working directory:

- results_with_nk.npz: raw simulation results for the system with NK cells
- results_without_nk.npz: raw simulation results for the system without NK cells
- simulation_summary.jpg: a summary plot comparing both simulations.


----------------------------------------------------
POLISH VERSION
----------------------------------------------------

ZAWARTOŚĆ PROJEKTU
----------------------------------------------------
	- `nk_virus_model.py`: skrypt zawierający implementację modelu agentowego tkanki nabłonkowej na dwuwymiarowej siatce heksagonalnej wraz z poruszającymi się na niej komórkami NK (Natural Killers). W układzie komórek nabłonkowych dochodzi do zarażania komórek wirusem, rozprzestrzeniania się wirusa na sąsiadujące komórki, progresji infekcji oraz śmierci komórek. Dynamika układu jest symulowana algorytmem Gillespiego dla ciągłego w czasie procesu Markowa. W celu optymalizacji do efektywnego losowania węzła wykorzystano reprezentację możliwych typów zdarzeń w postaci drzew Fenwicka aktualizowanych lokalnie po każdym wydarzeniu.
	-`run_experiments.py`: skrypt uruchamiający dwie sparowane symulacje na podstawie tych samych parametrów początkowych pobieranych z pliku `params.json`. Jedna symulacja przeprowadzana jest z obecnością komórek NK, druga - bez nich. Następnie generowany jest zbiorczy wykres porównujący przebieg infekcji w obu przypadkach. Porównanie przedstawione jest na podstawie liczebności populacji komórek, rozkładu stanów infekcji, liczby martwych komórek w czasie oraz liczby aktywnych stanów infekcji w czasie.
	- `params.json`: plik konfiguracyjny z parametrami symulacji.
	- plik 'README'.


WYMAGANIA
----------------------------------------------------
Projekt pisany był pod wersję Pythona 3.12 (64 bit), powinien jednak być kompatybilny z wersjami 3.9+. W celu poprawnego działania wymagane są następujące biblioteki:
	- numpy
	- scipy
	- matplotlib
	- numba.


URUCHOMIENIE
----------------------------------------------------
Wystarczy uruchomić skrypt `run_experiments.py` z katalogu, w którym znajduje się również skrypt `nk_virus_model.py` oraz plik `params.json`:

    python run_experiments.py

Domyślnie skrypt wczytuje parametry symulacji z pliku `params.json`. Znaczenie poszczególnych parametrów symulacji opisane jest w skrypcie modelu. Pierwsze uruchomienie skryptu powinno potrwać dłużej niż kolejne z uwagi na konieczność kompilacji funkcji oznaczonych dekoratorem @njit(cache=True).

WYNIKI
----------------------------------------------------
Po uruchomieniu `run_experiments.py` w katalogu roboczym pojawią się następujące pliki:

	- results_with_nk.npz: surowe wyniki symulacji układu z komórkami NK
	- results_without_nk.npz: surowe wyniki symulacji układu bez komórek NK
	- simulation_summary.jpg: zbiorczy wykres porównujący obie symulacje.