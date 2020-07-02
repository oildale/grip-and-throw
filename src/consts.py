# Nie należy używać poniższych zmiennych jako parametrów,
# które można swobodnie konfigurować.
# Niestety moduł problem.py zawiera wiele stałych
# wyznaczonych eksperymentalnie i zależnych
# od konkretnych wartości poniższych zmiennych.
# Zostały one wprowadzone, aby w miarę możliwości zwiększyć
# czytelność kodu.

WIDTH = 150  # szerokość (a zarazem wysokość) obrazka, na którym operują model
NUM_ROTS = 16  # liczba obrotów używanych między innnymi przez MLGripperModule
TRANS_WIDTH = 214  # szerokość (a zarazem wysokość) obrazka po wykonaniu obrotu
TRAY_OFFSET = 0.6  # odległość podstawy robota od środka tacy
TARGET_OFFSET = 1.6  # odległość podstawy robota od środka pojemnika z przegrodami
