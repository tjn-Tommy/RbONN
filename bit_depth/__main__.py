"""Entry point: ``python -m bit_depth`` runs the full Monte-Carlo sweep + figures."""
from .config import Cfg
from .montecarlo import simulation_grid

if __name__ == '__main__':
    c = Cfg()
    simulation_grid(c, n_trials=300)
