"""Entry point: runs all experiments and saves all figures."""

import sys
import os
import time

# Ensure sim/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from experiments import (experiment_1_cp_assignment,
                         experiment_2_validator_placement,
                         experiment_3_nstar,
                         experiment_4_representation,
                         experiment_5_adaptive,
                         experiment_6_novel_methods,
                         experiment_7_composed_controller)


def main():
    print("=" * 60)
    print("Multi-Agent Coordination Optimization — Simulation Suite")
    print("=" * 60)
    print()

    t0 = time.time()

    print("[1/7] " + "-" * 50)
    df1 = experiment_1_cp_assignment(verbose=True)

    print("[2/7] " + "-" * 50)
    df2 = experiment_2_validator_placement(verbose=True)

    print("[3/7] " + "-" * 50)
    df3, nstar = experiment_3_nstar(verbose=True)

    print("[4/7] " + "-" * 50)
    df4 = experiment_4_representation(verbose=True)

    print("[5/7] " + "-" * 50)
    df5 = experiment_5_adaptive(verbose=True)

    print("[6/7] " + "-" * 50)
    results6 = experiment_6_novel_methods(verbose=True)

    print("[7/7] " + "-" * 50)
    df7, components7 = experiment_7_composed_controller(verbose=True)

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"All experiments complete. Total time: {elapsed:.1f}s")
    print(f"Figures saved to: paper/figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()
