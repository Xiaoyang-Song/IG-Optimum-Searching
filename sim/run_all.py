"""Run all four experiments end to end and write every figure to sim/figures/."""
import time

import toy2d
import ig_attribution_exp
import mnist_exp
import budget_exp


def main():
    t0 = time.time()

    print("=" * 70)
    print("Experiment 1: 2D synthetic staircase-saturation problem")
    print("=" * 70)
    toy2d.run()

    print("\n" + "=" * 70)
    print("Experiment 3: IG attribution validation (6-D synthetic model)")
    print("=" * 70)
    ig_attribution_exp.run()

    print("\n" + "=" * 70)
    print("Experiment 4: budget-constrained attribution-aware escape")
    print("=" * 70)
    budget_exp.run()

    print("\n" + "=" * 70)
    print("Experiment 2: MNIST target-confidence attack")
    print("=" * 70)
    mnist_exp.run()

    print(f"\nAll experiments done in {time.time()-t0:.1f}s. Figures in sim/figures/.")


if __name__ == "__main__":
    main()
