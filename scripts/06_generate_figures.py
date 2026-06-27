#!/usr/bin/env python3
"""Generate manuscript-ready figures from experiment outputs."""

from pinn_qi_radar.plotting import generate_all_figures


def main() -> None:
    figures = generate_all_figures()
    if not figures:
        print("No figures generated. Run experiment scripts first.")
        return
    print("Generated figures:")
    for fig in figures:
        print(f" - {fig}")


if __name__ == "__main__":
    main()
