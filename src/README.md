# CDM Spike and the Final Parsec Problem

This repository contains a set of exploratory Jupyter notebooks for studying how a cold dark matter (CDM) spike can affect the hardening of a supermassive black hole (SMBH) binary.

The simulations use `rebound` to evolve two SMBHs embedded in a finite-mass dark matter spike. As the binary interacts with the spike, dark matter particles can exchange energy and angular momentum with the SMBHs through gravitational slingshot interactions. The main diagnostic is the SMBH separation as a function of time. Once the spike is depleted, the separation reaches a plateau, which is interpreted here as the stalling distance.

## Notebook Guide

Shared setup code lives in `cdm_spike_utils.py`. The notebooks import this module so the repeated REBOUND initialization, CDM spike sampling, collision setup, moving average, and sweep plotting code is kept in one place.

1. `jupyter_orbital_bh.ipynb`
   - First exploratory simulation.
   - Builds one SMBH binary plus a CDM particle spike.
   - Tracks the SMBH separation and optionally generates a movie of the evolution.

2. `parameter_sweep_spike_mass.ipynb`
   - Sweeps over the total spike mass.
   - Tests how efficiently different CDM spike masses shrink the SMBH binary.

3. `parameter_sweep_number_of_particles.ipynb`
   - Sweeps over the numerical resolution of the spike.
   - Checks how the result depends on the number of simulated CDM particles.

4. `parameter_sweep_mass_ratio.ipynb`
   - Sweeps over the SMBH mass ratio.
   - Studies how unequal-mass binaries respond to the same CDM environment.

5. `Stalling_law_fit.ipynb`
   - Loads the saved results from the spike-mass sweep.
   - Fits an empirical stalling law as a function of spike mass.

## Dependencies

The notebooks require:

- Python 3
- `rebound`
- `numpy`
- `matplotlib`
- `scipy`
- `tqdm`
- `imageio`
- `pyav` for MP4 video export in the exploratory notebook

Install the core dependencies with:

```bash
pip install rebound numpy matplotlib scipy tqdm imageio pyav
```

## Suggested Reading Order

Start with `jupyter_orbital_bh.ipynb` to understand the physical setup and the measured quantity. Then run the three parameter sweeps independently. Finish with `Stalling_law_fit.ipynb`, which turns the sweep results into an empirical relation for the stalling distance.

## Notes

These notebooks are exploratory research code. The simulations use finite numbers of massive tracer particles for the CDM spike, so the absolute values should be interpreted with care. The parameter sweeps are intended to reveal qualitative trends and numerical sensitivity rather than provide a fully converged astrophysical prediction.
