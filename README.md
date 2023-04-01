# continuous_lc
Continuous lagged coherence, phase-locking value, and amplitude coherence

## Requirements
joblib, scipy, numpy

# Python files
- lagged_coherence.py: core functions
- demo.ipynb: demonstration of the new lagged coherence algorithm
- sims.ipynb: tests with simulated data
- meg_sensor_data.ipynb: analysis of MEG sensor data
- lfp_data.ipynb: analysis of LFP monkey data

# Matlab files
- lagged_hilbert_coherence.m: core function
- temporal_shuffle.m: temporal shuffling
- demo.m: demonstration of new lagged coherence algorithm
- rfft.m: Fourier transform of real signal
- irfft.m: inverse Fourier transform of real signal
- hilbert.m: hilbert transform
