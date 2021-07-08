# NanoScreener

KNOWN BUG:

The plotting functions use np.mean and np.std to calculate the expected binding free energies, 
but the data processing functions use a Boltzmann-weighted average (DEPENDENCIES.processing.boltz_mean).

--> The plots are not representative of the data!

When this bug is fixed, the correlation between computed and experimental free energies is more clear.
I know this because I temporarily fixed it locally, but did not push the changes.
