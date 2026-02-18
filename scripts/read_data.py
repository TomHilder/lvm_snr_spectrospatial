# Testing that we can build the model

from pathlib import Path

import cmasher as cmr  # noqa: F401
import jax
import matplotdrip  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from lvm_tools import DataConfig, FitDataBuilder, LVMTile
from lvm_tools.fit_data.filtering import BAD_FLUX_THRESHOLD
from matplotdrip import colormaps  # noqa: F401

# Some declaration-y stuff
plt.style.use("drip")
rng = np.random.default_rng(0)
DATA_LOC = Path("../data/W28")
assert DATA_LOC.is_dir()
jax.config.update("jax_enable_x64", True)

# Line wavelength (halpha)
# LINE_λ = 6562.8
LINE_λ = 6583.45

# Load the data
drp_files = list(DATA_LOC.glob("*.fits"))
drp_file = drp_files[0]
tile = LVMTile.from_file(drp_file=drp_file)
print(tile)
fd = FitDataBuilder(
    tiles=tile,
    config=DataConfig.from_tiles(
        tile,
        λ_range=(
            LINE_λ - 8,
            LINE_λ + 8,
        ),
        normalise_F_scale=2e-13,
        # normalise_F_offset=0.0,
        F_range=(BAD_FLUX_THRESHOLD, 1e-12),
    ),
).build()

peak_int_map = np.nanmax(fd.flux, axis=0)
sum_int_map = np.nansum(fd.flux, axis=0)
vmin, vmax = np.nanpercentile(sum_int_map, [5, 95])

# Plot the data
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
sc = ax.scatter(fd.α, fd.δ, c=sum_int_map, s=70, vmin=vmin, vmax=vmax)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("Data from LVM tile")
ax.set_aspect("equal")
plt.show()

# Plot a bunch of spectra from random spaxels
spax_inds = rng.choice(fd.α.shape[0], size=20, replace=False)
fig, axes = plt.subplots(10, 2, figsize=(15, 15), dpi=100, sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    spax_ind = spax_inds[i]
    ax.plot(fd.λ, fd.flux[:, spax_ind], color="C0")
    # ax.set_title(f"Spaxel {spax_ind}")
    # ax.set_xlabel(r"$\lambda$ [Å]")
    # ax.set_ylabel(r"$F_\lambda$ [erg/s/cm²/Å]")
    # ax.set_xticks([])
    # ax.set_yticks([])
plt.tight_layout()
plt.show()
