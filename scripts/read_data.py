# Testing that we can build the model

from pathlib import Path

import cmasher as cmr  # noqa: F401
import jax
import jax.numpy as jnp  # noqa: F401
import jax.random as jr
import matplotdrip  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from lvm_tools import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection
from lvm_tools.fit_data.filtering import BAD_FLUX_THRESHOLD
from lvm_tools.utils.mask import mask_near_points
from matplotdrip import colormaps  # noqa: F401
from models.two_lines import TwoComponentEmissionLine, neg_ln_posterior
from spectracles import (
    ConstrainedParameter,
    Known,
    Matern12,  # noqa: F401
    Matern32,  # noqa: F401
    Matern52,  # noqa: F401
    Parameter,
    SpatialDataGeneric,
    build_model,
    build_schedule,
    free_in,
    init_normal,
)

# Some declaration-y stuff
plt.style.use("drip")
rng = np.random.default_rng(0)
DATA_LOC = Path("../data/W28/THOR")
assert DATA_LOC.is_dir()
jax.config.update("jax_enable_x64", True)

LINE_NAME = "NII"
# LINE_NAME = "HALPHA"

if LINE_NAME == "HALPHA":
    LINE_λ = 6562.8
    λ_EXTENT = 8.0
    norm_F = 5e-13
elif LINE_NAME == "NII":
    LINE_λ = 6583.45
    λ_EXTENT = 8.0
    norm_F = 2e-13
elif LINE_NAME == "SII":
    LINE_λ = 6716.44
    λ_EXTENT = 8.0
    norm_F = 1e-13
elif LINE_NAME == "SIII":
    LINE_λ = 9531.1
    λ_EXTENT = 8.0
    norm_F = 0.5e-13
else:
    raise ValueError(f"Unknown line name: {LINE_NAME}")

# ============================
# Load and plot the data
# ============================

# Load the data
drp_files = list(DATA_LOC.glob("*.fits"))
# drp_file = drp_files[0]
# tile = LVMTile.from_file(drp_file=drp_file)
tiles = LVMTileCollection.from_tiles(
    [LVMTile.from_file(drp_file=drp_file) for drp_file in drp_files]
)
fd = FitDataBuilder(
    tiles=tiles,
    config=DataConfig.from_tiles(
        tiles,
        λ_range=(
            LINE_λ - λ_EXTENT,
            LINE_λ + λ_EXTENT,
        ),
        normalise_F_scale=norm_F,
        F_range=(BAD_FLUX_THRESHOLD, 1e-12),
    ),
).build()

peak_int_map = np.nanmax(fd.flux, axis=0)
sum_int_map = np.nansum(fd.flux, axis=0)
vmin, vmax = np.nanpercentile(sum_int_map, [5, 95])

# Plot the data
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
sc = ax.scatter(fd.α, fd.δ, c=sum_int_map, s=20, vmin=vmin, vmax=vmax)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("Data from LVM tile")
ax.set_aspect("equal")
plt.savefig(f"summed_intensity_map_{LINE_NAME}.pdf", bbox_inches="tight")
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
    ax.set_ylim(-0.1, 1.3)
plt.tight_layout()
plt.savefig(f"example_spectra_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

# Plot the summed spectrum across all spaxels
summed_spectrum = np.nansum(fd.flux, axis=1)
fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
ax.plot(fd.λ, summed_spectrum, color="C0")
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"Summed $F_\lambda$")
ax.set_title("Summed spectrum across all spaxels")
plt.savefig(f"summed_spectrum_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

# ============================
# Build and configure the model
# ============================

# Known values
n_spaxels = fd.α.shape[0]
n_modes = (201, 201)
line_centre = Known(LINE_λ)
idx_λ = np.argmin(np.abs(fd.λ - LINE_λ))
σ_lsf = Known(fd.lsf_σ[idx_λ])
v_bary = Known(fd.v_bary)

# Kernel choice
kernel = Matern12
# kernel = Matern32

# Kernel hyperparameters
# kernel_kwargs = dict(fixed=True, lower=0.0)
kernel_kwargs = dict(fixed=True, log=True)
A_var = ConstrainedParameter(10.0, **kernel_kwargs)
# A_var = ConstrainedParameter(1.0, **kernel_kwargs)
v_var = ConstrainedParameter(10.0, **kernel_kwargs)
# v_var = ConstrainedParameter(1.0, **kernel_kwargs)
σ_var = ConstrainedParameter(10.0, **kernel_kwargs)
# σ_var = ConstrainedParameter(1.0, **kernel_kwargs)
A_len = ConstrainedParameter(1.0, **kernel_kwargs)
# A_len = ConstrainedParameter(0.3, **kernel_kwargs)
v_len = ConstrainedParameter(1.0, **kernel_kwargs)
# v_len = ConstrainedParameter(0.3, **kernel_kwargs)
σ_len = ConstrainedParameter(1.0, **kernel_kwargs)
# σ_len = ConstrainedParameter(0.3, **kernel_kwargs)

# Parameters with initial values
offsets = Parameter(np.zeros(n_spaxels))
v_syst = Parameter(-100.0)
Δv_syst = ConstrainedParameter(100.0, lower=0)

# Instantiate the model
model = build_model(
    TwoComponentEmissionLine,
    n_spaxels=n_spaxels,
    offsets=offsets,
    line_centre=line_centre,
    n_modes=n_modes,
    A_kernel=kernel(variance=A_var, length_scale=A_len),
    v_kernel=kernel(variance=v_var, length_scale=v_len),
    σ_kernel=kernel(variance=σ_var, length_scale=σ_len),
    σ_lsf=σ_lsf,
    v_bary=v_bary,
    v_syst=v_syst,
    Δv_syst=Δv_syst,
)
# init_model = model.get_locked_model()

### Check things look right
# model.print_model_tree(show_sharing=True)
model.get_parameter_summary(show_shared=False)
# fig, ax = plt.subplots(figsize=(24, 24), dpi=100)
# model.plot_model_graph(ax=ax, show=False)
# plt.show()

# ============================
# Optimisation
# ============================

initial_loss = neg_ln_posterior(
    model,
    λ=fd.λ,
    xy_data=fd.αδ_data,
    data=fd.flux,
    u_data=fd.u_flux,
    mask=fd.mask,
)
print(f"Initial loss: {initial_loss:.2f}")

# Build an optimisation schedule
schedule = build_schedule(
    model=model,
    loss_fn=neg_ln_posterior,
    phases=[
        (4000, 0.01),  # steps, lr
        (4000, 0.01),
        (4000, 0.01),
        (4000, 0.01),
        (4000, 0.01),
        (4000, 0.01),
        (4000, 0.01),
        (4000, 0.01),
        # (500, 0.01),
        (4000, 0.01),
        # (500, 0.001),
        (4000, 0.001),
    ],
    params={
        "*.A.gp.coefficients": init_normal(0) | free_in(0, 3, 4, 5, 8, 9, 10),
        # "*.A.gp.coefficients": init_normal(0) | free_in(0, 3, 5, 8, 9, 10),
        "*.v.gp.coefficients": init_normal(1) | free_in(1, 3, 4, 6, 8, 9, 10),
        # "*.v.gp.coefficients": init_normal(1) | free_in(1, 3, 6, 8, 9, 10),
        "*.vσ.gp.coefficients": init_normal(2) | free_in(2, 4, 7, 8, 9, 10),
        "*.A.gp.kernel.length_scale": free_in(5, 8, 9, 10),
        "*.A.gp.kernel.variance": free_in(5, 8, 9, 10),
        "*.v.gp.kernel.length_scale": free_in(6, 8, 9, 10),
        "*.v.gp.kernel.variance": free_in(6, 8, 9, 10),
        "*.vσ.gp.kernel.length_scale": free_in(7, 8, 9, 10),
        "*.vσ.gp.kernel.variance": free_in(7, 8, 9, 10),
    },
    Δloss_criterion=1e-4,
    key=jr.key(42),
)

schedule.run_all(
    λ=fd.λ,
    xy_data=fd.αδ_data,
    data=fd.flux,
    u_data=fd.u_flux,
    mask=fd.mask,
)

# Attempt to evaluate the loss with the starting parameters to check for issues


# Plot the loss history
plt.figure()
plt.title("Loss history")
plt.plot(schedule.loss_history)
# plt.xscale("log")
plt.xlabel("Step")
plt.ylabel("neg. log posterior")
plt.savefig(f"loss_history_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

final_stage_idx = 6
plt.figure()
plt.title("Final stage loss history")
hist = schedule.loss_histories[final_stage_idx]
sign = np.sign(np.min(hist))
print(sign)
plt.plot(sign * hist)
plt.xlabel("Step")
if sign < 0:
    plt.ylabel("log posterior (higher is better)")
else:
    plt.ylabel("neg. log posterior (lower is better)")
plt.yscale("log")
plt.savefig(f"final_stage_loss_history_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

pred_model = schedule.model_history[7].get_locked_model()

print("Final parameter values:")
print("v_syst_1:", pred_model.line_1.v_syst.val)
print("v_syst_2:", pred_model.line_2.v_syst.val)

print("Final kernel hyperparameters:")
print("A_var:", pred_model.line_1.A.gp.kernel.variance.val)
print("A_len:", pred_model.line_1.A.gp.kernel.length_scale.val)
print("v_var:", pred_model.line_1.v.gp.kernel.variance.val)
print("v_len:", pred_model.line_1.v.gp.kernel.length_scale.val)
print("vσ_var:", pred_model.line_1.vσ.gp.kernel.variance.val)
print("vσ_len:", pred_model.line_1.vσ.gp.kernel.length_scale.val)

# ============================
# Plot the results
# ============================


### First, plot the fields

n_dense = 1600
α_dense_1D = np.linspace(fd.α.min(), fd.α.max(), n_dense)
δ_dense_1D = np.linspace(fd.δ.min(), fd.δ.max(), n_dense)

mask = mask_near_points(
    xgrid=α_dense_1D,
    ygrid=δ_dense_1D,
    xpoints=fd.α,
    ypoints=fd.δ,
    threshold=0.05,
)
α_dense, δ_dense = np.meshgrid(α_dense_1D, δ_dense_1D)
αδ_dense = SpatialDataGeneric(
    α_dense.flatten(),
    δ_dense.flatten(),
    idx=np.arange(n_dense**2, dtype=int),
)

A_pred_1 = pred_model.line_1.A(αδ_dense).reshape(n_dense, n_dense)
v_pred_1 = pred_model.line_1.v(αδ_dense).reshape(n_dense, n_dense)
v_pred_1_with_syst = v_pred_1 + pred_model.line_1.v_syst.val
vσ_pred_1 = pred_model.line_1.vσ(αδ_dense).reshape(n_dense, n_dense)
A_pred_2 = pred_model.line_2.A(αδ_dense).reshape(n_dense, n_dense)
v_pred_2 = pred_model.line_2.v(αδ_dense).reshape(n_dense, n_dense)
v_pred_2_with_syst = v_pred_2 + pred_model.line_2.v_syst.val
vσ_pred_2 = pred_model.line_2.vσ(αδ_dense).reshape(n_dense, n_dense)

A_pred_1 = np.where(mask, A_pred_1, np.nan)
A_pred_2 = np.where(mask, A_pred_2, np.nan)
v_pred_1 = np.where(mask, v_pred_1, np.nan)
v_pred_2 = np.where(mask, v_pred_2, np.nan)
vσ_pred_1 = np.where(mask, vσ_pred_1, np.nan)
vσ_pred_2 = np.where(mask, vσ_pred_2, np.nan)

log_A = False

if log_A:
    f_A = np.log10
else:
    f_A = lambda x: x  # noqa: E731

A_vmin, A_vmax = f_A(1e-3), f_A(3)  # max(A_pred_1.max(), A_pred_2.max())
max_abs_v = max(np.nanmax(np.abs(v_pred_1)), np.nanmax(np.abs(v_pred_2)))
v_vmin, v_vmax = -max_abs_v, max_abs_v
vσ_vmin, vσ_vmax = 0, max(np.nanmax(vσ_pred_1), np.nanmax(vσ_pred_2))

A_imshow_kwargs = dict(origin="lower", vmin=A_vmin, vmax=A_vmax, cmap="cmr.voltage_r")
v_imshow_kwargs = dict(
    origin="lower", vmin=v_vmin, vmax=v_vmax, cmap="red_white_blue_r"
)
σ_imshow_kwargs = dict(origin="lower", vmin=vσ_vmin, vmax=vσ_vmax, cmap="cmr.torch_r")
cbar_kwargs = dict(orientation="horizontal", location="top", pad=0.01)
fig, ax = plt.subplots(2, 3, figsize=(18, 12), dpi=100, layout="compressed")
im = ax[0, 0].imshow(f_A(A_pred_1), **A_imshow_kwargs)
im = ax[1, 0].imshow(f_A(A_pred_2), **A_imshow_kwargs)
plt.colorbar(im, ax=ax[:, 0], label="Line Flux", **cbar_kwargs)
im = ax[0, 1].imshow(v_pred_1, **v_imshow_kwargs)
im = ax[1, 1].imshow(v_pred_2, **v_imshow_kwargs)
plt.colorbar(im, ax=ax[:, 1], label="Velocity", **cbar_kwargs)
im = ax[0, 2].imshow(vσ_pred_1, **σ_imshow_kwargs)
im = ax[1, 2].imshow(vσ_pred_2, **σ_imshow_kwargs)
plt.colorbar(im, ax=ax[:, 2], label="Velocity Dispersion", **cbar_kwargs)
for ax_flat in ax.flat:
    ax_flat.set_xticks([])
    ax_flat.set_yticks([])
plt.savefig(f"predicted_fields_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

### Plot the velocity separation
v_sep = v_pred_2_with_syst - v_pred_1_with_syst
v_sep = np.where(mask, v_sep, np.nan)
max_v_sep = np.nanmax(np.abs(v_sep))
fig, ax = plt.subplots(figsize=(8, 8), dpi=100, layout="compressed")
im = ax.imshow(v_sep, cmap="RdBu", vmin=-max_v_sep, vmax=max_v_sep)
plt.title("Velocity separation")
plt.colorbar(im, ax=ax)
plt.savefig(f"velocity_separation_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()


### Plot a bunch of random spectra with the model overlaid

λ_dense = np.linspace(fd.λ.min(), fd.λ.max(), 200)
pred_flux_λ_dense = jax.vmap(pred_model, in_axes=(0, None))(λ_dense, fd.αδ_data)

fig, axes = plt.subplots(10, 2, figsize=(15, 15), dpi=100, sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    spax_ind = spax_inds[i]
    ax.plot(fd.λ, fd._flux[:, spax_ind], color="C0", label="Data")
    ax.plot(λ_dense, pred_flux_λ_dense[:, spax_ind], color="C1", label="Model")
    # ax.set_title(f"Spaxel {spax_ind}")
    # ax.set_xlabel(r"$\lambda$ [Å]")
    # ax.set_ylabel(r"$F_\lambda$ [erg/s/cm²/Å]")
    # ax.set_xticks([])
    # ax.set_yticks([])
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.savefig(f"example_spectra_with_model_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

### Evaluate on the actual data points
A_pred_1_dp = pred_model.line_1.A(fd.αδ_data)
v_pred_1_dp = pred_model.line_1.v(fd.αδ_data) + pred_model.line_1.v_syst.val
vσ_pred_1_dp = pred_model.line_1.vσ(fd.αδ_data)
A_pred_2_dp = pred_model.line_2.A(fd.αδ_data)
v_pred_2_dp = pred_model.line_2.v(fd.αδ_data) + pred_model.line_2.v_syst.val
vσ_pred_2_dp = pred_model.line_2.vσ(fd.αδ_data)

# Filer spaxel indices for which the model predicts a line flux in both components above the normalisation scale (i.e. where the model is "active" in both components)
threshold = 0.2
active_spax_inds = np.where(
    (A_pred_1_dp.flatten() > threshold) & (A_pred_2_dp.flatten() > threshold)
)[0]
print(
    f"Number of spaxels where model is active in both components: {len(active_spax_inds)}"
)

### Plot in spaxel indices where both lines are active
spax_inds_both_lines_active = rng.choice(active_spax_inds, size=20, replace=False)
fig, axes = plt.subplots(10, 2, figsize=(15, 20), dpi=100, sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    spax_ind = spax_inds_both_lines_active[i]
    ax.plot(fd.λ, fd._flux[:, spax_ind], color="C0", label="Data")
    ax.plot(λ_dense, pred_flux_λ_dense[:, spax_ind], color="C1", label="Model")
    # ax.set_title(f"Spaxel {spax_ind}")
    # ax.set_xlabel(r"$\lambda$ [Å]")
    # ax.set_ylabel(r"$F_\lambda$ [erg/s/cm²/Å]")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_yscale("log")
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.savefig(f"example_spectra_both_lines_active_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

# Plot the continuum offsets inferred
offs = pred_model.offs.const.spaxel_values.val
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
ax.scatter(fd.α, fd.δ, c=np.log10(offs), s=20)
ax.set_aspect(1)
plt.savefig(f"continuum_offsets_log_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
ax.scatter(fd.α, fd.δ, c=offs, s=20)
ax.set_aspect(1)
plt.savefig(f"continuum_offsets_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

# Filter spaxels to those only where the offsets are above a certain values
offs_threshold = 0.1
offs_spax_inds = np.where((offs.flatten() > threshold))[0]
print(f"Number of spaxels with large offsets: {len(offs_spax_inds)}")


fig, axes = plt.subplots(10, 2, figsize=(15, 20), dpi=100, sharex=True, sharey=True)
# fig, axes = plt.subplots(10, 2, figsize=(4, 5), dpi=100, sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    try:
        spax_ind = offs_spax_inds[i]
        # ax.text(1, 1, s=f"x: {fd.α[spax_ind]}, y: {fd.δ[spax_ind]}")
        ax.plot(fd.λ, fd._flux[:, spax_ind], color="C0", label="Data")
        ax.plot(λ_dense, pred_flux_λ_dense[:, spax_ind], color="C1", label="Model")
    except IndexError:
        pass
    # ax.set_title(f"Spaxel {spax_ind}")
    # ax.set_xlabel(r"$\lambda$ [Å]")
    # ax.set_ylabel(r"$F_\lambda$ [erg/s/cm²/Å]")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_yscale("log")
    if i == 0:
        ax.legend()
plt.savefig(f"example_spectra_large_offsets_{LINE_NAME}.pdf", bbox_inches="tight")
plt.show()

star_mask = np.ones_like(fd.flux, dtype=bool)
print(star_mask.shape, fd.mask.shape)
star_mask[]
