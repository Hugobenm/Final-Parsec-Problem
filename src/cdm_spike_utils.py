"""Shared helpers for the CDM spike SMBH-binary notebooks."""

import random

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rebound


AU_PER_PC = 206265
SECONDS_PER_YEAR = 3600 * 24 * 365.2425
AU_IN_METERS = 1 / 6.68459e-12
MSUN_IN_KG = 1.989e30


def schwarzschild_radius_au(mass_msun):
    """Return the Schwarzschild radius of a black hole mass in AU."""
    c_au_per_year = 2.99792458e8 * 6.68459e-12 * SECONDS_PER_YEAR
    g_au3_per_msun_year2 = 6.674e-11 * (6.68459e-12) ** 3 * SECONDS_PER_YEAR**2
    mass_kg = mass_msun * MSUN_IN_KG
    return 2 * g_au3_per_msun_year2 * mass_kg / c_au_per_year**2


def create_smbh_binary(m_primary, q=1.0, separation_au=AU_PER_PC):
    """Create a REBOUND simulation containing only the SMBH binary."""
    m_secondary = q * m_primary
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=m_primary)
    sim.add(m=m_secondary, a=separation_au)
    sim.move_to_com()
    return sim, sim.particles[0], sim.particles[1], m_secondary, m_primary + m_secondary


def configure_smbh_collisions(sim, r_primary, r_secondary=None, softening=True):
    """Set collision radii and merge particles that cross an SMBH radius."""
    if r_secondary is None:
        r_secondary = r_primary

    sim.particles[0].r = r_primary
    sim.particles[1].r = r_secondary
    sim.collision = "direct"
    sim.collision_resolve = "merge"

    if softening:
        sim.softening = max(r_primary, r_secondary)


def _add_particle_on_circular_orbit(sim, rng, mass, radius, m_central):
    """Add one particle at radius with a random tangential circular velocity."""
    phi = rng.uniform(0, 2 * np.pi)
    costheta = rng.uniform(-1, 1)
    theta = np.arccos(costheta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    v_circ = np.sqrt(sim.G * m_central / radius)
    pos_vec = np.array([x, y, z])
    random_vec = np.random.randn(3)
    perp_vec = np.cross(pos_vec, random_vec)
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    vel_vec = perp_vec * v_circ

    sim.add(m=mass, x=x, y=y, z=z, vx=vel_vec[0], vy=vel_vec[1], vz=vel_vec[2])


def add_radial_uniform_spike(
    sim,
    n_particles,
    total_spike_mass,
    m_central,
    r_min=1000,
    r_max=400000,
    seed=0,
):
    """Add CDM particles with radii drawn uniformly between r_min and r_max."""
    rng = random.Random(seed)
    particle_mass = total_spike_mass / n_particles

    for _ in range(int(n_particles)):
        radius = rng.uniform(r_min, r_max)
        _add_particle_on_circular_orbit(sim, rng, particle_mass, radius, m_central)

    return particle_mass


def add_volume_uniform_spike(
    sim,
    n_particles,
    total_spike_mass,
    m_central,
    r_min=1000,
    r_max=400000,
    seed=0,
):
    """Add CDM particles by rejection sampling a uniform volume density shell."""
    rng = random.Random(seed)
    n_particles = int(n_particles)
    particle_mass = total_spike_mass / n_particles

    particles_added = 0
    while particles_added < n_particles:
        x = rng.uniform(-r_max, r_max)
        y = rng.uniform(-r_max, r_max)
        z = rng.uniform(-r_max, r_max)
        radius = np.sqrt(x**2 + y**2 + z**2)

        if radius < r_min or radius > r_max:
            continue

        v_circ = np.sqrt(sim.G * m_central / radius)
        pos_vec = np.array([x, y, z])
        random_vec = np.random.randn(3)
        perp_vec = np.cross(pos_vec, random_vec)
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
        vel_vec = perp_vec * v_circ

        sim.add(m=particle_mass, x=x, y=y, z=z, vx=vel_vec[0], vy=vel_vec[1], vz=vel_vec[2])
        particles_added += 1

    return particle_mass


def add_power_law_spike(
    sim,
    n_particles,
    total_spike_mass,
    m_central,
    gamma,
    r_min=1000,
    r_max=400000,
    seed=0,
):
    """Add CDM particles sampled from rho(r) proportional to r^(-gamma)."""
    rng = random.Random(seed)
    n_particles = int(n_particles)
    particle_mass = total_spike_mass / n_particles
    alpha = 3.0 - gamma

    for _ in range(n_particles):
        u_radius = rng.uniform(0, 1)
        if abs(alpha) < 1e-5:
            radius = r_min * np.exp(u_radius * np.log(r_max / r_min))
        else:
            radius = (r_min**alpha + u_radius * (r_max**alpha - r_min**alpha)) ** (1.0 / alpha)

        _add_particle_on_circular_orbit(sim, rng, particle_mass, radius, m_central)

    return particle_mass


def separation_history(sim, black_hole_1, black_hole_2, n_frames, delta_time, frame_callback=None):
    """Integrate the simulation and return the SMBH separation history in parsec."""
    separations_au = []

    for frame_index in range(n_frames):
        sim.integrate(sim.t + delta_time)

        if frame_callback is not None:
            frame_callback(sim, frame_index)

        separations_au.append(black_hole_1 ** black_hole_2)

    return np.array(separations_au) / AU_PER_PC


def moving_average(values, window=50):
    """Return a trailing moving average with the same early-time behavior as the notebooks."""
    values = np.asarray(values)
    averaged = [values[0]]

    for i in range(1, len(values)):
        n_previous = min(i, window)
        averaged.append(np.sum(values[i - n_previous + 1 : i + 1]) / n_previous)

    return np.array(averaged)


def apply_publication_style(use_tex=True):
    """Apply common matplotlib settings used by the sweep notebooks."""
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 12,
        }
    )


def plot_parameter_sweep(
    t_array,
    distance_histories,
    parameter_values,
    colorbar_label,
    cmap_name="viridis",
    ylim=(0, 1.05),
    output_path=None,
):
    """Plot one separation curve per scanned parameter value."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=min(parameter_values), vmax=max(parameter_values))

    for value, distance_history in zip(parameter_values, distance_histories):
        ax.plot(t_array, distance_history, color=cmap(norm(value)), linewidth=2)

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Distance between the BHs (pc)")
    ax.set_ylim(*ylim)
    ax.grid()

    scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax)
    colorbar.set_label(colorbar_label, fontsize=11)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)

    return fig, ax


def plot_spike_3d(sim, output_path=None, title="3D Distribution of the Dark Matter Spike"):
    """Plot the current CDM spike and SMBH positions in 3D."""
    x_dm = [p.x for p in sim.particles[2:]]
    y_dm = [p.y for p in sim.particles[2:]]
    z_dm = [p.z for p in sim.particles[2:]]

    x_bh = [sim.particles[0].x, sim.particles[1].x]
    y_bh = [sim.particles[0].y, sim.particles[1].y]
    z_bh = [sim.particles[0].z, sim.particles[1].z]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_dm, y_dm, z_dm, s=5, c="blue", alpha=0.3, label="Dark Matter")
    ax.scatter(x_bh, y_bh, z_bh, c="red", s=100, marker="X", label="SMBHs")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")

    max_range = max(
        np.max(np.abs(x_dm + x_bh)),
        np.max(np.abs(y_dm + y_bh)),
        np.max(np.abs(z_dm + z_bh)),
    )
    max_range = max(max_range, 400000)
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.legend()
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)

    return fig, ax
