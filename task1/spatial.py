"""Electrode geometry for B2: simulated cap displacement by spherical-spline interpolation from the true positions."""
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix   # MNE's spherical spline (Perrin et al. 1989), as used by interpolate_bads
import data


def positions(ch_names):
    """standard_1005 positions, centred on the least-squares sphere through the 64 electrodes, plus that radius (m).
    Centring matters: the spline and the rotation both assume the head is a sphere around the origin."""
    p = data.montage_pos(ch_names)
    A = np.c_[2 * p, np.ones(len(p))]
    sol = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    c = sol[:3]
    r = np.sqrt(sol[3] + c @ c)
    return p - c, r


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def displacement_matrix(ch_names, shift_mm, direction):
    """M (64 x 64) such that M @ x is what the cap would record if every electrode slid `shift_mm` along the scalp in
    `direction` (unit vector in the tangent plane at Cz: +x = towards the right ear, +y = towards the nose).
    A rigid cap slide is a rotation of all positions about the head centre by angle = arc length / radius."""
    p, r = positions(ch_names)
    d = np.array([direction[0], direction[1], 0.0])
    if shift_mm == 0 or not d.any():
        return np.eye(len(p))
    axis = np.cross([0, 0, 1.0], d)          # moving the top of the head towards d = rotating about z x d
    R = rotation(axis, shift_mm / 1000 / r)
    # alpha=None: exact spline at the electrodes. MNE's default 1e-5 smooths even at 0 mm (max |M - I| = 0.67, own check),
    # which would mix a smoothing effect into the displacement curve; exact gives M = I at 0 mm.
    return _make_interpolation_matrix(p, p @ R.T, alpha=None)
