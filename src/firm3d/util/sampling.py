import math

import numpy as np

__all__ = ["s_density", "sample_s", "sample_tz", "sample_stz"]


def s_density(s):
    r"""
    Compute function proportional to fusion reaction rate in s

    Args:
        s: float or double

    Returns:
        function value at s
    """
    return (
        ((1 - s**5) ** 2)
        * ((1 - s) ** (-2 / 3))
        * np.exp(-19.94 * (12 * (1 - s)) ** (-1 / 3))
    )


# Rejection sample s
def sample_s():
    r"""
    Sample s from a distribution proportional to fusion reaction rate
        via rejection sampling

    Returns:
        A sample of s
    """
    bound = 3e-4
    x = np.random.uniform()
    y = bound * np.random.uniform()

    while s_density(x) < y:
        assert s_density(x) <= bound
        x = np.random.uniform()
        y = bound * np.random.uniform()
    return x


def sample_tz(s, J_max, field):
    r"""
    Sample theta and zeta proportional to Jacobian for a given s

    Args:
        s: BoozerMagneticField object
        J_max: maximum of observed J on mangetic field
        field: BoozerMagneticField object

    Returns:
        theta, zeta
    """
    J = rand_J = 0
    while rand_J >= J:
        theta = np.random.uniform(low=0, high=2 * math.pi, size=1)
        zeta = np.random.uniform(low=0, high=2 * math.pi, size=1)
        rand_J = np.random.uniform(low=0, high=J_max, size=1)

        loc = np.array([s, theta[0], zeta[0]]).reshape(1, 3)
        field.set_points(loc)

        G = field.G()
        iota = field.iota()
        I = field.I()
        modB = field.modB()
        J = (G + iota * I) / (modB**2)
        J = J[0][0]
        assert J_max >= J
    return theta[0], zeta[0]


# Sample s,t,z
def sample_stz(field, J_max):
    r"""
    Sample s proportional to fusion reaction rate
    Sample theta and zeta proportional to Jacobian conditional on s

    Args:
        field: BoozerMagneticField object
        J_max: maximum of observed J on mangetic field

    Returns:
        s, theta, zeta
    """
    s = sample_s()
    theta, zeta = sample_tz(s, J_max, field)
    return np.array([s, theta, zeta])
