# Credits: scikit-guess project
# Source: https://gitlab.com/madphysicist/scikit-guess/-/blob/master/src/skg/nsphere.py

# BSD 2-Clause License

# Copyright (c) 2018, Joseph Fox-Rabinovitz
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
N-sphere fit with center and radius.

This module is a little different from the others because it fits an
n-dimensional surface and because it does not have a model function
because of the non-functional nature of n-spheres.
"""

from numpy import empty, sqrt, square
from scipy.linalg import lstsq

# from .util import preprocess

__all__ = ["nsphere_fit"]


def nsphere_fit(x, scaling=False):
    r"""
    Fit an n-sphere to ND data.

    The center and radius of the n-sphere are optimized using the Coope
    method. The sphere is described by

    .. math::

       \left \lVert \vec{x} - \vec{c} \right \rVert_2 = r

    Parameters
    ----------
    x : array-like
        The n-vectors describing the data. Usually this will be a nxm
        array containing m n-dimensional data points.
    scaling : bool
        If `True`, scale and offset the data to a bounding box of -1 to
        +1 during computations for numerical stability. Default is
        `False`.

    Return
    ------
    r : scalar
        The optimal radius of the best-fit n-sphere for `x`.
    c : array
        An array of size `x.shape[axis]` with the optimized center of
        the best-fit n-sphere.

    References
    ----------
    - [Coope]_ "\ :ref:`ref-cfblanls`\ "
    """
    n = x.shape[-1]
    x = x.reshape(-1, n)
    m = x.shape[0]

    B = empty((m, n + 1), dtype=x.dtype)
    X = B[:, :-1]
    X[:] = x
    B[:, -1] = 1

    if scaling:
        xmin = X.min()
        xmax = X.max()
        scale = 0.5 * (xmax - xmin)
        offset = 0.5 * (xmax + xmin)
        X -= offset
        X /= scale

    d = square(X).sum(axis=-1)

    y, *_ = lstsq(B, d, overwrite_a=True, overwrite_b=True)

    c = 0.5 * y[:-1]
    r = sqrt(y[-1] + square(c).sum())

    if scaling:
        r *= scale
        c *= scale
        c += offset

    return r, c
