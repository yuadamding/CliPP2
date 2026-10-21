"""Resumable bounded Brent steps for batching independent objective evaluations.

Adapted from SciPy's _minimize_scalar_bounded; the numerical decisions, stopping
test and evaluation limit are unchanged. This iterator changes scheduling only.

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np


def bounded_steps(bounds, xatol, maxiter=500):
    """Yield each requested x; send f(x); return (minimizer, minimum, success)."""
    a, b = bounds
    if not (np.isfinite(a) and np.isfinite(b) and a <= b):
        raise ValueError("Finite ordered scalar bounds are required")
    sqrt_eps, golden_mean = np.sqrt(2.2e-16), .5 * (3. - np.sqrt(5.))
    fulc = a + golden_mean * (b - a)
    nfc = xf = fulc
    rat = e = 0.
    fx = yield xf
    num, fu = 1, np.inf
    ffulc = fnfc = fx
    xm = .5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.
    tol2 = 2. * tol1
    success = True
    while np.abs(xf - xm) > tol2 - .5 * (b - a):
        golden = True
        if np.abs(e) > tol1:
            golden = False
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2. * (q - r)
            if q > 0.:
                p = -p
            q = np.abs(q)
            r, e = e, rat
            if np.abs(p) < np.abs(.5*q*r) and p > q*(a-xf) and p < q*(b-xf):
                rat = (p + 0.) / q
                x = xf + rat
                if x-a < tol2 or b-x < tol2:
                    rat = tol1 * (np.sign(xm-xf) + ((xm-xf) == 0))
            else:
                golden = True
        if golden:
            e = a-xf if xf >= xm else b-xf
            rat = golden_mean * e
        x = xf + (np.sign(rat) + (rat == 0)) * np.maximum(np.abs(rat), tol1)
        fu = yield x
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fnfc or nfc == xf:
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif fu <= ffulc or fulc == xf or fulc == nfc:
                fulc, ffulc = x, fu
        xm = .5 * (a+b)
        tol1 = sqrt_eps * np.abs(xf) + xatol/3.
        tol2 = 2.*tol1
        if num >= maxiter:
            success = False
            break
    return float(xf), float(fx), bool(success and not (
        np.isnan(xf) or np.isnan(fx) or np.isnan(fu)))
