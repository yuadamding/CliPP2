"""Fixed pairwise-fusion estimator internals.

Deliberately empty: every consumer imports the submodule it needs directly
(``from .fusion.solver import ...``, ``from ..core.fusion.types import ...``).
Re-exporting here would force the solver and the whole Torch backend to load on
any ``core.fusion`` import, which defeats the leaf-module layering that
``core/bic.py`` and ``core/fusion/types.py`` are designed around.
"""
