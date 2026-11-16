# This file is part of Patsy
# Copyright (C) 2011-2026 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

"""Migrate fitted linear-model designs to a revised categorical dictionary.

The motivating scenario: you fit a linear model through patsy (for example
with :func:`dmatrix` plus your favourite least-squares routine) and stored
the coefficients.  Later the *business dictionary* for one or more
categorical variables is revised -- new levels are added and/or the level
order is changed -- but you do **not** want to refit on the historical
sample.  :func:`migrate_design` rebuilds the design metadata and the
coefficients against the new dictionary so that:

* predictions for the original levels are unchanged (for arbitrary numeric
  inputs), and
* each newly added level predicts exactly like the original level it is
  declared to fall back to.

This works for additive and interaction formulas built from named numeric
columns and :func:`C` categorical columns coded with patsy's native default
(:class:`Treatment`) contrasts, and for several categorical variables at
once.  The returned :class:`MigratedDesign` carries a genuine
:class:`DesignInfo` (usable directly with :func:`build_design_matrices`),
the migrated coefficients, and a :meth:`~MigratedDesign.predict` helper that
follows patsy's native missing-data (``drop``/``raise``) semantics.  The
original :class:`DesignInfo` is never modified and remains usable.
"""

# These are made available in the patsy.* namespace
__all__ = ["migrate_design", "migrate_design_info", "MigratedDesign"]

import itertools

import numpy as np

from patsy import PatsyError
from patsy.build import _subterm_column_names_iter, _build_subterm, build_design_matrices
from patsy.compat import OrderedDict
from patsy.contrasts import Treatment
from patsy.design_info import DesignInfo, FactorInfo, SubtermInfo
from patsy.highlevel import dmatrix
from patsy.util import no_pickling, assert_no_pickling


def _factor_keys(factor):
    # The keys a user may use to refer to a categorical factor: its full
    # factor name (e.g. "C(a)"), plus -- for a simple C(column) call -- the
    # bare column name (e.g. "a").
    name = factor.name()
    keys = {name}
    if name.startswith("C(") and name.endswith(")"):
        inner = name[2:-1]
        if inner.isidentifier():
            keys.add(inner)
    return keys


def test__factor_keys():
    class MockFactor(object):
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

    assert _factor_keys(MockFactor("C(a)")) == {"C(a)", "a"}
    assert _factor_keys(MockFactor("a")) == {"a"}
    # not a simple single-argument C() call -> no bare alias
    assert _factor_keys(MockFactor("C(a, Treatment)")) == {"C(a, Treatment)"}
    assert _factor_keys(MockFactor("C(f(x))")) == {"C(f(x))"}


def _resolve_categorical_factors(design_info, mapping, description):
    # Resolve user-supplied keys (column names or factor names) to the
    # categorical factor objects of design_info. Returns {factor: value}.
    key_to_factor = {}
    ambiguous = set()
    for factor, factor_info in design_info.factor_infos.items():
        if factor_info.type != "categorical":
            continue
        for key in _factor_keys(factor):
            if key in key_to_factor and key_to_factor[key] is not factor:
                ambiguous.add(key)
            else:
                key_to_factor[key] = factor
    resolved = {}
    for key, value in mapping.items():
        if key in ambiguous:
            raise PatsyError(
                "%s %r is ambiguous; use the full factor name instead"
                % (description, key)
            )
        if key not in key_to_factor:
            available = sorted(key_to_factor)
            raise PatsyError(
                "%s %r does not match any categorical factor in the design "
                "(available: %s)" % (description, key, ", ".join(map(repr, available)))
            )
        resolved[key_to_factor[key]] = value
    return resolved


def _validate_new_level_list(factor, levels):
    if isinstance(levels, (str, bytes)):
        raise PatsyError(
            "new levels for factor %r must be given as a sequence of levels, "
            "not a single string" % (factor.name(),)
        )
    levels = list(levels)
    if not levels:
        raise PatsyError(
            "new levels for factor %r must contain at least one level"
            % (factor.name(),)
        )
    seen = set()
    for level in levels:
        try:
            hash(level)
        except TypeError:
            raise PatsyError(
                "categorical levels must be hashable; got unhashable level "
                "%r for factor %r" % (level, factor.name())
            )
        if level in seen:
            raise PatsyError(
                "duplicate level %r in the new levels for factor %r"
                % (level, factor.name())
            )
        seen.add(level)
    return levels


def _validate_fallbacks(design_info, nl_map, fb_map):
    for factor, factor_fallbacks in fb_map.items():
        if factor not in nl_map:
            raise PatsyError(
                "fallbacks were given for factor %r, but no new levels were "
                "given for it" % (factor.name(),)
            )
        old_levels = set(design_info.factor_infos[factor].categories)
        new_levels = set(nl_map[factor])
        for new_level, target in factor_fallbacks.items():
            if new_level in old_levels:
                raise PatsyError(
                    "a fallback was declared for level %r of factor %r, but "
                    "that level already exists in the original design; "
                    "fallbacks are only meaningful for newly added levels"
                    % (new_level, factor.name())
                )
            if new_level not in new_levels:
                raise PatsyError(
                    "a fallback was declared for level %r of factor %r, but "
                    "that level is not present in the new levels"
                    % (new_level, factor.name())
                )
            if target not in old_levels:
                raise PatsyError(
                    "fallback target %r for new level %r of factor %r is not "
                    "one of the original levels %r"
                    % (target, new_level, factor.name(), sorted(old_levels))
                )


def _check_treatment_coding(design_info, nl_map):
    # The coefficient migration is only guaranteed for patsy's native default
    # (Treatment) coding of the migrated factors; check every subterm contrast
    # that involves a migrated factor.
    migrated = set(nl_map)
    treatment = Treatment()
    for term, subterms in design_info.term_codings.items():
        for subterm in subterms:
            for factor, contrast_matrix in subterm.contrast_matrices.items():
                if factor not in migrated:
                    continue
                levels = list(design_info.factor_infos[factor].categories)
                matrix = np.asarray(contrast_matrix.matrix)
                full = treatment.code_with_intercept(levels).matrix
                reduced = treatment.code_without_intercept(levels).matrix
                if np.array_equal(matrix, full) or np.array_equal(matrix, reduced):
                    continue
                raise PatsyError(
                    "factor %r is coded with a non-default contrast; "
                    "migrate_design only supports the native default "
                    "(Treatment) coding" % (factor.name(),)
                )


def _validate_coefficients(design_info, coefficients):
    names = design_info.column_names
    if hasattr(coefficients, "index"):
        # pandas.Series-like: align by column name.
        index = list(coefficients.index)
        if len(index) != len(names) or set(index) != set(names):
            raise PatsyError(
                "the coefficient index does not match the design column "
                "names %r" % (names,)
            )
        coef = np.asarray(coefficients.loc[list(names)], dtype=float)
    else:
        coef = np.asarray(coefficients, dtype=float).ravel()
        if coef.shape[0] != len(names):
            raise PatsyError(
                "expected %d coefficients (one per design column), but got %d"
                % (len(names), coef.shape[0])
            )
    return coef


def _rebuild_design_info(design_info, nl_map):
    # Build a new DesignInfo identical to design_info except that every
    # migrated categorical factor uses the new levels, coded with patsy's
    # native default (Treatment) contrasts, preserving each subterm's
    # full-/reduced-rank structure. Numeric factors and all factor states are
    # reused unchanged, so numeric columns (including stateful transforms)
    # behave exactly as in the original design.
    factor_infos = {}
    for factor, factor_info in design_info.factor_infos.items():
        if factor_info.type == "categorical" and factor in nl_map:
            factor_infos[factor] = FactorInfo(
                factor,
                "categorical",
                factor_info.state,
                categories=tuple(nl_map[factor]),
            )
        else:
            factor_infos[factor] = factor_info
    term_codings = OrderedDict()
    for term, subterms in design_info.term_codings.items():
        new_subterms = []
        for subterm in subterms:
            contrast_matrices = {}
            num_columns = 1
            for factor in subterm.factors:
                factor_info = factor_infos[factor]
                if factor_info.type == "categorical":
                    if factor in nl_map:
                        old_contrast = subterm.contrast_matrices[factor]
                        n_old = len(design_info.factor_infos[factor].categories)
                        full_rank = old_contrast.matrix.shape[1] == n_old
                        levels = list(factor_info.categories)
                        if full_rank:
                            contrast = Treatment().code_with_intercept(levels)
                        else:
                            contrast = Treatment().code_without_intercept(levels)
                    else:
                        contrast = subterm.contrast_matrices[factor]
                    contrast_matrices[factor] = contrast
                    num_columns *= contrast.matrix.shape[1]
                else:
                    num_columns *= factor_info.num_columns
            new_subterms.append(
                SubtermInfo(subterm.factors, contrast_matrices, num_columns)
            )
        term_codings[term] = new_subterms
    column_names = []
    for term, subterms in term_codings.items():
        for subterm in subterms:
            column_names.extend(_subterm_column_names_iter(factor_infos, subterm))
    return DesignInfo(
        column_names, factor_infos=factor_infos, term_codings=term_codings
    )


def _collapse_indices(design_info, nl_map, fb_map):
    # For every categorical factor, an array mapping each *new* level index to
    # the index of the original level it should be predicted as: the level
    # itself when it already existed, otherwise its declared fallback.
    collapse = {}
    for factor, factor_info in design_info.factor_infos.items():
        if factor_info.type != "categorical":
            continue
        old_levels = list(factor_info.categories)
        old_index = {}
        for i, level in enumerate(old_levels):
            old_index[level] = i
        if factor in nl_map:
            factor_fallbacks = fb_map.get(factor, {})
            idx = []
            for level in nl_map[factor]:
                if level in old_index:
                    idx.append(old_index[level])
                elif level in factor_fallbacks:
                    idx.append(old_index[factor_fallbacks[level]])
                else:
                    raise PatsyError(
                        "new level %r of factor %r is not one of the original "
                        "levels, so a fallback original level must be declared "
                        "for it" % (level, factor.name())
                    )
            collapse[factor] = np.asarray(idx, dtype=int)
        else:
            collapse[factor] = np.arange(len(old_levels), dtype=int)
    return collapse


def _proto_vectors(num_columns):
    # num_columns+1 vectors in R^num_columns which, together with the constant
    # vector, are linearly independent (a Vandermonde-like construction). Used
    # as stand-in values for numeric factor columns when probing the design's
    # linear structure; the exact values are irrelevant because numeric
    # columns are identical in the old and new designs.
    return [
        np.array([float(v) ** (j + 1) for j in range(num_columns)])
        for v in range(1, num_columns + 2)
    ]


def _probe_assignments(factor_infos, term_codings, full_cross):
    # Yield dicts mapping each factor to a probe value: a level index (int)
    # for categorical factors, or a numeric column vector for numerical
    # factors. Categorical factors are enumerated over all their (new) levels;
    # factors are crossed within each connected component of the
    # "appears-in-a-term-together" graph (or over all factors if full_cross),
    # with remaining factors held at a fixed background value.
    factors = list(factor_infos)
    if full_cross:
        components = [factors]
    else:
        parent = {factor: factor for factor in factors}

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for term in term_codings:
            term_factors = list(term.factors)
            for other in term_factors[1:]:
                parent[find(term_factors[0])] = find(other)
        component_map = OrderedDict()
        for factor in factors:
            component_map.setdefault(find(factor), []).append(factor)
        components = list(component_map.values())

    def background(factor):
        factor_info = factor_infos[factor]
        if factor_info.type == "categorical":
            return 0
        return _proto_vectors(factor_info.num_columns)[0]

    assignments = []
    for component in components:
        cats = [
            f for f in component if factor_infos[f].type == "categorical"
        ]
        nums = [f for f in component if factor_infos[f].type == "numerical"]
        value_lists = [range(len(factor_infos[f].categories)) for f in cats]
        value_lists += [_proto_vectors(factor_infos[f].num_columns) for f in nums]
        for combo in itertools.product(*[list(vs) for vs in value_lists]):
            assignment = {factor: background(factor) for factor in factors}
            for factor, value in zip(cats + nums, combo):
                assignment[factor] = value
            assignments.append(assignment)
    return assignments


def _build_probe_matrix(design_info, assignments, collapse):
    # Assemble the design matrix for the given probe assignments, reusing
    # patsy's own subterm builder so the result matches build_design_matrices
    # exactly. When collapse is given, categorical level indices are first
    # mapped back to original-level indices (evaluating the old design on
    # "collapsed" data).
    n = len(assignments)
    factor_values = {}
    for factor, factor_info in design_info.factor_infos.items():
        if factor_info.type == "categorical":
            indices = np.empty(n, dtype=int)
            for i, assignment in enumerate(assignments):
                level = assignment[factor]
                if collapse is not None:
                    level = collapse[factor][level]
                indices[i] = level
            factor_values[factor] = indices
        else:
            factor_values[factor] = np.asarray(
                [assignment[factor] for assignment in assignments], dtype=float
            )
    out = np.empty((n, len(design_info.column_names)))
    start = 0
    for term, subterms in design_info.term_codings.items():
        for subterm in subterms:
            stop = start + subterm.num_columns
            _build_subterm(
                subterm, design_info.factor_infos, factor_values, out[:, start:stop]
            )
            start = stop
    return out


def _compute_transform(old_design_info, new_design_info, collapse):
    # Find the matrix A such that, for every possible input row,
    #     x_new @ A == x_old_collapsed
    # where x_new is the new design row and x_old_collapsed is the old design
    # row evaluated with each new level replaced by its fallback original
    # level. Then migrating coefficients is just ``coef_new = A @ coef_old``,
    # because x_new @ coef_new == x_new @ A @ coef_old == x_old_collapsed @
    # coef_old for every input.
    p_new = len(new_design_info.column_names)
    p_old = len(old_design_info.column_names)
    if p_new == 0:
        # Degenerate design with no columns at all.
        return np.zeros((0, p_old))
    assignments = _probe_assignments(
        new_design_info.factor_infos, new_design_info.term_codings, full_cross=False
    )
    X_new = _build_probe_matrix(new_design_info, assignments, collapse=None)
    if np.linalg.matrix_rank(X_new) < p_new:
        # The per-component probe set did not exercise every column; fall back
        # to the full cross of all factors, which always spans the design's
        # function space.
        assignments = _probe_assignments(
            new_design_info.factor_infos,
            new_design_info.term_codings,
            full_cross=True,
        )
        X_new = _build_probe_matrix(new_design_info, assignments, collapse=None)
        if np.linalg.matrix_rank(X_new) < p_new:
            raise PatsyError(
                "could not construct a full-rank probe of the new design; "
                "coefficient migration is not possible for this design"
            )
    X_old = _build_probe_matrix(old_design_info, assignments, collapse=collapse)
    transform, _, _, _ = np.linalg.lstsq(X_new, X_old, rcond=None)
    if not np.allclose(X_new @ transform, X_old, rtol=1e-6, atol=1e-8):
        raise PatsyError(
            "the original model's predictions cannot be represented exactly "
            "in the new design; coefficient migration is not possible"
        )
    # For Treatment coding the exact transform is integer-valued; snap to
    # integers so migrated coefficients (and hence predictions) reproduce the
    # original ones as exactly as floating point allows.
    rounded = np.round(transform)
    if np.allclose(transform, rounded, rtol=0.0, atol=1e-8):
        transform = rounded
    return transform


class MigratedDesign(object):
    """The result of :func:`migrate_design`: a fitted design re-expressed
    against a revised categorical dictionary.

    Attributes:

    .. attribute:: design_info

       A :class:`DesignInfo` describing the new design. It can be passed
       directly to :func:`build_design_matrices` to score new data (which may
       use the new levels), with patsy's usual missing-data handling.

    .. attribute:: coefficients

       A 1-d ndarray of coefficients aligned with
       ``design_info.column_names``, such that ``design_matrix @
       coefficients`` reproduces the original model's predictions (with new
       levels predicted as their declared fallbacks).

    .. attribute:: transform

       The matrix ``A`` such that ``coefficients == A @ old_coefficients``.
       Useful if you want to migrate additional coefficient vectors (e.g.
       several regularization paths) without recomputing the transform.

    .. versionadded:: 1.1.0
    """

    def __init__(self, design_info, coefficients, transform=None):
        self.design_info = design_info
        self.coefficients = coefficients
        self.transform = transform

    @property
    def column_names(self):
        "The new design's column names, in order."
        return self.design_info.column_names

    def predict(self, data, NA_action="drop", return_type="matrix", dtype=np.dtype(float)):
        """Predict with the migrated model on new data.

        :arg data: A dict-like object of new data, which may use the new
          categorical levels.
        :arg NA_action: How to handle rows containing missing values, exactly
          as in :func:`build_design_matrices`: ``"drop"`` (the default) drops
          them, ``"raise"`` raises an error, or pass an :class:`NAAction`.
        :arg return_type: ``"matrix"`` (the default) returns a 1-d ndarray of
          predictions; ``"dataframe"`` returns a :class:`pandas.Series` whose
          index reflects any dropped rows.
        :arg dtype: The dtype used when building the design matrix.
        :returns: The predictions ``X @ coefficients`` for the design matrix
          ``X`` built from ``data``.
        """
        (matrix,) = build_design_matrices(
            [self.design_info],
            data,
            NA_action=NA_action,
            return_type=return_type,
            dtype=dtype,
        )
        if return_type == "dataframe":
            return matrix @ self.coefficients
        return np.asarray(matrix) @ self.coefficients

    def __repr__(self):
        return "%s(%r, %r)" % (
            self.__class__.__name__,
            self.design_info.column_names,
            self.coefficients,
        )

    __getstate__ = no_pickling


def migrate_design_info(design_info, new_levels):
    """Rebuild a :class:`DesignInfo` against a revised categorical dictionary.

    This is the structural half of :func:`migrate_design`: every categorical
    factor named in `new_levels` is recoded with patsy's native default
    (:class:`Treatment`) contrasts over the given level order, while numeric
    factors, factor states, and the term/subterm structure are preserved. The
    result is usable directly with :func:`build_design_matrices`.

    :arg design_info: The original :class:`DesignInfo` (e.g. the
      ``.design_info`` attribute of a matrix returned by :func:`dmatrix`).
    :arg new_levels: A dict mapping each categorical factor to migrate --
      identified either by its factor name (``"C(a)"``) or, for a simple
      ``C(a)`` call, by the bare column name (``"a"``) -- to the complete
      sequence of levels in their new order. New levels may be added and old
      levels reordered (or dropped); factors not mentioned are left unchanged.
    :returns: A new :class:`DesignInfo`. The original is not modified.

    Note that this only rebuilds the design metadata; mapping fitted
    coefficients onto the new columns (including fallback behaviour for newly
    added levels) is what :func:`migrate_design` does.

    .. versionadded:: 1.1.0
    """
    _check_design_info(design_info)
    nl_map = _resolve_categorical_factors(design_info, new_levels, "new_levels key")
    nl_map = {
        factor: _validate_new_level_list(factor, levels)
        for factor, levels in nl_map.items()
    }
    return _rebuild_design_info(design_info, nl_map)


def migrate_design(design_info, coefficients, new_levels, fallbacks=None):
    """Migrate a fitted linear model to a revised categorical dictionary,
    without refitting.

    :arg design_info: The :class:`DesignInfo` of the design the model was
      fitted on (e.g. ``fitted_matrix.design_info``).
    :arg coefficients: The fitted coefficients, one per column of
      ``design_info``. Either a 1-d array-like in column order, or a
      :class:`pandas.Series` indexed by the design's column names.
    :arg new_levels: A dict mapping each categorical factor to migrate --
      identified either by its factor name (``"C(a)"``) or, for a simple
      ``C(a)`` call, by the bare column name (``"a"``) -- to the complete
      sequence of levels in their new order. Several categorical factors may
      be migrated at once.
    :arg fallbacks: A dict mapping each migrated factor to a further dict
      ``{new_level: original_level}`` declaring, for every newly added level,
      which original level it should be predicted as. Levels that already
      existed need (and accept) no fallback.
    :returns: A :class:`MigratedDesign` carrying the new :class:`DesignInfo`,
      the migrated coefficients, and a :meth:`~MigratedDesign.predict`
      helper. The original ``design_info`` is left untouched and remains
      usable.

    Guarantees (up to floating-point precision):

    * for any original level and any numeric inputs, predictions are
      identical to the original model;
    * each newly added level predicts exactly like the original level it
      falls back to, including under interactions and when several
      categorical factors are migrated simultaneously.

    Only patsy's native default (:class:`Treatment`) coding of the migrated
    factors is supported; a :class:`PatsyError` is raised otherwise.

    Example::

      import numpy as np
      from patsy import dmatrices, migrate_design

      y, X = dmatrices("y ~ x + C(region)", train, return_type="dataframe")
      beta = np.linalg.lstsq(X, y, rcond=None)[0]

      migrated = migrate_design(
          X.design_info, beta,
          new_levels={"region": ["north", "south", "east", "west", "overseas"]},
          fallbacks={"region": {"overseas": "west"}},
      )
      migrated.predict(new_data)          # uses the new dictionary
      # or, equivalently, by hand:
      #   X_new = build_design_matrices([migrated.design_info], new_data)[0]
      #   y_hat = np.asarray(X_new) @ migrated.coefficients

    .. versionadded:: 1.1.0
    """
    _check_design_info(design_info)
    if fallbacks is None:
        fallbacks = {}
    nl_map = _resolve_categorical_factors(design_info, new_levels, "new_levels key")
    fb_map = _resolve_categorical_factors(design_info, fallbacks, "fallbacks key")
    nl_map = {
        factor: _validate_new_level_list(factor, levels)
        for factor, levels in nl_map.items()
    }
    _validate_fallbacks(design_info, nl_map, fb_map)
    _check_treatment_coding(design_info, nl_map)
    coef = _validate_coefficients(design_info, coefficients)
    new_design_info = _rebuild_design_info(design_info, nl_map)
    collapse = _collapse_indices(design_info, nl_map, fb_map)
    transform = _compute_transform(design_info, new_design_info, collapse)
    new_coefficients = transform @ coef
    return MigratedDesign(new_design_info, new_coefficients, transform=transform)


def _check_design_info(design_info):
    if not isinstance(design_info, DesignInfo):
        raise PatsyError("expected a DesignInfo object, got %r" % (design_info,))
    if design_info.term_codings is None or design_info.factor_infos is None:
        raise PatsyError(
            "design_info does not carry term/factor metadata; use the "
            "DesignInfo produced by dmatrix/dmatrices/design_matrix_builders"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _fit(formula, data, seed=0):
    rng = np.random.RandomState(seed)
    matrix = dmatrix(formula, data)
    coef = rng.randn(len(matrix.design_info.column_names))
    return matrix.design_info, coef


def _collapsed_data(data, fallbacks):
    out = dict(data)
    for column, factor_fallbacks in fallbacks.items():
        out[column] = [
            value if value not in factor_fallbacks else factor_fallbacks[value]
            for value in out[column]
        ]
    return out


def _check_migration(
    formula,
    train_data,
    new_levels,
    fallbacks,
    test_data,
    coef=None,
    seed=0,
):
    from patsy import dmatrix, build_design_matrices

    design_info, fitted_coef = _fit(formula, train_data, seed=seed)
    if coef is not None:
        fitted_coef = coef
    migrated = migrate_design(design_info, fitted_coef, new_levels, fallbacks)
    # Reference: collapse new levels to their fallbacks, predict with the
    # original model.
    collapsed = _collapsed_data(test_data, fallbacks)
    expected = np.asarray(build_design_matrices([design_info], collapsed)[0]) @ fitted_coef
    got = migrated.predict(test_data)
    assert np.allclose(got, expected, rtol=1e-8, atol=1e-10)
    # And the new design_info must be directly usable with
    # build_design_matrices, giving the same predictions.
    direct = np.asarray(build_design_matrices([migrated.design_info], test_data)[0])
    assert np.allclose(direct @ migrated.coefficients, expected, rtol=1e-8, atol=1e-10)
    return design_info, migrated


def test_migrate_design_additive_reorder_and_add():
    rng = np.random.RandomState(1)
    train = {
        "x": rng.randn(40),
        "a": rng.choice(["a1", "a2", "a3"], 40),
    }
    test = {"x": rng.randn(50), "a": rng.choice(["a1", "a2", "a3", "a_new"], 50)}
    # reorder levels (a2 becomes the reference) and add a_new
    _check_migration(
        "1 + x + C(a)",
        train,
        {"a": ["a2", "a1", "a3", "a_new"]},
        {"a": {"a_new": "a1"}},
        test,
    )


def test_migrate_design_new_level_as_reference():
    rng = np.random.RandomState(2)
    train = {"a": rng.choice(["a1", "a2", "a3"], 40)}
    test = {"a": rng.choice(["a1", "a2", "a3", "a_new"], 60)}
    # the brand-new level becomes the reference level of the new dictionary
    _check_migration(
        "1 + C(a)",
        train,
        {"a": ["a_new", "a1", "a2", "a3"]},
        {"a": {"a_new": "a2"}},
        test,
    )


def test_migrate_design_full_rank():
    rng = np.random.RandomState(3)
    train = {"a": rng.choice(["a1", "a2", "a3"], 40)}
    test = {"a": rng.choice(["a1", "a2", "a3", "a_new"], 60)}
    _check_migration(
        "0 + C(a)",
        train,
        {"a": ["a3", "a2", "a1", "a_new"]},
        {"a": {"a_new": "a3"}},
        test,
    )


def test_migrate_design_categorical_interaction_two_factors():
    rng = np.random.RandomState(4)
    train = {
        "a": rng.choice(["a1", "a2", "a3"], 60),
        "b": rng.choice(["b1", "b2"], 60),
    }
    test = {
        "a": rng.choice(["a1", "a2", "a3", "a_new"], 80),
        "b": rng.choice(["b1", "b2", "b_new"], 80),
    }
    # both categorical variables gain a level and are reordered, simultaneously
    _check_migration(
        "1 + C(a) + C(b) + C(a):C(b)",
        train,
        {"a": ["a2", "a1", "a3", "a_new"], "b": ["b2", "b1", "b_new"]},
        {"a": {"a_new": "a2"}, "b": {"b_new": "b1"}},
        test,
    )


def test_migrate_design_categorical_numeric_interaction():
    rng = np.random.RandomState(5)
    train = {"x": rng.randn(60), "a": rng.choice(["a1", "a2", "a3"], 60)}
    test = {"x": rng.randn(80), "a": rng.choice(["a1", "a2", "a3", "a_new"], 80)}
    _check_migration(
        "1 + x + C(a) + C(a):x",
        train,
        {"a": ["a3", "a1", "a2", "a_new"]},
        {"a": {"a_new": "a3"}},
        test,
    )


def test_migrate_design_numeric_numeric_interaction():
    rng = np.random.RandomState(6)
    train = {
        "x": rng.randn(60),
        "z": rng.randn(60),
        "a": rng.choice(["a1", "a2", "a3"], 60),
    }
    test = {
        "x": rng.randn(80),
        "z": rng.randn(80),
        "a": rng.choice(["a1", "a2", "a3", "a_new"], 80),
    }
    _check_migration(
        "1 + x + z + x:z + C(a)",
        train,
        {"a": ["a2", "a1", "a3", "a_new"]},
        {"a": {"a_new": "a1"}},
        test,
    )


def test_migrate_design_multiple_new_levels():
    rng = np.random.RandomState(7)
    train = {"a": rng.choice(["a1", "a2", "a3"], 60)}
    test = {"a": rng.choice(["a1", "a2", "a3", "n1", "n2"], 100)}
    _check_migration(
        "1 + C(a)",
        train,
        {"a": ["a1", "a2", "a3", "n1", "n2"]},
        {"a": {"n1": "a1", "n2": "a3"}},
        test,
    )


def test_migrate_design_old_levels_unchanged_for_any_numeric_input():
    # Explicitly check the two headline guarantees separately.
    rng = np.random.RandomState(8)
    train = {"x": rng.randn(60), "a": rng.choice(["a1", "a2", "a3"], 60)}
    design_info, coef = _fit("1 + x + C(a) + C(a):x", train, seed=8)
    migrated = migrate_design(
        design_info,
        coef,
        {"a": ["a2", "a1", "a3", "a_new"]},
        {"a": {"a_new": "a1"}},
    )
    from patsy import build_design_matrices

    # (1) old levels: predictions unchanged for arbitrary numeric input
    xs = rng.randn(200)
    for level in ["a1", "a2", "a3"]:
        data = {"x": xs, "a": [level] * 200}
        old_pred = np.asarray(build_design_matrices([design_info], data)[0]) @ coef
        assert np.allclose(migrated.predict(data), old_pred, rtol=1e-8, atol=1e-10)
    # (2) new level: equals the original model at its declared fallback
    data_new = {"x": xs, "a": ["a_new"] * 200}
    data_fb = {"x": xs, "a": ["a1"] * 200}
    old_pred_fb = np.asarray(build_design_matrices([design_info], data_fb)[0]) @ coef
    assert np.allclose(migrated.predict(data_new), old_pred_fb, rtol=1e-8, atol=1e-10)


def test_migrate_design_native_design_info():
    # The rebuilt design must be exactly what patsy natively produces for the
    # new dictionary (verified against an ordered pandas.Categorical, the
    # canonical native way to fix a level order).
    from patsy.util import have_pandas

    rng = np.random.RandomState(9)
    train = {
        "x": rng.randn(60),
        "a": rng.choice(["a1", "a2", "a3"], 60),
        "b": rng.choice(["b1", "b2"], 60),
    }
    design_info, coef = _fit("1 + x + C(a) + C(b):x", train, seed=9)
    new_levels = {"a": ["a2", "a1", "a3", "a_new"], "b": ["b2", "b1", "b_new"]}
    migrated = migrate_design(
        design_info,
        coef,
        new_levels,
        {"a": {"a_new": "a1"}, "b": {"b_new": "b2"}},
    )
    assert migrated.design_info is not design_info
    if have_pandas:
        import pandas

        from patsy import dmatrix

        native_data = {
            "x": train["x"],
            "a": pandas.Categorical(train["a"], categories=new_levels["a"]),
            "b": pandas.Categorical(train["b"], categories=new_levels["b"]),
        }
        native = dmatrix("1 + x + C(a) + C(b):x", native_data).design_info
        assert migrated.design_info.column_names == native.column_names
        assert migrated.design_info.term_names == native.term_names
        for subterms_m, subterms_n in zip(
            migrated.design_info.term_codings.values(),
            native.term_codings.values(),
        ):
            for st_m, st_n in zip(subterms_m, subterms_n):
                suffixes_n = {
                    tuple(cm.column_suffixes) for cm in st_n.contrast_matrices.values()
                }
                suffixes_m = {
                    tuple(cm.column_suffixes) for cm in st_m.contrast_matrices.values()
                }
                assert suffixes_m == suffixes_n
                for cm_m in st_m.contrast_matrices.values():
                    for cm_n in st_n.contrast_matrices.values():
                        if cm_n.column_suffixes == cm_m.column_suffixes:
                            assert np.array_equal(cm_m.matrix, cm_n.matrix)


def test_migrate_design_na_semantics():
    import pytest

    rng = np.random.RandomState(10)
    train = {"x": rng.randn(40), "a": rng.choice(["a1", "a2", "a3"], 40)}
    design_info, coef = _fit("1 + x + C(a)", train, seed=10)
    migrated = migrate_design(
        design_info, coef, {"a": ["a2", "a1", "a3", "a_new"]}, {"a": {"a_new": "a1"}}
    )
    data = {"x": [1.0, np.nan, 3.0], "a": ["a1", "a_new", "a2"]}
    # native "drop": the NA row is dropped
    pred = migrated.predict(data, NA_action="drop")
    assert pred.shape == (2,)
    # native "raise": a PatsyError is raised
    pytest.raises(PatsyError, migrated.predict, data, NA_action="raise")
    # dataframe return type exposes the surviving index
    from patsy.util import have_pandas

    if have_pandas:
        import pandas

        s = migrated.predict(
            pandas.DataFrame(data), NA_action="drop", return_type="dataframe"
        )
        assert list(s.index) == [0, 2]


def test_migrate_design_old_design_still_usable():
    rng = np.random.RandomState(11)
    train = {"x": rng.randn(40), "a": rng.choice(["a1", "a2", "a3"], 40)}
    design_info, coef = _fit("1 + x + C(a)", train, seed=11)
    before_columns = list(design_info.column_names)
    before_categories = {
        f.name(): fi.categories
        for f, fi in design_info.factor_infos.items()
        if fi.type == "categorical"
    }
    migrated = migrate_design(
        design_info, coef, {"a": ["a2", "a1", "a3", "a_new"]}, {"a": {"a_new": "a1"}}
    )
    # original design_info untouched and still builds matrices
    assert design_info.column_names == before_columns
    assert {
        f.name(): fi.categories
        for f, fi in design_info.factor_infos.items()
        if fi.type == "categorical"
    } == before_categories
    from patsy import build_design_matrices

    data = {"x": [1.0, 2.0], "a": ["a1", "a3"]}
    old_pred = np.asarray(build_design_matrices([design_info], data)[0]) @ coef
    assert old_pred.shape == (2,)
    assert migrated.design_info is not design_info


def test_migrate_design_pandas_series_coefficients():
    from patsy.util import have_pandas

    if not have_pandas:
        return
    import pandas

    rng = np.random.RandomState(12)
    train = {"x": rng.randn(40), "a": rng.choice(["a1", "a2", "a3"], 40)}
    design_info, coef = _fit("1 + x + C(a)", train, seed=12)
    series = pandas.Series(coef, index=design_info.column_names)
    # shuffled order: must be aligned by name
    shuffled = series.iloc[::-1]
    migrated = migrate_design(
        design_info, shuffled, {"a": ["a2", "a1", "a3", "a_new"]}, {"a": {"a_new": "a1"}}
    )
    migrated_positional = migrate_design(
        design_info, coef, {"a": ["a2", "a1", "a3", "a_new"]}, {"a": {"a_new": "a1"}}
    )
    assert np.allclose(migrated.coefficients, migrated_positional.coefficients)
    # wrong labels -> error
    import pytest

    bad = pandas.Series(coef, index=["x%d" % i for i in range(len(coef))])
    pytest.raises(PatsyError, migrate_design, design_info, bad, {"a": ["a1", "a2", "a3"]})


def test_migrate_design_errors():
    import pytest

    rng = np.random.RandomState(13)
    train = {"x": rng.randn(40), "a": rng.choice(["a1", "a2", "a3"], 40)}
    design_info, coef = _fit("1 + x + C(a)", train, seed=13)
    levels = {"a": ["a1", "a2", "a3", "a_new"]}
    fb = {"a": {"a_new": "a1"}}

    # unknown factor
    pytest.raises(PatsyError, migrate_design, design_info, coef, {"nope": ["a1"]})
    # new level without a declared fallback
    pytest.raises(PatsyError, migrate_design, design_info, coef, levels)
    # fallback target is not an original level
    pytest.raises(
        PatsyError, migrate_design, design_info, coef, levels, {"a": {"a_new": "zz"}}
    )
    # fallback declared for a level that already existed
    pytest.raises(
        PatsyError, migrate_design, design_info, coef, levels, {"a": {"a1": "a2", "a_new": "a1"}}
    )
    # fallback for a factor that is not being migrated
    pytest.raises(
        PatsyError, migrate_design, design_info, coef, levels, {"x": {"a_new": "a1"}}
    )
    # duplicate new levels
    pytest.raises(
        PatsyError, migrate_design, design_info, coef, {"a": ["a1", "a1"]}
    )
    # new levels given as a bare string
    pytest.raises(PatsyError, migrate_design, design_info, coef, {"a": "a1"})
    # empty new levels
    pytest.raises(PatsyError, migrate_design, design_info, coef, {"a": []})
    # wrong number of coefficients
    pytest.raises(PatsyError, migrate_design, design_info, coef[:-1], levels, fb)
    # non-default contrast on the migrated factor
    sum_design = dmatrix("1 + C(a, Sum)", train).design_info
    sum_coef = np.zeros(len(sum_design.column_names))
    pytest.raises(
        PatsyError, migrate_design, sum_design, sum_coef, {"a": ["a1", "a2", "a3"]}
    )
    # not a full DesignInfo
    pytest.raises(
        PatsyError, migrate_design, DesignInfo(["a", "b"]), [1.0, 2.0], {"a": ["a1"]}
    )


def test_migrate_design_info():
    rng = np.random.RandomState(14)
    train = {"x": rng.randn(40), "a": rng.choice(["a1", "a2", "a3"], 40)}
    design_info = dmatrix("1 + x + C(a)", train).design_info
    new_di = migrate_design_info(design_info, {"a": ["a2", "a1", "a3", "a_new"]})
    assert new_di.column_names == ["Intercept", "C(a)[T.a1]", "C(a)[T.a3]", "C(a)[T.a_new]", "x"]
    assert design_info.column_names == ["Intercept", "C(a)[T.a2]", "C(a)[T.a3]", "x"]
    # original untouched
    assert design_info.factor_infos != new_di.factor_infos
    import pytest

    pytest.raises(PatsyError, migrate_design_info, design_info, {"unknown": ["a1"]})


def test_migratedDesign_no_pickling():
    rng = np.random.RandomState(15)
    train = {"a": rng.choice(["a1", "a2"], 20)}
    design_info, coef = _fit("1 + C(a)", train, seed=15)
    migrated = migrate_design(
        design_info, coef, {"a": ["a1", "a2", "a_new"]}, {"a": {"a_new": "a1"}}
    )
    assert_no_pickling(migrated)
