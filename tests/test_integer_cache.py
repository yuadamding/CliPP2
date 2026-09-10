"""Partition refit reuse is bound to the retained numerical input identity."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion.torch_backend import resolve_runtime
from CliPP2.core.fusion.partition_starts import generate_partition_initializer_pool
from CliPP2.core.objective import compile_observed_model, model_to_torch
from CliPP2.io.data import CNFilterRecord, CNFilterReport, tumor_data_fingerprint
from CliPP2.model_selection import candidates
from CliPP2.model_selection.partitions import _partition_signature
from test_integer_likelihood import integer_data
from test_curvature_preparation import _context


def _key(data):
    labels = np.zeros(data.num_mutations, dtype=np.int64)
    return candidates._selection_refit_cache_key(
        data=data,
        partition_signature=_partition_signature(labels, tuple(data.mutation_ids)),
        selection_options=resolve_fit_config(device="cpu"),
    )


@pytest.mark.parametrize("change", ["counts", "candidates", "observed", "bounds"])
def test_numeric_changes_invalidate_same_family_refit_cache(change):
    original = integer_data(((4,),))
    changed = replace(original)
    if change == "counts":
        changed = replace(original, alt_counts=original.alt_counts + 1)
    elif change == "candidates":
        changed = integer_data(((3,),))
    elif change == "observed":
        changed = replace(original, count_observed=np.zeros_like(original.alt_counts, dtype=bool))
    else:
        changed = replace(original, phi_upper=original.phi_upper * 0.9)
    assert compile_observed_model(original, eps=1e-6).model_id == compile_observed_model(changed, eps=1e-6).model_id
    assert _key(original) != _key(changed)


def test_removed_integer_specification_cannot_enter_a_cache():
    data = integer_data(((4,),))
    with pytest.raises(TypeError, match="path_likelihood"):
        replace(data, path_likelihood=object())


def test_fingerprint_is_computed_once_and_replacements_start_fresh(monkeypatch):
    from CliPP2.io import data as data_module

    original = integer_data(((4,),))
    model = compile_observed_model(original, eps=1e-6)
    original_hash = tumor_data_fingerprint(original)
    compute = data_module._retained_data_fingerprint
    calls = []

    def record(data):
        calls.append(data)
        return compute(data)

    monkeypatch.setattr(data_module, "_retained_data_fingerprint", record)
    assert tumor_data_fingerprint(original) == original_hash
    assert tumor_data_fingerprint(original) == original_hash
    assert not calls
    changed = replace(original, alt_counts=original.alt_counts + 1)
    assert len(calls) == 1 and calls[0] is changed
    assert not changed._compiled_models
    assert tumor_data_fingerprint(changed) != original_hash
    assert compile_observed_model(original, eps=1e-6) is model
    assert compile_observed_model(changed, eps=1e-6) is not model


@pytest.mark.parametrize("change", ["reads", "bounds", "eps"])
def test_same_shape_source_model_must_match_requested_data(change):
    data = integer_data(((4,),))
    changed = replace(data, alt_counts=data.alt_counts + 1) if change == "reads" else data
    source = compile_observed_model(changed, eps=0.01 if change == "eps" else 1e-6)
    if change == "bounds":
        source = replace(source, upper=source.upper * 0.8)
    runtime = resolve_runtime("cpu", dtype="float64")
    tensor = model_to_torch(source, runtime, eps=.01 if change == "eps" else 1e-6)
    context = replace(_context(data), source_model=source, model=tensor, _tensor_snapshot=())
    with pytest.raises(ValueError, match="likelihood or epsilon"):
        generate_partition_initializer_pool(
            context=context, pilot_phi=context.exact_pilot,
            fit_options=resolve_fit_config(device="cpu", dtype="float64"),
        )


def test_forged_runtime_input_label_cannot_authorize_wrong_source_model():
    from test_solver_request import _prepared
    data, context = _prepared()
    changed = replace(data, alt_counts=data.alt_counts + 1)
    runtime = resolve_runtime("cpu", dtype="float64")
    wrong = compile_observed_model(changed, eps=1e-6)
    forged = replace(context, source_model=wrong, model=model_to_torch(wrong, runtime, eps=1e-6),
                     _tensor_snapshot=())
    with pytest.raises(ValueError, match="likelihood or epsilon"):
        forged.validate()
    with pytest.raises(ValueError, match="likelihood or epsilon"):
        replace(context, source_model=None).validate()


def test_runtime_validation_supports_matching_nondefault_eps():
    data = integer_data(((4,),))
    context = _context(data, eps=.01)
    assert generate_partition_initializer_pool(
        context=context, pilot_phi=context.exact_pilot,
        fit_options=resolve_fit_config(device="cpu", dtype="float64", eps=.01),
    )
    with pytest.raises(ValueError, match="prepared likelihood epsilon"):
        generate_partition_initializer_pool(
            context=context, pilot_phi=context.exact_pilot,
            fit_options=resolve_fit_config(device="cpu", dtype="float64"),
        )


def test_exclusion_provenance_does_not_invalidate_retained_refit_cache():
    data = integer_data(((4,),))
    report = CNFilterReport(
        policy_id="clonal_cn_major_le6_whole_mutation_v1",
        input_mutation_count=2,
        retained_mutation_count=1,
        excluded_mutation_ids=("excluded",),
        records=(CNFilterRecord("excluded", "r0", "s0", "MAJOR_CN_GT_6", 1, 7),),
    )
    assert _key(data) == _key(replace(data, cn_filter_report=report))


def test_refit_cache_hit_never_reuses_results_for_changed_counts(monkeypatch):
    original = integer_data(((4,),))
    changed = replace(original, alt_counts=original.alt_counts + 1)
    calls = []

    def fake_refit(data, labels, **kwargs):
        calls.append(float(data.alt_counts[0, 0]))
        return SimpleNamespace(
            global_optimality_gap=0.0,
            loglik=-float(data.alt_counts[0, 0]),
            finite_candidate_found=True,
            refit_finite_coordinate_count=1,
            refit_coordinate_count=1,
            global_optimum_certified=True,
        )

    monkeypatch.setattr(candidates, "partition_constrained_observed_refit", fake_refit)
    options = resolve_fit_config(device="cpu")
    labels = np.array([0])
    signature = _partition_signature(labels, tuple(original.mutation_ids))
    cache = {}

    def refit(data):
        return candidates._fixed_labels_refit(
            data=data, labels=labels, partition_signature=signature,
            selection_options=options, cache=cache,
        )

    first = refit(original)
    assert refit(original) is first
    second = refit(changed)
    assert second is not first
    assert second.result.loglik != first.result.loglik
    assert refit(changed) is second
    assert calls == [24.0, 25.0]
    assert len(cache) == 2
