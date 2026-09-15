"""Regression coverage for the confirmed functionality-review findings."""
from __future__ import annotations

import math

import pytest

from mathkernel import MathKernel
from mathkernel.models import RealNode, TrustLevel
from mathkernel.numerics import compile_float64


@pytest.mark.parametrize(("source", "expected", "precision"), [
    ("1e-8", 1e-8, 1),
    ("6.582119569e-16", 6.582119569e-16, 10),
    ("1E+8", 1e8, 1),
])
def test_scientific_literals_remain_approximate(source, expected, precision):
    kernel = MathKernel()
    parsed = kernel.parse(source)
    assert parsed.ok and parsed.trust == TrustLevel.NUMERIC
    node = kernel.expressions[parsed.data["expr_id"]]
    assert isinstance(node, RealNode)
    assert node.value == source and node.precision == precision
    evaluated = kernel.numeric_evaluate(parsed.data["expr_id"])
    assert float(evaluated.data["value"]) == pytest.approx(expected)


def test_pi_call_and_user_symbol_have_backend_parity():
    kernel = MathKernel()
    pi_id = kernel.parse("2*pi()").data["expr_id"]
    assert compile_float64(kernel.expressions[pi_id], [])() == pytest.approx(2 * math.pi)
    numeric = kernel.numeric_evaluate(pi_id, dps=40)
    assert float(numeric.data["value"]) == pytest.approx(2 * math.pi)

    symbol_id = kernel.parse("pi").data["expr_id"]
    assert not kernel.numeric_evaluate(symbol_id).ok
    assert compile_float64(kernel.expressions[symbol_id], ["pi"])(2.5) == 2.5
    with pytest.raises(ValueError, match="unbound symbol"):
        compile_float64(kernel.expressions[symbol_id], [])


def test_equivalence_excludes_undefined_denominator_witnesses():
    kernel = MathKernel()
    result = kernel.prove_equivalence("x/(d*c)", "2*x/(2*d*c)", formal=False)
    assert result.status == "verified"
    assert any("d*c" in item.replace(" ", "") for item in result.data["admissibility_constraints"])

    refuted = kernel.prove_equivalence("1/x", "2/x", formal=False)
    assert refuted.status == "refuted"
    assert refuted.data["counterexample"]["x"] != "0"
    z3_evidence = next(item for item in refuted.evidence if item.engine == "z3")
    assert z3_evidence.detail["witness_validated"] is True


def test_positive_domains_are_constraints_in_equivalence():
    kernel = MathKernel()
    context = kernel.create_context(domains={"D": "positive", "c": "positive"})
    result = kernel.prove_equivalence(
        "(B*r*A/(D*c))^2",
        "12*B^2*(r^2*A^2/12)/(D^2*c^2)",
        context_id=context.context_id,
        formal=False,
    )
    assert result.status == "verified"


def test_adaptive_trajectory_matches_analytic_solution():
    kernel = MathKernel()
    rhs = kernel.parse("y0").data["expr_id"]
    result = kernel.ode_solve_numeric(
        [rhs], ["0", "1"], ["1"], t_eval=["0", "0.25", "0.5", "1"],
        rtol=1e-12, atol=1e-14, max_step=0.1,
    )
    assert result.ok and result.data["solver_status"] == "converged"
    trajectory = result.data["trajectory"]
    assert [float(row[0]) for row in trajectory["y"]] == pytest.approx(
        [math.exp(float(t)) for t in trajectory["t"]], rel=1e-10)
    assert result.data["controls"]["rtol"] == "0.000000000001"
    assert result.data["error_estimates"]["integration"]["method"]
    assert result.data["error_estimates"]["interpolation"]["method"] == "cubic-hermite"


def test_fixed_step_controls_and_ignored_tolerances():
    kernel = MathKernel()
    rhs = kernel.parse("y0").data["expr_id"]
    result = kernel.ode_solve_numeric(
        [rhs], ["0", "1"], ["1"], fast=True, steps=200,
        t_eval=[0, 0.5, 1],
    )
    assert result.ok and result.data["accepted_steps"] == 200
    assert float(result.data["trajectory"]["y"][1][0]) == pytest.approx(math.exp(0.5), rel=1e-8)
    rejected = kernel.ode_solve_numeric([rhs], ["0", "1"], ["1"], fast=True, tol=1e-6)
    assert not rejected.ok and "does not use" in rejected.errors[0]


def test_adaptive_float64_route_and_controls():
    pytest.importorskip("scipy")
    kernel = MathKernel()
    rhs = kernel.parse("y0").data["expr_id"]
    result = kernel.ode_solve_numeric(
        [rhs], ["0", "1"], ["1"], method="DOP853", rtol=1e-10,
        atol=1e-12, max_step=0.2, t_eval=[0, 0.5, 1],
    )
    assert result.ok and result.data["method"] == "DOP853"
    assert float(result.data["trajectory"]["y"][1][0]) == pytest.approx(math.exp(0.5), rel=1e-9)


def test_sampled_quadrature_irregular_grid_axis_and_cumulative():
    kernel = MathKernel()
    result = kernel.sampled_quadrature(
        [0, 0.5, 2], [[0, 0.25, 4], [0, 1, 2]], axis=1)
    assert result.ok and result.trust == TrustLevel.NUMERIC
    assert result.data["value"] == pytest.approx([3.25, 2.5])
    assert result.data["interpolation"] == "piecewise-linear"
    assert result.data["error_estimate"] is None

    cumulative = kernel.sampled_quadrature([0, 0.5, 2], [0, 0.25, 4], cumulative=True)
    assert cumulative.data["values"] == pytest.approx([0, 0.0625, 3.25])


@pytest.mark.parametrize("x", [[0, 0, 1], [0, 2, 1]])
def test_sampled_quadrature_rejects_duplicates_and_unordered_data(x):
    result = MathKernel().sampled_quadrature(x, [0, 1, 2])
    assert not result.ok and "strictly monotonic" in result.errors[0]


def test_documented_python_api_names_exist():
    kernel = MathKernel()
    expression_id = kernel.parse("x + 1").data["expr_id"]
    assert kernel.get_expression(expression_id).ok
    assert kernel.create_context(domains={"x": "positive"}).context_id
