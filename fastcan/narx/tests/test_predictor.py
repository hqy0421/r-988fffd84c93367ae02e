"""Test the stateful NARXPredictor."""

import numpy as np
import pytest
import sklearn
from numpy.testing import assert_array_equal

from fastcan.narx import NARX, NARXPredictor, make_narx


def _make_data(n_samples=120, multi_output=False):
    """Synthetic NARX-like data."""
    rng = np.random.default_rng(12345)
    max_delay = 3
    u0 = rng.uniform(0, 1, n_samples + max_delay)
    u1 = rng.normal(0, 0.1, n_samples)
    if multi_output:
        e0 = rng.normal(0, 0.1, n_samples)
        e1 = rng.normal(0, 0.02, n_samples)
        y0 = np.zeros(n_samples + max_delay)
        y1 = np.zeros(n_samples + max_delay)
        for i in range(max_delay, n_samples + max_delay):
            y0[i] = 0.5 * y0[i - 1] + 0.8 * y1[i - 1] + 0.3 * u0[i] ** 2 + 1
            y1[i] = 0.6 * y1[i - 1] - 0.2 * y0[i - 1] * y1[i - 2] + 0.5
        y = np.c_[y0[max_delay:] + e0, y1[max_delay:] + e1]
        X = np.c_[u0[max_delay:], u1]
    else:
        e = rng.normal(0, 0.1, n_samples)
        y = np.zeros(n_samples + max_delay)
        for i in range(max_delay, n_samples + max_delay):
            y[i] = 0.5 * y[i - 1] + 0.3 * u0[i] ** 2 + 2 * u0[i - 1] * u0[i - 3] + 1
        y = y[max_delay:] + e
        X = np.c_[u0[max_delay:], u1]
    return X, y


def _fit_narx(multi_output):
    X, y = _make_data(multi_output=multi_output)
    narx = make_narx(
        X,
        y,
        n_terms_to_select=3 if multi_output else 4,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    ).fit(X, y, coef_init="one_step_ahead")
    return narx, X, y


def _concat(parts, ndim):
    if ndim == 1:
        return np.r_[tuple(parts)]
    return np.concatenate(parts)


@pytest.mark.parametrize("multi_output", [False, True])
@pytest.mark.parametrize(
    "splits",
    [
        [120],  # single batch equals NARX.predict
        [60, 60],
        [1, 1, 1, 5, 17, 95],  # splits inside the warm-up region
        [2, 3, 4, 10, 101],
        [1] * 120,  # one sample per call
    ],
)
def test_batches_equal_one_shot(multi_output, splits):
    """Batched prediction through a predictor equals a one-shot prediction."""
    narx, X, y = _fit_narx(multi_output)
    y_init = y[: narx.max_delay_]
    one_shot = narx.predict(X, y_init=y_init)

    predictor = narx.init_predictor(y_init=y_init)
    parts = []
    start = 0
    for size in splits:
        parts.append(predictor.predict(X[start : start + size]))
        start += size
    batched = _concat(parts, one_shot.ndim)

    # Exact equality: both run the same Cython recursion.
    assert_array_equal(batched, one_shot)
    assert predictor.n_predicted_ == X.shape[0]


def test_predictor_class_and_factory_equivalent():
    """NARXPredictor(narx) and narx.init_predictor() behave identically."""
    narx, X, y = _fit_narx(False)
    y_init = y[: narx.max_delay_]
    p1 = NARXPredictor(narx, y_init=y_init)
    p2 = narx.init_predictor(y_init=y_init)
    assert_array_equal(p1.predict(X[:25]), p2.predict(X[:25]))


def test_interleaved_sessions_do_not_contaminate():
    """Two predictors (two data sources) keep independent states."""
    narx, X, y = _fit_narx(False)
    y_init = y[: narx.max_delay_]
    rng = np.random.default_rng(0)
    X_other = rng.random(X.shape)
    other_init = np.zeros_like(y_init)

    line_a = NARXPredictor(narx, y_init=y_init)
    line_b = NARXPredictor(narx, y_init=other_init)
    parts_a = []
    parts_b = []
    for lo in range(0, X.shape[0], 13):
        hi = min(lo + 13, X.shape[0])
        parts_b.append(line_b.predict(X_other[lo:hi]))
        parts_a.append(line_a.predict(X[lo:hi]))

    assert_array_equal(np.r_[tuple(parts_a)], narx.predict(X, y_init=y_init))
    assert_array_equal(np.r_[tuple(parts_b)], narx.predict(X_other, y_init=other_init))
    # The underlying fitted model must remain untouched.
    assert not hasattr(narx, "_x_hist")


def test_reset_starts_new_session():
    """reset() consumes new initial conditions on a line change."""
    narx, X, y = _fit_narx(False)
    rng = np.random.default_rng(7)
    predictor = NARXPredictor(narx, y_init=y[: narx.max_delay_])
    predictor.predict(X[:40])

    new_init = rng.random((narx.max_delay_, narx.n_outputs_))
    returned = predictor.reset(y_init=new_init)
    assert returned is predictor
    assert predictor.n_predicted_ == 0

    expected = narx.predict(X[40:], y_init=new_init)
    assert_array_equal(predictor.predict(X[40:]), expected)


def test_reset_default_zeros():
    """reset() without arguments uses zero initial conditions."""
    narx, X, y = _fit_narx(False)
    predictor = NARXPredictor(narx, y_init=y[: narx.max_delay_])
    predictor.predict(X[:10])
    predictor.reset()
    assert_array_equal(
        predictor.predict(X[10:20]),
        narx.predict(X[10:20], y_init=np.zeros((narx.max_delay_, 1))),
    )


def test_auto_regressive_integer_x():
    """AR models predict a number of steps via integer X."""
    _X, y = _make_data()
    arm = make_narx(
        None,
        y,
        n_terms_to_select=3,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    ).fit(None, y, coef_init="one_step_ahead")
    y_init = y[: arm.max_delay_]
    one_shot = arm.predict(len(y), y_init=y_init)

    predictor = NARXPredictor(arm, y_init=y_init)
    batched = np.r_[
        predictor.predict(7),
        predictor.predict(50),
        predictor.predict(len(y) - 57),
    ]
    assert_array_equal(batched, one_shot)


def test_short_y_init_cyclic_semantics():
    """A y_init shorter than max_delay keeps NARX.predict cyclic semantics."""
    _X, y = _make_data()
    arm = make_narx(
        None,
        y,
        n_terms_to_select=3,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    ).fit(None, y, coef_init="one_step_ahead")
    y_init = y[:2]  # shorter than max_delay
    one_shot = arm.predict(len(y), y_init=y_init)

    predictor = NARXPredictor(arm, y_init=y_init)
    batched = np.r_[
        predictor.predict(1),
        predictor.predict(40),
        predictor.predict(len(y) - 41),
    ]
    assert_array_equal(batched, one_shot)


def test_zero_max_delay():
    """A static (max_delay == 0) model chains batches without any state."""
    X, y = _make_data()
    narx = NARX().fit(X, y)
    assert narx.max_delay_ == 0
    predictor = NARXPredictor(narx)
    one_shot = narx.predict(X)
    batched = np.r_[predictor.predict(X[:10]), predictor.predict(X[10:])]
    assert_array_equal(batched, one_shot)


@pytest.mark.parametrize("ndim_y", [1, 2])
def test_predict_output_ndim(ndim_y):
    """The predictor preserves the 1d/2d output shape convention."""
    X, y = _make_data()
    y_fit = y if ndim_y == 1 else y.reshape(-1, 1)
    narx = make_narx(
        X,
        y_fit,
        n_terms_to_select=3,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    ).fit(X, y_fit)
    predictor = NARXPredictor(narx, y_init=y_fit[: narx.max_delay_])
    assert predictor.predict(X[:5]).ndim == ndim_y


def test_validation_errors():
    """Wrong y_init shape, empty y_init, and integer X for exogenous model."""
    narx, _X, y = _fit_narx(False)
    with pytest.raises(ValueError, match="`y_init` should have"):
        NARXPredictor(narx, y_init=np.ones((3, 2)))

    arm = make_narx(
        None,
        y,
        n_terms_to_select=3,
        max_delay=3,
        poly_degree=2,
        verbose=0,
    ).fit(None, y)
    with pytest.raises(ValueError, match="at least have one sample"):
        NARXPredictor(arm, y_init=np.empty((0, 1)))

    with pytest.raises(ValueError, match="integer"):
        NARXPredictor(narx).predict(5)

    with pytest.raises(sklearn.exceptions.NotFittedError):
        NARXPredictor(NARX())


def test_nan_input_raises_without_corrupting_state():
    """A NaN batch is rejected before the session state is advanced."""
    narx, X, y = _fit_narx(False)
    predictor = NARXPredictor(narx, y_init=y[: narx.max_delay_])
    good = predictor.predict(X[:10])
    X_bad = X[10:20].copy()
    X_bad[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        predictor.predict(X_bad)
    # State is untouched: continuing with finite data matches one-shot.
    rest = predictor.predict(X[10:])
    one_shot = narx.predict(X, y_init=y[: narx.max_delay_])
    assert_array_equal(np.r_[good, rest], one_shot)


def test_divergence_raises_and_reset_recovers():
    """A diverging recursion is reported; reset() makes the session usable."""
    # Build an unstable model whose output exceeds 1e20 quickly.
    rng = np.random.default_rng(0)
    X = rng.random((50, 1))
    y = rng.random(50)
    feat_ids = np.array([[-1, 1]])  # auto-regressive y(k-1) term
    delay_ids = np.array([[-1, 1]])
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(X, y)
    narx.coef_ = np.array([50.0])  # unstable feedback
    narx.intercept_ = np.array([0.0])

    predictor = NARXPredictor(narx, y_init=np.ones((1, 1)))
    with pytest.raises(FloatingPointError, match="diverged"):
        predictor.predict(X)
    # The broken session must be explicitly reset.
    predictor.reset(y_init=np.zeros((1, 1)))
    narx.coef_ = np.array([0.0])
    out = predictor.predict(X[:5])
    assert out.shape == (5,)
