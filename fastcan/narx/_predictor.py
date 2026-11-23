"""
Stateful (streaming) prediction for NARX models.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from numbers import Integral

import numpy as np
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_is_fitted, validate_data

from ._narx_fast import _predict


class NARXPredictor:
    """Stateful multiple-step-ahead predictor for a fitted :class:`NARX` model.

    A :class:`NARX` model itself is stateless: every call to
    :meth:`NARX.predict` restarts the recursion from the given `y_init`.
    In streaming applications, e.g. forecasting the outlet temperature of a
    factory process batch by batch (shift by shift), the tail of one
    prediction must be carried over as the initial condition of the next
    batch. :class:`NARXPredictor` keeps this recursion state for one
    measurement session (e.g. one production line).

    The predictor stores the last ``max_delay_`` input samples and the last
    ``max_delay_`` predicted outputs. Each :meth:`predict` call appends the
    new batch to this state and runs the same recursion as
    :meth:`NARX.predict`. Consequently, splitting one input sequence into
    batches and predicting them through one predictor gives results
    identical (bit for bit) to predicting the whole sequence in a single
    :meth:`NARX.predict` call with the same `y_init`.

    Different data sources (e.g. different production lines) should use
    different predictor instances. The instances hold independent states, so
    interleaved requests never contaminate each other. When a line change
    requires new initial conditions, call :meth:`reset`.

    Each predictor is intended for a single measurement session and is not
    guaranteed to be thread-safe; concurrent requests for the same source
    should be serialized.

    Parameters
    ----------
    narx : NARX
        A fitted NARX model.

    y_init : array-like of shape (n_init,) or (n_init, `n_outputs_`), \
            default=None
        The initial values of the output for the measurement session.
        It should at least have one sample. If None, zeros are used, which
        is the same default as :meth:`NARX.predict`.
        Batched inputs `X` are expected to contain finite values; missing
        data inside a stream cannot be reconciled with the carried-over
        state. When a gap occurs, split the stream there and start a new
        session with :meth:`reset`.

    Attributes
    ----------
    n_predicted_ : int
        The number of samples predicted by this predictor since the last
        :meth:`reset` (or construction).

    Examples
    --------
    >>> import numpy as np
    >>> from fastcan.narx import NARX, NARXPredictor
    >>> rng = np.random.default_rng(12345)
    >>> X = rng.random((20, 1))
    >>> y = rng.random(20)
    >>> narx = NARX().fit(X, y)
    >>> y_init = y[: narx.max_delay_]
    >>> # One-shot prediction
    >>> y_one_shot = narx.predict(X, y_init=y_init)
    >>> # Batch-wise prediction through a stateful predictor
    >>> predictor = NARXPredictor(narx, y_init=y_init)
    >>> y_batches = np.r_[
    ...     predictor.predict(X[:10]),
    ...     predictor.predict(X[10:]),
    ... ]
    >>> np.allclose(y_one_shot, y_batches)
    True
    """

    @validate_params(
        {
            "narx": "no_validation",  # checked via check_is_fitted below
            "y_init": [None, "array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, narx, y_init=None):
        check_is_fitted(narx)
        self._narx = narx
        self._reset_state(y_init)

    def _check_y_init(self, y_init):
        """Validate `y_init` and return a 2D C-contiguous array."""
        narx = self._narx
        if y_init is None:
            return np.zeros((narx.max_delay_, narx.n_outputs_), dtype=float, order="C")
        y_init = check_array(
            y_init,
            ensure_2d=False,
            dtype=float,
            order="C",
            ensure_min_samples=0,
            ensure_all_finite="allow-nan",
        )
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)
        if y_init.shape[1] != narx.n_outputs_:
            raise ValueError(
                f"`y_init` should have {narx.n_outputs_} outputs "
                f"but got {y_init.shape[1]}."
            )
        if narx.max_delay_ > 0 and y_init.shape[0] == 0:
            raise ValueError(
                "`y_init` should at least have one sample when "
                f"`max_delay_` is {narx.max_delay_}, but got 0 samples."
            )
        return np.ascontiguousarray(y_init, dtype=float)

    def _reset_state(self, y_init):
        """(Re)initialize the session state."""
        narx = self._narx
        self._y_init_full = self._check_y_init(y_init)
        self._x_hist = np.empty((0, narx.n_features_in_), dtype=float, order="C")
        self._y_hist = np.empty((0, narx.n_outputs_), dtype=float, order="C")
        self.n_predicted_ = 0

    @validate_params(
        {
            "y_init": [None, "array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def reset(self, y_init=None):
        """Reset the predictor to a new measurement session.

        Use this method on a line change, where the recursion must start from
        new initial conditions instead of the tail of the previous session.

        Parameters
        ----------
        y_init : array-like of shape (n_init,) or (n_init, `n_outputs_`), \
                default=None
            The new initial values of the output. If None, zeros are used.

        Returns
        -------
        self : object
            The reset predictor.
        """
        self._reset_state(y_init)
        return self

    @validate_params(
        {
            "X": ["array-like", Interval(Integral, 1, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(self, X):
        """Predict the next batch and advance the recursion state.

        The tail of the previous batches is automatically consumed as the
        initial condition of this batch, so consecutive calls chain together
        as if the whole sequence were predicted at once.

        Parameters
        ----------
        X : array-like of shape (n_samples, `n_features_in_`) or int
            When X is an array, it is the input data of the next batch.
            When (nonlinear) AR model is adopted, where `n_features_in_` is
            0, X can be a positive integer indicating the number of steps to
            predict.

        Returns
        -------
        y_hat : array-like of shape (n_samples,) or (n_samples, `n_outputs_`)
            Predicted values of this batch. The number of dimensions is the
            same as that of y in :meth:`NARX.fit`.
        """
        narx = self._narx
        max_delay = narx.max_delay_

        if isinstance(X, Integral):
            if narx.n_features_in_ == 0:
                X = np.empty((int(X), 0), dtype=float, order="C")
            else:
                raise ValueError(
                    "X should be an array-like of shape "
                    "(n_samples, n_features_in_) but got an integer "
                    f"{X}, when `n_features_in_` is not 0."
                )
        else:
            X = validate_data(
                narx,
                X,
                dtype=float,
                order="C",
                reset=False,
                ensure_all_finite="allow-nan",
                ensure_min_features=0,
            )
        n_samples = X.shape[0]

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Batched inputs X must contain finite values. A missing value "
                "cannot be reconciled with the carried-over session state; "
                "split the stream at the gap and call reset() with new initial "
                "conditions to start a new session."
            )

        # Number of history rows actually needed by the recursion.
        n_hist = min(max_delay, self.n_predicted_)
        x_hist = self._x_hist[-n_hist:] if n_hist > 0 else self._x_hist
        y_hist = self._y_hist[-n_hist:] if n_hist > 0 else self._y_hist

        # Prepend the carried-over state so the recursion sees an unbroken
        # time series. The first n_hist outputs of the augmented call are
        # re-emitted history and discarded afterwards.
        X_aug = np.ascontiguousarray(np.r_[x_hist, X]) if n_hist > 0 else X

        if self.n_predicted_ < max_delay:
            # Warm-up is not finished yet. The original y_init is used as a
            # cyclic reference, exactly reproducing NARX.predict's
            # `y_ref[k % n_ref]` rule. History rows were themselves emitted
            # under the same rule, so they match the cyclic reference.
            y_ref = self._y_init_full
        else:
            # Fully warmed up: recurse from the predicted tail only.
            y_ref = y_hist

        y_hat_aug = np.zeros((X_aug.shape[0], narx.n_outputs_), dtype=float)
        session_sizes_cumsum = np.array([X_aug.shape[0]], dtype=np.int32)
        _predict(
            X_aug,
            y_ref,
            narx.coef_,
            narx.intercept_,
            narx.feat_ids_,
            narx.delay_ids_,
            narx.output_ids_,
            session_sizes_cumsum,
            max_delay,
            y_hat_aug,
        )

        if (
            not np.all(np.isfinite(y_hat_aug))
            or np.max(np.abs(np.nan_to_num(y_hat_aug))) > 1e20
        ):
            raise FloatingPointError(
                "The NARX prediction diverged (non-finite values or values "
                "larger than 1e20). The session state is no longer valid; "
                "call reset() with new initial conditions before predicting "
                "again."
            )

        y_hat = np.ascontiguousarray(y_hat_aug[n_hist:])

        # Advance the state with the inputs and outputs of this batch.
        # Only the last max_delay rows are needed by future recursions.
        self._x_hist = np.ascontiguousarray(
            np.r_[self._x_hist, X][-max_delay:] if max_delay > 0 else self._x_hist[:0]
        )
        self._y_hist = np.ascontiguousarray(
            np.r_[self._y_hist, y_hat][-max_delay:]
            if max_delay > 0
            else self._y_hist[:0]
        )
        self.n_predicted_ += n_samples
        if self.n_predicted_ >= max_delay:
            # Warm-up finished: the initial condition is no longer needed.
            self._y_init_full = None

        if narx._y_ndim == 1:
            y_hat = y_hat.flatten()
        return y_hat
