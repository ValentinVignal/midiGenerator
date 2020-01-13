from functools import partial
from itertools import product

import numpy as np
from scipy.optimize import OptimizeResult

import pytest

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns_message

from skopt import dummy_minimize
from skopt import gp_minimize
from skopt import forest_minimize
from skopt import gbrt_minimize
from skopt.benchmarks import branin
from skopt.benchmarks import bench1
from skopt.benchmarks import bench4
from skopt.benchmarks import bench5
from skopt.callbacks import DeltaXStopper
from skopt.space import Space


# dummy_minimize does not support same parameters so
# treated separately
MINIMIZERS = [gp_minimize]
ACQUISITION = ["LCB", "PI", "EI"]
ACQ_FUNCS_PS = ["PIps", "EIps"]

for est, acq in product(["ET", "RF"], ACQUISITION):
    MINIMIZERS.append(
        partial(forest_minimize, base_estimator=est, acq_func=acq))
for acq in ACQUISITION:
    MINIMIZERS.append(partial(gbrt_minimize, acq_func=acq))


def check_minimizer_api(result, n_calls, n_models=None):
    # assumes the result was produced on branin
    assert(isinstance(result.space, Space))

    if n_models is not None:
        assert_equal(len(result.models), n_models)

    assert_equal(len(result.x_iters), n_calls)
    assert_array_equal(result.func_vals.shape, (n_calls,))

    assert(isinstance(result.x, list))
    assert_equal(len(result.x), 2)

    assert(isinstance(result.x_iters, list))
    for n in range(n_calls):
        assert(isinstance(result.x_iters[n], list))
        assert_equal(len(result.x_iters[n]), 2)

        assert(isinstance(result.func_vals[n], float))
        assert_almost_equal(result.func_vals[n], branin(result.x_iters[n]))

    assert_array_equal(result.x, result.x_iters[np.argmin(result.func_vals)])
    assert_almost_equal(result.fun, branin(result.x))

    assert(isinstance(result.specs, dict))
    assert("args" in result.specs)
    assert("function" in result.specs)


def check_minimizer_bounds(result, n_calls):
    # no values should be below or above the bounds
    eps = 10e-9  # check for assert_array_less OR equal
    assert_array_less(result.x_iters, np.tile([10+eps, 15+eps], (n_calls, 1)))
    assert_array_less(np.tile([-5-eps, 0-eps], (n_calls, 1)), result.x_iters)


def check_result_callable(res):
    """
    Check that the result instance is set right at every callable call.
    """
    assert(isinstance(res, OptimizeResult))
    assert_equal(len(res.x_iters), len(res.func_vals))
    assert_equal(np.min(res.func_vals), res.fun)


def call_single(res):
    pass


@pytest.mark.fast_test
@pytest.mark.parametrize("verbose, call",
                         product(
                            [True, False],
                            [call_single, [call_single,
                                           check_result_callable]]))
def test_minimizer_api_dummy_minimize(verbose, call):
    # dummy_minimize is special as it does not support all parameters
    # and does not fit any models
    n_calls = 7
    result = dummy_minimize(branin, [(-5.0, 10.0), (0.0, 15.0)],
                            n_calls=n_calls, random_state=1,
                            verbose=verbose, callback=call)

    assert result.models == []
    check_minimizer_api(result, n_calls)
    check_minimizer_bounds(result, n_calls)
    assert_raise_message(ValueError,
                         "return a scalar",
                         dummy_minimize, lambda x: x, [[-5, 10]])


@pytest.mark.slow_test
@pytest.mark.parametrize("verbose, call, minimizer",
                         product(
                            [True, False],
                            [call_single, [call_single,
                                           check_result_callable]],
                            MINIMIZERS))
def test_minimizer_api(verbose, call, minimizer):
    n_calls = 7
    n_random_starts = 3
    n_models = n_calls - n_random_starts + 1

    result = minimizer(branin, [(-5.0, 10.0), (0.0, 15.0)],
                       n_random_starts=n_random_starts,
                       n_calls=n_calls,
                       random_state=1,
                       verbose=verbose, callback=call)

    check_minimizer_api(result, n_calls, n_models)
    check_minimizer_bounds(result, n_calls)
    assert_raise_message(ValueError,
                         "return a scalar",
                         minimizer, lambda x: x, [[-5, 10]])


@pytest.mark.fast_test
@pytest.mark.parametrize("minimizer", MINIMIZERS)
def test_minimizer_api_random_only(minimizer):
    # no models should be fit as we only evaluate at random points
    n_calls = 5
    n_random_starts = 5

    result = minimizer(branin, [(-5.0, 10.0), (0.0, 15.0)],
                       n_random_starts=n_random_starts,
                       n_calls=n_calls,
                       random_state=1)

    check_minimizer_api(result, n_calls)
    check_minimizer_bounds(result, n_calls)


@pytest.mark.slow_test
@pytest.mark.parametrize("minimizer", MINIMIZERS)
def test_fixed_random_states(minimizer):
    # check that two runs produce exactly same results, if not there is a
    # random state somewhere that is not reproducible
    n_calls = 4
    n_random_starts = 2

    space = [(-5.0, 10.0), (0.0, 15.0)]
    result1 = minimizer(branin, space, n_calls=n_calls,
                        n_random_starts=n_random_starts, random_state=1)

    dimensions = [(-5.0, 10.0), (0.0, 15.0)]
    result2 = minimizer(branin, dimensions, n_calls=n_calls,
                        n_random_starts=n_random_starts, random_state=1)

    assert_array_almost_equal(result1.x_iters, result2.x_iters)
    assert_array_almost_equal(result1.func_vals, result2.func_vals)


@pytest.mark.slow_test
@pytest.mark.parametrize("minimizer", MINIMIZERS)
def test_minimizer_with_space(minimizer):
    # check we can pass a Space instance as dimensions argument and get same
    # result
    n_calls = 4
    n_random_starts = 2

    space = Space([(-5.0, 10.0), (0.0, 15.0)])
    space_result = minimizer(branin, space, n_calls=n_calls,
                             n_random_starts=n_random_starts, random_state=1)

    check_minimizer_api(space_result, n_calls)
    check_minimizer_bounds(space_result, n_calls)

    dimensions = [(-5.0, 10.0), (0.0, 15.0)]
    result = minimizer(branin, dimensions, n_calls=n_calls,
                       n_random_starts=n_random_starts, random_state=1)

    assert_array_almost_equal(space_result.x_iters, result.x_iters)
    assert_array_almost_equal(space_result.func_vals, result.func_vals)


@pytest.mark.slow_test
@pytest.mark.parametrize("n_random_starts, optimizer_func",
                         product([0, 1, 2, 3, 4],
                                 [gp_minimize,
                                  forest_minimize,
                                  gbrt_minimize]))
def test_init_vals_and_models(n_random_starts, optimizer_func):
    # test how many models are fitted when using initial points, y0 values
    # and random starts
    space = [(-5.0, 10.0), (0.0, 15.0)]
    x0 = [[1, 2], [3, 4], [5, 6]]
    y0 = list(map(branin, x0))
    n_calls = 7

    optimizer = partial(optimizer_func, n_random_starts=n_random_starts)
    res = optimizer(branin, space, x0=x0, y0=y0, random_state=0,
                    n_calls=n_calls)

    assert_equal(len(res.models), n_calls - n_random_starts + 1)


@pytest.mark.slow_test
@pytest.mark.parametrize("n_random_starts, optimizer_func",
                         product([0, 1, 2, 3, 4],
                                 [gp_minimize,
                                  forest_minimize,
                                  gbrt_minimize]))
def test_init_points_and_models(n_random_starts, optimizer_func):
    # test how many models are fitted when using initial points and random
    # starts (no y0 in this case)
    space = [(-5.0, 10.0), (0.0, 15.0)]
    x0 = [[1, 2], [3, 4], [5, 6]]
    n_calls = 7

    optimizer = partial(optimizer_func, n_random_starts=n_random_starts)
    res = optimizer(branin, space, x0=x0, random_state=0,
                    n_calls=n_calls)
    assert_equal(len(res.models), n_calls - len(x0) - n_random_starts + 1)


@pytest.mark.slow_test
@pytest.mark.parametrize("n_random_starts, optimizer_func",
                         product([0, 5],
                                 [gp_minimize,
                                  forest_minimize,
                                  gbrt_minimize]))
def test_init_vals(n_random_starts, optimizer_func):
    space = [(-5.0, 10.0), (0.0, 15.0)]
    x0 = [[1, 2], [3, 4], [5, 6]]
    n_calls = len(x0) + n_random_starts + 1

    optimizer = partial(optimizer_func, n_random_starts=n_random_starts)
    check_init_vals(optimizer, branin, space, x0, n_calls)


@pytest.mark.fast_test
def test_init_vals_dummy_minimize():
    space = [(-5.0, 10.0), (0.0, 15.0)]
    x0 = [[1, 2], [3, 4], [5, 6]]
    n_calls = 10
    check_init_vals(dummy_minimize, branin, space, x0, n_calls)


@pytest.mark.slow_test
@pytest.mark.parametrize("optimizer", [
        dummy_minimize,
        partial(gp_minimize, n_random_starts=0),
        partial(forest_minimize, n_random_starts=0),
        partial(gbrt_minimize, n_random_starts=0)])
def test_categorical_init_vals(optimizer):
    space = [("-2", "-1", "0", "1", "2")]
    x0 = [["0"], ["1"], ["2"]]
    n_calls = 4
    check_init_vals(optimizer, bench4, space, x0, n_calls)


@pytest.mark.slow_test
@pytest.mark.parametrize("optimizer", [
        dummy_minimize,
        partial(gp_minimize, n_random_starts=0),
        partial(forest_minimize, n_random_starts=0),
        partial(gbrt_minimize, n_random_starts=0)])
def test_mixed_spaces(optimizer):
    space = [("-2", "-1", "0", "1", "2"), (-2.0, 2.0)]
    x0 = [["0", 2.0], ["1", 1.0], ["2", 1.0]]
    n_calls = 4
    check_init_vals(optimizer, bench5, space, x0, n_calls)


def check_init_vals(optimizer, func, space, x0, n_calls):
    y0 = list(map(func, x0))
    # testing whether the provided points with their evaluations
    # are taken into account
    res = optimizer(
        func, space, x0=x0, y0=y0,
        random_state=0, n_calls=n_calls)
    assert_array_equal(res.x_iters[0:len(x0)], x0)
    assert_array_equal(res.func_vals[0:len(y0)], y0)
    assert_equal(len(res.x_iters), len(x0) + n_calls)
    assert_equal(len(res.func_vals), len(x0) + n_calls)

    # testing whether the provided points are taken into account
    res = optimizer(
        func, space, x0=x0,
        random_state=0, n_calls=n_calls)
    assert_array_equal(res.x_iters[0:len(x0)], x0)
    assert_array_equal(res.func_vals[0:len(y0)], y0)
    assert_equal(len(res.x_iters), n_calls)
    assert_equal(len(res.func_vals), n_calls)

    # testing whether providing a single point instead of a list
    # of points works correctly
    res = optimizer(
        func, space, x0=x0[0],
        random_state=0, n_calls=n_calls)
    assert_array_equal(res.x_iters[0], x0[0])
    assert_array_equal(res.func_vals[0], y0[0])
    assert_equal(len(res.x_iters), n_calls)
    assert_equal(len(res.func_vals), n_calls)

    # testing whether providing a single point and its evaluation
    # instead of a list of points and their evaluations works correctly
    res = optimizer(
        func, space, x0=x0[0], y0=y0[0],
        random_state=0, n_calls=n_calls)
    assert_array_equal(res.x_iters[0], x0[0])
    assert_array_equal(res.func_vals[0], y0[0])
    assert_equal(len(res.x_iters), 1 + n_calls)
    assert_equal(len(res.func_vals), 1 + n_calls)

    # testing whether it correctly raises an exception when
    # the number of input points and the number of evaluations differ
    assert_raises(ValueError, dummy_minimize, func,
                  space, x0=x0, y0=[1])


@pytest.mark.fast_test
@pytest.mark.parametrize("minimizer", MINIMIZERS)
def test_invalid_n_calls_arguments(minimizer):
    assert_raise_message(ValueError,
                         "Expected `n_calls` >= 10, got 0",
                         minimizer,
                         branin, [(-5.0, 10.0), (0.0, 15.0)],
                         n_calls=0,
                         random_state=1)

    assert_raise_message(ValueError,
                         "set `n_random_starts` > 0, or provide `x0`",
                         minimizer,
                         branin, [(-5.0, 10.0), (0.0, 15.0)],
                         n_random_starts=0,
                         random_state=1)

    # n_calls >= n_random_starts
    assert_raise_message(ValueError,
                         "Expected `n_calls` >= 10",
                         minimizer, branin, [(-5.0, 10.0), (0.0, 15.0)],
                         n_calls=1, n_random_starts=10, random_state=1)

    # n_calls >= n_random_starts + len(x0)
    assert_raise_message(ValueError,
                         "Expected `n_calls` >= 10",
                         minimizer, branin, [(-5.0, 10.0), (0.0, 15.0)],
                         n_calls=1, x0=[[-1, 2], [-3, 3], [2, 5]],
                         random_state=1, n_random_starts=7)

    # n_calls >= n_random_starts
    assert_raise_message(ValueError,
                         "Expected `n_calls` >= 7",
                         minimizer, branin, [(-5.0, 10.0), (0.0, 15.0)],
                         n_calls=1, x0=[[-1, 2], [-3, 3], [2, 5]],
                         y0=[2.0, 3.0, 5.0],
                         random_state=1, n_random_starts=7)


@pytest.mark.fast_test
@pytest.mark.parametrize("minimizer", MINIMIZERS)
def test_repeated_x(minimizer):
    assert_warns_message(
        UserWarning, "has been evaluated at", minimizer, lambda x: x[0],
        dimensions=[[0, 1]], x0=[[0], [1]], n_random_starts=0, n_calls=3)

    assert_warns_message(
        UserWarning, "has been evaluated at", minimizer, bench4,
        dimensions=[("0", "1")], x0=[["0"], ["1"]], n_calls=3,
        n_random_starts=0)


@pytest.mark.fast_test
@pytest.mark.parametrize("minimizer", MINIMIZERS)
def test_consistent_x_iter_dimensions(minimizer):
    # check that all entries in x_iters have the same dimensions
    # two dmensional problem, bench1 is a 1D function but in this
    # instance we do not really care about the objective, could be
    # a total dummy
    res = minimizer(bench1,
                    dimensions=[(0, 1), (2, 3)],
                    x0=[[0, 2], [1, 2]], n_calls=3,
                    n_random_starts=0)
    assert len(set(len(x) for x in res.x_iters)) == 1
    assert len(res.x_iters[0]) == 2

    # one dimensional problem
    res = minimizer(bench1, dimensions=[(0, 1)], x0=[[0], [1]], n_calls=3,
                    n_random_starts=0)
    assert len(set(len(x) for x in res.x_iters)) == 1
    assert len(res.x_iters[0]) == 1

    assert_raise_message(RuntimeError,
                         "use inconsistent dimensions",
                         minimizer, bench1, dimensions=[(0, 1)],
                         x0=[[0, 1]], n_calls=3, n_random_starts=0)

    assert_raise_message(RuntimeError,
                         "use inconsistent dimensions",
                         minimizer, bench1, dimensions=[(0, 1)],
                         x0=[0, 1], n_calls=3, n_random_starts=0)


@pytest.mark.slow_test
@pytest.mark.parametrize("minimizer",
                         [gp_minimize, forest_minimize, gbrt_minimize])
def test_early_stopping_delta_x(minimizer):
    n_calls = 11
    res = minimizer(bench1,
                    callback=DeltaXStopper(0.1),
                    dimensions=[(-1., 1.)],
                    x0=[[-0.1], [0.1], [-0.9]],
                    n_calls=n_calls,
                    n_random_starts=0, random_state=1)
    assert len(res.x_iters) < n_calls


@pytest.mark.slow_test
@pytest.mark.parametrize("minimizer",
                         [gp_minimize, forest_minimize, gbrt_minimize])
def test_early_stopping_delta_x_empty_result_object(minimizer):
    # check that the callback handles the case of being passed an empty
    # results object, e.g. at the start of the optimization loop
    n_calls = 15
    res = minimizer(bench1,
                    callback=DeltaXStopper(0.1),
                    dimensions=[(-1., 1.)],
                    n_calls=n_calls,
                    n_random_starts=1, random_state=1)
    assert len(res.x_iters) < n_calls


@pytest.mark.parametrize("acq_func, minimizer", product(ACQ_FUNCS_PS,
                                                        [gp_minimize,
                                                         forest_minimize,
                                                         gbrt_minimize]))
def test_per_second_api(acq_func, minimizer):
    def bench1_with_time(x):
        return bench1(x), np.abs(x[0])

    n_calls = 3
    res = minimizer(bench1_with_time, [(-2.0, 2.0)],
                    acq_func=acq_func, n_calls=n_calls, n_random_starts=1,
                    random_state=1)
    assert len(res.log_time) == n_calls
