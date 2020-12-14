import pytest

from ml.solver import StackSolver


@pytest.fixture
def solver() -> StackSolver:
    return StackSolver()


@pytest.mark.parametrize('expression, result',
                         [('4 + 2', 6),
                          ('(3 + 5) / 2', 4),
                          ('3 * (4 + (3 * (4 / 2))) - 1', 29),
                          ('(4 - 3 * (2 - (1*0)))', -2),
                          ("(15 + 3) - (4 * 2)", 10)])
def test_solvable(solver, expression, result):
    assert solver.solve(expression) == result


@pytest.mark.parametrize('expression',
                         ['4 + (',
                          '1 / 0'])
def test_unsolvable(solver, expression):
    with pytest.raises(ValueError):
        solver.solve(expression)
