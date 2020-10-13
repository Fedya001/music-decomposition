import pytest
import dataclasses


@dataclasses.dataclass
class Case:
    a: int
    b: int
    sum_: int

    def __str__(self) -> str:
        return f"{self.a}+{self.b}"


TEST_CASES = [
    Case(2, 2, 3),
    Case(-42, 100, 58)
]


@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_plus(t: Case) -> None:
    assert t.a + t.b == t.sum_
