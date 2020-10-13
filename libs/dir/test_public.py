import sys

from .some_local_code import some_useful_function


def test_version() -> None:
    """
    To do this task you need python=3.8.5
    """
    assert '3.6.9' == sys.version.split(' ', maxsplit=1)[0]


def test_useful_function() -> None:
    assert 9 == some_useful_function(3)
