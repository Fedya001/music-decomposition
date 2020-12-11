import sys


def test_python_version() -> None:
    assert sys.version_info.major == 3
