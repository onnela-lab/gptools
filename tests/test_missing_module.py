from gptools.missing_module import MissingModule
import pytest


def test_missing_module():
    module = MissingModule(ModuleNotFoundError("foobar"))
    with pytest.raises(ModuleNotFoundError):
        module.member
