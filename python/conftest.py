import pytest
from pathlib import Path


@pytest.fixture
def data_root() -> Path:
    return Path(__file__).parent.parent / "data"
