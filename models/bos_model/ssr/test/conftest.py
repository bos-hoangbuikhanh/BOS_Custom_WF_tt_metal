import pytest
from bos_metal.core.infra import device as tt_device

@pytest.fixture(scope="session")
def tt_dev():
    """Open the Tenstorrent device once for the entire test session."""
    dev = tt_device.open(0)          # or pass device_id from env
    yield dev
    tt_device.close(dev)


