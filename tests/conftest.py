import pytest
import torch

from ._fixtures import execution_device as execution_device


@pytest.fixture(autouse=True, scope='session')
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
