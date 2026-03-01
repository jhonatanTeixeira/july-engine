import pytest
import os

def pytest_configure(config):
    config.addinivalue_line("markers", "cpu: mark test as cpu-bound")
    config.addinivalue_line("markers", "gpu: mark test as gpu-bound")
    config.addinivalue_line("markers", "api: mark test as api-bound")

def pytest_addoption(parser):
    parser.addoption("--with-ollama", action="store_true", default=False, help="run tests with Ollama backend")
    parser.addoption("--with-vllm", action="store_true", default=False, help="run tests with vLLM backend")
    parser.addoption("--gpu-only", action="store_true", default=False, help="run only GPU tests")
    parser.addoption("--cpu-only", action="store_true", default=False, help="run only CPU tests")
    parser.addoption("--api-only", action="store_true", default=False, help="run only API tests")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu-only"):
        skip_non_gpu = pytest.mark.skip(reason="only running GPU tests")
        for item in items:
            if "gpu" not in item.keywords:
                item.add_marker(skip_non_gpu)
    
    elif config.getoption("--cpu-only"):
        skip_non_cpu = pytest.mark.skip(reason="only running CPU tests")
        for item in items:
            if "cpu" not in item.keywords:
                item.add_marker(skip_non_cpu)
                
    elif config.getoption("--api-only"):
        skip_non_api = pytest.mark.skip(reason="only running API tests")
        for item in items:
            if "api" not in item.keywords:
                item.add_marker(skip_non_api)

@pytest.fixture
def with_ollama(request):
    return request.config.getoption("--with-ollama")

@pytest.fixture
def with_vllm(request):
    return request.config.getoption("--with-vllm")

@pytest.fixture
def gpu_only(request):
    return request.config.getoption("--gpu-only")

@pytest.fixture
def cpu_only(request):
    return request.config.getoption("--cpu-only")

@pytest.fixture
def api_only(request):
    return request.config.getoption("--api-only")
