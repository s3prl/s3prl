import pytest
import logging

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption(
        "--runcorpus", action="store_true", help="run tests with corpus path dependency"
    )
    parser.addoption(
        "--practice",
        action="store_true",
        help="for test scripts only for practice and not real test cases.",
    )
    parser.addoption(
        "--runextra", action="store_true", help="run tests with extra dependencies"
    )
    parser.addoption(
        "--fairseq", action="store_true", help="run tests with fairseq dependencies"
    )
    parser.addoption("--upstream_name", action="store")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.upstream_name
    if "upstream_name" in metafunc.fixturenames:
        metafunc.parametrize("upstream_name", [option_value])


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "corpus: mark test as required corpus path dependency"
    )
    config.addinivalue_line(
        "markers", "extra_dependency: mask test requiring extra dependencies to run"
    )
    config.addinivalue_line("markers", "practice: mark test as a practice")
    config.addinivalue_line("markers", "fairseq: mark test as a fairseq")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--runcorpus"):
        skip_corpus = pytest.mark.skip(reason="need --runcorpus option to run")
        for item in items:
            if "corpus" in item.keywords:
                item.add_marker(skip_corpus)

    if not config.getoption("--practice"):
        skip_practice = pytest.mark.skip(reason="need --practice option to run")
        for item in items:
            if "practice" in item.keywords:
                item.add_marker(skip_practice)

    if not config.getoption("--runextra"):
        skip_extra = pytest.mark.skip(reason="need --runextra option to run")
        for item in items:
            if "extra_dependency" in item.keywords:
                item.add_marker(skip_extra)

    if not config.getoption("--fairseq"):
        skip_extra = pytest.mark.skip(reason="need --fairseq option to run")
        for item in items:
            if "fairseq" in item.keywords:
                item.add_marker(skip_extra)


class Helper:
    pass


@pytest.fixture
def helpers():
    return Helper
