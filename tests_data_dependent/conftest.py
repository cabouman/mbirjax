import os, sys

def _explicit_m_passed(argv):
    return any(a == "-m" or a.startswith("-m=") for a in argv)

def pytest_configure(config):
    if os.getenv("PYCHARM_HOSTED") == "1" and not _explicit_m_passed(sys.argv):
        config.option.markexpr = "data_dependent or not data_dependent"