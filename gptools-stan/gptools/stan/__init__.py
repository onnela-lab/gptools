import os


def get_include() -> str:
    """
    Get the include directory for the graph Gaussian process library.
    """
    return os.path.dirname(__file__)


if __name__ == "__main__":
    print(get_include())
