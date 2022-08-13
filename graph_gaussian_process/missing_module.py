class MissingModule:
    """
    Placeholder for missing modules.
    """
    def __init__(self, ex: ModuleNotFoundError):
        self.ex = ex

    def __getattr__(self, name: str) -> None:
        raise ModuleNotFoundError(
            f"cannot access attribute '{name}' of missing module '{self.ex.name}'"
        )
