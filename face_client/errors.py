class ClientError(Exception):
    """Fatal, user-facing client error.

    Raised anywhere in the package instead of calling sys.exit(). cli.main()
    is the single place that catches it and translates it to a clean exit(1),
    which keeps every function importable and unit-testable (no
    process-killing sys.exit in library code).
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
