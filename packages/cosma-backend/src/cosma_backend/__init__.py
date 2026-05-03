from .app import App as App, create_app as create_app, run as run

# ---------------------------------------------------------------------------
# Version handshake between backend and frontend
# ---------------------------------------------------------------------------
#
# `__version__` mirrors the package version in pyproject.toml. It moves on
# every PyPI release (including no-op patch bumps).
#
# `__api_version__` is the *contract* the frontend pins against. Bump it
# only when the wire format changes in a way that an older frontend can't
# handle: a renamed JSON field, a removed endpoint, a new ProcessingStatus
# string the older Swift enum can't decode, etc. Adding a new optional
# field is NOT a bump.
#
# `__min_frontend_api_version__` is the floor: if a too-old frontend
# connects, the backend tells it "you need to upgrade." Usually equal to
# __api_version__ unless we deliberately keep a transition window open.
#
# Frontend declares the same number in Swift as `kRequiredBackendApiVersion`.
# On launch, it fetches /api/version and refuses to operate on a mismatch.
# This kills the class of bug where `uv tool upgrade cosma` silently moves
# the backend past what the bundled frontend understands.
#
# Bump procedure:
#   1. Make the breaking API change.
#   2. __api_version__ += 1.
#   3. Update fileSearchForntend Swift `kRequiredBackendApiVersion`.
#   4. Cut a coordinated frontend+backend release with matching __version__.
#   5. (Optional) leave __min_frontend_api_version__ at the previous value
#      for one release if you want to give users a soft transition.
__version__ = "1.0.0"
__api_version__ = 1
__min_frontend_api_version__ = 1


class _BrokenPipeSafe:
    """Wrap a text stream so writes to a closed pipe don't raise.

    structlog/uvicorn/transformers all emit log lines during a bootstrap
    install; any one of them hitting a closed parent pipe would abort the
    whole run with "[Errno 32] Broken pipe". We can't reasonably wrap every
    logger call, so we intercept the two sinks those loggers share.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, data):
        try:
            return self._stream.write(data)
        except (BrokenPipeError, OSError):
            return len(data) if isinstance(data, (str, bytes)) else 0

    def flush(self):
        try:
            self._stream.flush()
        except (BrokenPipeError, OSError):
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def serve():
    import signal
    import sys
    import uvicorn

    # When launched as a child of the Swift app, stdout/stderr are pipes.
    # If the parent stops reading (window closed, UI refactor, etc.) the
    # next write raises BrokenPipeError, which surfaces as "[Errno 32]
    # Broken pipe" and aborts downloads mid-flight. Ignoring SIGPIPE makes
    # writes fail silently at the OS level, and wrapping the Python streams
    # catches any BrokenPipeError Python still raises at write time. We
    # keep stderr functional locally (via the wrapper) so crash tracebacks
    # aren't swallowed when running in a terminal.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass  # SIGPIPE isn't available on Windows; fine.

    sys.stdout = _BrokenPipeSafe(sys.stdout)
    sys.stderr = _BrokenPipeSafe(sys.stderr)

    app = create_app()
    uvicorn.run(
        app,
        host=app.config["HOST"],
        port=app.config["PORT"],
        log_level="warning",
        # Drop the "INFO: 127.0.0.1:.. - GET /api/... 200" access line
        # for every request — under normal frontend chatter that's
        # ~200 lines/min of pure noise. Quart's own before/after_request
        # hooks already log at DEBUG when needed.
        access_log=False,
        # I can't find a way to gracefully shut down SSE connections,
        # so this bullshit will have to do for now
        # 10 s is comfortable now that after_serving signals SSE streams to
        # return immediately and teardown runs in parallel. Must stay <= the
        # Swift CosmaManager's SIGTERM→SIGKILL window (~12 s) so we exit
        # cleanly instead of getting force-killed.
        timeout_graceful_shutdown=10,
    )
