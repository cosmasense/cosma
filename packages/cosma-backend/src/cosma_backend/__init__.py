from .app import App as App, create_app as create_app, run as run


def serve():
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=app.config["HOST"],
        port=app.config["PORT"],
        log_level="info",
        # I can't find a way to gracefully shut down SSE connections,
        # so this bullshit will have to do for now
        timeout_graceful_shutdown=5,
    )
