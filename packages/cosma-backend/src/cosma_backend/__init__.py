from .app import app as app
from .app import run as run

def serve():
    import uvicorn
    
    uvicorn.run(
        app, host="127.0.0.1",
        port=60534,
        log_level="info",
        # I can't find a way to gracefully shut down SSE connections,
        # so this bullshit will have to do for now
        timeout_graceful_shutdown=5
    )
