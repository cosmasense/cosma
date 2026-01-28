"""
Serve command for Cosma CLI.
"""

import click


@click.command("serve")
def serve_command():
    """Start the Cosma backend server."""
    from cosma_backend import serve as serve_backend
    serve_backend()
