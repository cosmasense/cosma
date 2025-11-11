import click
from click_help_colors import HelpColorsGroup


@click.group(
    cls=HelpColorsGroup,
    help_headers_color='cyan',
    help_options_color='green',
)
def cli():
    """Search engine for your files!"""
    pass

@cli.command()
@click.argument('directory', default='.')
def search(directory: str):
    """Search through files in a directory using the TUI interface"""
    from cosma_tui import start_tui
    result = start_tui(directory)
    if result:
        print(result)

@cli.command()
def serve():
    """Start the Cosma backend server"""
    from cosma_backend import serve as serve_backend
    serve_backend()
