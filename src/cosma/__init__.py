import click
from click_help_colors import HelpColorsGroup

from .cli.commands.search import search_command
from .cli.commands.index import index_command
from .cli.commands.status import status_command
from .cli.commands.watch import watch_group
from .cli.commands.files import files_group
from .cli.commands.filters import filters_group
from .cli.commands.updates import updates_command
from .cli.commands.serve import serve_command
from .cli.commands.tui import tui_command


@click.group(
    cls=HelpColorsGroup,
    help_headers_color='cyan',
    help_options_color='green',
)
def cli():
    """Search engine for your files!"""
    pass


# Register commands
cli.add_command(search_command, name="search")
cli.add_command(index_command, name="index")
cli.add_command(status_command, name="status")
cli.add_command(watch_group, name="watch")
cli.add_command(files_group, name="files")
cli.add_command(filters_group, name="filters")
cli.add_command(updates_command, name="updates")
cli.add_command(serve_command, name="serve")
cli.add_command(tui_command, name="tui")
