from importlib.metadata import version, PackageNotFoundError

import click
from click_help_colors import HelpColorsGroup

from .cli.commands.search import search_command
from .cli.commands.index import index_command
from .cli.commands.status import status_command
from .cli.commands.watch import watch_group
from .cli.commands.files import files_group
from .cli.commands.filters import filters_group
from .cli.commands.settings import settings_group
from .cli.commands.queue import queue_group
from .cli.commands.updates import updates_command
from .cli.commands.serve import serve_command
from .cli.commands.tui import tui_command


def _get_version(package: str) -> str:
    """Get version for a package, or 'not installed' if not found."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _print_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Print version information for all cosma packages."""
    if not value or ctx.resilient_parsing:
        return

    packages = [
        ("cosma", "cosma"),
        ("cosma-backend", "cosma-backend"),
        ("cosma-tui", "cosma-tui"),
        ("cosma-client", "cosma-client"),
    ]

    click.echo("cosma version info:")
    click.echo()
    for display_name, package_name in packages:
        ver = _get_version(package_name)
        click.echo(f"  {display_name}: {ver}")

    ctx.exit()


@click.group(
    cls=HelpColorsGroup,
    help_headers_color='cyan',
    help_options_color='green',
)
@click.option(
    "--version", "-V",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show version information for all packages.",
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
cli.add_command(settings_group, name="settings")
cli.add_command(queue_group, name="queue")
cli.add_command(updates_command, name="updates")
cli.add_command(serve_command, name="serve")
cli.add_command(tui_command, name="tui")
