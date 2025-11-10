import click

from cosma_backend import serve as serve_backend
from cosma_tui import start_tui

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        result = start_tui()
        if result:
            print(result)
    
@cli.command()
def serve():
    serve_backend()

