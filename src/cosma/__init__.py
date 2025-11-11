import click

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    # Store whether we should check for directory argument
    ctx.ensure_object(dict)
    
@cli.command()
@click.argument('directory', default='.')
def tui(directory: str):
    from cosma_tui import start_tui
    result = start_tui(directory)
    if result:
        print(result)

@cli.command()
def serve():
    from cosma_backend import serve as serve_backend
    serve_backend()

# Make tui the default command
@cli.result_callback()
@click.pass_context
def process_result(ctx: click.Context, result, **kwargs):
    if ctx.invoked_subcommand is None:
        # If no subcommand was given, invoke tui with remaining args
        ctx.invoke(tui, directory='.')
