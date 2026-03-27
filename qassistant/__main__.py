"""
CLI entrypoint for qassistant using Click.
"""
import click
import sys

from . import __version__


@click.group()
@click.version_option(version=__version__)
def cli():
    """
    qassistant command line interface.
    """
    pass


@cli.command()
def run():
    """
    Run the GUI application.
    """
    from qassistant.gui.application import run_app
    run_app()


if __name__ == "__main__":
    cli()
