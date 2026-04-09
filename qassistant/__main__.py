"""
CLI entrypoint for qassistant using Click.
"""
import click

from . import __version__


@click.command()
@click.version_option(version=__version__)
def run():
    """
    Run the GUI application.
    """
    from qassistant.gui.application import run_app
    run_app()


if __name__ == "__main__":
    run()
