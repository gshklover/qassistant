"""
Python execution shell impementation for use with agent tools.
"""
import dataclasses
from IPython.core.interactiveshell import InteractiveShell
from typing import Any


@dataclasses.dataclass(slots=True)
class ExecutionResult:
    """
    Standardized result format for code execution.
    """
    output: list[str]
    result: Any


class PythonShell:
    """
    Implements python shell using ipython's InteractiveShell.
    """
    def __init__(self, shell: InteractiveShell = None):
        self.shell = shell or InteractiveShell()

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute the given code and return the output.
        """
        result = self.shell.run_cell(code)
        if result.error_in_exec:
            raise result.error_in_exec
        return result.result
