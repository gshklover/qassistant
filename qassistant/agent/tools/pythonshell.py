"""
Python execution shell impementation for use with agent tools.
"""
import ast
import dataclasses
import sys
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
from typing import Any

from .utils import truncate


class _OrderedCapture:
    """
    Context manager that captures stdout, stderr, and IPython display outputs
    as ordered text chunks and exposes them as output lines.
    """

    def __init__(self, shell: InteractiveShell):
        self._shell = shell
        self._chunks: list[str] = []
        self._pending_text = ''
        self._pending_source = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._orig_display = None

    @property
    def outputs(self) -> list[str]:
        """
        Returns captured lines (stdout, stderr, and display outputs) in the order they were produced.
        """
        return [line for chunk in self._chunks for line in chunk.splitlines() or [chunk]]

    def _append(self, text: str, source: str | None = None):
        if not text:
            return 0

        if self._pending_source not in (None, source):
            self._flush_pending()

        self._pending_source = source
        self._pending_text += text
        return len(text)

    def _flush_pending(self):
        if self._pending_text:
            self._chunks.append(self._pending_text)
            self._pending_text = ''
        self._pending_source = None

    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        capture = self

        class _StreamProxy:
            def __init__(self, source: str, original):
                self._source = source
                self._original = original

            @property
            def encoding(self):
                return getattr(self._original, 'encoding', 'utf-8')

            def write(self, text: str):
                return capture._append(text, self._source)

            def flush(self):
                capture._flush_pending()

            def isatty(self):
                return bool(getattr(self._original, 'isatty', lambda: False)())

        sys.stdout = _StreamProxy('stdout', self._orig_stdout)
        sys.stderr = _StreamProxy('stderr', self._orig_stderr)

        orig_display = self._shell.display_pub.publish

        def _display_hook(data, metadata=None, source=None, **kwargs):
            self._flush_pending()
            text = (
                data.get('text/plain')
                or data.get('text/html')
                or str(data)
            )
            self._chunks.append(str(text))

        self._orig_display = orig_display
        self._shell.display_pub.publish = _display_hook
        return self

    def __exit__(self, *_):
        self._flush_pending()
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._shell.display_pub.publish = self._orig_display


@dataclasses.dataclass(slots=True)
class ExecutionResult:
    """
    Standardized result format for code execution.
    """
    success: bool
    error: str = ''
    output: list[str] = dataclasses.field(default_factory=list)
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the execution result to a dictionary format for easier serialization.
        """
        return {
            "success": self.success,
            "error": self.error,
            "output": self.output,
            "result": truncate(str(self.result)),
        }

    def __str__(self):
        return str(self.to_dict())
        
    def __repr__(self):
        return str(self)


class PythonShell:
    """
    Implements python shell using ipython's InteractiveShell.
    """
    def __init__(self, shell: InteractiveShell = None):
        self.shell = shell or InteractiveShell()

    @property
    def currentWorkArea(self) -> str:
        """
        Return current shell working directory.
        """
        return str(Path.cwd())

    def execute(self, code: str) -> ExecutionResult:
        """
        Executes specified code and returns the result.

        :param code: the python code to execute
        
        :return: ExecutionResult with the following fields:
            - success: indicates if execution was successful
            - error: error message if execution failed
            - output: list of printed output lines
            - result: the value of the last expression in the code
        """        
        self.shell.user_ns.pop('_', None)  # clear previous '_' value
        
        with _OrderedCapture(self.shell) as captured:
            result = self.shell.run_cell(code)
        
        if '_' not in self.shell.user_ns:
            self.shell.user_ns['_'] = self._extract_assignment(code)

        if result.error_before_exec:
            return ExecutionResult(success=False, error=str(result.error_before_exec), output=captured.outputs)
        elif result.error_in_exec:
            return ExecutionResult(success=False, error=str(result.error_in_exec), output=captured.outputs)
        else:
            return ExecutionResult(
                success=True,
                output=captured.outputs,
                result=self.shell.user_ns.get('_', ''),
            )
    
    def get_variables(self) -> list[dict]:
        """
        Get currently defined variables, their types and values.
        """
        return [
            {
                "name": name,
                "type": type(value).__name__,
                "value": truncate(repr(value)),
            }
            for name, value in self.shell.user_ns.items()
            if not name.startswith('_')  # filter out internal variables
        ]
    
    def _extract_assignment(self, code: str) -> Any:
        """
        Naive extraction of the assignment in the code.
        This is a best-effort approach and may not cover all edge cases.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        
        # take the last expression and check if the expression is an assignment:
        last_node = tree.body[-1]
        if isinstance(last_node, ast.Assign):
            if isinstance(last_node.targets[0], ast.Name):
                return self.shell.user_ns.get(last_node.targets[0].id, None)
            if isinstance(last_node.targets[0], ast.Subscript) and isinstance(last_node.targets[0].value, ast.Name):
                return self.shell.user_ns.get(last_node.targets[0].value.id, None)
        return None
