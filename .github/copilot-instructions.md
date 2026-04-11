## Python Coding Guidelines

* Use modular project structure with consistent python styling, separate concepts such as: data vs UI, command line vs core modules...
* For Qt-derived classes, use Qt-style naming convention (camelCase function names)
* Generate unit tests using 'unittest' python framework in 'tests' sub-directory for core modules. Use per-module test_ files (not per-class).
* Perform incremental changes to fulfil user requests, do not introduce unnecessary changes to existing code.
* Use the following packages:
  - `click` for command line processing
  - `pyside6` for Qt GUI
  - `ruff` for lint testing and fixes
  - `unittest` for testing
  - `pathlib.Path` for file system paths instead of `os.path`
* Use type hints for input arguments and return values. Do not generate -> None annotation if a function does not return a value.
* Do not use import aliases. Example: use 'import pandas', do not use 'import pandas as pd'.
* Sort imports by name, group standard package imports first, then local imports separated by an empty line
* Use tripple-quote multi-line docstrings. Example:
  """
  This is a function docstring.
  """
* Use declarative Qt syntax, preferring to pass widget properties / signal handlers in constructors instead of explicit functional calls.
  Example: QPushButton("Click me", clicked=self._onClick)
