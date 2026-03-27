## Python Coding Guidelines

* Use modular project structure with consistent python styling, separate concepts such as: data vs UI, command line vs core modules...
* For Qt-derived classes, use Qt-style naming convention (camelCase function names)
* Generate unit tests using 'unittest' python framework in 'tests' sub-directory for core modules.
* Perform incremental changes to fullfil user requests, do not introduce unnecessary changes to existing code.
* Use the following packages:
  - `click` for command line processing
  - `pyside6` for Qt GUI
  - `ruff` for lint testing and fixes
  - `unittest` for testing
* Use type hints for input arguments and return values. Do not generate -> None annotation if a function does not return a value.
* Do not use import aliases. Example: use 'import pandas', do not use 'import pandas as pd'.
* Sort imports by name, group standard package imports first, then local imports separated by empty line
* Use tripple-quote multi-line docstrings. Example:
  """
  This is a function docstring.
  """
