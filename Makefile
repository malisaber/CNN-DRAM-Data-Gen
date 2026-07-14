PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: all clean deps check-tools numpy

all: deps

check-tools:
	@command -v "$(PYTHON)" >/dev/null 2>&1 || { printf '%s\n' "Missing Python interpreter: $(PYTHON)" >&2; exit 1; }
	@$(PYTHON) -m pip --version >/dev/null 2>&1 || { printf '%s\n' "Missing pip for $(PYTHON)" >&2; exit 1; }

numpy:
	@$(PYTHON) -c "import numpy" >/dev/null 2>&1 || { printf '%s\n' "Installing numpy" ; $(PIP) install --user numpy; }

deps: check-tools numpy

clean:
	@printf '%s\n' "Nothing to clean for Python generators."
