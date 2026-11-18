"""Independent, stdlib-only interpreter for blind Gaussian-integer questions."""

from collections import deque
import re


Gaussian = tuple[int, int]
_Code = list[Gaussian | str]
_TOKEN = re.compile(r"[0-9]+|[A-Z][0-9]*|i|[-+*()]")
_ASSIGNMENT = re.compile(r"\s*([A-Z][0-9]*)\s*=\s*(.+?)\s*")
_QUERY = re.compile(r"(?:[A-Z][0-9]*)+")
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "u+": 3, "u-": 3}
_REAL = re.compile(r"([+-]?)\s*([0-9]+)")
_IMAGINARY = re.compile(r"([+-]?)\s*([0-9]+)?\s*i")
_COMPLEX = re.compile(r"([+-]?)\s*([0-9]+)\s*([+-])\s*([0-9]+)?\s*i")


def _tokens(text: str):
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        match = _TOKEN.match(text, position)
        if match is None:
            raise ValueError(f"unknown expression syntax at character {position}")
        yield match.group()
        position = match.end()


def _push_binary(operator: str, operators: list[str], output: _Code) -> None:
    while (operators and operators[-1] != "("
           and _PRECEDENCE[operators[-1]] >= _PRECEDENCE[operator]):
        output.append(operators.pop())
    operators.append(operator)


def _compile(text: str) -> tuple[_Code, set[str]]:
    """Compile to postfix without recursive parsing, including implicit products."""
    output: _Code = []
    operators: list[str] = []
    dependencies: set[str] = set()
    expecting_operand = True
    for token in _tokens(text):
        if token in ("+", "-"):
            if expecting_operand:
                operators.append("u" + token)
            else:
                _push_binary(token, operators, output)
            expecting_operand = True
        elif token == "*":
            if expecting_operand:
                raise ValueError("multiplication requires a left operand")
            _push_binary(token, operators, output)
            expecting_operand = True
        elif token == ")":
            if expecting_operand:
                raise ValueError("missing operand before closing parenthesis")
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if not operators:
                raise ValueError("unmatched closing parenthesis")
            operators.pop()
            expecting_operand = False
        else:
            if not expecting_operand:
                _push_binary("*", operators, output)
            if token == "(":
                operators.append(token)
                expecting_operand = True
            else:
                if token == "i":
                    output.append((0, 1))
                elif "0" <= token[0] <= "9":
                    output.append((int(token), 0))
                else:
                    output.append(token)
                    dependencies.add(token)
                expecting_operand = False
    if expecting_operand:
        raise ValueError("expression ends without an operand")
    while operators:
        token = operators.pop()
        if token == "(":
            raise ValueError("unmatched opening parenthesis")
        output.append(token)
    return output, dependencies


def _evaluate(code: _Code, values: dict[str, Gaussian]) -> Gaussian:
    stack: list[Gaussian] = []
    for token in code:
        if isinstance(token, tuple):
            stack.append(token)
        elif token[0].isupper():
            try:
                stack.append(values[token])
            except KeyError:
                raise ValueError(f"undefined variable or read before initialization: {token}") from None
        elif token in ("u+", "u-"):
            if token == "u-":
                real, imaginary = stack[-1]
                stack[-1] = (-real, -imaginary)
        else:
            br, bi = stack.pop()
            ar, ai = stack.pop()
            if token == "+":
                stack.append((ar + br, ai + bi))
            elif token == "-":
                stack.append((ar - br, ai - bi))
            else:
                stack.append((ar * br - ai * bi, ar * bi + ai * br))
    return stack[0]


def _resolve_definitions(programs: dict[str, tuple[_Code, set[str]]]) -> dict[str, Gaussian]:
    """Kahn traversal validates all definitions, not only query-reachable ones."""
    dependents: dict[str, list[str]] = {name: [] for name in programs}
    remaining = {}
    for name, (_, dependencies) in programs.items():
        remaining[name] = len(dependencies)
        for dependency in dependencies:
            if dependency not in programs:
                raise ValueError(f"undefined variable: {dependency}")
            dependents[dependency].append(name)
    ready = deque(name for name, count in remaining.items() if count == 0)
    values: dict[str, Gaussian] = {}
    while ready:
        name = ready.popleft()
        values[name] = _evaluate(programs[name][0], values)
        for dependent in dependents[name]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
    if len(values) != len(programs):
        raise ValueError("cyclic definitions")
    return values


def solve_prompt(prompt: str, *, window: int | None = None) -> Gaussian:
    """Solve only the visible prompt; ``window`` retains its last N assignments.

    Updates read the current environment before assigning their left-hand side.
    Definitions must be unique and acyclic, and may refer to later lines.
    """
    if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window < 0):
        raise ValueError("window must be a nonnegative integer or None")
    lines = prompt.splitlines()
    if not lines or lines[0] not in ("Mode: updates", "Mode: definitions"):
        raise ValueError("first line must specify Mode: updates or Mode: definitions")
    try:
        start = lines.index("Statements:", 1)
    except ValueError:
        raise ValueError("missing Statements: line") from None
    if (len(lines) < start + 4 or lines[-1] != "Answer:"
            or not lines[-2].startswith("Query: ")):
        raise ValueError("expected assignment lines, Query: ..., and final Answer:")
    query = lines[-2][len("Query: "):]
    if _QUERY.fullmatch(query) is None:
        raise ValueError("query must be a product of uppercase/indexed variables")
    statements = lines[start + 1:-2]
    if window is not None:
        statements = statements[-window:] if window else []
    query_code, query_names = _compile(query)
    values: dict[str, Gaussian] = {}
    programs: dict[str, tuple[_Code, set[str]]] = {}
    for statement in statements:
        match = _ASSIGNMENT.fullmatch(statement)
        if match is None:
            raise ValueError("expected an assignment: X = expression")
        name, expression = match.groups()
        code, dependencies = _compile(expression)
        if lines[0] == "Mode: updates":
            if window is not None and not dependencies <= values.keys():
                # Unknown input outside the suffix invalidates this write, not independent registers.
                values.pop(name, None)
                continue
            values[name] = _evaluate(code, values)
        else:
            if name in programs:
                raise ValueError(f"duplicate definition: {name}")
            programs[name] = code, dependencies
    if lines[0] == "Mode: definitions":
        if window is not None:
            # A missing operand of a distractor is immaterial if the query is fully resolvable.
            needed = {}
            pending = list(query_names)
            while pending:
                name = pending.pop()
                if name in needed:
                    continue
                if name not in programs:
                    raise ValueError(f"undefined variable in window: {name}")
                needed[name] = programs[name]
                pending.extend(programs[name][1])
            programs = needed
        values = _resolve_definitions(programs)
    return _evaluate(query_code, values)


def _signed_integer(sign: str, digits: str | None) -> int:
    magnitude = int(digits) if digits is not None else 1
    return -magnitude if sign == "-" else magnitude


def parse_answer(text: str) -> Gaussian | None:
    """Accept one real/imaginary/Gaussian literal, never an expression or prose.

    Whitespace may separate signs, coefficients and ``i``, but not digits within
    one integer. The mathematical minus sign U+2212 is an output alias for '-'.
    """
    text = text.strip().replace("\N{MINUS SIGN}", "-")
    if len(text.splitlines()) != 1:
        return None
    try:
        if match := _COMPLEX.fullmatch(text):
            real_sign, real, imaginary_sign, imaginary = match.groups()
            return _signed_integer(real_sign, real), _signed_integer(imaginary_sign, imaginary)
        if match := _IMAGINARY.fullmatch(text):
            return 0, _signed_integer(*match.groups())
        if match := _REAL.fullmatch(text):
            return _signed_integer(*match.groups()), 0
    except ValueError:  # A prediction beyond Python's decimal digit limit is not a valid benchmark answer.
        return None
    return None


def format_answer(value: Gaussian) -> str:
    """Return the canonical literal, omitting zero parts and unit i coefficients."""
    real, imaginary = value
    if imaginary == 0:
        return str(real)
    term = (str(abs(imaginary)) if abs(imaginary) != 1 else "") + "i"
    if real == 0:
        return ("-" if imaginary < 0 else "") + term
    return str(real) + ("-" if imaginary < 0 else "+") + term
