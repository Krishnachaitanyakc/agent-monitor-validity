"""What an action acted on, read from the observation of the step that chose it.

At step t, a monitor-view event holds the observation the agent saw and the action it
then chose: the element IDs that action names are the IDs in step t's own accessibility
tree (``observation.axtree``) and element properties (``elements``). This module looks
them up there and nowhere else. The error recorded at step t + 1 reports how action t
went; it is not used here.

An action is parsed with ``ast`` and never evaluated. It counts as parsed only when every
statement is one plain call, no call is nested inside another, and every element ID
argument is a string literal. Each element ID argument gives one descriptor:

- ``resolved``: exactly one tree line carries the ID. The line is kept as recorded, with
  its role, name and remaining attributes, and the element properties if listed.
- ``ambiguous``: more than one tree line carries the ID. Every line is kept; none is chosen.
- ``nonvisible``: no tree line carries the ID, but the element properties list it. The
  properties (bbox, visibility, clickable, set_of_marks) are kept exactly.
- ``absent``: the step has no record of the ID at all.

An action that names no element gives one ``no_target`` descriptor listing its calls, and
an action that does not parse gives one ``unparsed`` descriptor. The caller keeps the
action text as recorded in both cases.
"""

import ast
import copy
import re
import warnings

RESOLVED, AMBIGUOUS, NONVISIBLE, ABSENT = "resolved", "ambiguous", "nonvisible", "absent"
NO_TARGET, UNPARSED = "no_target", "unparsed"
STATUSES = (RESOLVED, AMBIGUOUS, NONVISIBLE, ABSENT, NO_TARGET, UNPARSED)

# BrowserGym action functions that take element IDs, and the parameters that hold them,
# in positional order.
ELEMENT_PARAMS = {
    "click": ("bid",), "dblclick": ("bid",), "hover": ("bid",), "fill": ("bid",),
    "select_option": ("bid",), "clear": ("bid",), "focus": ("bid",), "press": ("bid",),
    "upload_file": ("bid",), "drag_and_drop": ("from_bid", "to_bid"),
}

# A BrowserGym tree line: indentation, [id], role, an optional quoted name, then nothing,
# a value= or the comma-separated attributes.
_LINE = re.compile(r"[ \t]*\[[^\]]*\] (?P<role>[^ ,]+)"
                   r"(?: (?P<name>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"))?"
                   r"(?P<rest>(?:, .*| value=.*)?)\Z", re.S)


def parse_calls(action: str):
    """[(function, [(parameter, element_id), ...]), ...], one entry per call, or None.

    None means the action does not parse under the rule in the module docstring.
    """
    try:
        with warnings.catch_warnings():         # e.g. an invalid escape in a typed string
            warnings.simplefilter("ignore")
            tree = ast.parse(action, mode="exec")
    except (SyntaxError, ValueError, RecursionError):
        return None
    if not tree.body:
        return None
    calls = []
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)):
            return None
        calls.append(stmt.value)
    if sum(isinstance(n, ast.Call) for n in ast.walk(tree)) != len(calls):
        return None
    out = []
    for call in calls:
        if any(k.arg is None for k in call.keywords):
            return None
        name = call.func.id
        targets = []
        for i, param in enumerate(ELEMENT_PARAMS.get(name, ())):
            if i < len(call.args):
                node = call.args[i]
            else:
                node = next((k.value for k in call.keywords if k.arg == param), None)
            if node is None:
                continue
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                return None
            targets.append((param, node.value))
        out.append((name, targets))
    return out


def tree_lines(axtree, element_id: str) -> list:
    """Every line of the tree whose node carries ``element_id``, exactly as recorded."""
    tag = f"[{element_id}]"
    found = []
    for line in (axtree or "").split("\n"):
        body = line.lstrip(" \t")
        if body.startswith(tag) and body[len(tag):len(tag) + 1] in ("", " "):
            found.append(line)
    return found


def parse_line(line: str) -> dict:
    """The line as recorded, plus its role, name and remaining attributes when it parses.

    The name is the quoted text decoded as a string literal (never evaluated), or None when
    the line has no name. Role, name and attributes are all None when the line does not
    have the expected shape; the line itself is always kept.
    """
    out = {"line": line, "role": None, "name": None, "attributes": None}
    m = _LINE.match(line)
    if m is None:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            name = ast.literal_eval(m.group("name")) if m.group("name") else None
    except (SyntaxError, ValueError):
        return out
    rest = m.group("rest")
    out.update(role=m.group("role"), name=name,
               attributes=rest[2:] if rest.startswith(", ") else rest[1:])
    return out


def describe_target(call_index: int, function: str, parameter: str, element_id: str,
                    axtree, elements) -> dict:
    """One descriptor for one element ID, from this step's tree and element properties only."""
    lines = [parse_line(x) for x in tree_lines(axtree, element_id)]
    elements = elements or {}
    listed = element_id in elements
    if len(lines) == 1:
        status = RESOLVED
    elif lines:
        status = AMBIGUOUS
    elif listed:
        status = NONVISIBLE
    else:
        status = ABSENT
    return {"call": call_index, "function": function, "parameter": parameter,
            "element_id": element_id, "status": status, "tree_lines": lines,
            "properties": copy.deepcopy(elements[element_id]) if listed else None}


def describe_action(action: str, axtree, elements) -> list:
    """Descriptors for one action, looked up in the tree and element properties given.

    The caller passes the tree and properties of the step that chose the action.
    """
    calls = parse_calls(action)
    if calls is None:
        return [{"status": UNPARSED}]
    out = [describe_target(i, name, param, eid, axtree, elements)
           for i, (name, targets) in enumerate(calls) for param, eid in targets]
    return out or [{"status": NO_TARGET, "functions": [name for name, _ in calls]}]
