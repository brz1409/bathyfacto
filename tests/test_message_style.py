# Copyright 2026 Markus Brezovsky, TU Wien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""No em dash, en dash or semicolon in any ``CONSOLE.log``/``CONSOLE.print`` message or any
``raise ...(...)`` exception message, anywhere under ``bathyfacto/``.

This is an AST scan, not a text grep, so it survives reformatting and does not fire on the
character appearing in a comment, a docstring, or unrelated code. F-string constant parts are
checked; the ``{...}`` expression slots are not literal text and are skipped.
"""

import ast
from pathlib import Path
from typing import List

BATHYFACTO_DIR = Path(__file__).resolve().parents[1] / "bathyfacto"
BANNED_CHARACTERS = ("—", "–", ";")  # em dash, en dash, semicolon


def _literal_string_pieces(node: ast.AST) -> List[str]:
    """Every literal (non-interpolated) string piece inside a call-argument expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        pieces: List[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
            # ast.FormattedValue is a `{...}` slot: not literal text, nothing to check.
        return pieces
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_string_pieces(node.left) + _literal_string_pieces(node.right)
    return []


def _is_console_call(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in ("log", "print")
        and isinstance(func.value, ast.Name)
        and func.value.id == "CONSOLE"
    )


def _iter_message_call_sites(tree: ast.AST):
    """Yield every ``CONSOLE.log(...)``/``CONSOLE.print(...)`` call and every raised
    exception call (``raise SomeError(...)``), each paired with a short location label."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_console_call(node):
            yield f"CONSOLE.{node.func.attr}() at line {node.lineno}", node
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            yield f"raise {ast.dump(node.exc.func)[:60]} at line {node.lineno}", node.exc


def _messages_in_file(path: Path) -> List[tuple]:
    tree = ast.parse(path.read_text(), filename=str(path))
    results = []
    for label, call in _iter_message_call_sites(tree):
        for arg in call.args:
            for piece in _literal_string_pieces(arg):
                results.append((label, piece))
    return results


def _all_bathyfacto_py_files() -> List[Path]:
    return sorted(BATHYFACTO_DIR.rglob("*.py"))


def test_scan_covers_at_least_the_known_message_modules():
    """Guards the scan itself: if the package layout changes, this must still find files."""
    scanned = _all_bathyfacto_py_files()
    scanned_names = {p.name for p in scanned}
    for expected in ("build_dataset.py", "metashape_xml.py", "normalization.py", "bathyfacto_dataparser.py"):
        assert expected in scanned_names, f"{expected} not found under {BATHYFACTO_DIR}, scan root is wrong"


def test_no_console_or_exception_message_uses_em_dash_en_dash_or_semicolon():
    violations = []
    for path in _all_bathyfacto_py_files():
        for label, piece in _messages_in_file(path):
            for banned in BANNED_CHARACTERS:
                if banned in piece:
                    violations.append(f"{path.relative_to(BATHYFACTO_DIR.parent)} {label}: {piece!r}")
    assert violations == [], "banned punctuation (em dash / en dash / semicolon) in a message:\n" + "\n".join(
        violations
    )


def test_literal_string_pieces_extracts_fstring_constant_parts_not_expression_slots():
    """Guards the extractor itself against a change that silently stops checking f-strings.

    Python's parser merges adjacent string literals at parse time, including the trailing
    constant part of an f-string with an adjacent plain string (``f"a {b} c" "d"`` parses to
    one ``JoinedStr`` whose last constant piece is already ``" cd"``), so that is what the
    extractor sees and must return.
    """
    tree = ast.parse('CONSOLE.log(f"a {b} c" "d")')
    (call,) = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    pieces = _literal_string_pieces(call.args[0])
    assert pieces == ["a ", " cd"]
    assert "b" not in "".join(pieces), "the {b} expression slot must not leak into the literal pieces"
