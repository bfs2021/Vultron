#!/usr/bin/env python

#  Copyright (c) 2026 Carnegie Mellon University and Contributors.
#  - see Contributors.md for a full list of Contributors
#  - see ContributionInstructions.md for information on how you can Contribute to this project
#  Vultron Multiparty Coordinated Vulnerability Disclosure Protocol Prototype is
#  licensed under a MIT (SEI)-style license, please see LICENSE.md distributed
#  with this Software or contact permission@sei.cmu.edu for full terms.
#  Created, in part, with funding and support from the United States Government
#  (see Acknowledgments file). This program may include and/or can make use of
#  certain third party source code, object code, documentation and other files
#  ("Third Party Software"). See LICENSE.md for more details.
#  Carnegie Mellon®, CERT® and CERT Coordination Center® are registered in the
#  U.S. Patent and Trademark Office by Carnegie Mellon University
"""Architecture invariant: received-side execute_with_setup must use receiving_actor_id.

Spec BT-17-006 (MUST): any received-side use case that calls
``execute_with_setup`` MUST pass the *receiving* actor — the one
``resolve_receiving_actor_id`` returns (``request.receiving_actor_id``, else
the store's owner, BT-17-009) — and never any other actor expression: not
``actor_id=request.actor_id`` (the sender), not the sender under another
binding, not a literal, not another attribute of the request, not a
conditional that can fall back to the sender.  Passing the sender ID scopes
the DataLayer and GuardedCommit to the wrong actor, silently attributing
ledger entries and state transitions to the remote peer.

The check is positive: every received-side ``execute()`` call of
``execute_with_setup`` must pass the resolved receiving actor — a call to
``resolve_receiving_actor_id``, a local bound from it, a
``receiving_actor_id`` attribute, or a ``receiving_actor_id`` name — and the
negative fixtures prove every other shape is rejected.

Issue: #2338
"""

import ast
from pathlib import Path

from test.architecture import _corpus

_RECEIVED_ROOT = (
    _corpus.REPO_ROOT / "vultron" / "core" / "use_cases" / "received"
)


def _is_execute_with_setup_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute) and func.attr == "execute_with_setup"
    ) or (isinstance(func, ast.Name) and func.id == "execute_with_setup")


def _walk_own_scope(node: ast.AST):
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield from _walk_own_scope(child)


def _is_resolve_receiving_call(node: ast.expr) -> bool:
    """Return True for ``resolve_receiving_actor_id(...)`` (bare or method form)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Name) and func.id == "resolve_receiving_actor_id"
    ) or (
        isinstance(func, ast.Attribute)
        and func.attr == "resolve_receiving_actor_id"
    )


def _resolved_receiver_names(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Names the function binds to the receiving-actor resolver's result."""
    names: set[str] = set()
    for node in ast.walk(func):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and _is_resolve_receiving_call(
                node.value
            ):
                names.add(target.id)
    return names


def _passes_receiving_actor(call: ast.Call, resolved: set[str]) -> bool:
    """True when the call's ``actor_id`` is the resolved receiving actor."""
    for kw in call.keywords:
        if kw.arg != "actor_id":
            continue
        value = kw.value
        if _is_resolve_receiving_call(value):
            return True
        is_receiving_attr = (
            isinstance(value, ast.Attribute)
            and value.attr == "receiving_actor_id"
        )
        is_receiver_name = isinstance(value, ast.Name) and (
            value.id == "receiving_actor_id" or value.id in resolved
        )
        return is_receiving_attr or is_receiver_name
    return False


def _label(source_path: Path) -> str:
    try:
        return source_path.relative_to(_corpus.REPO_ROOT).as_posix()
    except ValueError:
        return str(source_path)


def _violations_in_execute(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    label: str,
) -> list[str]:
    """Return 'label:lineno' for every execute_with_setup without the receiving actor."""
    results: list[str] = []
    resolved = _resolved_receiver_names(func)
    for child in _walk_own_scope(func):
        if not _is_execute_with_setup_call(child):
            continue
        assert isinstance(child, ast.Call)
        if not _passes_receiving_actor(child, resolved):
            results.append(f"{label}:{child.lineno}")
    return results


def _collect_violations_from_tree(tree: ast.AST, label: str) -> list[str]:
    """Return 'file:lineno' entries where execute_with_setup misses the receiving actor."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "execute":
            continue
        violations.extend(_violations_in_execute(node, label))
    return violations


def _collect_violations(source_path: Path) -> list[str]:
    """Scan a single file (including tmp_path synthetic files not in corpus)."""
    try:
        source = source_path.read_text(encoding="utf-8")
        tree = _corpus.parse_inline(source, filename=str(source_path))
    except (OSError, SyntaxError):
        return []
    return _collect_violations_from_tree(tree, _label(source_path))


def test_received_side_execute_with_setup_uses_the_receiving_actor():
    """Every received-side execute() passes the resolved receiving actor to execute_with_setup.

    Spec BT-17-006. Issue #2338.
    """
    all_violations: list[str] = []
    for py_file, tree in _corpus.files_mentioning(
        "execute_with_setup", under=_RECEIVED_ROOT
    ):
        all_violations.extend(
            _collect_violations_from_tree(tree, _label(py_file))
        )

    assert not all_violations, (
        "received-side execute() calls pass something other than the resolved"
        " receiving actor (the resolver's call, a binding of its result, or"
        " request.receiving_actor_id — BT-17-006, issue #2338):\n"
        + "\n".join(f"  {v}" for v in sorted(all_violations))
    )


def test_the_positive_check_rejects_any_other_actor_expression(tmp_path):
    """Negative fixtures: a literal, the sender (directly, under another binding, via
    ``self``, or behind a conditional fallback), or another request attribute must all
    fail the check — only the resolved receiving actor passes."""
    bad = [
        'actor_id="hard-coded-id"',
        "actor_id=req.actor_id",
        "actor_id=self._request.actor_id",
        "actor_id=request.object_id",
        # A conditional that can still hand the call the sender is not the receiving actor.
        "actor_id=request.receiving_actor_id\n"
        "                if request.receiving_actor_id is not None\n"
        "                else request.actor_id",
    ]
    for arg in bad:
        source = (
            "class FakeUseCase:\n"
            "    def execute(self, request, req):\n"
            "        resolved = resolve_receiving_actor_id(dl, request.receiving_actor_id)\n"
            f"        bridge.execute_with_setup(tree=tree, {arg}, activity=request)\n"
        )
        path = tmp_path / "bad.py"
        path.write_text(source, encoding="utf-8")
        assert _collect_violations(path), (
            f"not rejected: {arg.splitlines()[0]}"
        )


def test_the_positive_check_accepts_the_resolved_receiving_actor(tmp_path):
    """Positive fixtures: the resolver's call, a binding of its result, a
    ``receiving_actor_id`` attribute and a ``receiving_actor_id`` name all pass."""
    good = [
        "actor_id=request.receiving_actor_id",
        "actor_id=resolve_receiving_actor_id(dl, request.receiving_actor_id)",
        "actor_id=receiving_actor_id",
        "actor_id=resolved",
    ]
    for arg in good:
        source = (
            "class FakeUseCase:\n"
            "    def execute(self, request, req):\n"
            "        resolved = resolve_receiving_actor_id(dl, request.receiving_actor_id)\n"
            f"        bridge.execute_with_setup(tree=tree, {arg}, activity=request)\n"
        )
        path = tmp_path / "good.py"
        path.write_text(source, encoding="utf-8")
        assert not _collect_violations(path), f"wrongly rejected: {arg}"
