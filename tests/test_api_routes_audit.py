"""Audit fixes for src/docai/api/routes.py."""
import ast
import pathlib


ROUTES_PATH = pathlib.Path(__file__).parent.parent / "src" / "docai" / "api" / "routes.py"


def _get_routes_source():
    return ROUTES_PATH.read_text(encoding="utf-8")


def test_logger_defined_at_module_level():
    """routes.py must define `logger = logging.getLogger(...)` at module level (not inside a function or try/except)."""
    source = _get_routes_source()
    tree = ast.parse(source)
    # Only look at top-level (module-level) statements — not inside functions, classes, or try blocks
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "logger":
                    # Found a module-level assignment to 'logger' — test passes
                    return
    raise AssertionError(
        "routes.py has no module-level `logger = ...` assignment. "
        "Add `logger = logging.getLogger(__name__)` at module level."
    )


def test_get_document_markdown_has_auth_dependency():
    """get_document_markdown must declare user_id: str = Depends(get_current_user_id)."""
    source = _get_routes_source()
    assert "Depends(get_current_user_id)" in source, (
        "get_document_markdown is missing Depends(get_current_user_id) — endpoint is unauthenticated"
    )
    # Verify it's in the right function by checking the source around that function
    lines = source.splitlines()
    in_func = False
    for line in lines:
        if "async def get_document_markdown" in line:
            in_func = True
        if in_func and "Depends(get_current_user_id)" in line:
            return  # found in the right function
        if in_func and line.strip().startswith("async def ") and "get_document_markdown" not in line:
            break  # left the function without finding it
    raise AssertionError(
        "get_document_markdown does not have Depends(get_current_user_id) in its signature"
    )


def test_delete_document_uses_transaction():
    """delete_document must use async with conn.transaction() around the DELETE statements."""
    source = _get_routes_source()
    lines = source.splitlines()
    in_delete_func = False
    found_transaction = False
    for line in lines:
        if "async def delete_document" in line:
            in_delete_func = True
        if in_delete_func and "conn.transaction()" in line:
            found_transaction = True
            break
        if in_delete_func and line.strip().startswith("async def ") and "delete_document" not in line:
            break
    assert found_transaction, (
        "delete_document must wrap SQL deletes in `async with conn.transaction()`"
    )
