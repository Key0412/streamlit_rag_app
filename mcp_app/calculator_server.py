from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")

# ── In-memory state ───────────────────────────────────────────────────────────
# This dict persists for the lifetime of the server process.
_notes: dict[str, str] = {}

# ── Arithmetic tools ──────────────────────────────────────────────────────────

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# ── Note tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def save_note(key: str, value: str) -> str:
    """Save a note under the given key. Returns a confirmation message."""
    _notes[key] = value
    return f"Note saved under key '{key}'"

@mcp.tool()
def get_notes() -> dict:
    """Return all saved notes as a dictionary."""
    return _notes

# ── Resource ──────────────────────────────────────────────────────────────────

@mcp.resource("cheatsheet://math")
def math_cheatsheet() -> str:
    """A short math reference for the LLM to consult."""
    return """
    Math Cheat-Sheet
    ================
    • Addition       : a + b
    • Subtraction    : a - b
    • Multiplication : a * b
    • Division       : a / b  (b ≠ 0)
    • Square root    : a ** 0.5
    """

if __name__ == "__main__":
    mcp.run(transport="stdio")