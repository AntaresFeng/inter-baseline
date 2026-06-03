"""Deep-merge utility for nested config dicts.

highway-env uses shallow dict.update() which replaces entire nested dicts
instead of merging them.  This module provides a recursive merge that
preserves structure at every nesting level.
"""


def deep_merge(base: dict, *overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (left-to-right, last wins).

    Rules
    -----
    - Both values are dicts  → recurse.
    - Both values are lists  → override replaces base (no index merge).
    - Otherwise              → override replaces base.

    Returns a **new** dict; *base* and *overrides* are never mutated.
    """
    result = base.copy()
    for override in overrides:
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    return result
