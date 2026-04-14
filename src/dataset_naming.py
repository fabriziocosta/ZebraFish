from __future__ import annotations


def build_mechanism_filename_prefix(selected_mechanisms: list[str]) -> str:
    if not selected_mechanisms:
        raise ValueError("selected_mechanisms must not be empty")

    mechanism_parts: list[str] = []
    for mechanism in selected_mechanisms:
        terms = [term for term in mechanism.split("_") if term]
        if not terms:
            raise ValueError(f"mechanism {mechanism!r} must contain at least one non-empty term")
        mechanism_parts.extend(term[:2] for term in terms[:2])
    return "_".join(mechanism_parts)
