def explain(concept: str, context: str = "") -> str:
    """STUB: Explain a concept across papers"""
    context_str = f" (Context: {context})" if context else ""
    return f"[STUB] Explaining concept: '{concept}'{context_str}"
