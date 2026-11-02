"""Local Smart Guided scheduling policies, not provider prices or quota accounting."""

from dataclasses import dataclass

from h3_app.errors import H3Error

AUTO = "Auto"
REQUESTS = "Request optimized"
TOKENS = "Token optimized"
BUDGET_PROFILES = (AUTO, REQUESTS, TOKENS)


@dataclass(frozen=True)
class Budget:
    name: str
    lines: int
    characters: int
    neighbors: int
    voices: int
    compact: bool


REQUEST_BUDGET = Budget(REQUESTS, 32, 9000, 2, 4, False)
TOKEN_BUDGET = Budget(TOKENS, 24, 6500, 1, 3, True)


def resolve_budget(profile=AUTO, *, model="", backend="Gemini", openai_model=""):
    if profile not in BUDGET_PROFILES:
        raise H3Error(f"Unknown Smart budget profile: {profile!r}.")
    if profile == AUTO:
        selected = (openai_model if backend == "OpenAI-compatible" else model).strip().lower()
        # Route by the active model, including compatible gateways exposing Gemini.
        # Unknown/custom models use the conservative token policy; explicit overrides win.
        profile = REQUESTS if "gemini" in selected and "flash-lite" in selected else TOKENS
    return REQUEST_BUDGET if profile == REQUESTS else TOKEN_BUDGET
