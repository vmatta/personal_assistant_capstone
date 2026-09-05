"""Shared helpers: retry wrapper for transient upstream LLM provider errors
(e.g. free-tier shared-pool rate limits on OpenRouter)."""

import json
import re

from openai import APIStatusError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_TEMPERATURE, SETTINGS, ACTIVE_PROVIDER


_OPENROUTER_FALLBACK_MODELS = (
    SETTINGS.get("llm", {})
    .get("openrouter", {})
    .get(
        "fallback_models",
        [
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
        ],
    )
)


def _is_openrouter_credit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "error code: 402" in text or "insufficient credits" in text


def _invoke_with_openrouter_fallback(messages):
    from langchain_openai import ChatOpenAI

    last_exc = None
    for model in _OPENROUTER_FALLBACK_MODELS:
        try:
            fallback_llm = ChatOpenAI(
                model=model,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return fallback_llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("No OpenRouter fallback models configured")


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=3, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def invoke_with_retry(llm, messages):
    """Invoke a LangChain chat model, retrying on transient upstream rate limits."""
    try:
        return llm.invoke(messages)
    except APIStatusError as exc:
        if ACTIVE_PROVIDER == "openrouter" and _is_openrouter_credit_error(exc):
            return _invoke_with_openrouter_fallback(messages)
        raise


def parse_json_response(text: str) -> dict:
    """Best-effort JSON parsing for LLM responses that may be wrapped in markdown code
    fences or preceded by explanatory text, instead of being strict raw JSON."""
    text = (text or "").strip()

    fence_match = re.search(r"```(?:json)?([^`]*)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))  # let this raise if still invalid

    raise ValueError("Could not parse JSON from LLM response")
