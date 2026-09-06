import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONFIG_DIR = Path(__file__).parent
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"

def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found at {SETTINGS_FILE}")
    
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

SETTINGS = load_settings()

# Resolve provider configuration dynamically
LLM_CONFIG = SETTINGS.get("llm", {})
ACTIVE_PROVIDER = LLM_CONFIG.get("provider", "local").lower()

if ACTIVE_PROVIDER == "local":
    LLM_API_KEY = ""
    LLM_BASE_URL = "local"
    LLM_MODEL = LLM_CONFIG.get("local", {}).get("model", "Qwen/Qwen3-1.7B")
    LLM_DEVICE = LLM_CONFIG.get("local", {}).get("device", "cpu")
elif ACTIVE_PROVIDER == "openai":
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_BASE_URL = LLM_CONFIG.get("openai", {}).get("base_url", "https://api.openai.com/v1")
    LLM_MODEL = LLM_CONFIG.get("openai", {}).get("model", "gpt-4o-mini")
    LLM_DEVICE = None
else:  # openrouter
    LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    LLM_BASE_URL = LLM_CONFIG.get("openrouter", {}).get("base_url", "https://openrouter.ai/api/v1")
    LLM_MODEL = LLM_CONFIG.get("openrouter", {}).get("model", "meta-llama/llama-2-7b-chat:free")
    LLM_DEVICE = None

LLM_TEMPERATURE = LLM_CONFIG.get("temperature", 0.2)
LLM_MAX_TOKENS = LLM_CONFIG.get("max_tokens", 512)


def get_llm():
    """Factory function to create LLM instance based on configured provider."""
    if ACTIVE_PROVIDER == "local":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        device = LLM_DEVICE if LLM_DEVICE in ["cpu", "cuda"] else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer from cache
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            device=-1 if device == "cpu" else 0,
        )
        return HuggingFacePipeline(model_kwargs={"temperature": LLM_TEMPERATURE}, pipeline=pipe)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )