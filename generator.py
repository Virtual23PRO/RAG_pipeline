from typing import List, Optional
import os

from google import genai
from google.genai import types

from config import GenerationConfig


class LLMGenerator:
    def __init__(self, cfg: GenerationConfig, device: Optional[str] = None):
        self.cfg = cfg

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it before running the program."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = cfg.llm_name

        self.gen_config = types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )

        print(f"Ładowanie LLM (Gemini): {self.model_name}")

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        ctx_text = ""
        for i, c in enumerate(contexts, start=1):
            ctx_text += f"[Fragment {i}]\n{c}\n\n"

        prompt = (
            "Otrzymasz pytanie użytkownika oraz listę fragmentów tekstu z różnych dokumentów.\n"
            "Twoje zadanie:\n"
            "1. Odpowiadaj WYŁĄCZNIE na podstawie podanych fragmentów. "
            "   Nie korzystaj z żadnej wiedzy zewnętrznej ani własnych domysłów.\n"
            "2. Jeśli fragmenty pozwalają odpowiedzieć na pytanie, odpowiedz po polsku w 1–3 zdaniach, "
            "   opierając się tylko na tych fragmentach.\n"
            "3. Jeśli fragmenty nie zawierają wystarczających informacji, aby odpowiedzieć, "
            "   napisz dokładnie jedno zdanie: \"Nie wiem na podstawie podanych dokumentów\".\n\n"
            "FRAGMENTY DOKUMENTÓW:\n"
            f"{ctx_text}\n"
            "PYTANIE UŻYTKOWNIKA:\n"
            f"{question}\n\n"
            "ODPOWIEDŹ (1–3 zdania po polsku, tylko na podstawie dokumentów):"
        )

        approx_max_chars = self.cfg.max_context_tokens * 4
        if len(prompt) > approx_max_chars:
            prompt = prompt[-approx_max_chars:]

        return prompt

    def generate(self, question: str, contexts: List[str]) -> str:
        prompt = self._build_prompt(question, contexts)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.gen_config,
            )

            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return "[Model did not generate a response]"

            return text

        except Exception as e:
            print(f"[GEMINI ERROR] {e}")
            return "[An error occurred while generating the response]"