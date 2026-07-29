from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()  # keep the console clean


class BaseLLM(ABC):
    """Abstract base class for a chat-tuned causal LLM
    used to answer queries."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        """Load the tokenizer and model weights.

        Args:
            model_name: The HuggingFace model identifier to load.
            device: Compute device ('cpu', 'cuda', 'mps'). Auto-selected
                when not provided or invalid.
            trust_remote_code: Passed through to `from_pretrained`.

        Raises:
            OSError: If the model or tokenizer cannot be downloaded/loaded.
        """
        self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def _resolve_device(self, device: str | None) -> None:
        """Validate the requested device and set `self.device`/`self.dtype`.

        Args:
            device: The device requested by the caller, or None to
                auto-select. If it is set but not one of the supported
                values, a warning is printed and auto-selection is used
                instead.
        """
        valid_devices = ("mps", "cuda", "cpu")

        if device in valid_devices:
            self.device = device
        else:
            self.device = self._select_device()
            if device is not None:
                print(f"[WARNING]: '{device}' not found, "
                      f"continuing using '{self.device}'")

        if self.device in ("cuda", "mps"):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    @staticmethod
    def _select_device() -> str:
        """Pick the best available compute device: mps > cuda > cpu."""
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @abstractmethod
    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format a (system, user) message pair into the model's prompt string.

        Args:
            system_prompt: Instructions describing the assistant's role
                (e.g. "answer only from the given context").
            user_prompt: The actual question, typically with retrieved
                context inlined.

        Returns:
            A single string ready to be tokenized and fed to the model.
        """

    def truncate_to_token_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to at most `max_tokens` tokens,
        using this model's tokenizer.

        Used to enforce a hard, real token limit on the context passed
        into the prompt (as opposed to a character-count approximation,
        which is only a rough proxy for token count).

        Args:
            text: The text to truncate (typically the assembled context).
            max_tokens: Maximum number of tokens to keep.

        Returns:
            `text` unchanged if it already fits; otherwise the prefix of
            `text` that fits within `max_tokens` tokens. Returns an
            empty string if truncation itself fails, instead of raising.
        """
        try:
            token_ids = self.tokenizer(
                text, add_special_tokens=False)["input_ids"]
            if len(token_ids) <= max_tokens:
                return text
            return str(
                self.tokenizer.decode(token_ids[:max_tokens],
                                      skip_special_tokens=True)
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[llm] token truncation failed: {exc}")
            return ""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate a natural-language answer for a given prompt.

        Args:
            system_prompt: Instructions describing the assistant's role.
            user_prompt: The question, typically with retrieved context.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The generated answer text. Returns an empty string if
            generation fails, instead of crashing the pipeline.
        """
        try:
            prompt = self.build_prompt(system_prompt, user_prompt)
            inputs = self.tokenizer(
                prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(  # type: ignore[misc]
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Only decode the newly generated tokens, not the prompt itself.
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            return str(self.tokenizer.decode(new_tokens,
                                             skip_special_tokens=True)).strip()
        except (RuntimeError, ValueError) as exc:
            print(f"[llm] generation failed: {exc}")
            return ""


class Qwen3LLM(BaseLLM):
    """LLM wrapper for Qwen/Qwen3-0.6B."""

    def __init__(self, device: str | None = None) -> None:
        """Initialize Qwen3-0.6B.

        Args:
            device: Compute device ('cpu', 'cuda', 'mps'). Auto-selected
                when not provided.
        """
        super().__init__(model_name="Qwen/Qwen3-0.6B", device=device)

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Build the prompt using Qwen3's chat template, thinking mode off.

        Thinking mode is disabled since RAG answers should be short and
        source-grounded rather than accompanied by a long reasoning trace.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return str(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
