"""
IsoCortex Desktop App — LLM Wrapper (llama.cpp)
=================================================
Loads a GGUF model via llama-cpp-python and provides
streaming token generation for the RAG chat pipeline.

Model: Qwen 2.5 1.5B Instruct Q4_K_M (~950 MB)
Storage: ~/.isocortex/models/
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from pathlib import Path
from typing import Callable, Generator, Optional

logger = logging.getLogger("IsoCortex.llm")

# ── Configuration ────────────────────────────────────────────────────

MODELS_DIR = Path.home() / ".isocortex" / "models"
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH = MODELS_DIR / MODEL_FILE

# Generation defaults — tuned for Qwen 2.5 1.5B Instruct (small model)
DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.45
DEFAULT_TOP_P = 0.85
DEFAULT_TOP_K = 40
DEFAULT_REPETITION_PENALTY = 1.2
# NOTE: llama-cpp-python does NOT support frequency_penalty — do NOT add it.
CONTEXT_WINDOW = 4096

# System prompt for RAG — optimized for small model context budget
SYSTEM_PROMPT = (
    "You are IsoCortex AI, a helpful offline document assistant.\n\n"
    "RESPONSE STYLE:\n"
    "- Be direct and concise. Answer the question, then stop.\n"
    "- Use short paragraphs (2-3 sentences max).\n"
    "- Use bullet points for lists. Keep each bullet to 1 line.\n"
    "- When you have fully answered, STOP. Never add filler.\n"
    "- NEVER repeat any sentence, bullet, or paragraph twice.\n"
    "- Each point you make must be unique and add new information.\n\n"
    "DOCUMENT QUESTIONS (when context is provided):\n"
    "- Answer using ONLY the provided document context.\n"
    "- Cite sources as [Source N].\n"
    '- If the context lacks information, say: "Based on the indexed documents, '
    'I don\'t have enough information to answer this." Then give a brief '
    'general answer if you know one, labeled [General Knowledge].\n'
    "- Do NOT fabricate information or cite sources not provided.\n\n"
    "META QUESTIONS (about IsoCortex itself, file counts, indexes):\n"
    "- Use the [Library Stats] section if provided.\n"
    "- IsoCortex is an offline desktop app for indexing, searching, and chatting "
    "with personal documents. It uses local AI models (Qwen 2.5 1.5B + MiniLM embeddings) "
    "with zero cloud dependencies. Supports PDF, DOCX, PPTX, XLSX, code, text.\n"
)

# HuggingFace download URL pattern
_HF_BASE = "https://huggingface.co"


# ======================================================================
# Model Download
# ======================================================================

def model_exists() -> bool:
    """Check if the GGUF model file is already downloaded."""
    return MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000_000


def download_model(
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    """
    Download the GGUF model from HuggingFace.

    Args:
        on_progress: Callback(downloaded_bytes, total_bytes, status_text)

    Returns:
        Path to the downloaded model file.

    Raises:
        RuntimeError if download fails.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading model: %s/%s", MODEL_REPO, MODEL_FILE)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for model download. "
            "Install it with: pip install huggingface_hub"
        )

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,  # type: ignore[arg-type]
    )

    # huggingface_hub may download to a blob path; move to expected name
    downloaded = Path(downloaded_path)
    if downloaded.name != MODEL_FILE:
        target = MODELS_DIR / MODEL_FILE
        if not target.exists():
            downloaded.rename(target)
            downloaded = target

    logger.info("Model downloaded to: %s", downloaded)
    return downloaded


def download_model_with_progress(
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    """
    Download with manual progress tracking using requests.
    Falls back to hf_hub_download if requests fails.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{_HF_BASE}/{MODEL_REPO}/resolve/main/{MODEL_FILE}"

    try:
        import requests

        if on_progress:
            on_progress(0, 1, "Connecting to HuggingFace...")

        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        total_size = int(resp.headers.get("content-length", 0))
        if total_size == 0:
            # Unknown size — fall back to hf_hub_download
            logger.warning("Unknown content-length, falling back to hf_hub_download")
            if on_progress:
                on_progress(0, 1, "Downloading (size unknown)...")
            return download_model(on_progress)

        downloaded = 0
        tmp_path = MODELS_DIR / f"{MODEL_FILE}.tmp"

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total_size, "Downloading model...")

        # Atomic replace
        os.replace(str(tmp_path), str(MODEL_PATH))

        logger.info("Model downloaded to: %s (%d bytes)", MODEL_PATH, total_size)
        return MODEL_PATH

    except Exception as exc:
        logger.warning("Requests download failed (%s), falling back to hf_hub_download", exc)
        if on_progress:
            on_progress(0, 1, "Downloading via HuggingFace Hub...")
        return download_model(on_progress)


def _detect_gpu_layers() -> int:
    """Auto-detect how many model layers to offload to GPU.

    Returns:
        -1 for full GPU offload (Metal on macOS, CUDA on Windows/Linux),
        or 0 for CPU fallback.
    """
    system = platform.system()
    if system == "Darwin":
        # macOS — Metal GPU via llama-cpp-python Metal build
        logger.info("macOS detected — enabling Metal GPU offload (all layers)")
        return -1
    if system == "Windows" or system == "Linux":
        # Check for CUDA / Vulkan support by probing llama_cpp
        try:
            from llama_cpp import llama_cpp  # type: ignore[import-untyped]
            # If llama-cpp-python was compiled with CUDA, offloading works
            if hasattr(llama_cpp, "llama_init_backend"):
                logger.info("%s detected — enabling GPU offload (all layers)", system)
                return -1
        except Exception:
            pass
        logger.info("%s detected — no GPU backend found, using CPU", system)
        return 0
    logger.info("Unknown platform %s — using CPU", system)
    return 0


# ======================================================================
# LLM Wrapper
# ======================================================================

class LLM:
    """
    Wraps llama-cpp-python for streaming text generation.

    Usage:
        llm = LLM()
        llm.load_model()  # blocking, ~10-20s

        for token_text in llm.stream("Hello", system="You are helpful."):
            print(token_text, end="", flush=True)
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or MODEL_PATH
        self._llm = None  # llama-cpp-python Llama instance
        self._loaded = False
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._llm is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load_model(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Load the GGUF model into memory. Thread-safe.

        Args:
            on_progress: Callback(status_text) for UI feedback.

        Returns:
            True if loaded successfully, False otherwise.
        """
        with self._lock:
            if self._loaded:
                return True

            if not self._model_path.exists():
                self._load_error = (
                    f"Model file not found: {self._model_path}\n"
                    "Please download the model first."
                )
                logger.error(self._load_error)
                return False

            try:
                if on_progress:
                    on_progress("Loading AI model...")

                from llama_cpp import Llama as _Llama

                if on_progress:
                    on_progress("Initializing llama.cpp...")

                # Auto-detect GPU offload for faster loading & inference
                n_gpu = _detect_gpu_layers()
                if on_progress:
                    on_progress(f"Initializing llama.cpp (GPU layers: {n_gpu})...")

                self._llm = _Llama(
                    model_path=str(self._model_path),
                    n_ctx=CONTEXT_WINDOW,
                    n_gpu_layers=n_gpu,
                    verbose=False,
                )

                self._loaded = True
                self._load_error = None
                logger.info(
                    "LLM loaded  model=%s  ctx=%d",
                    self._model_path.name,
                    CONTEXT_WINDOW,
                )
                return True

            except ImportError:
                self._load_error = (
                    "llama-cpp-python is not installed. "
                    "Install with: pip install llama-cpp-python"
                )
                logger.error(self._load_error)
                return False
            except Exception as exc:
                self._load_error = f"Failed to load model: {exc}"
                logger.error(self._load_error, exc_info=True)
                return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Generate a complete response (non-streaming).

        Returns the full generated text.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        assert self._llm is not None
        llm = self._llm  # local ref to prevent race with unload()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=DEFAULT_TOP_K,
            repeat_penalty=DEFAULT_REPETITION_PENALTY,
            stop=stop or [],
            stream=False,
        )

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error("Malformed LLM response: %s", e)
            return ""

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        stop: Optional[list[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens one by one.

        Yields:
            Token text fragments as they are generated.

        Raises:
            RuntimeError if model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        assert self._llm is not None
        llm = self._llm  # local ref to prevent race with unload()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream_response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=DEFAULT_TOP_K,
            repeat_penalty=DEFAULT_REPETITION_PENALTY,
            stop=stop or [],
            stream=True,
        )

        for chunk in stream_response:
            try:
                delta = chunk["choices"][0].get("delta", {})
                token_text = delta.get("content", "")
                if token_text:
                    yield token_text
            except (KeyError, IndexError):
                continue

    def stream_chat(
        self,
        messages: list[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        stop: Optional[list[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens for a full chat conversation (multi-turn).

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
                      The caller is responsible for managing conversation history.

        Yields:
            Token text fragments.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        assert self._llm is not None
        llm = self._llm  # local ref to prevent race with unload()

        stream_response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=DEFAULT_TOP_K,
            repeat_penalty=DEFAULT_REPETITION_PENALTY,
            stop=stop or [],
            stream=True,
        )

        for chunk in stream_response:
            try:
                delta = chunk["choices"][0].get("delta", {})
                token_text = delta.get("content", "")
                if token_text:
                    yield token_text
            except (KeyError, IndexError):
                continue

    def unload(self):
        """Free the model from memory."""
        with self._lock:
            if self._llm is not None:
                try:
                    # llama-cpp-python doesn't have an explicit unload,
                    # but deleting the object frees the C memory
                    del self._llm
                except Exception:
                    pass
                self._llm = None
                self._loaded = False
                logger.info("LLM unloaded")