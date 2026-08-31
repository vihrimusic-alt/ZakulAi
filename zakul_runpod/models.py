"""Lazy, warm-reused ACE-Step models; never silently fall back to CPU."""

from pathlib import Path
from typing import Callable

from loguru import logger

from .config import MODELS, Settings
from .validation import GenerateRequest


class Models:
    """Own one DiT and one optional LM for a synchronous RunPod worker."""

    def __init__(self, settings: Settings):
        """Keep imports and model loading outside the worker's lightweight health path."""
        self.settings = settings
        self.dit = None
        self.lm = None

    def health(self) -> dict:
        """Report CUDA and actual model readiness without downloading weights."""
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "cuda_available": available,
            "gpu": torch.cuda.get_device_name(0) if available else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            if available else 0,
            "torch_version": str(torch.__version__),
            "configured_model": self.settings.model,
            "configured_lm_model": self.settings.lm_model,
            "models_initialized": self.dit is not None,
            "llm_initialized": self.lm is not None,
        }

    def ensure_loaded(self, need_lm: bool, progress: Callable[[str], None]) -> None:
        """Load models once, failing explicitly if CUDA or initialization is unavailable."""
        import torch
        from acestep.gpu_config import get_gpu_config, set_global_gpu_config

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a GPU worker; CPU fallback is disabled")
        set_global_gpu_config(get_gpu_config())
        self.settings.prepare()
        if self.dit is None:
            from acestep.handler import AceStepHandler

            progress("Loading ACE-Step and checking model downloads")
            candidate = AceStepHandler()
            status, success = candidate.initialize_service(
                project_root=str(self.settings.root), config_path=self.settings.model,
                device="cuda", use_flash_attention=False, compile_model=False,
                offload_to_cpu=self.settings.offload, offload_dit_to_cpu=False,
                prefer_source="huggingface",
            )
            if not success:
                logger.error("ACE-Step initialization failed: {}", status)
                raise RuntimeError("ACE-Step model initialization failed; check worker logs")
            self.dit = candidate
        if need_lm and self.lm is None:
            self._load_lm(progress)

    def _load_lm(self, progress: Callable[[str], None]) -> None:
        from acestep.llm_inference import LLMHandler
        from acestep.model_downloader import ensure_lm_model, ensure_main_model

        progress("Loading the ACE-Step language model")
        if self.settings.lm_model == "acestep-5Hz-lm-1.7B":
            success, status = ensure_main_model(
                checkpoints_dir=self.settings.checkpoints, prefer_source="huggingface",
            )
        else:
            success, status = ensure_lm_model(
                model_name=self.settings.lm_model, checkpoints_dir=self.settings.checkpoints,
                prefer_source="huggingface",
            )
        if not success:
            logger.error("LM download failed: {}", status)
            raise RuntimeError("Language model download failed; check worker logs")
        candidate = LLMHandler()
        status, success = candidate.initialize(
            checkpoint_dir=str(self.settings.checkpoints), lm_model_path=self.settings.lm_model,
            backend="pt", device="cuda", offload_to_cpu=self.settings.lm_offload,
        )
        if not success:
            logger.error("LM initialization failed: {}", status)
            raise RuntimeError("Language model initialization failed; check worker logs")
        self.lm = candidate

    def plan_duration(self, request: GenerateRequest) -> float:
        """Let the loaded ACE-Step LM plan duration; preserve the original lyrics."""
        from acestep.inference import format_sample
        from .validation import number

        if self.lm is None:
            raise RuntimeError("Automatic duration needs the language model")
        metadata = {"vocal_language": request.language}
        if request.bpm is not None:
            metadata["bpm"] = request.bpm
        result = format_sample(
            llm_handler=self.lm, caption=request.prompt + (f". Create a short preview with a musical ending, at most {request.max_duration:g} seconds." if request.max_duration < 240 else ""), lyrics=request.lyrics,
            user_metadata=metadata, temperature=0.68,
        )
        if not result.success:
            raise RuntimeError("AI duration planning failed; no fixed duration was substituted")
        try:
            return min(number(result.duration, "AI planned duration", 10, 600 if request.max_duration < 240 else 240), request.max_duration)
        except ValueError as exc:
            raise ValueError("AI did not return a duration within 10–240 seconds. Shorten the lyrics or review the worker limit; the song was not truncated.") from exc

    def assist(self, data: dict) -> dict:
        """Generate actual text with the already-loaded music LM, not templates."""
        import re
        from acestep.inference import create_sample
        action = data["action"]
        query = data["query"] + "\nMusic caption must describe sound only: no nationalities, countries, artists or singing-language labels. Return original content, not instructions to the user."
        language = data.get("vocal_language", "unknown")
        result = create_sample(
            llm_handler=self.lm, query=query, instrumental=action == "style" and data.get("instrumental", False),
            vocal_language=None if language == "unknown" else language,
            temperature=0.95 if action == "style" else 0.88, top_p=0.92,
        )
        if not result.success:
            raise RuntimeError("AI writing failed; no template substituted")
        caption = str(result.caption or "").strip()
        caption = re.sub(r"\b(Ukrainian|English|Polish|German|French|Spanish|Portuguese|Russian)(?:[- ]language)?\b", "", caption, flags=re.I)
        caption = re.sub(r"[ \t]{2,}", " ", caption).strip(" ,.")[:2400]
        lyrics = str(result.lyrics or "").strip() if action == "lyrics" else ""
        if not caption or (action == "lyrics" and (len(lyrics) < 80 or len(lyrics)>4096)):
            raise RuntimeError("AI returned incomplete text; no template substituted")
        return {"operation":"assist", "caption":caption, "lyrics":lyrics, "model":self.settings.lm_model}

    def generate(
        self,
        request: GenerateRequest,
        seed: int,
        folder: Path,
        reference_audio: Path | None = None,
    ) -> Path:
        """Generate one lossless source, optionally conditioned by private reference audio."""
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        steps, guidance = MODELS[self.settings.model]
        instruction = "Strictly instrumental, no singing or speech." if request.instrumental else (
            "Sing the supplied lyrics verbatim in their original language; do not translate."
        )
        params = GenerationParams(
            task_type="text2music", caption=f"{request.prompt}\n{instruction}",
            reference_audio=str(reference_audio) if reference_audio else None,
            lyrics=request.lyrics, instrumental=request.instrumental,
            duration=max(10.0, request.duration), bpm=request.bpm, keyscale=request.keyscale,
            vocal_language=request.language, inference_steps=steps, guidance_scale=guidance,
            shift=3.0, seed=seed, thinking=request.thinking,
            use_cot_caption=False, use_cot_language=False, use_cot_metas=False,
            lm_temperature=0.68, lm_cfg_scale=2.5, lm_top_p=0.9,
        )
        config = GenerationConfig(
            batch_size=1, allow_lm_batch=False, use_random_seed=False,
            seeds=[seed], audio_format="wav",
        )
        result = generate_music(self.dit, self.lm, params, config, save_dir=str(folder))
        if not result.success or len(result.audios) != 1:
            logger.error("ACE-Step generation failed: {}", result.error or result.status_message)
            raise RuntimeError("ACE-Step could not generate one audio take; check worker logs")
        raw_path = result.audios[0].get("path")
        if not raw_path:
            raise RuntimeError("ACE-Step returned no saved audio file")
        source = Path(raw_path).resolve()
        if not source.is_relative_to(folder.resolve()) or not source.is_file():
            raise RuntimeError("ACE-Step returned an invalid output path")
        return source
