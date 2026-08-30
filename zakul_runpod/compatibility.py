"""Check the uploaded ACE-Step API contract during image build, without GPU imports."""

import ast
from pathlib import Path


def check_compatibility(root: Path) -> None:
    """Raise on missing API fields instead of deploying an incompatible engine silently."""
    requirements = {
        "acestep/inference.py": {
            "GenerationParams": {
                "task_type", "caption", "lyrics", "instrumental", "duration", "bpm",
                "keyscale", "vocal_language", "inference_steps", "guidance_scale", "shift",
                "seed", "thinking", "use_cot_caption", "use_cot_language", "use_cot_metas",
                "lm_temperature", "lm_cfg_scale", "lm_top_p",
            },
            "GenerationConfig": {"batch_size", "allow_lm_batch", "use_random_seed", "seeds", "audio_format"},
            "GenerationResult": {"success", "audios", "error", "status_message"},
            "generate_music": {"dit_handler", "llm_handler", "params", "config", "save_dir"},
        },
        "acestep/core/generation/handler/init_service_orchestrator.py": {
            "initialize_service": {
                "project_root", "config_path", "device", "use_flash_attention", "compile_model",
                "offload_to_cpu", "offload_dit_to_cpu", "prefer_source",
            },
        },
        "acestep/llm_inference.py": {
            "initialize": {"checkpoint_dir", "lm_model_path", "backend", "device", "offload_to_cpu"},
        },
        "acestep/model_downloader.py": {
            "ensure_main_model": {"checkpoints_dir", "prefer_source"},
            "ensure_lm_model": {"model_name", "checkpoints_dir", "prefer_source"},
        },
    }
    errors = []
    for filename, symbols in requirements.items():
        path = root / filename
        if not path.is_file():
            errors.append(f"Missing {filename}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
        nodes = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for name, expected in symbols.items():
            node = nodes.get(name)
            if isinstance(node, ast.ClassDef):
                available = {child.target.id for child in node.body
                             if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)}
            elif isinstance(node, ast.FunctionDef):
                available = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            else:
                available = set()
            if expected - available:
                errors.append(f"{filename}: {name} is missing {sorted(expected - available)}")
    if errors:
        raise RuntimeError("Unsupported ACE-Step snapshot:\n" + "\n".join(errors))


if __name__ == "__main__":
    check_compatibility(Path(__file__).resolve().parents[1])
    print("ACE-Step API contract validated for ZaKul RunPod Queue")
