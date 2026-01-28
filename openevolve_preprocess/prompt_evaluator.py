#!/usr/bin/env python3
"""
Prompt Configuration Evaluator for Phase 0: Meta-Prompt Evolution.

This evaluator assesses prompt configurations by:
1. Parsing the evolved YAML prompt configuration
2. Using the prompt (with template variations) to generate preprocessing code via LLM
3. Executing the generated code on the proxy dataset
4. Evaluating the preprocessing quality using frozen backbone

The evaluation flow:
    Prompt Config (YAML) → Parse → Generate Code → Execute → Evaluate

This allows us to evolve prompt configurations that consistently produce good code.

Usage (called by OpenEvolve):
    from prompt_evaluator import evaluate
    result = evaluate("path/to/prompt_config.yaml")
"""

import hashlib
import logging
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths
PROXY_DATA_PATH = os.environ.get(
    "PROXY_DATA_PATH",
    str(SCRIPT_DIR / "data" / "proxy_raw.h5ad")
)
GENEMAP_PATH = os.environ.get(
    "GENEMAP_PATH",
    str(PROJECT_ROOT / "data" / "genemap.csv")
)
BACKBONE_PATH = os.environ.get(
    "BACKBONE_PATH",
    str(PROJECT_ROOT / "examples" / "save" / "cellFM" / "CellFM_80M_weight.ckpt")
)

# LLM Configuration
LLM_API_BASE = os.environ.get(
    "LLM_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
LLM_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Cache for generated code
CODE_CACHE_DIR = SCRIPT_DIR / "output" / "code_cache"


def get_llm_client():
    """Get OpenAI-compatible client for code generation."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    if not LLM_API_KEY:
        raise ValueError(
            "No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY environment variable."
        )

    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
    )


def parse_prompt_config(config_path: str) -> Dict[str, Any]:
    """
    Parse and validate a prompt configuration YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle nested "prompt:" key
    if "prompt" in config:
        config = config["prompt"]

    # Validate required fields
    required_fields = ["system_message"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # Set defaults
    config.setdefault("num_top_programs", 3)
    config.setdefault("num_diverse_programs", 2)
    config.setdefault("include_artifacts", True)
    config.setdefault("use_template_stochasticity", False)
    config.setdefault("template_variations", {})

    return config


def apply_template_variations(system_message: str, variations: Dict[str, list]) -> str:
    """
    Apply stochastic template variations to the system message.

    Replaces placeholders like {task_framing} with random selections
    from the variations dictionary.

    Args:
        system_message: System message with {placeholders}
        variations: Dict mapping placeholder names to lists of options

    Returns:
        System message with placeholders replaced
    """
    result = system_message

    for placeholder, options in variations.items():
        if options and len(options) > 0:
            # Random selection from options
            selected = random.choice(options)
            # Replace placeholder (handle both {name} and {{name}} formats)
            result = result.replace(f"{{{placeholder}}}", selected)
            result = result.replace(f"{{{{{placeholder}}}}}", selected)

    return result


def generate_code_from_config(
    config: Dict[str, Any],
    max_retries: int = 3
) -> Tuple[str, Dict]:
    """
    Generate preprocessing code from a prompt configuration.

    Args:
        config: Parsed prompt configuration
        max_retries: Number of retries on failure

    Returns:
        Tuple of (generated_code, generation_metadata)
    """
    # Apply template variations if enabled
    system_message = config["system_message"]
    if config.get("use_template_stochasticity", False):
        variations = config.get("template_variations", {})
        system_message = apply_template_variations(system_message, variations)

    # Create cache key from the resolved system message
    cache_key = hashlib.md5(system_message.encode()).hexdigest()[:12]
    cache_path = CODE_CACHE_DIR / f"code_{cache_key}.py"

    # Check cache (disabled for stochastic prompts to allow variation)
    if not config.get("use_template_stochasticity", False) and cache_path.exists():
        logger.info(f"Using cached code: {cache_path}")
        with open(cache_path, "r") as f:
            code = f.read()
        return code, {"cached": True, "cache_path": str(cache_path)}

    # Generate code via LLM
    client = get_llm_client()

    # Code generation prompt
    code_gen_prompt = """Write a Python preprocessing function for scRNA-seq data.

The function must:
1. Be named `preprocess(raw_adata, genemap_path)`
2. Return an AnnData object with exactly 27,934 genes (genemap aligned)
3. Preserve obs['label'] column
4. Include all necessary imports

Output ONLY the Python code, no explanations."""

    for attempt in range(max_retries):
        try:
            logger.info(f"Generating code (attempt {attempt + 1}/{max_retries})...")

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": code_gen_prompt},
                ],
                temperature=0.3,
                max_tokens=4000,
            )

            raw_response = response.choices[0].message.content
            code = extract_python_code(raw_response)

            if not code:
                logger.warning("No valid Python code extracted")
                continue

            # Validate syntax
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as e:
                logger.warning(f"Syntax error in generated code: {e}")
                continue

            # Cache the code
            CODE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                f.write(code)

            metadata = {
                "cached": False,
                "model": LLM_MODEL,
                "attempt": attempt + 1,
                "response_length": len(raw_response),
                "code_length": len(code),
                "template_variations_applied": config.get("use_template_stochasticity", False),
            }

            return code, metadata

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

    raise RuntimeError(f"Failed to generate valid code after {max_retries} attempts")


def extract_python_code(text: str) -> Optional[str]:
    """
    Extract Python code from LLM response.

    Handles:
    - Code in ```python ... ``` blocks
    - Code in ``` ... ``` blocks
    - Raw code without formatting
    """
    # Try ```python blocks
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try ``` blocks
    pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Check if entire response is valid Python
    try:
        compile(text, "<string>", "exec")
        return text.strip()
    except SyntaxError:
        pass

    # Extract code starting with import/def
    lines = text.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        if line.strip().startswith(("import ", "from ", "def ", "class ", "#")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        code = "\n".join(code_lines)
        try:
            compile(code, "<string>", "exec")
            return code
        except SyntaxError:
            pass

    return None


def evaluate_generated_code(code: str) -> Dict[str, Any]:
    """
    Evaluate the generated preprocessing code.

    Uses the same evaluation logic as the code evaluator (evaluator.py).

    Args:
        code: Python code string

    Returns:
        Evaluation result dictionary
    """
    import anndata as ad
    import importlib.util
    import traceback

    # Import evaluator functions
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from evaluator import (
            evaluate_statistics,
            evaluate_frozen_backbone,
            extract_features,
            STAGE1_THRESHOLD,
        )
    finally:
        if str(SCRIPT_DIR) in sys.path:
            sys.path.remove(str(SCRIPT_DIR))

    # Save code to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Load module
        spec = importlib.util.spec_from_file_location("generated_preprocess", temp_path)
        module = importlib.util.module_from_spec(spec)

        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            spec.loader.exec_module(module)
        finally:
            for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")]:
                if p in sys.path:
                    sys.path.remove(p)

        if not hasattr(module, "preprocess"):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": "Generated code missing 'preprocess' function",
            }

        # Load proxy data
        if not os.path.exists(PROXY_DATA_PATH):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": f"Proxy data not found: {PROXY_DATA_PATH}",
            }

        raw_adata = ad.read_h5ad(PROXY_DATA_PATH)

        # Execute preprocessing
        logger.info("Executing generated preprocessing code...")
        processed_adata = module.preprocess(raw_adata.copy(), GENEMAP_PATH)

        if not isinstance(processed_adata, ad.AnnData):
            return {
                "combined_score": 0.0,
                "stage": 0,
                "error": f"preprocess() returned {type(processed_adata)}, expected AnnData",
            }

        # Stage 1: Statistical evaluation
        logger.info("Running statistical evaluation...")
        stats_score, stats_metrics = evaluate_statistics(processed_adata)

        if stats_score < STAGE1_THRESHOLD:
            return {
                "combined_score": stats_score * 0.3,
                "stage": 1,
                "stats_score": stats_score,
                "model_score": 0.0,
                "metrics": {"stage1": stats_metrics},
            }

        # Stage 2: Frozen backbone evaluation
        logger.info("Running frozen backbone evaluation...")
        model_score, model_metrics = evaluate_frozen_backbone(processed_adata)

        # Combined score
        combined_score = 0.3 * stats_score + 0.7 * model_score

        # Extract features
        complexity, norm_strength = extract_features(code, processed_adata)

        return {
            "combined_score": combined_score,
            "stage": 2,
            "stats_score": stats_score,
            "model_score": model_score,
            "preprocessing_complexity": complexity,
            "normalization_strength": norm_strength,
            "metrics": {
                "stage1": stats_metrics,
                "stage2": model_metrics,
            },
        }

    except Exception as e:
        return {
            "combined_score": 0.0,
            "stage": 0,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def extract_prompt_features(config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Extract feature dimensions from a prompt config for MAP-Elites.

    Features:
    - prompt_complexity: Based on system_message length and structure (0-1)
    - variation_diversity: Based on template_variations count (0-1)

    Args:
        config: Parsed prompt configuration

    Returns:
        Tuple of (prompt_complexity, variation_diversity)
    """
    system_message = config.get("system_message", "")

    # Prompt complexity (based on length and structure)
    length = len(system_message)
    complexity = min(1.0, length / 3000)  # Normalize to 0-1

    # Variation diversity
    variations = config.get("template_variations", {})
    total_variations = sum(len(v) for v in variations.values() if isinstance(v, list))
    diversity = min(1.0, total_variations / 15)  # Normalize to 0-1

    return complexity, diversity


def evaluate(config_path: str) -> Dict[str, Any]:
    """
    Main evaluation function called by OpenEvolve.

    Evaluates a prompt configuration by:
    1. Parsing the YAML configuration
    2. Generating code from the prompt via LLM
    3. Executing and evaluating the generated code

    Args:
        config_path: Path to the prompt configuration YAML file

    Returns:
        Evaluation result dictionary
    """
    logger.info("=" * 60)
    logger.info(f"Evaluating prompt config: {config_path}")
    logger.info("=" * 60)

    # Parse configuration
    try:
        config = parse_prompt_config(config_path)
    except Exception as e:
        return {
            "combined_score": 0.0,
            "stage": 0,
            "error": f"Failed to parse config: {e}",
        }

    # Extract prompt features
    prompt_complexity, variation_diversity = extract_prompt_features(config)

    # Generate code from prompt
    logger.info("\n[Phase 1] Generating code from prompt config...")
    try:
        generated_code, gen_metadata = generate_code_from_config(config)
        logger.info(f"Code generated successfully ({len(generated_code)} chars)")
    except Exception as e:
        return {
            "combined_score": 0.0,
            "stage": 0,
            "error": f"Code generation failed: {e}",
            "prompt_complexity": prompt_complexity,
            "variation_diversity": variation_diversity,
        }

    # Evaluate the generated code
    logger.info("\n[Phase 2] Evaluating generated code...")
    result = evaluate_generated_code(generated_code)

    # Add prompt-specific metrics
    result["prompt_complexity"] = prompt_complexity
    result["variation_diversity"] = variation_diversity
    result["generation_metadata"] = gen_metadata

    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL SCORE: {result.get('combined_score', 0):.4f}")
    logger.info(f"  Stage reached: {result.get('stage', 0)}")
    if "stats_score" in result:
        logger.info(f"  Stats score: {result['stats_score']:.4f}")
    if "model_score" in result:
        logger.info(f"  Model score: {result['model_score']:.4f}")
    logger.info(f"  Prompt features: complexity={prompt_complexity:.3f}, diversity={variation_diversity:.3f}")
    logger.info("=" * 60)

    return result


# For testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prompt_evaluator.py <config_path>")
        print("\nExample:")
        print("  python prompt_evaluator.py prompts/baseline_prompt_config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    result = evaluate(config_path)

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)
    for key, value in result.items():
        if key == "metrics":
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif key == "traceback":
            print(f"  {key}: (see below)")
        else:
            print(f"  {key}: {value}")

    if "traceback" in result:
        print("\nTraceback:")
        print(result["traceback"])
