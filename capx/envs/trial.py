"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import base64
import copy
import gc
import io
import json
import os
import time
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.telemetry import BudgetExceeded, EpisodeBudget, EpisodeTelemetry, get_active_telemetry

from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10

HARNESS_MODES = ("baseline", "ours", "ablation")


def _budgeted_query(
    args: Any,
    prompt: list[dict[str, Any]],
    *,
    role: str,
    purpose: str,
    turn_idx: int | None = None,
) -> dict[str, Any]:
    """Query a model through the active episode telemetry (budget + usage accounting)."""
    tele = get_active_telemetry()
    if tele is None:
        return _query_model(args, prompt)
    return tele.query(_query_model, args, prompt, role=role, purpose=purpose, turn_idx=turn_idx)


def _prompt_record(prompt: list[dict[str, Any]], image_names: list[str]) -> list[dict[str, Any]]:
    """Loggable copy of a VLM prompt with base64 images replaced by saved file names."""
    names = iter(image_names)
    record = []
    for msg in prompt:
        content = msg["content"]
        if isinstance(content, str):
            record.append({"role": msg["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                parts.append({"type": "image", "file": next(names, "<image>")})
        record.append({"role": msg["role"], "content": parts})
    return record


def _set_last_vdm(purpose: str, prompt: list[dict[str, Any]], image_names: list[str], output: str) -> None:
    tele = get_active_telemetry()
    if tele is not None:
        tele.last_vdm = {
            "purpose": purpose,
            "vdm_prompt": _prompt_record(prompt, image_names),
            "vdm_output": output,
            "vdm_call_index": tele.num_model_calls - 1,
        }

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        description = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
        )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    content = _budgeted_query(
        visual_differencing_args, prompt, role="vdm", purpose="vdm_initial", turn_idx=0
    )["content"]
    _set_last_vdm("vdm_initial", prompt, ["visual_feedback_00.png"], content)
    return content


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    n = len(visual_feedback_base64_history)
    tele = get_active_telemetry()
    content = _budgeted_query(
        visual_differencing_args, prompt, role="vdm", purpose="vdm_diff",
        turn_idx=tele.num_code_executions - 1 if tele is not None else None,
    )["content"]
    _set_last_vdm(
        "vdm_diff", prompt, [f"visual_feedback_{n - 2:02d}.png", f"visual_feedback_{n - 1:02d}.png"], content
    )
    return content


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
) -> tuple[str, str | None, dict | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data)
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(str(obs["full_prompt"]))

    ensemble_data = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            out = _query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        out = _budgeted_query(args, obs["full_prompt"], role="coding", purpose="initial_code", turn_idx=0)

    return out["content"], out["reasoning"], ensemble_data


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step["stdout"],
        console_stderr=info_step["stderr"],
    )

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    # Build decision prompt
    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )

    # Query model
    multiturn_ensemble_entry = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            content = _query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        tele = get_active_telemetry()
        content = _budgeted_query(
            args, decision_prompt, role="coding", purpose="multi_turn_decision",
            turn_idx=tele.num_code_executions - 1 if tele is not None else None,
        )

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)

    # --- Episode telemetry and budgets (identical for all harness modes) ---
    budget: EpisodeBudget = config.get("budget") or EpisodeBudget()
    mode = config.get("harness_mode", "baseline")
    if mode not in HARNESS_MODES:
        raise ValueError(f"Unknown harness mode {mode!r}")
    if mode != "baseline":
        raise NotImplementedError(f"Harness mode {mode!r} is not implemented yet (Phase 2)")
    task_name = config.get("task_name", "unknown_task")
    tele = EpisodeTelemetry(
        task_name=task_name,
        seed=trial,
        mode=mode,
        budget=budget,
        coding_model=args.model,
        vdm_model=args.visual_differencing_model
        if (config["use_img_differencing"] or use_video_diff)
        else None,
        output_path=os.path.join(config["output_dir"], f"episode_{task_name}_seed{trial}_{mode}.json"),
    )
    env.tool_wrapper = tele.wrap_tool
    if partial_artifacts is not None:
        partial_artifacts["telemetry"] = tele

    # --- 1. Reset environment ---
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    # Reset the SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation.
    # The soft wall-time budget is checked by the telemetry; SIGALRM is the hard backstop.
    import signal
    remaining = signal.alarm(0)  # cancel current alarm
    if remaining > 0:
        signal.alarm(int(budget.max_wall_time_s) + 120)

    if hasattr(env.low_level_env, "get_initial_state_record"):
        tele.initial_state = {"seed": trial, **env.low_level_env.get_initial_state_record()}
    tele.sim_steps_fn = lambda: getattr(env.low_level_env, "_sim_step_count", 0)
    tele.sim_steps_at_start = tele.sim_steps_fn()
    tele.reset_wall_time = time.time() - trial_start_time
    tele.start_time = time.time()
    tele.activate()
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif use_video_diff and hasattr(env, "enable_video_capture"):
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model must be in the list of VLM models"
        )

    # --- 2. Capture initial visual feedback ---
    tele.last_vdm = None
    try:
        visual_feedback_imgs, visual_feedback_base64_history, task_description = (
            _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
        )
    except BudgetExceeded as exc:
        tele.set_termination(exc.reason, exc.detail)
        visual_feedback_imgs, visual_feedback_base64_history, task_description = [], [], ""
    initial_vdm = tele.last_vdm

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
    else:
        raw_code, reasoning, ensemble_data = "", None, None
        if tele.termination_reason is None:
            try:
                raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)
            except BudgetExceeded as exc:
                tele.set_termination(exc.reason, exc.detail)
    gen_call_idx = tele.num_model_calls - 1

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Parse initial code into blocks
    initial_blocks = _extract_code(raw_code) if tele.termination_reason is None else []
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
    })

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks):
        try:
            tele.check_can_execute()
        except BudgetExceeded as exc:
            tele.set_termination(exc.reason, exc.detail)
            break
        code = code_blocks[code_block_idx]
        code_block_idx += 1

        turn_start = time.time()
        tools_before = dict(tele.tool_calls)
        steps_before = tele.sim_steps()

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        obs_next, reward, terminated, truncated, info_step = env.step(code)

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        turn_record: dict[str, Any] = {
            "turn_idx": tele.num_code_executions,
            "generated_code": code,
            "generated_by_call": gen_call_idx,
            "stdout": info_step["stdout"],
            "stderr": info_step["stderr"],
            "code_exec_success": info_step["sandbox_rc"] == 0,
            "exec_wall_time": time.time() - turn_start,
            "sim_steps_delta": tele.sim_steps() - steps_before,
            "tool_calls_delta": {
                k: v - tools_before.get(k, 0) for k, v in tele.tool_calls.items() if v - tools_before.get(k, 0)
            },
            "vdm_prompt": None,
            "vdm_output": None,
            "vdm_call_index": None,
            "decision_after": None,
            "decision_call_index": None,
        }
        if turn_record["turn_idx"] == 0 and initial_vdm is not None:
            turn_record["initial_scene_vdm"] = initial_vdm
        tele.add_turn(turn_record)
        if tele.budget_exceeded is not None:
            # A budget guard fired inside the sandbox (e.g. simulator steps).
            tele.set_termination(tele.budget_exceeded, tele.termination_detail)
            break

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next

        # Multi-turn decision
        if multi_turn_prompt:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                tele.set_termination("env_horizon", "robosuite horizon reached")
                break
            if tele.num_code_executions >= budget.max_turns:
                # No further execution is allowed, so no decision/VDM call is made.
                tele.set_termination("max_turns", f"{tele.num_code_executions} code executions")
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            tele.last_vdm = None
            try:
                decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                    env, obs, args, config, visual_differencing_args,
                    multi_turn_prompt, code_blocks, code_block_idx, info_step,
                    task_description, visual_feedback_imgs, visual_feedback_base64_history,
                    stderr_history,
                    turn_frames=turn_frames,
                    wrist_turn_frames=wrist_turn_frames,
                    wrist_base64_history=wrist_base64_history,
                )
            except BudgetExceeded as exc:
                if tele.last_vdm is not None:
                    turn_record.update({k: tele.last_vdm[k] for k in ("vdm_prompt", "vdm_output", "vdm_call_index")})
                turn_record["turn_wall_time"] = time.time() - turn_start
                tele.set_termination(exc.reason, exc.detail)
                tele.save()
                break

            if tele.last_vdm is not None:
                turn_record.update({k: tele.last_vdm[k] for k in ("vdm_prompt", "vdm_output", "vdm_call_index")})
            turn_record["decision_after"] = decision
            turn_record["decision_call_index"] = tele.num_model_calls - 1
            turn_record["decision_raw_response"] = new_code if decision == "finish" else None
            turn_record["turn_wall_time"] = time.time() - turn_start
            gen_call_idx = tele.num_model_calls - 1
            tele.save()

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                print("Model chose to regenerate code")
                new_blocks = _extract_code(new_code)
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(
                    [{"generation": num_regenerations + 1, "regenerated": True,
                      "regenerated_at_idx": code_block_idx}]
                    * len(new_blocks)
                )
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                tele.set_termination("finish", "coding agent answered FINISH")
                break

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    print("Code blocks done")

    # --- Post-episode evaluation (official signals; never shown to the agent) ---
    if tele.termination_reason is None:
        tele.set_termination("no_more_code", "no code block left to execute")
    finalize_episode_telemetry(env, tele)

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)

    success = info_step["sandbox_rc"] == 0

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def finalize_episode_telemetry(env: CodeExecutionEnvBase, tele: EpisodeTelemetry) -> None:
    """Evaluate the final simulator state with the official checks and save the episode log."""
    evaluation: dict[str, Any] = {}
    try:
        if hasattr(env.low_level_env, "task_completed"):
            evaluation["official_task_success"] = bool(env.low_level_env.task_completed())
        evaluation["official_reward"] = float(env.compute_reward())
    except Exception as exc:  # evaluation must not crash log saving
        evaluation["evaluation_error"] = repr(exc)
    tele.finalize(evaluation)
    tele.deactivate()
    env.tool_wrapper = None


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    if not hasattr(env.low_level_env, "handle"):
        return
    handle = env.low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = (
            obs["full_prompt"][-1]["content"][0]["text"].format(
                libero_environment_goal=goal
            )
        )
