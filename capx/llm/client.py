"""LLM client utilities for querying language models.

Extracted from capx/utils/launch_utils.py to separate LLM query logic
from launch/config utilities.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import requests

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

GPT_MODELS = [
    "openai/gpt-5.4",
    "openai/o4-mini",
    "gpt-5.4",
]
VLM_MODELS = [
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.4",
    "openai/o1",
    "openai/o4-mini",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "qwen3.5-122b-a10b-w8a8",
    "qwen3.6-35b-a3b",
    "Qwen3.8-Flash-Next",
    "Qwen3-VL-235B-A22B",
    "moonshotai/kimi-k2",
    "gpt-5.4",
]
CLAUDE_MODELS = ["anthropic/claude-opus-4-5", "anthropic/claude-haiku-4-5"]
OSS_MODELS = [
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "deepseek-v4-flash-s1",
    "qwen/qwen3.5-122b-a10b",
    "qwen3.5-122b-a10b-w8a8",
    "qwen3.6-35b-a3b",
    "gemma4-31b",
    "moonshotai/kimi-k2",
]
OPENROUTER_MODELS = [
    "openrouter/google/gemini-2.5-pro-preview",
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-chat-v3-0324",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/o4-mini",
    "openrouter/meta-llama/llama-4-maverick",
    "openrouter/qwen/qwen3-235b-a22b",
]
OPENROUTER_SERVER_URL = "http://localhost:8110/chat/completions"
QWEN38_MODEL = "Qwen3.8-Flash-Next"
QWEN38_SERVER_URL = "http://10.1.110.29:30000/v1/chat/completions"

# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

ENSEMBLE_CONFIGS = [
    # Gemini-3-Pro only — best single model per CaP-Bench (Figure 1).
    # 3 temps for diversity; synthesis still uses Gemini-3-Pro.
    # ~45% faster than full multimodel (no Claude/GPT latency bottleneck).
    ("openai/gpt-5.4", [0.1, 0.5, 0.9]),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_openrouter_model(model: str) -> bool:
    """Return True if the model should be routed through the OpenRouter proxy."""
    return model.startswith("openrouter/") or model in OPENROUTER_MODELS


@dataclass
class ModelQueryArgs:
    """Arguments for querying a model."""

    model: str
    server_url: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    enable_thinking: bool | None = None
    thinking_effort: Literal["low", "medium", "xhigh"] = "low"
    debug: bool = False


def _is_qwen38_model(model: str) -> bool:
    return model.rsplit("/", 1)[-1].lower() == QWEN38_MODEL.lower()


def _model_server_url(args: "LaunchArgs | ModelQueryArgs") -> str:
    """Route Qwen3.8 directly to SGLang instead of the legacy shared proxy."""
    if _is_qwen38_model(args.model):
        return QWEN38_SERVER_URL
    return args.server_url


def _stored_model_api_key() -> str | None:
    """Read the user-level CaP-X API key file when it exists."""
    key_path = os.path.expanduser(
        os.getenv("CAPX_API_KEY_FILE", "~/.config/capx/api_key")
    )
    try:
        with open(key_path, encoding="utf-8") as key_file:
            return key_file.read().strip() or None
    except OSError:
        return None


def _model_api_key(args: "LaunchArgs | ModelQueryArgs") -> str | None:
    """Resolve an explicit key, environment key, or stored user key."""
    if args.api_key:
        return args.api_key
    if capx_api_key := os.getenv("CAPX_API_KEY"):
        return capx_api_key
    if stored_api_key := _stored_model_api_key():
        return stored_api_key
    if args.model in GPT_MODELS:
        return os.getenv("OPENAI_API_KEY")
    return None


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """
    Collapse a list of messages with sequential text into a single text input, images are still in the same relative position
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


def _completions_to_responses_convert_prompt(prompt: list[dict]) -> list[dict]:
    """Convert completions api format to responses api format.

    Args:
        prompt: The prompt in completions api format

    Returns:
        The prompt in responses api format

    Switch prompt structure to api responses api format e.g.:
    From
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    To
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the image in detail."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    """

    for message in prompt:
        for content in message["content"]:
            if type(content) == str:
                continue
            if content.get("type") == "text":
                content["type"] = "input_text"
                content["text"] = content.pop("text")

            elif content.get("type") == "image_url":
                content["type"] = "input_image"
                content["image_url"] = content["image_url"]["url"]
    return prompt


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


def _parse_content_and_reasoning(content) -> tuple[str, str | None]:
    """Normalize content from API response and split out embedded reasoning.

    Handles two quirks of reasoning-model responses:
    - content may be a list instead of a string (some vLLM deployments)
    - reasoning may be embedded as <think>...</think> tags inside the content
    """
    if content is None:
        content = ""
    if isinstance(content, list):
        content = "".join(
            item if isinstance(item, str)
            else item.get("text", "") if isinstance(item, dict)
            else str(item)
            for item in content
        )
    reasoning = None
    if "</think>" in content:
        think_end = content.index("</think>")
        think_start = content.find("<think>")
        if think_start != -1:
            reasoning = content[think_start + len("<think>"):think_end].strip()
        else:
            reasoning = content[:think_end].strip()
        content = content[think_end + len("</think>"):].strip()
    return content, reasoning


def _apply_thinking_settings(
    payload: dict[str, Any], args: "LaunchArgs | ModelQueryArgs"
) -> None:
    """Keep the thinking switch separate from Qwen3.8's effort level."""
    if (enable_thinking := getattr(args, "enable_thinking", None)) is not None:
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = enable_thinking
    if _is_qwen38_model(args.model):
        payload["model"] = QWEN38_MODEL
        effort = getattr(args, "thinking_effort", "low")
        if effort not in ("low", "medium", "xhigh"):
            raise ValueError("Qwen3.8 thinking_effort must be low, medium, or xhigh.")
        payload["reasoning_effort"] = effort


def _require_final_content(
    content: str,
    model: str,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> None:
    """Reject reasoning-only replies instead of silently executing empty code."""
    if not content.strip():
        raise RuntimeError(
            f"Model {model!r} returned empty final content "
            f"(finish_reason={finish_reason!r}, usage={usage!r}). "
            "Reasoning is not executable code. Check the token budget, thinking "
            "settings, and server reasoning parser."
        )


def _adjust_max_tokens_for_context_error(response: "requests.Response") -> int | None:
    """Parse a 400 context-length error and return a safe max_tokens value, or None if unrecognised."""
    import re
    try:
        body = response.json()
        msg = body.get("error", {}).get("message", "")
    except Exception:
        msg = response.text

    # Match: "maximum context length is X tokens ... prompt contains at least Y input tokens"
    ctx_match = re.search(r"maximum context length is (\d+)", msg)
    inp_match = re.search(r"prompt contains at least (\d+)", msg)
    if ctx_match and inp_match:
        try:
            context_limit = int(ctx_match.group(1))
            input_tokens = int(inp_match.group(1))
        except ValueError:
            return None
        # The server reports "at least N" input tokens (approximate), so use a
        # generous buffer (512) to ensure the retry doesn't also fail.
        safe_max = context_limit - input_tokens - 512
        if safe_max > 0:
            return safe_max
    return None


def query_model(
    args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]
) -> dict[str, Any]:
    """Query vLLM server for code generation.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
    Returns:
        Model response content and optional reasoning.
    """

    if is_openrouter_model(args.model):
        server_url = OPENROUTER_SERVER_URL
    else:
        server_url = _model_server_url(args)

    if args.model in GPT_MODELS:
        if "codex" in args.model:
            prompt = _completions_to_responses_convert_prompt(prompt)
            payload = {
                "model": args.model,
                "input": prompt,
            }
        else:
            payload = {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "max_completion_tokens": args.max_tokens,  # Total completion tokens = reasoning + output tokens
                "messages": prompt,
            }
    elif is_openrouter_model(args.model):
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
        }
    elif args.model in OSS_MODELS:
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            # "chat_template_kwargs": {"enable_thinking": False},
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
        }
    _apply_thinking_settings(payload, args)
    headers = {"Content-Type": "application/json"}
    if api_key := _model_api_key(args):
        headers["Authorization"] = f"Bearer {api_key}"
    start_time = time.time()

    # keep calling until it works
    response = requests.post(
        server_url, headers=headers, data=json.dumps(payload), timeout=1200
    )
    retry = 1
    while response.status_code in [404, 500, 502, 503, 504]:
        sleep_time = 240 + random.uniform(-90, 90)
        print(f"Retry {retry}. Model query failed with status code {response.status_code}. Error: {response.text}. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
        response = requests.post(
            server_url, headers=headers, data=json.dumps(payload), timeout=1200
        )
        retry += 1

    # Handle context-length 400 errors by reducing max_tokens and retrying once.
    if response.status_code == 400:
        adjusted = _adjust_max_tokens_for_context_error(response)
        if adjusted is not None:
            print(f"Context length exceeded; retrying with max_tokens={adjusted}")
            payload["max_tokens"] = adjusted
            response = requests.post(
                server_url, headers=headers, data=json.dumps(payload), timeout=1200
            )

    end_time = time.time()
    print(f"Time taken to query model: {end_time - start_time:.2f} seconds")
    if not response.ok:
        print(f"Server error {response.status_code}: {response.text}")
    response.raise_for_status()
    body = response.json()
    out = {}
    if args.debug:
        print(json.dumps(body, indent=2))
    try:
        if args.model in GPT_MODELS and "codex" in args.model:
            out["content"] = body["output_text"]
            out["reasoning"] = None
        else:
            raw_content = body["choices"][0]["message"]["content"]
            message = body["choices"][0]["message"]
            api_reasoning = message.get("reasoning") or message.get("reasoning_content")
            content, embedded_reasoning = _parse_content_and_reasoning(raw_content)
            out["content"] = content
            out["reasoning"] = api_reasoning or embedded_reasoning
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format: {body}") from exc
    _require_final_content(
        out["content"], args.model,
        body.get("choices", [{}])[0].get("finish_reason"), body.get("usage"),
    )
    return out


def query_model_streaming(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
) -> Iterable[dict]:
    """Query model with streaming enabled, yielding partial responses.

    Yields dictionaries with:
      - {"type": "content_delta", "content": "partial text"}
      - {"type": "reasoning_delta", "content": "partial reasoning"} (if supported)
      - {"type": "done", "content": "full content", "reasoning": "full reasoning or None"}

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation

    Yields:
        Partial response chunks as they arrive
    """
    server_url = _model_server_url(args)
    if args.model in GPT_MODELS:
        payload = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
            "stream": True,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
            # "chat_template_kwargs": {"enable_thinking": False},
        }

    _apply_thinking_settings(payload, args)
    headers = {"Content-Type": "application/json"}
    if api_key := _model_api_key(args):
        headers["Authorization"] = f"Bearer {api_key}"

    full_content = ""
    full_reasoning = ""
    finish_reason = None
    usage = None

    start_time = time.time()

    # Make the request; on context-length 400, reduce max_tokens and retry once.
    response = requests.post(
        server_url, headers=headers, data=json.dumps(payload), timeout=1200, stream=True
    )
    if response.status_code == 400:
        adjusted = _adjust_max_tokens_for_context_error(response)
        if adjusted is not None:
            print(f"Context length exceeded; retrying stream with max_tokens={adjusted}")
            response.close()
            payload["max_tokens"] = adjusted
            response = requests.post(
                server_url, headers=headers, data=json.dumps(payload), timeout=1200, stream=True
            )

    with response:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        is_sse = "text/event-stream" in content_type
        is_json = "application/json" in content_type

        # If it's a regular JSON response (server doesn't support streaming),
        # fall back to non-streaming behavior
        if is_json and not is_sse:
            print("Warning: Server returned JSON instead of SSE stream, falling back to non-streaming")
            body = response.json()
            try:
                raw_content = body["choices"][0]["message"]["content"]
                message = body["choices"][0]["message"]
                api_reasoning = message.get("reasoning") or message.get("reasoning_content")
                full_content, embedded_reasoning = _parse_content_and_reasoning(raw_content)
                full_reasoning = api_reasoning or embedded_reasoning or ""
                if full_reasoning:
                    print(f"Reasoning extracted ({len(full_reasoning)} chars)")
                else:
                    print("No reasoning returned by model")
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected response format: {body}") from exc

            _require_final_content(
                full_content, args.model,
                body["choices"][0].get("finish_reason"), body.get("usage"),
            )
            yield {"type": "content_delta", "content": full_content}
            yield {
                "type": "done",
                "content": full_content,
                "reasoning": full_reasoning if full_reasoning else None,
            }
            end_time = time.time()
            print(f"Time taken to query model (streaming fallback): {end_time - start_time:.2f} seconds")
            return

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            # SSE format: "data: {...}" or "data: [DONE]"
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    usage = data.get("usage") or usage
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    finish_reason = choices[0].get("finish_reason") or finish_reason
                    delta = choices[0].get("delta", {})

                    # Handle content delta
                    content_delta = delta.get("content", "")
                    if content_delta:
                        full_content += content_delta
                        yield {"type": "content_delta", "content": content_delta}

                    # Handle reasoning delta (some APIs support this)
                    reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content", "")
                    if reasoning_delta:
                        full_reasoning += reasoning_delta
                        yield {"type": "reasoning_delta", "content": reasoning_delta}

                except ValueError:
                    continue
            else:
                # Try parsing as raw JSON (non-SSE format)
                try:
                    data = json.loads(line_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content_delta = delta.get("content", "")
                        if content_delta:
                            full_content += content_delta
                            yield {"type": "content_delta", "content": content_delta}
                except ValueError:
                    continue

    end_time = time.time()
    print(f"Time taken to query model (streaming): {end_time - start_time:.2f} seconds")

    if not full_reasoning:
        full_content, embedded_reasoning = _parse_content_and_reasoning(full_content)
        if embedded_reasoning:
            full_reasoning = embedded_reasoning

    if full_reasoning:
        print(f"Reasoning extracted ({len(full_reasoning)} chars)")
    else:
        print("No reasoning returned by model")

    _require_final_content(full_content, args.model, finish_reason, usage)
    yield {
        "type": "done",
        "content": full_content,
        "reasoning": full_reasoning if full_reasoning else None,
    }


def query_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    synthesis_model: str = "openai/gpt-5.4",
    is_multiturn = False
) -> dict[str, Any]:
    """Query 9 models (3 models x 3 temperatures) and synthesize final output."""

    def query_single(model: str, temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Multimodel Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Build all (model, temp) pairs and query in parallel
    tasks = [(m, t) for m, temps in ENSEMBLE_CONFIGS for t in temps]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, m, t): (m, t) for m, t in tasks}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Multimodel Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    synth_args = ModelQueryArgs(
        model=synthesis_model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {synthesis_model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


def query_single_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    model: str,
    is_multiturn = False,
) -> dict[str, Any]:
    """Query the same model 9 times (with temperatures 0.1 to 0.9) and synthesize final output.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
        model: The model to use for both candidate generation and synthesis

    Returns:
        Dictionary containing synthesized content, reasoning, all responses, and text artifacts
    """

    def query_single(temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            result = query_model(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            error_msg = str(e)
            print(f"[Single Model Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False}

    # Query same model with 9 different temperatures (0.1 to 0.9)
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, t): t for t in temperatures}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Single Model Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All single model ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError("All single model ensemble queries failed")

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate (temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else: # first generation has no REGEN/FINISH candidates
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    # Use the same model for synthesis
    synth_args = ModelQueryArgs(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final = query_model(synth_args, synthesis_prompt)

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (underscore-prefixed names)
# ---------------------------------------------------------------------------

_query_model = query_model
_query_model_streaming = query_model_streaming
_query_model_ensemble = query_model_ensemble
_query_single_model_ensemble = query_single_model_ensemble
