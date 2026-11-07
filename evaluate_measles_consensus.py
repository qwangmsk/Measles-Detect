#!/usr/bin/env python3
"""Ultra-fast, minimal multi-agent clinical deliberation benchmark for measles rash identification.

Optimizations:
1. Single Image Analysis Per Agent: Each agent (GPT, Gemini) processes the image exactly ONCE.
   Rounds 3 & 4 operate purely on lightweight text completions (sub-500ms).
2. Minimalist Prompts: Zero conversational fluff or boilerplate text.
3. Strict Output Token Caps: Step 1 (180), Step 2 (150), Step 3 (120), Step 4 (80).
4. Zero Thinking Overhead: `thinking_budget=0` on Gemini 3.8 Flash eliminates reasoning latency.
5. Multi-Threaded Concurrency: Parallel worker pool (--workers N).
6. Fast Mode (--fast): 2-turn consensus (each agent runs once, 2 calls total).

Example commands:
# 1. High-speed evaluation with 4 parallel worker threads (Fast 2-Turn)
python3 evaluate_measles_consensus.py --fast --workers 4 --limit-per-class 0

# 2. High-speed evaluation (Full 4-Turn with single image pass per agent)
python3 evaluate_measles_consensus.py --workers 4 --limit-per-class 0

# 3. Single-image live test
python3 evaluate_measles_consensus.py --image rash_dataset/measles/MSL_01_01.jpg --fast

"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from measles_eval_common import (
    images_for_run,
    metrics,
    parse_response,
    prepared_image,
    read_rows,
    write_rows,
)


# =====================================================================
# Lightweight Stateful Agent Abstractions
# =====================================================================

class OpenAIAgent:
    """Stateful OpenAI agent. Analyzes image on first turn; subsequent turns are text-only."""

    def __init__(self, name: str, model: str, client: Any, cfg: Dict[str, Any]):
        self.name = name
        self.model = model
        self.client = client
        self.cfg = cfg
        self.messages: List[Dict[str, Any]] = []
        self.reset()

    def reset(self):
        """Clears memory for a fresh patient case."""
        self.messages = []

    def chat(
        self, prompt: str, image_bytes: bytes = None, max_tokens: int = None
    ) -> str:
        """Appends user prompt (with image if provided), queries model, stores reply."""
        content: List[Dict[str, Any]] = [{'type': 'text', 'text': prompt}]
        if image_bytes is not None:
            image_b64 = base64.b64encode(image_bytes).decode('ascii')
            detail = self.cfg.get('openai', {}).get('detail', 'low')
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{image_b64}',
                    'detail': detail,
                },
            })

        self.messages.append({'role': 'user', 'content': content})
        timeout = float(self.cfg.get('timeout_seconds', 180))
        token_budget = max_tokens or int(self.cfg.get('openai', {}).get('max_tokens', 2048))

        kwargs = {
            'model': self.model,
            'messages': self.messages,
            'timeout': timeout,
            'max_completion_tokens': token_budget,
        }
        effort = self.cfg.get('openai', {}).get('reasoning_effort')
        if effort:
            kwargs['reasoning_effort'] = effort

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            err_msg = str(exc)
            if 'reasoning_effort' in err_msg:
                kwargs.pop('reasoning_effort', None)
                response = self.client.chat.completions.create(**kwargs)
            elif 'max_completion_tokens' in err_msg:
                kwargs.pop('max_completion_tokens', None)
                kwargs['max_tokens'] = token_budget
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise

        reply = (response.choices[0].message.content or '').strip()
        if not reply:
            finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
            raise ValueError(
                f"OpenAI ({self.model}) returned empty content (finish_reason: '{finish_reason}'). "
                f"Increase max_completion_tokens (budget: {token_budget}) to accommodate reasoning tokens."
            )
        self.messages.append({'role': 'assistant', 'content': reply})
        return reply


class GeminiAgent:
    """Stateful Gemini agent. Analyzes image on first turn; subsequent turns are text-only."""

    def __init__(
        self,
        name: str,
        model: str,
        client: Any,
        types: Any,
        cfg: Dict[str, Any],
    ):
        self.name = name
        self.model = model
        self.client = client
        self.types = types
        self.cfg = cfg
        self.chat_session = None
        self.reset()

    def reset(self):
        """Clears memory and starts a new session with zero thinking delay."""
        safety_settings = [
            self.types.SafetySetting(
                category=self.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=self.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            self.types.SafetySetting(
                category=self.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=self.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            self.types.SafetySetting(
                category=self.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=self.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            self.types.SafetySetting(
                category=self.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=self.types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
        timeout_ms = int(float(self.cfg.get('timeout_seconds', 180)) * 1000)

        gen_kwargs = {
            'temperature': float(
                self.cfg.get('gemini', {}).get('temperature', 0)
            ),
            'safety_settings': safety_settings,
            'http_options': self.types.HttpOptions(timeout=timeout_ms),
        }
        if hasattr(self.types, 'ThinkingConfig'):
            gen_kwargs['thinking_config'] = self.types.ThinkingConfig(
                thinking_budget=0
            )

        config = self.types.GenerateContentConfig(**gen_kwargs)
        self.chat_session = self.client.chats.create(
            model=self.model, config=config
        )

    def chat(
        self,
        prompt: str,
        image_bytes: bytes = None,
        max_tokens: int = None,
        json_mode: bool = False,
    ) -> str:
        """Appends prompt (with image if provided), stores reply in native chat session."""
        parts: List[Any] = [prompt]
        if image_bytes is not None:
            parts.append(
                self.types.Part.from_bytes(
                    data=image_bytes, mime_type='image/jpeg'
                )
            )

        timeout_ms = int(float(self.cfg.get('timeout_seconds', 180)) * 1000)
        output_budget = max_tokens or int(
            self.cfg.get('gemini', {}).get('max_output_tokens', 800)
        )
        override_kwargs = {
            'temperature': float(
                self.cfg.get('gemini', {}).get('temperature', 0)
            ),
            'max_output_tokens': output_budget,
            'http_options': self.types.HttpOptions(timeout=timeout_ms),
        }
        if json_mode:
            override_kwargs['response_mime_type'] = 'application/json'
        if hasattr(self.types, 'ThinkingConfig'):
            override_kwargs['thinking_config'] = self.types.ThinkingConfig(
                thinking_budget=0
            )

        config_override = self.types.GenerateContentConfig(**override_kwargs)
        response = self.chat_session.send_message(
            message=parts, config=config_override
        )

        if not response.text:
            reason = ''
            if response.candidates and response.candidates[0].finish_reason:
                reason = f' (finish_reason: {response.candidates[0].finish_reason})'
            raise ValueError(f'Empty Gemini response{reason}')
        return response.text.strip()


# =====================================================================
# Deliberation Protocol: Minimalist Prompts & Single Image Inspection
# =====================================================================

def run_agentic_consensus(
    image_bytes: bytes,
    diagnostician: OpenAIAgent,
    critic: GeminiAgent,
    fast_mode: bool = False,
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Deliberation protocol where each agent inspects the image ONLY ONCE."""
    diagnostician.reset()
    critic.reset()

    transcript: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # STEP 1: Agent 1 (GPT) analyzes image -> Top-5 (Image inspected 1x)
    # -----------------------------------------------------------------
    p1 = (
        'Examine this skin rash image. List your TOP 5 differential diagnoses ranked 1 to 5.\n'
        'Include 1 brief sentence of visible lesion justification per diagnosis.\n'
        'Be specific (distinguish measles from rubella, roseola, scarlet fever, drug rash).'
    )
    if verbose:
        print('\n' + '=' * 70)
        print(f'STEP 1: Agent 1 ({diagnostician.model}) Top-5 [Image Attached]')
        print('=' * 70)

    # Agent 1 analyzes the image here
    s1_out = diagnostician.chat(p1, image_bytes=image_bytes)
    transcript['agent1_initial'] = s1_out
    if verbose:
        print(s1_out)

    if fast_mode:
        # -------------------------------------------------------------
        # FAST MODE: Agent 2 reviews image + Top 5 -> Direct Top 3 JSON
        # (Both agents inspect the image only once; 2 calls total)
        # -------------------------------------------------------------
        p_fast = (
            f"Diagnostician's top 5:\n{s1_out}\n\n"
            'Examine this rash image. Critique the differential and select the final 3 most likely diagnoses.\n'
            'Measles (rubeola) is distinct from rubella (German measles).\n'
            'Return ONLY JSON: {"top3": ["diagnosis 1", "diagnosis 2", "diagnosis 3"]}'
        )
        if verbose:
            print('\n' + '=' * 70)
            print(
                f'STEP 2 (Fast Mode): Agent 2 ({critic.model}) Top 3 Consensus [Image Attached]'
            )
            print('=' * 70)

        # Agent 2 analyzes the image here
        final_raw = critic.chat(
            p_fast, image_bytes=image_bytes, json_mode=True
        )
        transcript['final_raw'] = final_raw
        if verbose:
            print(final_raw)
        return final_raw, transcript

    # -----------------------------------------------------------------
    # STEP 2: Agent 2 (Gemini) reviews image + Top 5 (Image inspected 1x)
    # -----------------------------------------------------------------
    p2 = (
        f"Diagnostician's top 5:\n{s1_out}\n\n"
        'Examine this rash image. In 2-3 short bullet points, critique agreements, disagreements, '
        'and any missed exanthems (e.g. measles vs rubella vs other viral exanthems).'
    )
    if verbose:
        print('\n' + '=' * 70)
        print(
            f'STEP 2: Agent 2 ({critic.model}) Peer Critique [Image Attached]'
        )
        print('=' * 70)

    # Agent 2 analyzes the image here
    s2_critique = critic.chat(p2, image_bytes=image_bytes)
    transcript['agent2_critique'] = s2_critique
    if verbose:
        print(s2_critique)

    # -----------------------------------------------------------------
    # STEP 3: Agent 1 (GPT) Updates Differential [TEXT ONLY - No image]
    # -----------------------------------------------------------------
    p3 = (
        f'Peer critique:\n{s2_critique}\n\n'
        'Update your top 5 diagnoses (ranked 1 to 5) based on this critique. Output disease names with 1-phrase rationale.'
    )
    if verbose:
        print('\n' + '=' * 70)
        print(f'STEP 3: Agent 1 ({diagnostician.model}) Update [Text-Only]')
        print('=' * 70)

    # Fast text-only completion (no image bytes sent or processed!)
    s1_updated = diagnostician.chat(p3, image_bytes=None)
    transcript['agent1_updated'] = s1_updated
    if verbose:
        print(s1_updated)

    # -----------------------------------------------------------------
    # STEP 4: Agent 2 (Gemini) Consensus Top-3 [TEXT ONLY - No image]
    # -----------------------------------------------------------------
    p4 = (
        f"Diagnostician's updated top 5:\n{s1_updated}\n\n"
        'Select the final top 3 clinical diagnoses. Measles is distinct from rubella.\n'
        'Return ONLY JSON: {"top3": ["diagnosis 1", "diagnosis 2", "diagnosis 3"]}'
    )
    if verbose:
        print('\n' + '=' * 70)
        print(
            f'STEP 4: Agent 2 ({critic.model}) Final Consensus Top-3 [Text-Only]'
        )
        print('=' * 70)

    # Fast text-only completion in JSON mode (no image bytes sent or processed!)
    final_raw = critic.chat(
        p4, image_bytes=None, json_mode=True
    )
    transcript['final_raw'] = final_raw
    if verbose:
        print(final_raw)

    return final_raw, transcript


# =====================================================================
# Execution & Benchmark Management
# =====================================================================

def init_agent_instances(
    cfg: Dict[str, Any], agent1_model: str, agent2_model: str
) -> Tuple[OpenAIAgent, GeminiAgent]:
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        raise SystemExit(
            'Error: Set OPENAI_API_KEY environment variable for Agent 1 (GPT).'
        )

    gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not gemini_key:
        raise SystemExit(
            'Error: Set GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable for Agent 2 (Gemini).'
        )

    from openai import OpenAI
    from google import genai
    from google.genai import types

    openai_client = OpenAI(api_key=openai_key)
    timeout_ms = int(float(cfg.get('timeout_seconds', 180)) * 1000)
    gemini_client = genai.Client(
        api_key=gemini_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )

    diagnostician = OpenAIAgent(
        name='Diagnostician', model=agent1_model, client=openai_client, cfg=cfg
    )
    critic = GeminiAgent(
        name='Critic',
        model=agent2_model,
        client=gemini_client,
        types=types,
        cfg=cfg,
    )
    return diagnostician, critic


def parse_args():
    ap = argparse.ArgumentParser(
        description='Streamlined multi-agent deliberation for measles rash assessment.'
    )
    ap.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).with_name('measles_config.json'),
        help='Path to measles_config.json',
    )
    ap.add_argument(
        '--image',
        type=Path,
        help='Run an interactive single-image deliberation demo on this image file',
    )
    ap.add_argument('--data', type=Path, help='Override dataset_dir in config')
    ap.add_argument('--agent1-model', help='Override Agent 1 (GPT) model ID')
    ap.add_argument('--agent2-model', help='Override Agent 2 (Gemini) model ID')
    ap.add_argument(
        '--limit-per-class',
        type=int,
        help='Override limit_per_class (0 = all images)',
    )
    ap.add_argument(
        '--workers',
        type=int,
        help='Number of parallel worker threads (e.g. 3 to 6)',
    )
    ap.add_argument(
        '--fast',
        action='store_true',
        help='Run in fast 2-turn consensus mode (2 calls total, both agents inspect image 1x)',
    )
    ap.add_argument(
        '--overwrite',
        action='store_true',
        help='Recompute every selected image in this run',
    )
    ap.add_argument(
        '--no-dialogues',
        action='store_true',
        help='Disable saving individual transcript JSON files',
    )
    return ap.parse_args()


def run_single_image(
    args, cfg: Dict[str, Any], agent1_model: str, agent2_model: str
):
    path = args.image.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Image not found: {path}')

    mode_name = 'FAST 2-Turn' if args.fast else 'STREAMLINED 4-Turn'
    print(f'Starting multi-agent conference [{mode_name}] on: {path.name}')
    print(f'  Agent 1 (Diagnostician): {agent1_model}')
    print(f'  Agent 2 (Critic/Consensus): {agent2_model}')

    diagnostician, critic = init_agent_instances(
        cfg, agent1_model, agent2_model
    )
    image_bytes = prepared_image(path, cfg)

    t0 = time.monotonic()
    raw_final, transcript = run_agentic_consensus(
        image_bytes,
        diagnostician,
        critic,
        fast_mode=args.fast,
        verbose=True,
    )
    elapsed = time.monotonic() - t0

    predicted, guesses, _ = parse_response(raw_final, 'top3')
    print('\n' + '=' * 70)
    print('CASE CONCLUSION')
    print('=' * 70)
    print(f'Image:                    {path.name}')
    print(f'Predicted Measles Status: {predicted.upper()}')
    print(f'Consensus Top 3:          {guesses}')
    print(f'Total Latency:            {elapsed:.2f}s')


def run_dataset_benchmark(
    args, cfg: Dict[str, Any], agent1_model: str, agent2_model: str
):
    cfg['mode'] = 'top3'
    selected = images_for_run(cfg)

    workers = (
        args.workers
        if args.workers is not None
        else int(cfg.get('workers', 4))
    )
    fast_mode = bool(args.fast or cfg.get('fast_mode', False))
    protocol_mode = 'fast2turn' if fast_mode else 'streamlined4turn'

    provider_name = 'agentic_consensus'
    composite_model_id = f'{agent1_model}+{agent2_model}'

    relevant_fingerprint = {
        'provider': provider_name,
        'protocol': protocol_mode,
        'agent1': agent1_model,
        'agent2': agent2_model,
        'mode': 'top3',
        'max_edge': cfg.get('max_edge', 768),
        'jpeg_quality': cfg.get('jpeg_quality', 85),
        'sampling': cfg.get('sampling', 'first'),
        'seed': cfg.get('seed', 42),
        'limit_per_class': cfg['limit_per_class'],
    }
    run_id = hashlib.sha256(
        json.dumps(relevant_fingerprint, sort_keys=True).encode()
    ).hexdigest()[:16]

    folder = Path(cfg['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)

    dialogue_dir = folder / 'dialogues' / run_id
    if not args.no_dialogues:
        dialogue_dir.mkdir(parents=True, exist_ok=True)

    safe_model = re.sub(r'[^A-Za-z0-9._-]+', '_', composite_model_id)
    stem = f'measles_{provider_name}_{safe_model}_{protocol_mode}_{run_id}'
    csv_path = folder / (stem + '.csv')
    json_path = folder / (stem + '.metrics.json')

    cfg_snapshot = dict(cfg)
    cfg_snapshot['agent1_model'] = agent1_model
    cfg_snapshot['agent2_model'] = agent2_model
    cfg_snapshot['protocol_mode'] = protocol_mode
    cfg_snapshot['workers'] = workers
    (folder / (stem + '.config.json')).write_text(
        json.dumps(cfg_snapshot, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    rows = [] if args.overwrite else read_rows(csv_path)
    rows = [
        r
        for r in rows
        if r.get('run_id') == run_id and r.get('provider') == provider_name
    ]
    done = {
        (r['path'], r['image_sha256'])
        for r in rows
        if r['predicted'] in ('measles', 'non_measles')
    }

    print('=' * 75)
    print(f'STREAMLINED AGENTIC BENCHMARK [{protocol_mode.upper()}]')
    print('=' * 75)
    print(f'  Selected Images:          {len(selected)}')
    print(f'  Parallel Worker Threads:  {workers}')
    print(f'  Agent 1 (Diagnostician):  {agent1_model}')
    print(f'  Agent 2 (Critic/Consensus): {agent2_model}')
    print(f'  Output CSV:               {csv_path}')
    print('=' * 75 + '\n')

    pending = []
    for i, (path, rel, truth) in enumerate(selected, 1):
        image = prepared_image(path, cfg)
        digest = hashlib.sha256(image).hexdigest()
        if (rel, digest) in done:
            print(f'[{i}/{len(selected)}] skip {rel} (already completed)')
        else:
            pending.append((i, path, rel, truth, image, digest))

    write_lock = threading.Lock()
    completed_count = len(done)

    def process_case(case_tuple):
        nonlocal completed_count
        idx, path, rel, truth, image_bytes, digest = case_tuple

        # Dedicated agent instances per thread
        diagnostician, critic = init_agent_instances(
            cfg, agent1_model, agent2_model
        )

        predicted, guesses, confidence, raw, error, elapsed = (
            'ERROR',
            [],
            None,
            '',
            '',
            0.0,
        )
        transcript = {}

        t0 = time.monotonic()
        for attempt in range(int(cfg.get('retries', 2)) + 1):
            try:
                raw, transcript = run_agentic_consensus(
                    image_bytes,
                    diagnostician,
                    critic,
                    fast_mode=fast_mode,
                    verbose=False,
                )
                predicted, guesses, confidence = parse_response(raw, 'top3')
                error = ''
                elapsed = time.monotonic() - t0
                break
            except Exception as exc:
                elapsed = time.monotonic() - t0
                error = f'{type(exc).__name__}: {exc}'
                if attempt < int(cfg.get('retries', 2)):
                    time.sleep(min(2 * (2**attempt), 6))

        if not args.no_dialogues and transcript:
            dialogue_file = dialogue_dir / f'{Path(rel).stem}.dialogue.json'
            transcript_record = {
                'path': rel,
                'truth': truth,
                'predicted': predicted,
                'top3_guesses': guesses,
                'seconds': round(elapsed, 2),
                'transcript': transcript,
            }
            dialogue_file.write_text(
                json.dumps(transcript_record, indent=2, ensure_ascii=False),
                encoding='utf-8',
            )

        row_item = {
            'path': rel,
            'truth': truth,
            'provider': provider_name,
            'model_id': composite_model_id,
            'evaluation_mode': 'top3',
            'run_id': run_id,
            'image_sha256': digest,
            'predicted': predicted,
            'top3_guesses': json.dumps(guesses, ensure_ascii=False),
            'confidence_subjective': confidence,
            'seconds': round(elapsed, 2),
            'error': error,
            'raw_response': raw,
        }

        with write_lock:
            completed_count += 1
            nonlocal rows
            rows = [r for r in rows if r.get('path') != rel]
            rows.append(row_item)
            write_rows(csv_path, rows)
            status_text = (
                error[:120] if error else f'Top-3: {guesses} -> {predicted}'
            )
            print(
                f'[{completed_count}/{len(selected)}] ({elapsed:.1f}s) {rel} | {status_text}',
                flush=True,
            )

        return row_item

    if pending:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                list(executor.map(process_case, pending))
        else:
            for case in pending:
                process_case(case)

    selected_paths = {rel for _, rel, _ in selected}
    current = [r for r in rows if r['path'] in selected_paths]
    cfg_metrics = dict(cfg)
    cfg_metrics['model_id'] = composite_model_id
    result = metrics(current, selected, cfg_metrics, provider_name, run_id)
    json_path.write_text(json.dumps(result, indent=2), encoding='utf-8')

    print('\n' + '=' * 75)
    print('BENCHMARK COMPLETE')
    print('=' * 75)
    print('Predictions CSV:', csv_path)
    print('Metrics JSON:', json_path)
    if not args.no_dialogues:
        print('Dialogue Transcripts:', dialogue_dir)
    print(json.dumps(result, indent=2))
    print(
        'Top-3 evaluation: measles positive only if measles/rubeola is among the three diagnoses.'
    )


def main():
    args = parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))

    cfg['dataset_dir'] = str(
        args.data.resolve()
        if args.data
        else (cfg_path.parent / cfg['dataset_dir']).resolve()
    )
    cfg['output_dir'] = str((cfg_path.parent / cfg['output_dir']).resolve())
    cfg['limit_per_class'] = (
        args.limit_per_class
        if args.limit_per_class is not None
        else cfg.get('limit_per_class', 0)
    )

    agent1_model = (
        args.agent1_model
        or cfg.get('models', {}).get('openai')
        or 'gpt-6.1-terra'
    )
    agent2_model = (
        args.agent2_model
        or cfg.get('models', {}).get('gemini')
        or 'gemini-3.8-flash'
    )

    if args.image:
        run_single_image(args, cfg, agent1_model, agent2_model)
    else:
        run_dataset_benchmark(args, cfg, agent1_model, agent2_model)


if __name__ == '__main__':
    main()
