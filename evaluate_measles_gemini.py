#!/usr/bin/env python3
"""Gemini API measles benchmark. Uses measles_config.json and measles_eval_common.py.

Configuration explanation (in measles_config.json):
    "limit_per_class": 0.   # Full evaluation
    "limit_per_class": 10,  # Number of images to evaluate per class
    
Example commands:
    python evaluate_measles_gemini.py \
        --mode binary
    python evaluate_measles_gemini.py \
        --mode top3
    python evaluate_measles_gemini.py \
        --model gemini-2.5-pro \
        --mode binary

"""
import os
from measles_eval_common import run


def main():
    key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not key:
        raise SystemExit('Set GEMINI_API_KEY (or GOOGLE_API_KEY) first')
    from google import genai
    from google.genai import types

    client = None

    def call(cfg, image):
        nonlocal client
        timeout_ms = int(float(cfg.get('timeout_seconds', 180)) * 1000)
        if client is None:
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=timeout_ms),
            )

        # Permissive safety settings for research clinical dermatology photographs
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        response = client.models.generate_content(
            model=cfg['model_id'],
            contents=[
                cfg['prompts'][cfg['mode']],
                types.Part.from_bytes(data=image, mime_type='image/jpeg'),
            ],
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                temperature=float(cfg.get('gemini', {}).get('temperature', 0)),
                safety_settings=safety_settings,
                http_options=types.HttpOptions(timeout=timeout_ms),
            ),
        )

        if not response.text:
            reason = ''
            if response.candidates and response.candidates[0].finish_reason:
                reason = f' (finish_reason: {response.candidates[0].finish_reason})'
            raise ValueError(f'Empty Gemini response{reason}')

        return response.text

    run('gemini', call)


if __name__ == '__main__':
    main()
