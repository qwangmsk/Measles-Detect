#!/usr/bin/env python3
"""OpenAI API measles benchmark. Uses measles_config.json and measles_eval_common.py.

Configuration explanation (in measles_config.json):
    "limit_per_class": 0.   # Full evaluation
    "limit_per_class": 10,  # Number of images to evaluate per class
    
Example commands:
    python evaluate_measles_openai.py \
        --mode binary
    python evaluate_measles_openai.py \
        --mode top3
    python evaluate_measles_openai.py \
        --model gpt-5.6 \
        --mode binary

"""
import base64
import os
from measles_eval_common import run


def main():
    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('Set OPENAI_API_KEY first')
    from openai import OpenAI
    client = OpenAI()

    def call(cfg, image):
        data_url = 'data:image/jpeg;base64,' + base64.b64encode(image).decode('ascii')
        response = client.chat.completions.create(
            model=cfg['model_id'],
            messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': cfg['prompts'][cfg['mode']]},
                {'type': 'image_url', 'image_url': {
                    'url': data_url, 'detail': cfg.get('openai', {}).get('detail', 'high')}}]}],
            response_format={'type': 'json_object'},
            timeout=float(cfg.get('timeout_seconds', 180)),
        )
        return response.choices[0].message.content or ''

    run('openai', call)


if __name__ == '__main__':
    main()
