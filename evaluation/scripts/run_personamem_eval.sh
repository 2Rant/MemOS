export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --repo-type dataset bowen-upenn/ImplicitPersona --local-dir ./data/personamem

python .scripts/personamam/processed.py --limit 200

python .scripts/personamam/memos_run.py
