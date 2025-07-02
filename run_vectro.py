import subprocess
from pathlib import Path

import yaml


def main():
    index_path = Path('codex/vectro-index.yaml')
    text = index_path.read_text().splitlines()
    # Skip non-YAML header lines like "yaml", "Copy", "Edit" if present
    while text and not (text[0].startswith('trigger:') or text[0].startswith('steps:')):
        text.pop(0)
    data = yaml.safe_load('\n'.join(text))

    steps = data.get('steps', []) if isinstance(data, dict) else []

    for step in steps:
        task = step.get('task')
        if not task:
            continue
        print(f'Running task: {task}')
        result = subprocess.run(['python', 'vectro_test.py', '--task', task])
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
