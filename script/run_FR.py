#!/usr/bin/env python3
"""Launch train.py with the FR rule config."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    rule_config = repo_root / 'config' / 'rule_config' / 'rule_patterns_FR.json'

    if not rule_config.exists():
        print(f"Missing rule config: {rule_config}")
        return 1

    cmd = [
        sys.executable,
        str(repo_root / 'train.py'),
        '--rules_list', str(rule_config),
        '--rule_config_path', str(rule_config),
    ]
    cmd.extend(sys.argv[1:])

    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == '__main__':
    raise SystemExit(main())
