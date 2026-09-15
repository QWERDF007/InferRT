#!/usr/bin/env python3
"""Build and run the real dependency-free C++ header tests, then evaluator tests."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--sanitizers', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'evidence/core_tests.json')
    args = parser.parse_args()
    started = time.perf_counter()
    results = []
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix='dino_core_') as tmp:
        binary = str(Path(tmp) / ('test_core.exe' if os.name == 'nt' else 'test_core'))
        flags = ['-std=c++17', '-O3', '-Wall', '-Wextra', '-Werror']
        if args.sanitizers:
            flags = ['-std=c++17', '-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        compile_command = [args.compiler, *flags, '-I', str(ROOT / 'src/features/priv/dino'),
                           str(ROOT / 'tests/standalone/test_retrieval_core.cpp'), '-o', binary]
        commands = [('compile_core', compile_command), ('core', [binary]),
                    ('evaluator', [sys.executable, str(ROOT / 'tests/standalone/test_evaluator.py')])]
        for label, command in commands:
            run = subprocess.run(command, text=True, encoding='utf-8', errors='replace', capture_output=True, env=environment)
            results.append({'name': label, 'return_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr})
            if run.returncode:
                break
    success = len(results) == 3 and all(r['return_code'] == 0 for r in results)
    report = {'status': 'PASS' if success else 'FAIL', 'platform': platform.platform(),
              'scope': 'actual_production_header_and_evaluation_script',
              'sanitizers_requested': args.sanitizers,
              'asan_options': environment.get('ASAN_OPTIONS'),
              'elapsed_seconds': time.perf_counter() - started, 'checks': results,
              'full_inferrt_build': 'NOT_RUN_MISSING_PARENT_PROJECT_AND_DEPENDENCIES',
              'cuda_runtime_test': 'NOT_RUN_NO_CUDA', 'real_gallery_recall': None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report['status'])
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
