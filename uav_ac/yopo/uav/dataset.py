"""Prepare or run the pinned official CUDA depth dataset collector."""
import argparse
from pathlib import Path
import subprocess

from uav_ac.yopo.common.collection import PROJECT_ROOT, RUNS_ROOT, prepare_collection


def prepare(config_path, run_dir):
    return prepare_collection(config_path, run_dir, runs_root=RUNS_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--binary', type=Path,
                        default=RUNS_ROOT / 'yopo/build/dataset_generator')
    parser.add_argument('--prepare-only', action='store_true',
                        help='write the resolved config only; do not build or generate data')
    args = parser.parse_args()
    if not args.prepare_only and not args.binary.is_file():
        parser.error('collector binary missing; build the vendored simulator first')
    config = prepare(args.config, args.run_dir)
    print(config, flush=True)
    if not args.prepare_only:
        subprocess.run([str(args.binary.resolve()), str(config)], check=True,
                       cwd=PROJECT_ROOT)


if __name__ == '__main__':
    main()
