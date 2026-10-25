"""CLI dispatch: preserve the original code and environment for baseline runs."""
import argparse
import json
from pathlib import Path
from options import normalized_config, validate_options, modern_runtime


def main(config):
    c = validate_options(normalized_config(config))
    if not modern_runtime(c):
        import run
        # One fixed-point iteration with unit relaxation is direct DDIM.
        if c['inversion_method'] == 'direct':
            c['inversion_method'], c['inversion_iterations'] = 'fixed_point', 1
        run.main(c)
    else:
        from run_extended import main as extended
        extended(c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(json.loads(Path(args.config).read_text()))
