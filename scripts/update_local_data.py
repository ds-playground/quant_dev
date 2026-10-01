"""Save or update live Yahoo data in data/local (git-ignored), for offline analysis.

    python scripts/update_local_data.py              # every ticker in configs/tickers.yaml
    python scripts/update_local_data.py ES=F ^GSPC   # just these

The first run for a symbol downloads its history from --start; later runs add the new bars and
report any saved bar Yahoo has revised. A symbol that fails is reported and the rest continue;
the exit code is 1 if any failed, so a scheduler can notice.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.price_return import save_all  # noqa: E402


def report(entry):
    """One line per symbol as it finishes, plus any saved values Yahoo has revised."""
    if 'error' in entry:
        print(f"{entry['symbol']}: FAILED, {entry['error']}")
        return
    action = 'saved' if entry['created'] else f"+{entry['added']} bars"
    revised = f", {len(entry['revised'])} revised value(s)" if entry['revised'] else ''
    print(f"{entry['symbol']}: {action}, {entry['rows']} bars {entry['first']} to {entry['last']}{revised}")
    for change in entry['revised']:
        print(f"    {change['date']} {change['column']}: {change['saved']} -> {change['new']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('symbols', nargs='*', help='Yahoo symbols (default: configs/tickers.yaml)')
    parser.add_argument('--start', default='2016-01-01', help='first date for a new symbol')
    args = parser.parse_args(argv)

    result = save_all(args.symbols or None, start_date=args.start, on_result=report)
    failed = [f['symbol'] for f in result['failed']]
    if failed:
        total = len(failed) + len(result['saved'])
        print(f'{len(failed)} of {total} failed: {", ".join(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
