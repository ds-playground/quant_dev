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

from src.tools.price_return import load_ticker_config, save_local  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('symbols', nargs='*', help='Yahoo symbols (default: configs/tickers.yaml)')
    parser.add_argument('--start', default='2016-01-01', help='first date for a new symbol')
    args = parser.parse_args(argv)
    symbols = args.symbols or list(load_ticker_config(verbose=False))

    failed = []
    for symbol in symbols:
        try:
            r = save_local(symbol, start_date=args.start)
        except Exception as exc:  # noqa: BLE001 - report and carry on with the rest
            failed.append(symbol)
            print(f'{symbol}: FAILED, {type(exc).__name__}: {exc}')
            continue
        action = 'saved' if r['created'] else f"+{r['added']} bars"
        revised = f", {len(r['revised'])} revised value(s)" if r['revised'] else ''
        print(f"{symbol}: {action}, {r['rows']} bars {r['first']} to {r['last']}{revised}")
        for change in r['revised']:
            print(f"    {change['date']} {change['column']}: {change['saved']} -> {change['new']}")
    if failed:
        print(f'{len(failed)} of {len(symbols)} failed: {", ".join(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
