"""Refresh aggregate reports; preserve final analysis after finalization."""
import json
from revision.safe_reporting import PUBLIC


def main():
    status_path = PUBLIC / 'FINALIZATION_STATUS.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'complete':
        from revision.final_analysis import main as generate
    else:
        from revision.report_progress import main as generate
    generate()


if __name__ == '__main__':
    main()
