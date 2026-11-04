"""Check the staged publication without printing file contents or secret matches."""
from pathlib import PurePosixPath
import re
import subprocess


def git(*arguments):
    return subprocess.check_output(['git', *arguments])


def main():
    staged = git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').decode().split('\0')
    blocked, secrets = [], []
    patterns = [re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
                re.compile(rb'github_pat_[A-Za-z0-9_]{50,}'),
                re.compile(rb'AKIA[0-9A-Z]{16}'),
                re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]
    extensions = {'.py', '.md', '.txt', '.sh', '.slurm'}
    special = {'.gitignore', '.gitattributes', 'revision/attacker_config.json'}
    for name in filter(None, staged):
        path = PurePosixPath(name)
        if (name.startswith(('data/', 'results/', 'figures/', '.venv', '.git/')) or
                path.suffix.lower() not in extensions and name not in special or
                path.name.startswith('.env')):
            blocked.append(name)
            continue
        blob = git('show', ':' + name)
        if any(pattern.search(blob) for pattern in patterns):
            secrets.append(name)
    if blocked or secrets:
        print('Blocked publication paths:', blocked)
        print('Potential credentials in paths:', secrets)
        return 1
    print('Staged publication checked: ' + str(sum(bool(name) for name in staged)) + ' code/configuration/documentation files; no data/output paths or credential patterns.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
