"""Pin distributions actually imported by an executed worker (never data values)."""
import json
import sys
from pathlib import Path
import importlib_metadata as metadata
from .safe_reporting import PUBLIC, write_json


def snapshot(environment_name):
    top_modules = {module.split('.')[0] for module in sys.modules}
    distributions = set()
    for package, names in metadata.packages_distributions().items():
        if package in top_modules:
            distributions.update(names)
    versions, requirements = {}, ['--extra-index-url https://download.pytorch.org/whl/cpu']
    for name in sorted(distributions, key=str.lower):
        distribution = metadata.distribution(name)
        version = distribution.version
        versions[name] = version
        direct = distribution.read_text('direct_url.json')
        direct = json.loads(direct) if direct else {}
        if direct.get('url', '').startswith('https://github.com/ryan112358/private-pgm'):
            requirements.append(name + ' @ git+' + direct['url'] + '@' + direct['vcs_info']['commit_id'])
        else:
            requirements.append(name + '==' + version)
    directory = PUBLIC / 'environments'
    write_json(directory / (environment_name + '_imported_versions.json'), {'python': sys.version, 'distributions': versions})
    return requirements


def save_lock(environment_name):
    if (PUBLIC / 'environments' / ('requirements-' + environment_name + '-pinned.txt')).exists():
        return
    requirements = snapshot(environment_name)
    directory = PUBLIC / 'environments'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ('requirements-' + environment_name + '-pinned.txt')).write_text('\n'.join(requirements) + '\n', encoding='utf-8')
