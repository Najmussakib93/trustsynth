"""Isolated import-only compatibility patch; DDPM algorithm files stay unchanged."""
import difflib
import hashlib
import importlib.util
import re
import shutil
import os
import sys
from pathlib import Path
from .safe_reporting import ROOT, PUBLIC, write_json


def main():
    library = Path('Lib') if os.name == 'nt' else Path('lib') / ('python' + '.'.join(map(str, sys.version_info[:2])))
    target = ROOT / '.venv-synthcity' / library / 'site-packages' / 'synthcity'
    spec = importlib.util.find_spec('synthcity')
    source = Path(spec.origin).parent if spec is not None else target
    if not (source / '__init__.py').exists():
        raise RuntimeError('Install the pinned synthcity package before applying its isolated import patch')
    if source.resolve() != target.resolve():
        shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    else:
        receipt = PUBLIC / 'synthcity_compatibility.json'
        if receipt.exists():
            previous = __import__('json').loads(receipt.read_text())
            if all(hashlib.sha256((target / item['file']).read_bytes()).hexdigest() == item['after_sha256'] for item in previous['changed']):
                print('Verified isolated import patch is already applied.', flush=True)
                return
    before_algorithms = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (target / 'plugins' / 'core' / 'models' / 'tabular_ddpm').rglob('*.py')}
    before_plugin = hashlib.sha256((target / 'plugins/generic/plugin_ddpm.py').read_bytes()).hexdigest()
    originals, replacements = {}, {}
    relative = 'metrics/__init__.py'
    originals[relative] = (target / relative).read_text(encoding='utf-8')
    replacements[relative] = '''"""Optional evaluation imports are lazy in this revision environment."""
from .weighted_metrics import WeightedMetrics
def __getattr__(name):
    if name == 'Metrics':
        from .eval import Metrics
        return Metrics
    raise AttributeError(name)
'''
    relative = 'metrics/weighted_metrics.py'
    originals[relative] = (target / relative).read_text(encoding='utf-8')
    text = originals[relative].replace('from .eval import standard_metrics', '')
    text = text.replace('        if len(metrics) != len(weights):',
                        '        from .eval import standard_metrics\n        if len(metrics) != len(weights):')
    replacements[relative] = text
    relative = 'plugins/core/models/factory.py'
    originals[relative] = (target / relative).read_text(encoding='utf-8')
    text = originals[relative]
    imports = re.findall(r'^from (tsai\.[\w.]+) import (\w+)$', text, flags=re.MULTILINE)
    for module, name in imports:
        text = text.replace('from ' + module + ' import ' + name + '\n', '')
        text = re.sub(r'(=)' + re.escape(name) + r'(,)', r'\1"' + module + '.' + name + r'"\2', text)
    replacements[relative] = text
    diff, changes = [], []
    for relative, text in replacements.items():
        old = originals[relative]
        if old == text:
            raise RuntimeError('Expected import patch did not apply')
        (target / relative).write_text(text, encoding='utf-8')
        diff.extend(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                    fromfile='a/' + relative, tofile='b/' + relative))
        changes.append({'file': relative, 'before_sha256': hashlib.sha256(old.encode()).hexdigest(),
                        'after_sha256': hashlib.sha256(text.encode()).hexdigest()})
    unchanged = {}
    for relative, before in before_algorithms.items():
        after = hashlib.sha256((target / relative).read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError('DDPM algorithm source changed unexpectedly')
        unchanged[relative] = before
    plugin = Path('plugins/generic/plugin_ddpm.py')
    if hashlib.sha256((target / plugin).read_bytes()).hexdigest() != before_plugin:
        raise RuntimeError('DDPM plugin source changed unexpectedly')
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / 'synthcity_import_compatibility.patch').write_text(''.join(diff), encoding='utf-8')
    write_json(PUBLIC / 'synthcity_compatibility.json', {'purpose': 'Defer optional metrics/time-series imports unused by DDPM',
        'changed': changes, 'ddpm_algorithm_sources_unchanged': unchanged, 'ddpm_plugin_unchanged': True})
    print('Isolated import patch written; original DDPM algorithm/plugin sources verified unchanged.', flush=True)


if __name__ == '__main__':
    main()
