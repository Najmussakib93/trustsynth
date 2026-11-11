"""Narrow SmartNoise 1.0.6 compatibility patch for OpenDP's non-NaN metrics.

Only Gaussian/Laplace scalar domains are changed. No noise sampler, calibration
formula, privacy parameter, seed, or AIM algorithm is replaced.
"""
import ast
import hashlib
import importlib
import importlib.util
import importlib.metadata as metadata
from pathlib import Path
import sys

from .safe_reporting import PUBLIC, write_json

FUNCTIONS = ('cdp_rho', 'gaussian_noise', 'laplace_noise')
CHANGE = 'smartnoise-1.0.6-opendp-non-nan-scalar-domains-v1'


def corrected_source(source):
    """Validate the three known call sites and edit only their nan keyword."""
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    replacements = []
    for name in FUNCTIONS:
        if name not in functions:
            raise RuntimeError('Unsupported SmartNoise utility source: missing ' + name)
        calls = [node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'atom_domain'
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == 'dp']
        if len(calls) != 1:
            raise RuntimeError('Unsupported SmartNoise domain call count in ' + name)
        call = calls[0]
        arguments = {kw.arg: kw.value for kw in call.keywords}
        if (call.args or set(arguments) not in [{'T'}, {'T', 'nan'}]
                or not isinstance(arguments.get('T'), ast.Name) or arguments['T'].id != 'float'):
            raise RuntimeError('Unsupported SmartNoise scalar domain in ' + name)
        if 'nan' in arguments:
            if not isinstance(arguments['nan'], ast.Constant) or arguments['nan'].value is not False:
                raise RuntimeError('Unexpected explicit NaN domain policy in ' + name)
            continue
        if call.lineno != call.end_lineno:
            raise RuntimeError('Unsupported multiline SmartNoise domain declaration')
        replacements.append((call.lineno - 1, call.col_offset, call.end_col_offset))
    for line, start, end in sorted(replacements, reverse=True):
        lines[line] = lines[line][:end - 1] + ', nan=False' + lines[line][end - 1:]
    return ''.join(lines)


def source_path():
    if any(metadata.version(name) != version for name, version in
           [('smartnoise-synth', '1.0.6'), ('opendp', '0.14.2'), ('opacus', '0.14.0')]):
        raise RuntimeError('This compatibility fix requires smartnoise-synth==1.0.6, opendp==0.14.2 and opacus==0.14.0')
    return Path(metadata.distribution('smartnoise-synth').locate_file('snsynth/utils.py')).resolve()


def provenance():
    path = source_path()
    source = path.read_text(encoding='utf-8')
    if corrected_source(source) != source:
        raise RuntimeError('AIM compatibility patch absent; run python -m revision.patch_aim in the DP environment')
    backup = path.with_suffix('.py.trustsynth-original')
    return {'change': CHANGE, 'functions': list(FUNCTIONS),
            'domain': 'dp.atom_domain(T=float, nan=False)',
            'utils_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'original_utils_sha256': hashlib.sha256(backup.read_text(encoding='utf-8').encode()).hexdigest() if backup.exists() else None,
            'patch_implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'software': {name: metadata.version(name) for name in
                         ['smartnoise-synth', 'opendp', 'opacus', 'private-pgm', 'numpy', 'pandas', 'scipy', 'torch']},
            'private_pgm_source': metadata.distribution('private-pgm').read_text('direct_url.json'),
            'python': sys.version}


def main():
    path = source_path()
    if Path(sys.prefix).resolve() not in path.parents or sys.prefix == sys.base_prefix:
        raise RuntimeError('Apply this patch inside the isolated DP environment, never to a shared installation')
    source = path.read_text(encoding='utf-8')
    patched = corrected_source(source)
    if source != patched:
        backup = path.with_suffix('.py.trustsynth-original')
        if not backup.exists():
            backup.write_text(source, encoding='utf-8')
        path.write_text(patched, encoding='utf-8')
        # A same-second rewrite must not reuse cached bytecode.
        cache = Path(importlib.util.cache_from_source(str(path)))
        if cache.exists():
            cache.unlink()
        importlib.invalidate_caches()
    details = provenance()
    write_json(PUBLIC / 'aim_compatibility.json', details)
    print('Verified AIM non-NaN scalar domains in cdp_rho, gaussian_noise and laplace_noise.', flush=True)
    return details


if __name__ == '__main__':
    main()
