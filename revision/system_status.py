"""Lightweight local resource diagnostics; emits only this revision's process metadata."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.venv-synthcity' / 'Lib' / 'site-packages'))
import psutil

memory = psutil.virtual_memory()
print('Available memory MiB:', round(memory.available / 2**20), flush=True)
for process in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline', 'cpu_times', 'memory_info']):
    try:
        info = process.info
        command = ' '.join(info['cmdline'] or [])
        if info['name'] == 'python.exe' and ('revision.' in command or '.venv-' in command or '-m pip' in command):
            print({'pid': info['pid'], 'parent': info['ppid'], 'command': command,
                   'cpu_seconds': info['cpu_times'].user + info['cpu_times'].system,
                   'resident_MiB': round(info['memory_info'].rss / 2**20),
                   'io_read_MiB': round(process.io_counters().read_bytes / 2**20),
                   'io_write_MiB': round(process.io_counters().write_bytes / 2**20)}, flush=True)
    except psutil.Error:
        pass
