"""Expose stack locations and exception class only; never echo private log text."""
import argparse
import re
from .safe_reporting import private_path

parser = argparse.ArgumentParser()
parser.add_argument('name', nargs='?', default='generator_failure.log')
args = parser.parse_args()
text = private_path('logs/' + args.name).read_text(encoding='utf-8')
for filename, line, function in re.findall(r'File "([^"]+)", line (\d+), in ([\w<>]+)', text):
    print({'file': filename, 'line': line, 'function': function})
last = text.strip().splitlines()[-1]
match = re.match(r'([A-Za-z]+(?:Error|Exception)):', last)
print('Exception class:', match.group(1) if match else 'unclassified')
print('Known implementation terms:', [word for word in ['TimedIterator', 'tqdm', 'deterministic', 'repeat', 'torch', 'metadata', 'NoneType', 'len', 'seed'] if word in last])
