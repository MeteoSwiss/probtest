import re
import sys

file_path = sys.argv[1]
builder = sys.argv[2]

regex = f'.*{builder}.*configureflags="(?P<flags>.*?)"'

with open(file_path, encoding="utf-8") as f:
    data = f.read()
    try:
        m = re.search(regex, data, re.IGNORECASE).groupdict()
        print(m["flags"])
    except AttributeError:
        print(f"ERROR: did not find configure flags for {builder}")
