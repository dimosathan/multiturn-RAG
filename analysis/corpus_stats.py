"""Count documents and passages per corpus directly from the passage-level JSONL files.

    python analysis/corpus_stats.py data/corpora/{clapnq,fiqa,govt,cloud}.jsonl

A document id is the passage id without its trailing ``-<start>-<end>`` offsets
(e.g. ``551325-0-398`` -> ``551325``).  Use the output to verify Table 1 / Table 7.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

OFFSETS = re.compile(r"-\d+-\d+$")


def stats(path: str):
    docs: Counter = Counter()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row.get("_id") or row.get("id") or row.get("document_id"))
            docs[OFFSETS.sub("", pid)] += 1
            n += 1
    return len(docs), n


if __name__ == "__main__":
    print(f"{'corpus':<10}{'documents':>12}{'passages':>12}{'avg p/d':>10}")
    for p in sys.argv[1:]:
        d, n = stats(p)
        print(f"{Path(p).stem:<10}{d:>12,}{n:>12,}{n / max(d, 1):>10.1f}")
