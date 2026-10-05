"""Ported from 4.ipynb cell 81. Input paths are taken from environment variables
(MTRAG_DEV_REFERENCE, MTRAG_TEST_REFERENCE, MTRAG_RESULTS_DIR, MTRAG_FIG_DIR); defaults follow data/README.md.
"""
"""
Figure 3: Non-standalone phenomena breakdown
Reads dev set JSONL and classifies each turn >1 using
heuristic regex patterns. Outputs:
  - horizontal bar chart (non_standalone_phenomena.pdf)
  - printed LaTeX table snippet

The QUESTION field is read from:
  entry['question'] OR entry['input'] OR entry['query']
"""

import json, re, os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEV_PATH   = os.environ.get('MTRAG_DEV_REFERENCE', 'data/generation/reference.jsonl')
OUTPUT_DIR = os.environ.get('MTRAG_FIG_DIR', 'results/figures/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
COLORS = {
    'blue': '#376EC4', 'teal': '#008C7D',
    'orange': '#D7820F', 'purple': '#783AA0',
    'red': '#BE3737', 'gray': '#6E6E6E', 'green': '#2E8B57',
}

# ── PHENOMENON DETECTORS ─────────────────────────────
# Each returns True if the pattern is present in the query.
# Applied only to non-first turns (turn > 1).

PHENOMENA = {
    'Pronoun coreference\n(it, they, this, that…)': lambda q: bool(re.search(
        r'\b(it|its|they|them|their|this|that|these|those|he|she|him|her)\b',
        q, re.I)),

    'Ellipsis / bare topic\n(≤4 tokens, no verb)': lambda q: (
        len(q.split()) <= 4 and
        not re.search(r'\b(is|are|was|were|do|does|did|have|has|had|can|could|would|will|should|what|why|how|when|where|who)\b', q, re.I)
    ),

    'Implicit topic carryover\n("What about…", "And…")': lambda q: bool(re.search(
        r'^(what about|how about|and (what|how|why|when|where)|tell me more|more (about|on|info)|anything else)',
        q.strip(), re.I)),

    'Temporal dependence\n(this week/month, now, current…)': lambda q: bool(re.search(
        r'\b(this week|this month|this year|right now|currently|today|latest|recent|upcoming|next week|last week)\b',
        q, re.I)),

    'Comparative follow-up\n(better, cheaper, vs, difference…)': lambda q: bool(re.search(
        r'\b(better|cheaper|faster|worse|more|less|vs\.?|versus|compared|difference|alternative|instead|rather)\b',
        q, re.I)),

    'Scope ambiguity\n("What about security/pricing?")': lambda q: bool(re.search(
        r'^(what about|how about)\s+\w+\??$',
        q.strip(), re.I)),
}

# ── LOAD AND CLASSIFY ─────────────────────────────────
def get_question(entry):
    for key in ['question', 'input', 'query', 'Question', 'utterance']:
        if key in entry and entry[key]:
            return str(entry[key]).strip()
    return ''

def get_turn_number(entry):
    return int(entry.get('turn', entry.get('turn_number', 1)))

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def analyze_non_standalone(data):
    non_first = [e for e in data if get_turn_number(e) > 1]
    total_ns  = len(non_first)

    print(f"\n  Non-first turns: {total_ns} / {len(data)}")

    # Count phenomena (multi-label: a turn can match multiple)
    counts   = Counter()
    examples = {}

    for entry in non_first:
        q = get_question(entry)
        for name, detector in PHENOMENA.items():
            if detector(q):
                counts[name] += 1
                if name not in examples:
                    examples[name] = q[:60] + ('…' if len(q) > 60 else '')

    print(f"\n  Phenomenon frequencies (% of non-first turns):")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_ns
        ex  = examples.get(name, '')
        print(f"    {pct:5.1f}%  {name.replace(chr(10), ' ')}")
        print(f"             e.g.: \"{ex}\"")

    return counts, total_ns, examples

# ── FIGURE ────────────────────────────────────────────
def fig_nonstandalone_phenomena():
    print("\n  Generating: non_standalone_phenomena.pdf")
    data = load_jsonl(DEV_PATH)
    counts, total_ns, examples = analyze_non_standalone(data)

    # Sort by frequency descending
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    labels = [item[0] for item in sorted_items]
    values = [100 * item[1] / total_ns for item in sorted_items]

    # Color gradient from most to least frequent
    bar_colors = [
        COLORS['blue'], COLORS['teal'], COLORS['orange'],
        COLORS['purple'], COLORS['green'], COLORS['gray']
    ][:len(labels)]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, values, color=bar_colors,
                   edgecolor='white', linewidth=0.5, height=0.6)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}%', va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('% of non-first turns (multi-label)')
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Footnote
    fig.text(0.02, -0.04,
             f'N={total_ns} non-first turns. Multi-label: turns may match '
             f'multiple phenomena.',
             fontsize=7, color=COLORS['gray'], style='italic')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'non_standalone_phenomena.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)

    # ── LaTeX table snippet ──
    print(f"\n  LaTeX table snippet:")
    print(r"  \begin{table}[H]")
    print(r"  \centering\small")
    print(r"  \begin{tabular}{lcp{4.5cm}}")
    print(r"  \toprule")
    print(r"  \textbf{Phenomenon} & \textbf{\% turns} & \textbf{Example} \\")
    print(r"  \midrule")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_ns
        ex  = examples.get(name, '').replace('…', r'\ldots')
        short_name = name.split('\n')[0]
        print(f"  {short_name} & {pct:.0f}\\% & "
              f"\\emph{{\"{ex}\"}} \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(f"  \\caption{{Non-standalone phenomena in dev set "
          f"non-first turns ($N={total_ns}$). Multi-label; "
          f"percentages sum to $>$100.}}")
    print(r"  \label{tab:nonstandalone_phenomena}")
    print(r"  \end{table}")


fig_nonstandalone_phenomena()
