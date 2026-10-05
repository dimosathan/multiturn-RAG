"""Ported from 4.ipynb cell 82. Input paths are taken from environment variables
(MTRAG_DEV_REFERENCE, MTRAG_TEST_REFERENCE, MTRAG_FIG_DIR); defaults follow data/README.md.
"""
"""
Figure 4: Turn position distribution — Dev vs Test
Shows WHERE in the conversation each evaluated turn falls.
Dev: mix of turn positions 1..10+
Test: turn positions of the 507 evaluated test turns

Outputs:
  - turn_position_distribution.pdf
  - printed summary text
"""

import json, os
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEV_PATH   = os.environ.get('MTRAG_DEV_REFERENCE', 'data/generation/reference.jsonl')
TEST_PATH  = os.environ.get('MTRAG_TEST_REFERENCE', 'data/test/reference_final.jsonl')
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
    'blue': '#376EC4', 'orange': '#D7820F',
    'teal': '#008C7D', 'gray': '#6E6E6E', 'red': '#BE3737',
}

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} turns from {path}")
    return data

def get_turn_number(entry):
    return int(entry.get('turn', entry.get('turn_number',
               entry.get('turn_index', 1))))

def fig_turn_position_distribution():
    print("\n  Generating: turn_position_distribution.pdf")

    dev_data  = load_jsonl(DEV_PATH)
    test_data = load_jsonl(TEST_PATH)

    dev_turns  = [get_turn_number(e) for e in dev_data]
    test_turns = [get_turn_number(e) for e in test_data]

    # Bin: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10+
    MAX_BIN = 10
    def bin_turn(t):
        return min(t, MAX_BIN)

    dev_binned  = [bin_turn(t) for t in dev_turns]
    test_binned = [bin_turn(t) for t in test_turns]

    bins     = list(range(1, MAX_BIN + 1))
    dev_cnt  = Counter(dev_binned)
    test_cnt = Counter(test_binned)

    dev_pct  = [100 * dev_cnt.get(b, 0) / len(dev_turns)  for b in bins]
    test_pct = [100 * test_cnt.get(b, 0) / len(test_turns) for b in bins]

    x_labels = [str(b) if b < MAX_BIN else f'{MAX_BIN}+' for b in bins]

    # ── Print stats ──
    dev_first_pct  = 100 * dev_cnt.get(1, 0) / len(dev_turns)
    test_first_pct = 100 * test_cnt.get(1, 0) / len(test_turns)
    dev_nonfirst   = 100 * sum(dev_cnt.get(b,0)
                                for b in bins if b > 1) / len(dev_turns)
    test_nonfirst  = 100 * sum(test_cnt.get(b,0)
                                for b in bins if b > 1) / len(test_turns)

    print(f"\n  Dev  — Turn 1: {dev_first_pct:.1f}%, "
          f"Non-first: {dev_nonfirst:.1f}%")
    print(f"  Test — Turn 1: {test_first_pct:.1f}%, "
          f"Non-first: {test_nonfirst:.1f}%")

    # ── Figure: side-by-side bars + cumulative line ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                    sharey=False)

    width = 0.36
    x = np.arange(len(bins))

    # Left: grouped bar
    ax1.bar(x - width/2, dev_pct,  width,
            label='Dev',  color=COLORS['blue'],
            edgecolor='white', linewidth=0.4)
    ax1.bar(x + width/2, test_pct, width,
            label='Test', color=COLORS['orange'],
            edgecolor='white', linewidth=0.4)

    ax1.set_xlabel('Turn Position in Conversation')
    ax1.set_ylabel('% of evaluated turns')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(frameon=False, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Annotate structural difference
    ax1.annotate(f'Test: {100 - test_pct[0]:.1f}% non-first',
                 xy=(0, test_pct[0] if test_pct[0] > 0 else 0.5),
                 xytext=(1.5, max(dev_pct)*0.7),
                 fontsize=7, color=COLORS['orange'],
                 arrowprops=dict(arrowstyle='->', lw=0.8,
                                 color=COLORS['orange']))

    # Right: cumulative % of turns by position
    dev_cum  = np.cumsum(dev_pct)
    test_cum = np.cumsum(test_pct)

    ax2.plot(x, dev_cum,  color=COLORS['blue'],
             linewidth=2, marker='o', markersize=4,
             label='Dev')
    ax2.plot(x, test_cum, color=COLORS['orange'],
             linewidth=2, marker='o', markersize=4,
             label='Test', linestyle='--')

    ax2.set_xlabel('Turn Position (≤)')
    ax2.set_ylabel('Cumulative % of turns')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim(0, 105)
    ax2.axhline(50, color=COLORS['gray'], linewidth=0.7,
                linestyle=':', alpha=0.7)
    ax2.text(9.1, 51, '50%', fontsize=7, color=COLORS['gray'])
    ax2.legend(frameon=False, loc='lower right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Cumulative distribution', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'turn_position_distribution.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)

    # ── LaTeX caption ──
    print(f"\n  Suggested caption:")
    print(f"  Turn position distribution for evaluated turns in the")
    print(f"  development (blue) and test (orange) sets. The dev set")
    print(f"  spans positions 1--10+; in the test set {100 - test_pct[0]:.1f}% of the")
    print(f"  evaluated turns are non-first turns (dev: {100 - dev_pct[0]:.1f}%).")


fig_turn_position_distribution()
