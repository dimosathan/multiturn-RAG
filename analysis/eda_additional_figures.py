"""Ported from 4.ipynb cell 80. Input paths are taken from environment variables
(MTRAG_DEV_REFERENCE, MTRAG_TEST_REFERENCE, MTRAG_FIG_DIR); defaults follow data/README.md.
"""
"""
Additional EDA Figures for MTRAG Paper Appendix (Colab version)
================================================================
Run AFTER the main eda_analysis_colab.py script.

Generates:
  1. dev_test_ans_per_domain.pdf  — Side-by-side answerability comparison
  2. conv_length_distribution.pdf — Conversation length histogram (dev vs test)
  3. dev_test_qtype_shift.pdf     — Question type shift (dev vs test)

Usage in Colab:
    Paste and run this entire cell after mounting Google Drive.
"""

import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# ⚙️ CONFIGURATION — SAME AS MAIN SCRIPT
# ============================================================
DEV_PATH   = os.environ.get('MTRAG_DEV_REFERENCE', 'data/generation/reference.jsonl')
TEST_PATH  = os.environ.get('MTRAG_TEST_REFERENCE', 'data/test/reference_final.jsonl')
OUTPUT_DIR = os.environ.get('MTRAG_FIG_DIR', 'results/figures/')
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'blue':   '#376EC4',
    'teal':   '#008C7D',
    'orange': '#D7820F',
    'purple': '#783AA0',
    'red':    '#BE3737',
    'gray':   '#6E6E6E',
    'green':  '#2E8B57',
}


# ============================================================
# Data helpers (same as main script)
# ============================================================
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} turns from {os.path.basename(path)}")
    return data


def normalize_domain(collection_str):
    c = collection_str.lower()
    if 'clapnq' in c: return 'ClapNQ'
    elif 'fiqa' in c: return 'FiQA'
    elif 'govt' in c or 'government' in c: return 'Govt'
    elif 'cloud' in c or 'ibmcloud' in c: return 'Cloud'
    return 'Unknown'


def get_answerability(entry):
    ans = entry.get('Answerability', entry.get('answerability', []))
    if not ans: return 'Unknown'
    val = ans[0].upper().strip()
    return {'ANSWERABLE': 'Answerable', 'UNANSWERABLE': 'Unanswerable',
            'PARTIALLY_ANSWERABLE': 'Partial', 'PARTIAL': 'Partial',
            'CONVERSATIONAL': 'Conversational'}.get(val, val)


def get_domain(entry):
    return normalize_domain(entry.get('Collection', ''))


def get_conversation_id(entry):
    return entry.get('conversation_id', '')


def get_turn_number(entry):
    return int(entry.get('turn', 1))


def get_question_types(entry):
    return entry.get('Question Type', [])


# ============================================================
# FIGURE 1: Dev vs Test Answerability Per Domain
# ============================================================
def fig_dev_test_ans_per_domain(dev_data, test_data):
    """
    Grouped bar chart: unanswerable rate per domain,
    dev vs test side by side.
    """
    domains = ['ClapNQ', 'FiQA', 'Govt', 'Cloud', 'Overall']

    def compute_unans_rate(data):
        domain_total = Counter()
        domain_unans = Counter()
        for e in data:
            d = get_domain(e)
            domain_total[d] += 1
            if get_answerability(e) == 'Unanswerable':
                domain_unans[d] += 1
        rates = {}
        for d in ['ClapNQ', 'FiQA', 'Govt', 'Cloud']:
            rates[d] = 100 * domain_unans[d] / domain_total[d] if domain_total[d] else 0
        total = sum(domain_total.values())
        total_u = sum(domain_unans.values())
        rates['Overall'] = 100 * total_u / total if total else 0
        return rates

    dev_rates = compute_unans_rate(dev_data)
    test_rates = compute_unans_rate(test_data)

    x = np.arange(len(domains))
    width = 0.32

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    dev_vals = [dev_rates[d] for d in domains]
    test_vals = [test_rates[d] for d in domains]

    bars1 = ax.bar(x - width/2, dev_vals, width,
                   label='Dev', color=COLORS['blue'],
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_vals, width,
                   label='Test', color=COLORS['red'],
                   edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Unanswerable Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend(frameon=False, loc='upper left')
    ax.set_ylim(0, max(test_vals) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.0f}%', ha='center', va='bottom', fontsize=7)

    # Add separator before Overall
    ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dev_test_ans_per_domain.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 2: Conversation Length Distribution
# ============================================================
def fig_conv_length_distribution(dev_data, test_data):
    """
    Histogram: number of evaluated turns per conversation,
    dev vs test.
    """
    def conv_lengths(data):
        conv_turns = Counter()
        for e in data:
            conv_turns[get_conversation_id(e)] += 1
        return list(conv_turns.values())

    dev_lengths = conv_lengths(dev_data)
    test_lengths = conv_lengths(test_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4),
                                     gridspec_kw={'width_ratios': [2, 1]})

    # Dev histogram
    bins_dev = np.arange(0.5, max(dev_lengths) + 1.5, 1)
    ax1.hist(dev_lengths, bins=bins_dev, color=COLORS['blue'],
             edgecolor='white', linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Evaluated turns per conversation')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Dev (n={len(dev_lengths)})', fontsize=10)
    ax1.axvline(x=np.mean(dev_lengths), color=COLORS['red'],
                linestyle='--', linewidth=1, label=f'Mean={np.mean(dev_lengths):.1f}')
    ax1.legend(frameon=False, fontsize=7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Test histogram
    bins_test = np.arange(0.5, max(test_lengths) + 1.5, 1)
    ax2.hist(test_lengths, bins=bins_test, color=COLORS['orange'],
             edgecolor='white', linewidth=0.5, alpha=0.8)
    ax2.set_xlabel('Evaluated turns per conversation')
    ax2.set_title(f'Test (n={len(test_lengths)})', fontsize=10)
    ax2.axvline(x=np.mean(test_lengths), color=COLORS['red'],
                linestyle='--', linewidth=1, label=f'Mean={np.mean(test_lengths):.1f}')
    ax2.legend(frameon=False, fontsize=7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'conv_length_distribution.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 3: Question Type Shift (Dev vs Test)
# ============================================================
def fig_qtype_shift(dev_data, test_data):
    """
    Horizontal bar chart: question type percentages,
    dev vs test (top-7 types).
    """
    def qtype_pcts(data):
        counts = Counter()
        for e in data:
            for qt in get_question_types(e):
                counts[qt] += 1
        total = len(data)
        return {k: 100 * v / total for k, v in counts.items()}

    dev_pcts = qtype_pcts(dev_data)
    test_pcts = qtype_pcts(test_data)

    # Top 7 by combined frequency
    all_types = set(dev_pcts.keys()) | set(test_pcts.keys())
    combined = {qt: dev_pcts.get(qt, 0) + test_pcts.get(qt, 0)
                for qt in all_types}
    top_types = sorted(combined, key=combined.get, reverse=True)[:7]
    top_types.reverse()  # so highest is on top in horizontal bars

    dev_vals = [dev_pcts.get(qt, 0) for qt in top_types]
    test_vals = [test_pcts.get(qt, 0) for qt in top_types]

    y = np.arange(len(top_types))
    height = 0.32

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.barh(y - height/2, dev_vals, height,
            label='Dev', color=COLORS['blue'],
            edgecolor='white', linewidth=0.5)
    ax.barh(y + height/2, test_vals, height,
            label='Test', color=COLORS['orange'],
            edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Percentage of turns (%)')
    ax.set_yticks(y)
    ax.set_yticklabels(top_types)
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate notable shifts
    for i, qt in enumerate(top_types):
        d = dev_pcts.get(qt, 0)
        t = test_pcts.get(qt, 0)
        shift = t - d
        if abs(shift) > 5:
            max_val = max(d, t)
            sign = '+' if shift > 0 else ''
            ax.text(max_val + 1, i, f'{sign}{shift:.0f}pp',
                    va='center', fontsize=7, color=COLORS['red'],
                    fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dev_test_qtype_shift.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 4: Answerability Shift Heatmap (Dev→Test per Domain)
# ============================================================
def fig_answerability_shift_heatmap(dev_data, test_data):
    """
    Heatmap showing the CHANGE in answerability rates
    from dev to test, per domain.
    Red = increase, Blue = decrease.
    """
    domains = ['ClapNQ', 'FiQA', 'Govt', 'Cloud']
    categories = ['Answerable', 'Partial', 'Unanswerable']

    def compute_rates(data):
        domain_ans = defaultdict(Counter)
        for e in data:
            domain_ans[get_domain(e)][get_answerability(e)] += 1
        rates = {}
        for d in domains:
            total = sum(domain_ans[d].values())
            rates[d] = {c: 100 * domain_ans[d].get(c, 0) / total
                       for c in categories} if total else {c: 0 for c in categories}
        return rates

    dev_rates = compute_rates(dev_data)
    test_rates = compute_rates(test_data)

    # Compute shift matrix
    matrix = np.zeros((len(categories), len(domains)))
    for j, d in enumerate(domains):
        for i, c in enumerate(categories):
            matrix[i, j] = test_rates[d].get(c, 0) - dev_rates[d].get(c, 0)

    fig, ax = plt.subplots(figsize=(4.0, 2.2))

    # Diverging colormap centered at 0
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    # Annotate cells with shift values
    for i in range(len(categories)):
        for j in range(len(domains)):
            val = matrix[i, j]
            sign = '+' if val > 0 else ''
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{sign}{val:.0f}pp',
                    ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Δ (test − dev) pp', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title('Answerability Shift: Dev → Test', fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'answerability_shift_heatmap.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)


# ============================================================
# Additional Stats (printed to console)
# ============================================================
def print_additional_stats(dev_data, test_data):
    """Print extra stats useful for filling in LaTeX text."""

    print(f"\n{'='*60}")
    print("  Additional Statistics for Paper Text")
    print(f"{'='*60}")

    # Turn position distribution in test
    test_turns = [get_turn_number(e) for e in test_data]
    print(f"\n  Test set turn positions:")
    turn_counts = Counter(test_turns)
    for t in sorted(turn_counts.keys()):
        print(f"    Turn {t:2d}: {turn_counts[t]:4d} "
              f"({100*turn_counts[t]/len(test_data):.1f}%)")
    print(f"    Mean turn position: {np.mean(test_turns):.1f}")
    print(f"    Median turn position: {np.median(test_turns):.0f}")

    # Non-standalone in test
    non_first = sum(1 for e in test_data if get_turn_number(e) > 1)
    print(f"\n  Test non-first turns: {non_first}/{len(test_data)} "
          f"({100*non_first/len(test_data):.0f}%)")

    # Unanswerable per turn position (dev)
    print(f"\n  Dev: Unanswerable rate by turn position:")
    turn_total = defaultdict(int)
    turn_unans = defaultdict(int)
    for e in dev_data:
        t = min(get_turn_number(e), 10)  # bucket 10+
        turn_total[t] += 1
        if get_answerability(e) == 'Unanswerable':
            turn_unans[t] += 1
    for t in sorted(turn_total.keys()):
        rate = 100 * turn_unans[t] / turn_total[t]
        label = f"Turn {t}" if t < 10 else "Turn 10+"
        print(f"    {label:10s}: {turn_unans[t]:3d}/{turn_total[t]:3d} "
              f"({rate:.1f}%)")

    # Reference answer length comparison
    dev_lens = [len(e.get('targets', [{}])[-1].get('text', '').split())
                for e in dev_data if e.get('targets')]
    test_lens = [len(e.get('targets', [{}])[-1].get('text', '').split())
                 for e in test_data if e.get('targets')]
    print(f"\n  Reference answer lengths:")
    print(f"    Dev:  mean={np.mean(dev_lens):.1f}, "
          f"median={np.median(dev_lens):.0f}")
    print(f"    Test: mean={np.mean(test_lens):.1f}, "
          f"median={np.median(test_lens):.0f}")
    print(f"    Test answers are {100*(1-np.mean(test_lens)/np.mean(dev_lens)):.0f}% "
          f"shorter on average")


# ============================================================
# RUN
# ============================================================
print("\n" + "="*60)
print("  MTRAG Additional EDA Figures")
print("="*60)

dev_data = load_jsonl(DEV_PATH)
test_data = load_jsonl(TEST_PATH)

print(f"\n{'='*60}")
print("  Generating Additional Figures...")
print(f"{'='*60}")

fig_dev_test_ans_per_domain(dev_data, test_data)
fig_conv_length_distribution(dev_data, test_data)
fig_qtype_shift(dev_data, test_data)
fig_answerability_shift_heatmap(dev_data, test_data)

print_additional_stats(dev_data, test_data)

print(f"\n{'='*60}")
print(f"  DONE! New figures in {OUTPUT_DIR}:")
print(f"    ├── dev_test_ans_per_domain.pdf")
print(f"    ├── conv_length_distribution.pdf")
print(f"    ├── dev_test_qtype_shift.pdf")
print(f"    └── answerability_shift_heatmap.pdf")
print(f"{'='*60}\n")