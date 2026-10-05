"""Ported from 4.ipynb cell 79. Input paths are taken from environment variables
(MTRAG_DEV_REFERENCE, MTRAG_TEST_REFERENCE, MTRAG_FIG_DIR); defaults follow data/README.md.
"""
"""
EDA Analysis Script for MTRAG Paper Appendix (Colab version)
=============================================================
Generates all tables (LaTeX) and figures (PDF) for the Dataset Details appendix.

Usage in Colab:
    1. Mount Google Drive
    2. Update DEV_PATH and TEST_PATH below
    3. Run this entire cell
"""

import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# ⚙️ CONFIGURATION — UPDATE THESE PATHS
# ============================================================
DEV_PATH  = os.environ.get('MTRAG_DEV_REFERENCE', 'data/generation/reference.jsonl')
TEST_PATH = os.environ.get('MTRAG_TEST_REFERENCE', 'data/test/reference_final.jsonl')
OUTPUT_DIR = os.environ.get('MTRAG_FIG_DIR', 'results/figures/')
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Paper-quality plot settings (ACL/SemEval style)
# ============================================================
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
DOMAIN_COLORS = {
    'ClapNQ': COLORS['blue'],
    'FiQA':   COLORS['orange'],
    'Govt':   COLORS['teal'],
    'Cloud':  COLORS['purple'],
}
ANS_COLORS = {
    'Answerable':     COLORS['blue'],
    'Partial':        COLORS['teal'],
    'Unanswerable':   COLORS['orange'],
    'Conversational': COLORS['gray'],
}


# ============================================================
# Data Loading & Normalization
# ============================================================
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} turns from {path}")
    return data


def normalize_domain(collection_str):
    c = collection_str.lower()
    if 'clapnq' in c:
        return 'ClapNQ'
    elif 'fiqa' in c:
        return 'FiQA'
    elif 'govt' in c or 'government' in c:
        return 'Govt'
    elif 'cloud' in c or 'ibmcloud' in c:
        return 'Cloud'
    else:
        return 'Unknown'


def get_answerability(entry):
    ans = entry.get('Answerability', entry.get('answerability', []))
    if not ans:
        return 'Unknown'
    val = ans[0].upper().strip()
    mapping = {
        'ANSWERABLE': 'Answerable',
        'UNANSWERABLE': 'Unanswerable',
        'PARTIALLY_ANSWERABLE': 'Partial',
        'PARTIAL': 'Partial',
        'CONVERSATIONAL': 'Conversational',
    }
    return mapping.get(val, val)


def get_question_types(entry):
    return entry.get('Question Type', [])


def get_multi_turn(entry):
    mt = entry.get('Multi-Turn', [])
    return mt[0] if mt else 'N/A'


def get_turn_number(entry):
    return int(entry.get('turn', 1))


def get_conversation_id(entry):
    return entry.get('conversation_id', '')


def get_ref_answer_text(entry):
    targets = entry.get('targets', [])
    if targets:
        return targets[-1].get('text', '')
    return ''


def get_domain(entry):
    return normalize_domain(entry.get('Collection', ''))


# ============================================================
# Analysis Functions
# ============================================================
def basic_stats(data, split_name='Dev'):
    convs = set(get_conversation_id(e) for e in data)
    turns = len(data)
    avg_turns = turns / len(convs) if convs else 0
    non_standalone = sum(1 for e in data if get_turn_number(e) > 1)
    pct_ns = 100 * non_standalone / turns if turns else 0

    print(f"\n{'='*60}")
    print(f"  {split_name} Set — Basic Statistics")
    print(f"{'='*60}")
    print(f"  Conversations:     {len(convs)}")
    print(f"  Total turns:       {turns}")
    print(f"  Avg turns/conv:    {avg_turns:.1f}")
    print(f"  Non-standalone:    {non_standalone} ({pct_ns:.0f}%)")

    return {
        'conversations': len(convs),
        'turns': turns,
        'avg_turns': round(avg_turns, 1),
        'non_standalone_pct': round(pct_ns, 1),
    }


def answerability_distribution(data, split_name='Dev'):
    ans_counts = Counter(get_answerability(e) for e in data)
    total = sum(ans_counts.values())

    print(f"\n  Answerability Distribution ({split_name}):")
    for label in ['Answerable', 'Partial', 'Unanswerable', 'Conversational']:
        count = ans_counts.get(label, 0)
        pct = 100 * count / total if total else 0
        print(f"    {label:20s}: {count:4d} ({pct:5.1f}%)")

    return {k: round(100 * v / total, 1) for k, v in ans_counts.items()}


def answerability_per_domain(data, split_name='Dev'):
    domain_ans = defaultdict(Counter)
    for e in data:
        domain_ans[get_domain(e)][get_answerability(e)] += 1

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  LaTeX: Answerability per Domain ({split_name})          │")
    print(f"  └─────────────────────────────────────────────────┘")
    print()
    print(r"  \begin{table}[H]")
    print(r"  \centering\small")
    print(r"  \begin{tabular}{lcccc}")
    print(r"  \toprule")
    print(r"  \textbf{Domain} & \textbf{Ans} & \textbf{Partial}")
    print(r"  & \textbf{Unans} & \textbf{Conv} \\")
    print(r"  \midrule")

    overall = Counter()
    for domain in ['ClapNQ', 'FiQA', 'Govt', 'Cloud']:
        counts = domain_ans[domain]
        total = sum(counts.values())
        if total == 0:
            continue
        row = []
        for label in ['Answerable', 'Partial', 'Unanswerable', 'Conversational']:
            c = counts.get(label, 0)
            pct = 100 * c / total
            row.append(f"{pct:.0f}\\%")
            overall[label] += c
        print(f"  {domain:8s} & {' & '.join(row)} \\\\")

    ov_total = sum(overall.values())
    ov_row = [f"{100*overall.get(l,0)/ov_total:.0f}\\%"
              for l in ['Answerable', 'Partial', 'Unanswerable', 'Conversational']]
    print(r"  \midrule")
    print(f"  Overall  & {' & '.join(ov_row)} \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(f"  \\caption{{Answerability distribution per domain "
          f"({split_name.lower()} set, {len(data)} turns). "
          f"Percentages are row-normalized.}}")
    print(r"  \label{tab:ans_per_domain}")
    print(r"  \end{table}")

    return dict(domain_ans)


def qtype_per_domain(data, split_name='Dev'):
    domain_qtypes = defaultdict(Counter)
    for e in data:
        domain = get_domain(e)
        for qt in get_question_types(e):
            domain_qtypes[domain][qt] += 1

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  LaTeX: Top-3 Question Types per Domain          │")
    print(f"  └─────────────────────────────────────────────────┘")
    print()
    print(r"  \begin{table}[H]")
    print(r"  \centering\small")
    print(r"  \begin{tabular}{llc}")
    print(r"  \toprule")
    print(r"  \textbf{Domain} & \textbf{Top-3 Question Types}")
    print(r"  & \textbf{Coverage} \\")
    print(r"  \midrule")

    for domain in ['ClapNQ', 'FiQA', 'Govt', 'Cloud']:
        counts = domain_qtypes[domain]
        total = sum(counts.values())
        if total == 0:
            continue
        top3 = counts.most_common(3)
        names = ', '.join(t[0] for t in top3)
        coverage = sum(t[1] for t in top3) / total * 100
        print(f"  {domain:8s} & {names:42s} & {coverage:.0f}\\% \\\\")

    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(f"  \\caption{{Top-3 question types per domain "
          f"({split_name.lower()} set). "
          f"Coverage = percentage of turns with one of the three types.}}")
    print(r"  \label{tab:qtype_per_domain}")
    print(r"  \end{table}")

    return dict(domain_qtypes)


def domain_distribution(data, split_name='Dev'):
    domain_counts = Counter(get_domain(e) for e in data)
    total = sum(domain_counts.values())

    print(f"\n  Domain Distribution ({split_name}):")
    for d in ['ClapNQ', 'FiQA', 'Govt', 'Cloud']:
        c = domain_counts.get(d, 0)
        print(f"    {d:8s}: {c:4d} ({100*c/total:.1f}%)")

    return {k: round(100 * v / total, 1) for k, v in domain_counts.items()}


def qtype_overall(data, split_name='Dev'):
    qtype_counts = Counter()
    for e in data:
        for qt in get_question_types(e):
            qtype_counts[qt] += 1

    total = len(data)
    print(f"\n  Question Type Distribution ({split_name}, top-5):")
    for qt, count in qtype_counts.most_common(5):
        print(f"    {qt:20s}: {count:4d} ({100*count/total:.1f}%)")

    return {k: round(100 * v / total, 1) for k, v in qtype_counts.items()}


def reference_answer_stats(data, split_name='Dev'):
    lengths = []
    qtype_lengths = defaultdict(list)

    for e in data:
        text = get_ref_answer_text(e)
        wc = len(text.split())
        if wc > 0:
            lengths.append(wc)
            for qt in get_question_types(e):
                qtype_lengths[qt].append(wc)

    lengths = np.array(lengths)

    print(f"\n  Reference Answer Length Statistics ({split_name}):")
    print(f"    Range:  {lengths.min()} — {lengths.max()} words")
    print(f"    Mean:   {lengths.mean():.1f}")
    print(f"    Median: {np.median(lengths):.0f}")
    print(f"    Std:    {lengths.std():.1f}")

    print(f"\n  LaTeX snippet:")
    print(f"  Reference answer lengths range from {lengths.min()} to "
          f"{lengths.max()} words")
    print(f"  (mean: {lengths.mean():.1f}, median: {np.median(lengths):.0f}, "
          f"std: {lengths.std():.1f}),")

    print(f"\n  Per Question Type (sorted by mean length):")
    sorted_qt = sorted(qtype_lengths.items(),
                       key=lambda x: np.mean(x[1]), reverse=True)
    for qt, vals in sorted_qt:
        vals = np.array(vals)
        print(f"    {qt:20s}: mean={vals.mean():.1f}, "
              f"median={np.median(vals):.0f}, n={len(vals)}")

    if sorted_qt:
        longest_qt = sorted_qt[0][0]
        longest_mean = np.mean(sorted_qt[0][1])
        shortest_qt = sorted_qt[-1][0]
        shortest_mean = np.mean(sorted_qt[-1][1])
        print(f"\n  LaTeX: ...with {longest_qt} producing the longest "
              f"references (mean: {longest_mean:.0f} words)")
        print(f"  and {shortest_qt} the shortest "
              f"(mean: {shortest_mean:.0f} words).")

    return {
        'min': int(lengths.min()),
        'max': int(lengths.max()),
        'mean': round(float(lengths.mean()), 1),
        'median': int(np.median(lengths)),
        'std': round(float(lengths.std()), 1),
        'per_qtype': {
            qt: {'mean': round(float(np.mean(v)), 1),
                 'median': int(np.median(v)), 'n': len(v)}
            for qt, v in qtype_lengths.items()
        }
    }


def multi_turn_stats(data, split_name='Dev'):
    mt_counts = Counter(get_multi_turn(e) for e in data)
    total = sum(mt_counts.values())

    print(f"\n  Multi-Turn Type Distribution ({split_name}):")
    for mt, count in mt_counts.most_common():
        print(f"    {mt:20s}: {count:4d} ({100*count/total:.1f}%)")

    return {k: round(100 * v / total, 1) for k, v in mt_counts.items()}


# ============================================================
# Dev vs Test Comparison (LaTeX)
# ============================================================
def print_dev_test_latex(dev_stats, test_stats, dev_ans, test_ans,
                         dev_domain, test_domain, dev_qtype, test_qtype):
    print(f"\n{'='*60}")
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  LaTeX: Dev vs Test Comparison Table              │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print(r"  \begin{table}[H]")
    print(r"  \centering\small")
    print(r"  \begin{tabular}{lcc}")
    print(r"  \toprule")
    print(r"  \textbf{Characteristic} & \textbf{Dev} & \textbf{Test} \\")
    print(r"  \midrule")
    print(f"  Conversations & {dev_stats['conversations']} "
          f"& {test_stats['conversations']} \\\\")
    print(f"  Total turns & {dev_stats['turns']} "
          f"& {test_stats['turns']} \\\\")
    print(f"  Avg turns/conv & {dev_stats['avg_turns']} "
          f"& {test_stats['avg_turns']} \\\\")
    print(r"  \midrule")

    for label in ['Answerable', 'Partial', 'Unanswerable', 'Conversational']:
        d_val = dev_ans.get(label, 0)
        t_val = test_ans.get(label, 0)
        shift = abs(t_val - d_val)
        t_str = f"\\textbf{{{t_val}}}" if shift > 3 else str(t_val)
        print(f"  {label} (\\%) & {d_val} & {t_str} \\\\")

    print(r"  \midrule")
    print(r"  \multicolumn{3}{l}{\emph{Domain distribution"
          r" (\% of turns)}} \\")
    for d in ['ClapNQ', 'FiQA', 'Govt', 'Cloud']:
        d_val = dev_domain.get(d, 0)
        t_val = test_domain.get(d, 0)
        shift = abs(t_val - d_val)
        t_str = f"\\textbf{{{t_val}}}" if shift > 5 else str(t_val)
        print(f"  {d} & {d_val} & {t_str} \\\\")

    print(r"  \midrule")
    print(r"  \multicolumn{3}{l}{\emph{Question type"
          r" (top-3, \% of turns)}} \\")
    top3 = sorted(dev_qtype.items(), key=lambda x: x[1], reverse=True)[:3]
    for qt, d_val in top3:
        t_val = test_qtype.get(qt, 0)
        print(f"  {qt} & {d_val} & {t_val} \\\\")

    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \caption{Development vs.\ test set comparison.")
    print(r"  Notable distributional shifts are shown in \textbf{bold}.}")
    print(r"  \label{tab:dev_test_comparison}")
    print(r"  \end{table}")

    # Suggested analysis text
    print(f"\n  ── Suggested text: ──")
    shifts = {l: test_ans.get(l, 0) - dev_ans.get(l, 0)
              for l in ['Answerable', 'Partial', 'Unanswerable', 'Conversational']}
    biggest = max(shifts.items(), key=lambda x: abs(x[1]))
    if abs(biggest[1]) > 2:
        direction = "higher" if biggest[1] > 0 else "lower"
        print(f"  The test set exhibits a notably {direction} "
              f"{biggest[0].lower()} rate")
        print(f"  ({test_ans.get(biggest[0], 0)}\\% vs.\\ "
              f"{dev_ans.get(biggest[0], 0)}\\% in dev).")


# ============================================================
# Figure Generation
# ============================================================
def fig_dev_test_answerability(dev_ans, test_ans):
    labels = ['Answerable', 'Partial', 'Unanswerable', 'Conv.']
    keys = ['Answerable', 'Partial', 'Unanswerable', 'Conversational']
    dev_vals = [dev_ans.get(k, 0) for k in keys]
    test_vals = [test_ans.get(k, 0) for k in keys]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bars1 = ax.bar(x - width/2, dev_vals, width,
                   label='Dev', color=COLORS['blue'],
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_vals, width,
                   label='Test', color=COLORS['orange'],
                   edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, max(max(dev_vals), max(test_vals)) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.8,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'dev_test_answerability.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)


def fig_ref_answer_lengths(data):
    qtype_lengths = defaultdict(list)
    qtype_counts = Counter()
    for e in data:
        text = get_ref_answer_text(e)
        wc = len(text.split())
        if wc == 0:
            continue
        for qt in get_question_types(e):
            qtype_lengths[qt].append(wc)
            qtype_counts[qt] += 1

    top_qtypes = [qt for qt, _ in qtype_counts.most_common(6)]
    top_qtypes.sort(key=lambda k: np.median(qtype_lengths[k]), reverse=True)

    box_data = [qtype_lengths[qt] for qt in top_qtypes]

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    palette = [COLORS['blue'], COLORS['teal'], COLORS['orange'],
               COLORS['purple'], COLORS['green'], COLORS['gray']]

    bp = ax.boxplot(
        box_data, labels=top_qtypes,
        patch_artist=True, vert=True, widths=0.6,
        medianprops=dict(color=COLORS['red'], linewidth=1.5),
        flierprops=dict(marker='.', markersize=2, alpha=0.3),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp['boxes'], palette[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor('gray')
        patch.set_linewidth(0.5)

    ax.set_ylabel('Word Count')
    ax.set_xticklabels(top_qtypes, rotation=25, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    means = [np.mean(d) for d in box_data]
    ax.scatter(range(1, len(means)+1), means,
               marker='D', color=COLORS['red'], s=15, zorder=5,
               label='Mean')
    ax.legend(frameon=False, fontsize=7, loc='upper right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'ref_answer_lengths.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)


def fig_answerability_per_domain(data):
    domain_ans = defaultdict(Counter)
    for e in data:
        domain_ans[get_domain(e)][get_answerability(e)] += 1

    domains = ['Cloud', 'Govt', 'FiQA', 'ClapNQ']
    categories = ['Answerable', 'Partial', 'Unanswerable', 'Conversational']

    pct_data = {}
    for d in domains:
        total = sum(domain_ans[d].values())
        pct_data[d] = [100 * domain_ans[d].get(c, 0) / total
                       for c in categories] if total else [0]*4

    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    y = np.arange(len(domains))
    left = np.zeros(len(domains))

    for i, (cat, color) in enumerate(zip(categories,
            [ANS_COLORS[c] for c in categories])):
        vals = [pct_data[d][i] for d in domains]
        ax.barh(y, vals, left=left, label=cat,
                color=color, edgecolor='white',
                linewidth=0.5, height=0.55)
        for j, (v, l) in enumerate(zip(vals, left)):
            if v > 8:
                ax.text(l + v/2, j, f'{v:.0f}%',
                        ha='center', va='center', fontsize=7,
                        color='white', fontweight='bold')
        left += np.array(vals)

    ax.set_xlabel('Percentage (%)')
    ax.set_yticks(y)
    ax.set_yticklabels(domains)
    ax.set_xlim(0, 102)
    ax.legend(frameon=False, fontsize=7, ncol=2,
              bbox_to_anchor=(0.5, 1.18), loc='upper center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'answerability_per_domain.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)


def fig_domain_qtype_heatmap(data):
    domain_qtypes = defaultdict(Counter)
    all_qtypes = Counter()
    for e in data:
        domain = get_domain(e)
        for qt in get_question_types(e):
            domain_qtypes[domain][qt] += 1
            all_qtypes[qt] += 1

    domains = ['ClapNQ', 'FiQA', 'Govt', 'Cloud']
    top_qtypes = [qt for qt, _ in all_qtypes.most_common(7)]

    matrix = np.zeros((len(top_qtypes), len(domains)))
    for j, d in enumerate(domains):
        total = sum(domain_qtypes[d].values())
        for i, qt in enumerate(top_qtypes):
            matrix[i, j] = (100 * domain_qtypes[d].get(qt, 0) / total
                           if total else 0)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    ax.set_yticks(range(len(top_qtypes)))
    ax.set_yticklabels(top_qtypes)

    for i in range(len(top_qtypes)):
        for j in range(len(domains)):
            val = matrix[i, j]
            color = 'white' if val > 25 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('% of domain turns', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'domain_qtype_heatmap.pdf')
    fig.savefig(path)
    print(f"  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)


# ============================================================
# RUN EVERYTHING
# ============================================================
print("\n" + "="*60)
print("  MTRAG EDA Analysis")
print("="*60)

dev_data = load_jsonl(DEV_PATH)
test_data = load_jsonl(TEST_PATH)

# ═══════════════════════════════════════════
#  DEV SET
# ═══════════════════════════════════════════
dev_stats  = basic_stats(dev_data, 'Dev')
dev_ans    = answerability_distribution(dev_data, 'Dev')
dev_domain = domain_distribution(dev_data, 'Dev')
dev_qtype  = qtype_overall(dev_data, 'Dev')
dev_mt     = multi_turn_stats(dev_data, 'Dev')

print("\n" + "─"*60)
answerability_per_domain(dev_data, 'Dev')
qtype_per_domain(dev_data, 'Dev')
ref_stats = reference_answer_stats(dev_data, 'Dev')

# ═══════════════════════════════════════════
#  TEST SET
# ═══════════════════════════════════════════
test_stats  = basic_stats(test_data, 'Test')
test_ans    = answerability_distribution(test_data, 'Test')
test_domain = domain_distribution(test_data, 'Test')
test_qtype  = qtype_overall(test_data, 'Test')
test_mt     = multi_turn_stats(test_data, 'Test')

print("\n" + "─"*60)
answerability_per_domain(test_data, 'Test')
test_ref_stats = reference_answer_stats(test_data, 'Test')

# ═══════════════════════════════════════════
#  DEV vs TEST COMPARISON
# ═══════════════════════════════════════════
print_dev_test_latex(dev_stats, test_stats,
                     dev_ans, test_ans,
                     dev_domain, test_domain,
                     dev_qtype, test_qtype)

# ═══════════════════════════════════════════
#  FIGURES
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("  Generating Figures...")
print(f"{'='*60}")

fig_dev_test_answerability(dev_ans, test_ans)
fig_ref_answer_lengths(dev_data)
fig_answerability_per_domain(dev_data)
fig_domain_qtype_heatmap(dev_data)

# ═══════════════════════════════════════════
#  SAVE STATS JSON
# ═══════════════════════════════════════════
all_stats = {
    'dev': {
        'basic': dev_stats,
        'answerability': dev_ans,
        'domain': dev_domain,
        'qtype': dev_qtype,
        'multi_turn': dev_mt,
        'ref_answer': ref_stats,
    },
    'test': {
        'basic': test_stats,
        'answerability': test_ans,
        'domain': test_domain,
        'qtype': test_qtype,
        'multi_turn': test_mt,
        'ref_answer': test_ref_stats,
    },
}
json_path = os.path.join(OUTPUT_DIR, 'eda_stats.json')
with open(json_path, 'w') as f:
    json.dump(all_stats, f, indent=2, ensure_ascii=False)
print(f"\n  ✓ Saved all stats: {json_path}")

print(f"\n{'='*60}")
print(f"  DONE!")
print(f"  Tables: copy LaTeX from console output above")
print(f"  Figures: {OUTPUT_DIR}")
print(f"    ├── dev_test_answerability.pdf")
print(f"    ├── ref_answer_lengths.pdf")
print(f"    ├── answerability_per_domain.pdf")
print(f"    ├── domain_qtype_heatmap.pdf")
print(f"    └── eda_stats.json")
print(f"{'='*60}\n")