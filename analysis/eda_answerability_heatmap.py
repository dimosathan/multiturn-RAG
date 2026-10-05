"""Ported from 4.ipynb cell 83. Input paths are taken from environment variables
(MTRAG_DEV_REFERENCE, MTRAG_TEST_REFERENCE, MTRAG_FIG_DIR); defaults follow data/README.md.
"""
"""
Figure 5: Answerability × Domain Heatmap — Dev vs Test
=======================================================
Two-panel heatmap (dev | test).
Rows: domains (ClapNQ, FiQA, Govt, Cloud)
Cols: Answerable / Partial / Unanswerable / Conv.

Reads the same JSONL files used in the main EDA script.
Output: eda_outputs/answerability_domain_heatmap.pdf
"""

import json
import os
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────
DEV_PATH   = os.environ.get('MTRAG_DEV_REFERENCE', 'data/generation/reference.jsonl')
TEST_PATH  = os.environ.get('MTRAG_TEST_REFERENCE', 'data/test/reference_final.jsonl')
OUTPUT_DIR = os.environ.get('MTRAG_FIG_DIR', 'results/figures/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':        10,
    'axes.labelsize':   11,
    'axes.titlesize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
})

# ── HELPERS ──────────────────────────────────────────────────
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  Loaded {len(data)} turns  ← {path}")
    return data


def normalize_domain(s):
    s = s.lower()
    if 'clapnq' in s:                    return 'ClapNQ'
    if 'fiqa'   in s:                    return 'FiQA'
    if 'govt'   in s or 'government' in s: return 'Govt'
    if 'cloud'  in s or 'ibmcloud'   in s: return 'Cloud'
    return 'Unknown'


def get_answerability(entry):
    ans = entry.get('Answerability', entry.get('answerability', []))
    if not ans:
        return 'Unknown'
    val = ans[0].upper().strip()
    return {
        'ANSWERABLE':          'Answerable',
        'UNANSWERABLE':        'Unanswerable',
        'PARTIALLY_ANSWERABLE':'Partial',
        'PARTIAL':             'Partial',
        'CONVERSATIONAL':      'Conv.',
    }.get(val, val)


def build_matrix(data, domains, categories):
    """
    Returns a (len(domains) × len(categories)) matrix of row-normalised %.
    Also returns raw counts per domain for the annotation tooltip.
    """
    domain_ans = defaultdict(Counter)
    for e in data:
        d = normalize_domain(e.get('Collection', ''))
        a = get_answerability(e)
        domain_ans[d][a] += 1

    matrix = np.zeros((len(domains), len(categories)))
    totals = {}
    for i, dom in enumerate(domains):
        total = sum(domain_ans[dom].values())
        totals[dom] = total
        for j, cat in enumerate(categories):
            matrix[i, j] = (
                100 * domain_ans[dom].get(cat, 0) / total if total else 0
            )
    return matrix, totals, domain_ans


# ── FIGURE ───────────────────────────────────────────────────
def fig_answerability_domain_heatmap():
    print("\n  Generating: answerability_domain_heatmap.pdf")

    dev_data  = load_jsonl(DEV_PATH)
    test_data = load_jsonl(TEST_PATH)

    DOMAINS    = ['ClapNQ', 'FiQA', 'Govt', 'Cloud']
    CATEGORIES = ['Answerable', 'Partial', 'Unanswerable', 'Conv.']

    dev_matrix,  dev_totals,  dev_raw  = build_matrix(dev_data,  DOMAINS, CATEGORIES)
    test_matrix, test_totals, test_raw = build_matrix(test_data, DOMAINS, CATEGORIES)

    # ── Print paper stats ──────────────────────────────────
    print("\n  Dev  answerability per domain (%):")
    for i, dom in enumerate(DOMAINS):
        vals = '  '.join(f"{CATEGORIES[j]}={dev_matrix[i,j]:.0f}%"
                         for j in range(len(CATEGORIES)))
        print(f"    {dom:8s}: {vals}")

    print("\n  Test answerability per domain (%):")
    for i, dom in enumerate(DOMAINS):
        vals = '  '.join(f"{CATEGORIES[j]}={test_matrix[i,j]:.0f}%"
                         for j in range(len(CATEGORIES)))
        print(f"    {dom:8s}: {vals}")

    # ── Identify biggest shift for caption ────────────────
    diff_matrix = test_matrix - dev_matrix
    max_shift_idx = np.unravel_index(np.abs(diff_matrix).argmax(),
                                     diff_matrix.shape)
    max_dom = DOMAINS[max_shift_idx[0]]
    max_cat = CATEGORIES[max_shift_idx[1]]
    dev_val  = dev_matrix[max_shift_idx]
    test_val = test_matrix[max_shift_idx]
    print(f"\n  Largest shift: {max_dom} / {max_cat}: "
          f"{dev_val:.0f}% (dev) → {test_val:.0f}% (test)")

    # ── Two-panel heatmap ──────────────────────────────────
    # Separate colormaps:
    #   Answerable  → Blues   (high = good)
    #   Unanswerable/Partial → Oranges (high = hard)
    #   Conv.       → Greys
    #
    # We use a single perceptually-uniform colormap per panel
    # with annotated cell values for readability.

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.2, 2.6),
        gridspec_kw={'wspace': 0.35}
    )

    panel_data = [
        (dev_matrix,  dev_totals,  'Dev set'),
        (test_matrix, test_totals, 'Test set'),
    ]

    # Shared colour scale: 0–100% so both panels are comparable
    vmin, vmax = 0, 100

    # Custom diverging-style: white at ~50%, blue for high answerable,
    # orange for high unanswerable.  We just use YlOrRd reversed so
    # low% = pale, high% = saturated — reader reads the numbers anyway.
    CMAP = 'YlOrRd'

    for ax, (matrix, totals, title) in zip(axes, panel_data):
        im = ax.imshow(matrix, cmap=CMAP, aspect='auto',
                       vmin=vmin, vmax=vmax)

        # Cell annotations
        for i in range(len(DOMAINS)):
            for j in range(len(CATEGORIES)):
                val   = matrix[i, j]
                # White text on dark cells, black on light
                color = 'white' if val > 55 else 'black'
                # Bold the Unanswerable column to highlight the key shift
                weight = 'bold' if CATEGORIES[j] == 'Unanswerable' else 'normal'
                ax.text(j, i, f'{val:.0f}%',
                        ha='center', va='center',
                        fontsize=8.5, color=color, fontweight=weight)

        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_xticklabels(CATEGORIES, rotation=25, ha='right')
        ax.set_yticks(range(len(DOMAINS)))
        ax.set_yticklabels(DOMAINS)
        ax.set_title(title, fontsize=10, pad=6)

        # Highlight Unanswerable column border
        unans_col = CATEGORIES.index('Unanswerable')
        for spine_pos in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine_pos].set_visible(False)

        # Draw a thin rectangle around the Unanswerable column
        from matplotlib.patches import FancyBboxPatch
        rect = plt.Rectangle(
            (unans_col - 0.5, -0.5),
            1, len(DOMAINS),
            linewidth=1.4, edgecolor='#BE3737',
            facecolor='none', zorder=5
        )
        ax.add_patch(rect)

        # Row totals as right-margin annotation
        for i, dom in enumerate(DOMAINS):
            ax.text(len(CATEGORIES) - 0.35, i,
                    f'n={totals[dom]}',
                    va='center', ha='left',
                    fontsize=6.5, color='#6E6E6E')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.65])
    sm = plt.cm.ScalarMappable(cmap=CMAP,
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('% of domain turns', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Difference annotation panel (optional) ──────────────
    # Uncomment the block below if you want a third "Δ" panel
    #
    # ax_diff = fig.add_subplot(...) — left as exercise

    plt.suptitle(
        'Answerability distribution per domain — Dev vs. Test',
        fontsize=10, y=1.04
    )

    path = os.path.join(OUTPUT_DIR, 'answerability_domain_heatmap.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"\n  ✓ Saved: {path}")
    plt.show()
    plt.close(fig)

    # ── BONUS: Δ (shift) heatmap ─────────────────────────────
    fig2, ax = plt.subplots(figsize=(3.8, 2.4))
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-45, vmax=45)

    for i in range(len(DOMAINS)):
        for j in range(len(CATEGORIES)):
            val   = diff_matrix[i, j]
            color = 'white' if abs(val) > 22 else 'black'
            sign  = '+' if val >= 0 else ''
            weight = 'bold' if CATEGORIES[j] == 'Unanswerable' else 'normal'
            ax.text(j, i, f'{sign}{val:.0f}pp',
                    ha='center', va='center',
                    fontsize=8, color=color, fontweight=weight)

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=25, ha='right')
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS)
    ax.set_title('Shift: Test − Dev (pp)', fontsize=10)

    for spine in ax.spines.values():
        ax.spines[spine.get_position_str() if hasattr(spine,'get_position_str') else 'top'].set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar2 = fig2.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar2.set_label('Δ pp (test − dev)', fontsize=8)
    cbar2.ax.tick_params(labelsize=7)

    path2 = os.path.join(OUTPUT_DIR, 'answerability_domain_shift.pdf')
    fig2.savefig(path2, bbox_inches='tight')
    print(f"  ✓ Saved (bonus shift panel): {path2}")
    plt.show()
    plt.close(fig2)

    # ── Suggested LaTeX caption ──────────────────────────────
    print(f"""
  ── Suggested caption ──
  \\caption{{Answerability distribution per domain (\\%
  of domain turns), development set (left) vs.\\ test set
  (right). Red borders highlight the Unanswerable column.
  Cloud exhibits the most severe shift: unanswerable turns
  rise from {dev_matrix[DOMAINS.index('Cloud'), CATEGORIES.index('Unanswerable')]:.0f}\\%
  (dev) to {test_matrix[DOMAINS.index('Cloud'), CATEGORIES.index('Unanswerable')]:.0f}\\%
  (test), the highest unanswerable rate across all domains.}}
  \\label{{fig:answerability_domain_heatmap}}
    """)


# ── ENTRY POINT ──────────────────────────────────────────────
fig_answerability_domain_heatmap()
