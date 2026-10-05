"""Ported from 4.ipynb cell 84. Input paths are taken from environment variables
(MTRAG_DEV_REFERENCE, MTRAG_TEST_REFERENCE, MTRAG_RESULTS_DIR, MTRAG_FIG_DIR); defaults follow data/README.md.
"""
"""
Task A Supplementary Figures & Tables
======================================
Generates:
  - figures/turn_index_vs_recall.pdf
  - figures/standalone_vs_non.pdf
  - LaTeX tables: standalone breakdown + cost
"""

import os, json, csv
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── CONFIG ──────────────────────────────────────────────────
BASE_PATH  = os.environ.get('MTRAG_RESULTS_DIR', 'outputs/dev_runs')
OUTPUT_DIR = os.environ.get('MTRAG_FIG_DIR', 'results/figures/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':          10,
    'axes.labelsize':     11,
    'axes.titlesize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})
COLORS = {
    'blue':   '#376EC4', 'teal':   '#008C7D',
    'orange': '#D7820F', 'purple': '#783AA0',
    'red':    '#BE3737', 'gray':   '#6E6E6E',
    'green':  '#2E8B57',
}
DOMAIN_COLORS = {
    'ClapNQ': COLORS['blue'],  'FiQA':  COLORS['orange'],
    'Govt':   COLORS['teal'],  'Cloud': COLORS['purple'],
}
CORPORA     = ['clapnq', 'fiqa', 'govt', 'cloud']
DOMAIN_LABEL = {'clapnq':'ClapNQ','fiqa':'FiQA','govt':'Govt','cloud':'Cloud'}
CONV_PATH   = f'{BASE_PATH}/retrieval/conversations.json'

# ── System paths ──────────────────────────────────────────────
BASELINE_PATHS = {
    'clapnq': f'{BASE_PATH}/elser_v1_no_rewrite_predictions/clapnq_elser_v1_no_rewrite_results.jsonl',
    'fiqa':   f'{BASE_PATH}/elser_v1_no_rewrite_predictions/fiqa_elser_v1_no_rewrite_results.jsonl',
    'govt':   f'{BASE_PATH}/elser_v1_no_rewrite_predictions/govt_elser_v1_no_rewrite_results.jsonl',
    'cloud':  f'{BASE_PATH}/elser_v1_no_rewrite_predictions/cloud_elser_v1_no_rewrite_results.jsonl',
}
FINAL_PATHS = {
    'clapnq': f'{BASE_PATH}/rrf_results/elserv1_clapnq_rrf_results.jsonl',
    'fiqa':   f'{BASE_PATH}/rrf_results/elserv1_fiqa_rrf_results.jsonl',
    'govt':   f'{BASE_PATH}/rrf_results/elserv1_govt_rrf_results.jsonl',
    'cloud':  f'{BASE_PATH}/rrf_results/elserv1_cloud_rrf_results.jsonl',
}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def load_gold_qrels(corpus):
    path   = f'{BASE_PATH}/retrieval/{corpus}/dev.tsv'
    qrels  = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) < 3: continue
            qid, doc_id, score = row[0].strip(), row[1].strip(), int(float(row[2]))
            if score > 0:
                qrels[qid].add(doc_id)
    return qrels


def load_predictions(path, k=100):
    if not path or not os.path.exists(path):
        print(f'  ⚠  File not found: {path}')
        return {}
    preds = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            qid  = str(item.get('task_id'))
            ctxs = item.get('contexts', []) or []
            docs = [str(c.get('document_id') or c.get('id') or '').strip()
                    for c in ctxs[:k]]
            preds[qid] = docs
    print(f'  ✓ Loaded {len(preds)} predictions ← {os.path.basename(path)}')
    return preds


def recall_at_k(preds, qrels, qid, k):
    gold = qrels.get(qid, set())
    if not gold: return None
    got  = set((preds.get(qid) or [])[:k])
    return len(got & gold) / len(gold)


def split_qid(qid):
    if '<::>' not in qid: return None, None
    base, t = qid.split('<::>', 1)
    try:    return base, int(t)
    except: return base, None


def first_list_val(x, default='Unknown'):
    if x is None:           return default
    if isinstance(x, list): return x[0] if x else default
    return x


def infer_corpus(name):
    name = (name or '').lower()
    for c in CORPORA:
        if c in name: return c
    return None


# ══════════════════════════════════════════════════════════════
# META MAP: qid → {turn_no, standalone, answerability, domain}
# Fixed: score() receives list-of-tuples, not list indices
# ══════════════════════════════════════════════════════════════
def build_meta_map(qrels, conversations_json, corpus):
    # ── group qrels by base ──
    base2turns = defaultdict(dict)   # base → {turn_no: gold_set}
    for qid, gold in qrels.items():
        base, t = split_qid(qid)
        if base is None or t is None: continue
        base2turns[base][t] = gold

    # ── filter conversations for this corpus ──
    convs = [c for c in conversations_json
             if infer_corpus(
                 (((c.get('retriever') or {}).get('collection') or {})
                  .get('name')) or '') == corpus]
    print(f'  [{corpus}] conversations: {len(convs)}  '
          f'qrel bases: {len(base2turns)}')

    # ── parse each conv → list of (turn_no:int, enrich:dict, ctx_docids:set) ──
    def parse_conv(conv):
        msgs   = conv.get('messages', []) or []
        result = []
        user_no = 0
        for i, msg in enumerate(msgs):
            if (msg.get('speaker') or '').lower() != 'user': continue
            user_no += 1
            enrich     = msg.get('enrichments') or {}
            ctx_docids = set()
            for j in range(i+1, min(i+6, len(msgs))):
                if (msgs[j].get('speaker') or '').lower() == 'agent':
                    for c in (msgs[j].get('contexts') or []):
                        did = c.get('document_id') or c.get('id')
                        if did: ctx_docids.add(str(did).strip())
                    break
            result.append((user_no, enrich, ctx_docids))
        return result   # list of (int, dict, set)

    conv_turns_list = [parse_conv(c) for c in convs]   # list[list[tuple]]

    # ── docid → set of conv indices ──
    doc2ci = defaultdict(set)
    for ci, turns in enumerate(conv_turns_list):
        for _, _, ctx in turns:          # ← unpack tuple correctly
            for d in ctx:
                doc2ci[d].add(ci)

    # ── score(base, conv_index): avg over gold turns of best ctx overlap ──
    def score_base_conv(turn2gold, conv_idx):
        turns = conv_turns_list[conv_idx]   # list of (int, dict, set)
        ctx_sets = [t[2] for t in turns]    # just the sets
        total, count = 0.0, 0
        for gold in turn2gold.values():
            if not gold: continue
            best = max((len(gold & ctx) / len(gold) for ctx in ctx_sets if ctx),
                       default=0.0)
            total += best; count += 1
        return total / count if count else 0.0

    # ── greedy assignment: base → best conv index ──
    # collect candidate conv indices for each base
    def get_candidates(base):
        cands = set()
        for gold in base2turns[base].values():
            for d in gold:
                cands |= doc2ci.get(d, set())
        return cands if cands else set(range(len(convs)))

    # sort bases by their best candidate score (descending)
    base_scores = {}
    for base in base2turns:
        cands = get_candidates(base)
        best  = max((score_base_conv(base2turns[base], ci) for ci in cands),
                    default=0.0)
        base_scores[base] = best

    assignments = {}
    used_convs  = set()
    for base in sorted(base2turns.keys(),
                       key=lambda b: base_scores[b], reverse=True):
        cands  = get_candidates(base)
        scored = sorted(
            [(score_base_conv(base2turns[base], ci), ci) for ci in cands],
            reverse=True
        )
        for s, ci in scored:
            if ci not in used_convs and s > 0:
                assignments[base] = ci
                used_convs.add(ci)
                break

    # ── build meta_map ──
    meta_map = {}
    for base, ci in assignments.items():
        turnno2data = {t[0]: t for t in conv_turns_list[ci]}
        for tno in base2turns[base]:
            qid = f'{base}<::>{tno}'
            if tno not in turnno2data: continue
            _, enrich, _ = turnno2data[tno]

            mt   = first_list_val(enrich.get('Multi-Turn') or
                                  enrich.get('Multi Turn') or
                                  enrich.get('MultiTurn'))
            mt_u = str(mt).strip().upper()
            if   mt_u in ['N/A','NA','NONE','SINGLE-TURN','SINGLE TURN','0']:
                standalone = 'Standalone'
            elif mt_u in ['','UNKNOWN']:
                standalone = 'Unknown'
            else:
                standalone = 'Non-standalone'

            meta_map[qid] = {
                'turn_no':       tno,
                'standalone':    standalone,
                'answerability': str(first_list_val(enrich.get('Answerability'))),
                'domain':        DOMAIN_LABEL.get(corpus, corpus),
            }

    covered = len(meta_map)
    total   = sum(len(v) for v in base2turns.values())
    print(f'  [{corpus}] meta_map coverage: {covered}/{total} = '
          f'{covered/max(1,total):.1%}  '
          f'(assigned {len(assignments)}/{len(base2turns)} bases)')
    return meta_map


# ══════════════════════════════════════════════════════════════
# COLLECT ALL DATA (one pass, reused by both figures)
# ══════════════════════════════════════════════════════════════
def collect_all(conversations_json):
    """
    Returns per_query_data: list of dicts with
      {qid, corpus, domain, turn_no, tbin, standalone,
       answerability, r5_base, r5_final}
    """
    print('\n  Loading conversations.json …')
    records = []

    for corpus in CORPORA:
        print(f'\n  ── {corpus.upper()} ──')
        qrels     = load_gold_qrels(corpus)
        meta_map  = build_meta_map(qrels, conversations_json, corpus)
        preds_b   = load_predictions(BASELINE_PATHS.get(corpus))
        preds_f   = load_predictions(FINAL_PATHS.get(corpus))

        for qid in qrels:
            if qid not in meta_map: continue
            meta = meta_map[qid]

            r5_b = recall_at_k(preds_b, qrels, qid, k=5)
            r5_f = recall_at_k(preds_f, qrels, qid, k=5)
            if r5_b is None and r5_f is None: continue

            tno  = meta['turn_no']
            records.append({
                'qid':           qid,
                'corpus':        corpus,
                'domain':        meta['domain'],
                'turn_no':       tno,
                'tbin':          min(tno, 6),          # 1..5, 6+
                'standalone':    meta['standalone'],
                'answerability': meta['answerability'],
                'r5_base':       r5_b if r5_b is not None else np.nan,
                'r5_final':      r5_f if r5_f is not None else np.nan,
            })

    df = pd.DataFrame(records)
    print(f'\n  Total records collected: {len(df)}')
    return df


# ══════════════════════════════════════════════════════════════
# FIGURE 1: Per-Turn R@5 Degradation
# ══════════════════════════════════════════════════════════════
def fig_turn_degradation(df):
    print('\n  Generating: turn_index_vs_recall.pdf')

    domains_ordered = ['Overall', 'ClapNQ', 'FiQA', 'Govt', 'Cloud']
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7), sharey=True)

    sys_cfg = [
        ('r5_base',  'ELSER (no rewrite)', COLORS['gray'],  '--', 1.5, 4),
        ('r5_final', 'Final system',        COLORS['blue'],  '-',  2.0, 5),
    ]

    all_stats = []

    for ax, dom in zip(axes, domains_ordered):
        sub = df if dom == 'Overall' else df[df['domain'] == dom]

        for col, label, color, ls, lw, ms in sys_cfg:
            grp = sub.dropna(subset=[col]).groupby('tbin')[col]
            bins  = sorted(grp.groups.keys())
            if not bins: continue

            means = [grp.get_group(b).mean() for b in bins]
            sems  = [grp.get_group(b).sem()  for b in bins]
            sems  = [s if not np.isnan(s) else 0 for s in sems]
            xlabs = [str(b) if b < 6 else '6+' for b in bins]
            x     = np.arange(len(bins))

            ax.plot(x, means, color=color, linestyle=ls,
                    linewidth=lw, marker='o', markersize=ms, label=label)
            ax.fill_between(x,
                            [m-e for m,e in zip(means,sems)],
                            [m+e for m,e in zip(means,sems)],
                            alpha=0.13, color=color)
            ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=8)

            # stats
            if dom == 'Overall' and bins:
                t1   = means[0]
                tlast = means[-1]
                drop = 100*(t1-tlast)/t1 if t1 > 0 else 0
                all_stats.append(
                    f'  {label}: Turn1={t1:.3f}  Turn6+={tlast:.3f}'
                    f'  Drop={drop:.0f}%'
                )

        ax.set_title(dom, fontsize=9,
                     color=DOMAIN_COLORS.get(dom, 'black'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Turn index', fontsize=8)
        if ax is axes[0]: ax.set_ylabel('R@5')
        ax.set_ylim(0.25, 1.02)
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.10),
               ncol=2, frameon=False, fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'turn_index_vs_recall.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'  ✓ Saved: {path}')
    plt.show(); plt.close(fig)

    print('\n  Per-turn stats (Overall):')
    for s in all_stats: print(s)


# ══════════════════════════════════════════════════════════════
# FIGURE 2 + TABLE: Standalone vs Non-Standalone
# ══════════════════════════════════════════════════════════════
def fig_standalone_vs_non(df):
    print('\n  Generating: standalone_vs_non.pdf')

    domains_ordered = ['Overall', 'ClapNQ', 'FiQA', 'Govt', 'Cloud']
    sa_labels       = ['Standalone', 'Non-standalone']
    sys_cols        = [('r5_base', 'ELSER'), ('r5_final', 'Final')]

    # ── Compute means ──
    def get_mean(sub, col):
        v = sub[col].dropna()
        return v.mean() if len(v) > 0 else 0.0

    rows_table = []
    bar_data   = []   # for figure

    for dom in domains_ordered:
        sub_dom = df if dom == 'Overall' else df[df['domain'] == dom]
        row = {'Domain': dom}
        for col, sys_name in sys_cols:
            for sa in sa_labels:
                sub_sa = sub_dom[sub_dom['standalone'] == sa]
                v = get_mean(sub_sa, col)
                key = f'{sys_name}_{sa}'
                row[key] = v
        # Gain on Non-standalone
        row['Gain_NonSA'] = row['Final_Non-standalone'] - row['ELSER_Non-standalone']
        row['Gain_SA']    = row['Final_Standalone']     - row['ELSER_Standalone']
        rows_table.append(row)
        bar_data.append(row)

    df_table = pd.DataFrame(rows_table)

    # ── Print LaTeX table ──
    print('\n  ── LaTeX Table: Standalone vs Non-Standalone ──\n')
    print(r'\begin{table}[t]')
    print(r'\centering\small')
    print(r'\begin{tabular}{lcccc|cc}')
    print(r'\toprule')
    print(r'& \multicolumn{2}{c}{\textbf{ELSER (no rewrite)}}')
    print(r'& \multicolumn{2}{c|}{\textbf{Final system}}')
    print(r'& \multicolumn{2}{c}{\textbf{Gain}} \\')
    print(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}')
    print(r'\textbf{Domain} & SA & Non-SA & SA & Non-SA')
    print(r'& $\Delta$SA & $\Delta$Non-SA \\')
    print(r'\midrule')

    for r in rows_table:
        dom   = r['Domain']
        sep   = r'\midrule' + '\n' if dom == 'Overall' else ''
        b_sa  = r['ELSER_Standalone']
        b_ns  = r['ELSER_Non-standalone']
        f_sa  = r['Final_Standalone']
        f_ns  = r['Final_Non-standalone']
        d_sa  = r['Gain_SA']
        d_ns  = r['Gain_NonSA']
        print(f'{sep}\\textbf{{{dom}}} '
              f'& {b_sa:.3f} & {b_ns:.3f} '
              f'& {f_sa:.3f} & {f_ns:.3f} '
              f'& $+${d_sa:.3f} & $+${d_ns:.3f} \\\\')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\caption{R@5 for standalone (SA) vs.\ non-standalone (Non-SA) queries,')
    print(r'ELSER baseline vs.\ final optimized system. Rewriting disproportionately')
    print(r'benefits non-standalone queries, which constitute 87\% of dev-set turns.}')
    print(r'\label{tab:standalone_breakdown}')
    print(r'\end{table}')

    # ── Figure: grouped bars ──
    n_groups = len(domains_ordered)
    n_bars   = 4    # base_SA, base_NS, final_SA, final_NS
    group_w  = 0.72
    bar_w    = group_w / n_bars
    x        = np.arange(n_groups)

    bar_cfgs = [
        ('ELSER_Standalone',     COLORS['gray'],  'ELSER SA'),
        ('ELSER_Non-standalone', '#BBBBBB',        'ELSER Non-SA'),
        ('Final_Standalone',     COLORS['blue'],  'Final SA'),
        ('Final_Non-standalone', '#7AAFF0',        'Final Non-SA'),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    all_vals = []

    for gi, dom in enumerate(domains_ordered):
        r = next(d for d in bar_data if d['Domain'] == dom)
        for bi, (key, color, _) in enumerate(bar_cfgs):
            v    = r[key]
            xpos = x[gi] - group_w/2 + bar_w*(bi+0.5)
            ax.bar(xpos, v, bar_w*0.88,
                   color=color, edgecolor='white', linewidth=0.3)
            all_vals.append(v)

    ax.set_xticks(x)
    ax.set_xticklabels(domains_ordered, fontsize=9)
    ax.set_ylabel('R@5')
    ax.set_ylim(0, max(all_vals)*1.20 if all_vals else 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate Non-SA gain for Overall
    r_ov = next(d for d in bar_data if d['Domain'] == 'Overall')
    gain_ns = r_ov['Gain_NonSA']
    ns_final_val = r_ov['Final_Non-standalone']
    ax.annotate(
        f'+{gain_ns:.3f}',
        xy=(x[0] - group_w/2 + bar_w*1.5, ns_final_val),
        xytext=(x[0] + 0.5, ns_final_val + 0.06),
        fontsize=8, color=COLORS['red'],
        arrowprops=dict(arrowstyle='->', lw=0.8, color=COLORS['red'])
    )

    legend_handles = [Patch(facecolor=c, label=l) for _, c, l in bar_cfgs]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8,
              ncol=2, bbox_to_anchor=(0.5, 1.22), loc='upper center')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'standalone_vs_non.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'\n  ✓ Saved: {path}')
    plt.show(); plt.close(fig)


# ══════════════════════════════════════════════════════════════
# COST TABLE (LaTeX only)
# ══════════════════════════════════════════════════════════════
def print_cost_table():
    print('\n  ── LaTeX Table: Cost Breakdown (Task A) ──\n')

    # (component, model, calls_per_query, cost_per_query_usd)
    components = [
        ('ELSER v1 retrieval',    'Elasticsearch',   '1',  0.0000),
        ('Query rewriting (×5)',  'DeepSeek-V3',     '5',  0.0008),
        ('Cohere reranking',      'Cohere Rerank v4','1',  0.0020),
    ]
    total_cpq = sum(r[3] for r in components)

    print(r'\begin{table}[t]')
    print(r'\centering\small')
    print(r'\begin{tabular}{llccc}')
    print(r'\toprule')
    print(r'\textbf{Component} & \textbf{Model}')
    print(r'& \textbf{Calls/Q} & \textbf{Cost/Q} & \textbf{Cost/777Q} \\')
    print(r'\midrule')

    for comp, model, calls, cpq in components:
        c777   = cpq * 777
        cpq_s  = f'\\${cpq:.4f}' if cpq > 0 else '---'
        c777_s = f'\\${c777:.2f}'  if cpq > 0 else '---'
        print(f'{comp} & {model} & {calls} & {cpq_s} & {c777_s} \\\\')

    print(r'\midrule')
    print(f'\\textbf{{Total}} & & & '
          f'\\${total_cpq:.4f} & \\${total_cpq*777:.2f} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\caption{Task~A per-query cost breakdown (dev set, 777~queries).')
    print(r'Elasticsearch retrieval incurs no per-query API cost.')
    print(r'DeepSeek-V3 rewriting at \$0.14/M input tokens;')
    print(r'Cohere Rerank~v4 at \$0.002/query.')
    print(r'Total estimated cost for one full dev-set run: \$\approx\$2.2.}')
    print(r'\label{tab:cost_breakdown_task_a}')
    print(r'\end{table}')

    print(f"""
  ── Suggested paragraph ──

  The full Task~A pipeline costs approximately \\$0.003 per
  query (\\$2.2 for the 777-query dev set), dominated by
  Cohere reranking (\\$0.002/Q) and five DeepSeek-V3
  rewriting calls (\\$0.001/Q combined).
  ELSER retrieval itself incurs no per-query API cost beyond
  infrastructure.
  Ablating reranking saves \\$0.002/Q but costs 8.7\\% in R@5
  (Table~\\ref{{tab:reranking_replacement}}); ablating rewriting
  saves \\$0.001/Q but costs 12.5\\% in R@5
  (Table~\\ref{{tab:rewriting_strategies}}).
  The full pipeline achieves R@5\\,=\\,0.607 vs.\\ 0.483
  for the unaugmented baseline, a 26\\% relative improvement
  at under half a cent per query.
    """)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('='*60)
    print('  Task A Supplementary Figures & Tables')
    print('='*60)

    conversations_json = json.load(open(CONV_PATH, encoding='utf-8'))
    df = collect_all(conversations_json)

    fig_turn_degradation(df)
    fig_standalone_vs_non(df)
    print_cost_table()

    print(f'\n{"="*60}')
    print(f'  DONE')
    print(f'  Figures  → {OUTPUT_DIR}turn_index_vs_recall.pdf')
    print(f'           → {OUTPUT_DIR}standalone_vs_non.pdf')
    print(f'  Tables   → copy LaTeX from console output')
    print(f'{"="*60}')
