# 03_network_analysis_flex.py
# 依赖: pandas numpy networkx community-louvain matplotlib seaborn pyvis
# pip install pandas numpy networkx community-louvain matplotlib seaborn pyvis

import pandas as pd, numpy as np, networkx as nx, seaborn as sns
import matplotlib.pyplot as plt, json, re, glob, os
from pathlib import Path

# --------- 全局参数 ---------
EXPLICIT_ROOT = "outputs/explicit-flex"
IMPLICIT_ROOT = "outputs/implicit-flex"
EDGE_GLOBS = [
    f"{EXPLICIT_ROOT}/*/Edges_dim_E.csv",
    f"{EXPLICIT_ROOT}/*/Edges_dim_V.csv",
    f"{EXPLICIT_ROOT}/*/Exp_word_edges.csv",
    f"{IMPLICIT_ROOT}/*/Imp_dim_edges.csv",
    f"{IMPLICIT_ROOT}/*/Imp_word_edges.csv",
]

DIRECTED = True            # 用有向图 (gender -> dimension/word)
HTML_NODES_LIMIT = 600     # 节点太多则不输出 HTML
ALL_DIMS = ['APP','CPT','DOM','EMO','LED','MOR','PHY']

# 可视化
sns.set_theme(style="whitegrid")
NODE_COLORS = {'male':'#4285F4','female':'#EA4335','dimension':'#34A853','word':'#FBBC04','other':'#9E9E9E'}
EDGE_COLORS = {'pos':'#2196F3','neg':'#F44336','zero':'#9E9E9E'}

# --------- 工具函数 ---------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def detect_src_tgt_wgt(df: pd.DataFrame):
    # 支持 source/target/weight 或任意三列，自动识别 weight 列
    cl = [c.lower() for c in df.columns]
    # 首选 weight 命名
    for key in ['weight','score','value','w']:
        if key in cl:
            w = df.columns[cl.index(key)]
            others = [c for c in df.columns if c != w]
            if len(others) < 2:
                raise ValueError("未找到两列节点列")
            return others[0], others[1], w
    # 兜底：最后三列
    if df.shape[1] < 3: raise ValueError("边表至少需三列")
    s,t,w = df.columns[:3]
    return s,t,w

def classify_node(n: str):
    s = str(n).lower()
    if s in ('male','man','men','男','男性'): return 'male'
    if s in ('female','woman','women','女','女性'): return 'female'
    if str(n).upper() in ALL_DIMS: return 'dimension'
    return 'word' if re.fullmatch(r"[A-Za-z][A-Za-z\-\']{1,}", str(n)) else 'other'

def build_graph_from_csv(path: Path):
    df = pd.read_csv(path)
    s,t,w = detect_src_tgt_wgt(df)
    G = nx.DiGraph() if DIRECTED else nx.Graph()
    for _,r in df.iterrows():
        try:
            wt = float(r[w])
        except Exception:
            continue
        G.add_edge(str(r[s]), str(r[t]), weight=wt)
    return G, df.rename(columns={s:'source', t:'target', w:'weight'})

def node_stats(G: nx.DiGraph):
    # 强度（绝对值聚合，兼容正负权）
    in_strength  = {n: sum(abs(d['weight']) for _,_,d in G.in_edges(n, data=True)) for n in G}
    out_strength = {n: sum(abs(d['weight']) for _,_,d in G.out_edges(n, data=True)) for n in G}
    # PageRank（带权，失败则度中心性）
    try:
        pr = nx.pagerank(G, weight='weight', max_iter=300, tol=1e-8)
    except nx.PowerIterationFailedConvergence:
        pr = nx.degree_centrality(G)
    # 以 |w| 构无向图算特征向量中心性
    G_abs = nx.Graph((u,v,{'weight':abs(d['weight'])}) for u,v,d in G.edges(data=True) if d['weight']!=0)
    try:
        eig = nx.eigenvector_centrality_numpy(G_abs, weight='weight')
    except Exception:
        eig = {n:0.0 for n in G}
    btw = nx.betweenness_centrality(G, weight='weight', normalized=True)
    rows = []
    for n in G:
        rows.append([n, in_strength[n], out_strength[n], pr.get(n,0.0), eig.get(n,0.0), btw.get(n,0.0), classify_node(n)])
    return pd.DataFrame(rows, columns=['node','in_strength','out_strength','pagerank','eigenvector','betweenness','type'])

def louvain_partition(G: nx.DiGraph):
    try:
        import community as louvain
    except Exception:
        from community import community_louvain as louvain
    G_abs = nx.Graph((u,v,{'weight':abs(d['weight'])}) for u,v,d in G.edges(data=True) if d['weight']!=0)
    if G_abs.number_of_edges() == 0:
        return {n:i for i,n in enumerate(G)}
    return louvain.best_partition(G_abs, weight='weight')

def cosine_similarity(G: nx.DiGraph):
    nodes = list(G)
    idx = {n:i for i,n in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for u,v,d in G.edges(data=True):
        M[idx[u], idx[v]] = d['weight']
    # 余弦
    norm = np.linalg.norm(M, axis=1, keepdims=True); norm[norm==0] = 1
    cos = (M/norm) @ (M/norm).T
    return pd.DataFrame(cos, index=nodes, columns=nodes)

def save_heatmap(mat: pd.DataFrame, out_png: Path, title: str):
    plt.figure(figsize=(7.2, 6.4))
    sns.heatmap(mat, cmap='coolwarm', center=0, xticklabels=False, yticklabels=False, square=True)
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def interactive_html(G: nx.DiGraph, out_html: Path, title: str):
    if len(G) > HTML_NODES_LIMIT:
        print(f"  · 节点 {len(G)} > {HTML_NODES_LIMIT}，跳过 HTML")
        return
    try:
        from pyvis.network import Network
    except Exception:
        print("  · 未安装 pyvis，跳过 HTML"); return
    net = Network(height="780px", width="100%", directed=True, notebook=False)
    # 节点
    for n in G:
        tp = classify_node(n)
        net.add_node(n, label=n, color=NODE_COLORS.get(tp,'#888'), size=20, title=f"type={tp}")
    # 边（宽度按 |w| 归一）
    ws = [abs(d['weight']) for _,_,d in G.edges(data=True)]
    wmin, wmax = (min(ws), max(ws)) if ws else (0,1)
    for u,v,d in G.edges(data=True):
        w = float(d['weight'])
        color = EDGE_COLORS['pos'] if w>0 else EDGE_COLORS['neg'] if w<0 else EDGE_COLORS['zero']
        width = 1 + 8 * ((abs(w)-wmin)/(wmax-wmin) if wmax > wmin else 0.5)
        net.add_edge(u, v, color=color, width=width, label=f"{w:.3f}",
                     title=f"{u} → {v}, w={w:.3f}", font={'size':10,'color':'#333'})
    net.set_options("""var options = { "physics": { "stabilization": true }, "edges": {"smooth": false } }""")
    net.save_graph(str(out_html))
    print("  · HTML:", out_html)

def process_one_edges_csv(path: Path):
    tag = path.stem
    model = path.parent.name
    family = path.parent.parent.name     # explicit-flex / implicit-flex
    outdir = path.parent / f"{tag}_analysis"
    ensure_dir(outdir)
    print(f"\n==> 处理: [{family}] / [{model}] / {path.name}")

    G, edges = build_graph_from_csv(path)
    print(f"   nodes={len(G)}, edges={G.number_of_edges()}")

    # 节点指标
    stats = node_stats(G)
    stats.to_csv(outdir/"node_stats.csv", index=False)

    # 社群
    com = louvain_partition(G)
    json.dump(com, open(outdir/"community.json","w"), ensure_ascii=False, indent=2)

    # 标准边表
    edges.to_csv(outdir/"edges_std.csv", index=False, float_format="%.6f")

    # 相似度矩阵 + 热图
    try:
        sim = cosine_similarity(G)
        sim.to_csv(outdir/"similarity.csv", float_format="%.6f")
        save_heatmap(sim, outdir/"matrix.png", f"{tag} cosine similarity")
    except Exception as e:
        print("  · 相似度计算失败：", e)

    # 交互 HTML
    interactive_html(G, outdir/(tag+".html"), tag)

# --------- 跨模型维度对比（热图） ---------
def summarize_explicit_dim(EXPLICIT_ROOT: str):
    # E/V 两套
    sumdir = Path(EXPLICIT_ROOT) / "_summary"
    ensure_dir(sumdir)

    for kind, pattern in [('E', f"{EXPLICIT_ROOT}/*/Edges_dim_E.csv"),
                          ('V', f"{EXPLICIT_ROOT}/*/Edges_dim_V.csv")]:
        files = list(glob.glob(pattern))
        if not files:
            continue
        rows=[]
        for f in files:
            df = pd.read_csv(f)
            # 取 male → dim 边作为该维度值（female 是负号镜像）
            male_rows = df[df['source'].str.lower().eq('male')]
            model = Path(f).parent.name
            for _,r in male_rows.iterrows():
                dim = str(r['target']).upper()
                if dim in ALL_DIMS:
                    rows.append([model, dim, float(r['weight'])])
        if not rows:
            continue
        mat = pd.DataFrame(rows, columns=['model','dim','val'])
        pivot = mat.pivot_table(index='dim', columns='model', values='val', aggfunc='mean')\
                   .reindex(ALL_DIMS)
        pivot.to_csv(sumdir/f"Explicit_{kind}_dim_matrix.csv", float_format="%.6f")
        # 热图
        plt.figure(figsize=(max(6, 0.9*len(pivot.columns)+2), 4.8))
        sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt=".2f",
                    cbar_kws={'label': f'{kind} weight (male→dim)'})
        plt.title(f"Explicit {kind} (male→dim) by model")
        plt.tight_layout(); plt.savefig(sumdir/f"Explicit_{kind}_dim_heatmap.png", dpi=300); plt.close()
        print(f"[Summary] 写出 {kind}: {sumdir/f'Explicit_{kind}_dim_matrix.csv'} / heatmap")

def summarize_implicit_dim(IMPLICIT_ROOT: str):
    sumdir = Path(IMPLICIT_ROOT) / "_summary"
    ensure_dir(sumdir)
    files = glob.glob(f"{IMPLICIT_ROOT}/*/Imp_dim_edges.csv")
    if not files:
        return
    rows=[]
    for f in files:
        df = pd.read_csv(f)
        model = Path(f).parent.name
        for dim in ALL_DIMS:
            m = df[(df['source'].str.lower()=='male') & (df['target'].str.upper()==dim)]
            f_ = df[(df['source'].str.lower()=='female') & (df['target'].str.upper()==dim)]
            mv = float(m['weight'].mean()) if len(m) else 0.0
            fv = float(f_['weight'].mean()) if len(f_) else 0.0
            diff = mv - fv         # >0 更倾向男性
            rows.append([model, dim, mv, fv, diff])
    mat = pd.DataFrame(rows, columns=['model','dim','male','female','diff'])
    # 三张矩阵
    for col in ['male','female','diff']:
        pivot = mat.pivot_table(index='dim', columns='model', values=col, aggfunc='mean')\
                   .reindex(ALL_DIMS)
        pivot.to_csv(sumdir/f"Implicit_dim_{col}_matrix.csv", float_format="%.6f")
        plt.figure(figsize=(max(6, 0.9*len(pivot.columns)+2), 4.8))
        sns.heatmap(pivot, cmap='RdBu_r', center=0 if col!='male' else None, annot=True, fmt=".2f",
                    cbar_kws={'label': f'Implicit {col}'})
        plt.title(f"Implicit {col} by model")
        plt.tight_layout(); plt.savefig(sumdir/f"Implicit_dim_{col}_heatmap.png", dpi=300); plt.close()
    print(f"[Summary] 写出隐性维度矩阵/热图 → {sumdir}")

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    all_edges = []
    for patt in EDGE_GLOBS:
        all_edges += glob.glob(patt)

    if not all_edges:
        print("未发现边表，请先运行 01/02 脚本生成 outputs/*-flex/**/edges.")
    else:
        for p in all_edges:
            process_one_edges_csv(Path(p))

    # 跨模型汇总
    summarize_explicit_dim(EXPLICIT_ROOT)
    summarize_implicit_dim(IMPLICIT_ROOT)

    print("\n[Network analysis flex] ✅ 全部完成")
