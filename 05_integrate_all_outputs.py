# 05_integrate_all_outputs.py  — 修正版(2)
# ---------------------------------------------------------------
# 将 01/02/03 生成的所有输出，汇总到单一 Excel 工作簿
# 写出：outputs/ALL/All_Bias_Integrated.xlsx
# 依赖：pip install pandas numpy openpyxl
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ========== 配置 ==========
EXPLICIT_ROOT = Path("outputs/explicit-flex")
IMPLICIT_ROOT = Path("outputs/implicit-flex")
OUT_DIR       = Path("outputs/ALL")
OUT_XLSX      = OUT_DIR / "All_Bias_Integrated.xlsx"

ALL_DIMS = ['APP','CPT','DOM','EMO','LED','MOR','PHY']
TOPK_WORD_EDGES = 1000

# ========== 工具 ==========
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_csv_safe(path: Path, **kw):
    try:
        if path.exists():
            return pd.read_csv(path, **kw)
    except Exception as e:
        print(f"[WARN] 读取失败: {path} - {e}")
    return None

def read_json_safe(path: Path):
    try:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] 读取失败: {path} - {e}")
    return None

def concat_nonempty(dfs):
    dfs = [d for d in dfs if d is not None and len(d)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def reorder_dims(df, dim_col='dim'):
    """仅排序，不改索引，不去重；未知维度放末尾。"""
    if df is None or len(df)==0 or dim_col not in df.columns:
        return df
    dfx = df.copy()
    dfx[dim_col] = dfx[dim_col].astype(str).str.upper()
    dfx[dim_col] = pd.Categorical(dfx[dim_col], categories=ALL_DIMS, ordered=True)
    return dfx.sort_values(dim_col).reset_index(drop=True)

def topk_by_abs(df, col='weight', k=TOPK_WORD_EDGES):
    if df is None or len(df)==0 or col not in df.columns:
        return df
    return df.iloc[df[col].abs().sort_values(ascending=False).index].head(k).reset_index(drop=True)

def safe_to_excel(xw, df, sheet_name):
    try:
        if df is not None and len(df):
            df.to_excel(xw, sheet_name=sheet_name, index=False)
        else:
            pd.DataFrame({"info":[f"{sheet_name} — empty"]}).to_excel(
                xw, sheet_name=sheet_name, index=False
            )
    except Exception as e:
        print(f"[WARN] 写入 {sheet_name} 失败：{e}")

# ========== 1) 显性 ==========
def collect_explicit(exp_root: Path):
    models = [p for p in exp_root.iterdir() if p.is_dir() and p.name != "_summary"]
    E_list, V_list, EV_list = [], [], []
    dimE_edges, dimV_edges, word_edges = [], [], []
    node_stats, communities = [], []
    missing = []

    for mdir in models:
        model = mdir.name

        E  = read_csv_safe(mdir/"ExplicitTruth_E_scores.csv")
        V  = read_csv_safe(mdir/"ExplicitValence_V_scores.csv")
        EV = read_csv_safe(mdir/"Explicit_EV_combined.csv")

        if E is not None:
            e = E.copy()
            if 'dim' not in e.columns:
                e.rename(columns={'index':'dim'}, inplace=True)
            if 'E_d' in e.columns:
                e.rename(columns={'E_d':'value'}, inplace=True)
            if 'E_explain' in e.columns:
                e.rename(columns={'E_explain':'explain'}, inplace=True)
            e['model']  = model
            e['metric'] = 'E'
            E_list.append(e)

        if V is not None:
            v = V.copy()
            if 'dim' not in v.columns:
                v.rename(columns={'index':'dim'}, inplace=True)
            if 'V_d' in v.columns:
                v.rename(columns={'V_d':'value'}, inplace=True)
            if 'V_explain' in v.columns:
                v.rename(columns={'V_explain':'explain'}, inplace=True)
            v['model']  = model
            v['metric'] = 'V'
            V_list.append(v)

        if EV is not None:
            EVx = EV.copy()
            EVx['model'] = model
            EV_list.append(EVx)

        # 边
        dE = read_csv_safe(mdir/"Edges_dim_E.csv")
        if dE is not None:
            de = dE.copy(); de['model']=model; de['layer']='E'
            dimE_edges.append(de)

        dV = read_csv_safe(mdir/"Edges_dim_V.csv")
        if dV is not None:
            dv = dV.copy(); dv['model']=model; dv['layer']='V'
            dimV_edges.append(dv)

        wE = read_csv_safe(mdir/"Exp_word_edges.csv")
        if wE is not None:
            w = topk_by_abs(wE)
            w['model']=model; w['family']='explicit'; w['layer']='V'
            word_edges.append(w)

        # 03 的分析输出（如存在）
        for tag in ["Edges_dim_E_analysis", "Edges_dim_V_analysis", "Exp_word_edges_analysis"]:
            adir = mdir / tag
            if not adir.exists():
                missing.append(str(adir))
                continue
            ns = read_csv_safe(adir/"node_stats.csv")
            if ns is not None:
                ns2 = ns.copy(); ns2['model']=model; ns2['graph']=tag
                node_stats.append(ns2)
            com = read_json_safe(adir/"community.json")
            if com:
                dfc = pd.DataFrame([{'node':k, 'community':v} for k,v in com.items()])
                dfc['model']=model; dfc['graph']=tag
                communities.append(dfc)

    # summary 矩阵
    sumdir = exp_root/"_summary"
    explicit_summary = {}
    for name in ["Explicit_E_dim_matrix.csv","Explicit_V_dim_matrix.csv"]:
        p = sumdir/name
        mat = read_csv_safe(p, index_col=0)
        if mat is not None:
            explicit_summary[name] = mat

    return {
        "scores": concat_nonempty([concat_nonempty(E_list), concat_nonempty(V_list)]),
        "EV_combined": concat_nonempty(EV_list),
        "dim_edges": concat_nonempty([concat_nonempty(dimE_edges), concat_nonempty(dimV_edges)]),
        "word_edges": concat_nonempty(word_edges),
        "node_stats": concat_nonempty(node_stats),
        "communities": concat_nonempty(communities),
        "summary_mats": explicit_summary,
        "missing": missing
    }

# ========== 2) 隐性 ==========
def collect_implicit(imp_root: Path):
    models = [p for p in imp_root.iterdir() if p.is_dir() and p.name != "_summary"]
    I_list = []
    dim_edges, word_edges = [], []
    node_stats, communities = [], []
    missing = []

    for mdir in models:
        model = mdir.name

        I = read_csv_safe(mdir/"ImplicitBias_I_scores.csv")
        if I is not None:
            Ix = I.copy()
            if 'dim' not in Ix.columns:
                Ix.rename(columns={'index':'dim'}, inplace=True)
            if 'I_d' not in Ix.columns and 'value' in Ix.columns:
                Ix.rename(columns={'value':'I_d'}, inplace=True)
            Ix['model']=model
            I_list.append(Ix)

        dE = read_csv_safe(mdir/"Imp_dim_edges.csv")
        if dE is not None:
            de = dE.copy(); de['model']=model; de['family']='implicit'; de['layer']='I'
            dim_edges.append(de)

        wE = read_csv_safe(mdir/"Imp_word_edges.csv")
        if wE is not None:
            w = topk_by_abs(wE)
            w['model']=model; w['family']='implicit'; w['layer']='I'
            word_edges.append(w)

        for tag in ["Imp_dim_edges_analysis", "Imp_word_edges_analysis"]:
            adir = mdir / tag
            if not adir.exists():
                missing.append(str(adir)); continue
            ns = read_csv_safe(adir/"node_stats.csv")
            if ns is not None:
                ns2 = ns.copy(); ns2['model']=model; ns2['graph']=tag
                node_stats.append(ns2)
            com = read_json_safe(adir/"community.json")
            if com:
                dfc = pd.DataFrame([{'node':k, 'community':v} for k,v in com.items()])
                dfc['model']=model; dfc['graph']=tag
                communities.append(dfc)

    # summary 矩阵
    sumdir = imp_root/"_summary"
    implicit_summary = {}
    for name in ["Implicit_dim_male_matrix.csv","Implicit_dim_female_matrix.csv","Implicit_dim_diff_matrix.csv"]:
        p = sumdir/name
        mat = read_csv_safe(p, index_col=0)
        if mat is not None:
            implicit_summary[name] = mat

    return {
        "scores": concat_nonempty(I_list),
        "dim_edges": concat_nonempty(dim_edges),
        "word_edges": concat_nonempty(word_edges),
        "node_stats": concat_nonempty(node_stats),
        "communities": concat_nonempty(communities),
        "summary_mats": implicit_summary,
        "missing": missing
    }

# ========== 3) 亮点 ==========
def highlights_explicit(scores_df, k=3):
    if scores_df is None or len(scores_df)==0: return pd.DataFrame()
    df = scores_df.copy()
    if 'value' not in df.columns and 'E_d' in df.columns:
        df['value'] = df['E_d']
    rows=[]
    for (model, metric), g in df.groupby(['model','metric']):
        g_sorted_pos = g.sort_values('value', ascending=False)
        g_sorted_neg = g.sort_values('value', ascending=True)
        for _,r in g_sorted_pos.head(k).iterrows():
            rows.append([model, metric, r['dim'], r['value'], 'male_adv'])
        for _,r in g_sorted_neg.head(k).iterrows():
            rows.append([model, metric, r['dim'], r['value'], 'female_adv'])
    return pd.DataFrame(rows, columns=['model','metric','dim','value','direction'])

def highlights_implicit(I_df, k=3):
    if I_df is None or len(I_df)==0: return pd.DataFrame()
    df = I_df.copy()
    if 'I_d' not in df.columns and 'value' in df.columns:
        df.rename(columns={'value':'I_d'}, inplace=True)
    rows=[]
    for model, g in df.groupby('model'):
        for _,r in g.sort_values('I_d', ascending=False).head(k).iterrows():
            rows.append([model, r['dim'], r['I_d'], 'female_adv'])
        for _,r in g.sort_values('I_d', ascending=True).head(k).iterrows():
            rows.append([model, r['dim'], r['I_d'], 'male_adv'])
    return pd.DataFrame(rows, columns=['model','dim','I_d','direction'])

# ========== 主流程 ==========
def main():
    ensure_dir(OUT_DIR)

    # 1) 显性
    exp = collect_explicit(EXPLICIT_ROOT)
    exp_scores    = exp["scores"]
    exp_EVcombo   = exp["EV_combined"]
    exp_dim_edges = exp["dim_edges"]
    exp_word_edges= exp["word_edges"]
    exp_nodes     = exp["node_stats"]
    exp_comms     = exp["communities"]
    exp_summary   = exp["summary_mats"]

    # 2) 隐性
    imp = collect_implicit(IMPLICIT_ROOT)
    imp_scores    = imp["scores"]
    imp_dim_edges = imp["dim_edges"]
    imp_word_edges= imp["word_edges"]
    imp_nodes     = imp["node_stats"]
    imp_comms     = imp["communities"]
    imp_summary   = imp["summary_mats"]

    # 3) 亮点
    hi_exp = highlights_explicit(exp_scores, k=3)
    hi_imp = highlights_implicit(imp_scores, k=3)

    # 预先准备 README 行（长度一致）
    readme_sheets = [
        "Explicit_Scores","Explicit_EV_Combined","Explicit_Dim_Edges","Explicit_Word_Edges",
        "Explicit_Node_Stats","Explicit_Communities",
        "Explicit_Explicit_E_dim_matrix","Explicit_Explicit_V_dim_matrix",
        "Implicit_Scores","Implicit_Dim_Edges","Implicit_Word_Edges",
        "Implicit_Node_Stats","Implicit_Communities",
        "Implicit_Implicit_dim_male_matrix","Implicit_Implicit_dim_female_matrix","Implicit_Implicit_dim_diff_matrix",
        "Highlights_Explicit","Highlights_Implicit","Missing_Logs"
    ]
    readme_notes = ["Auto-generated bias integration workbook"] * len(readme_sheets)
    readme = pd.DataFrame({"sheet": readme_sheets, "note": readme_notes})

    # 4) 写 Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        # 先写 README（防止无可见 sheet 报错）
        safe_to_excel(xw, readme, "README")

        # 显性
        safe_to_excel(xw, reorder_dims(exp_scores, 'dim'), "Explicit_Scores")
        if len(exp_EVcombo): safe_to_excel(xw, exp_EVcombo, "Explicit_EV_Combined")
        if len(exp_dim_edges): safe_to_excel(xw, exp_dim_edges, "Explicit_Dim_Edges")
        if len(exp_word_edges): safe_to_excel(xw, exp_word_edges, "Explicit_Word_Edges")
        if len(exp_nodes): safe_to_excel(xw, exp_nodes, "Explicit_Node_Stats")
        if len(exp_comms): safe_to_excel(xw, exp_comms, "Explicit_Communities")
        for name, mat in (exp_summary or {}).items():
            safe_to_excel(xw, mat.reset_index(), f"Explicit_{name.replace('.csv','')[:28]}")

        # 隐性
        safe_to_excel(xw, reorder_dims(imp_scores, 'dim'), "Implicit_Scores")
        if len(imp_dim_edges): safe_to_excel(xw, imp_dim_edges, "Implicit_Dim_Edges")
        if len(imp_word_edges): safe_to_excel(xw, imp_word_edges, "Implicit_Word_Edges")
        if len(imp_nodes): safe_to_excel(xw, imp_nodes, "Implicit_Node_Stats")
        if len(imp_comms): safe_to_excel(xw, imp_comms, "Implicit_Communities")
        for name, mat in (imp_summary or {}).items():
            safe_to_excel(xw, mat.reset_index(), f"Implicit_{name.replace('.csv','')[:28]}")

        # 亮点
        if len(hi_exp): safe_to_excel(xw, reorder_dims(hi_exp.rename(columns={'value':'score'}), 'dim'), "Highlights_Explicit")
        if len(hi_imp): safe_to_excel(xw, reorder_dims(hi_imp, 'dim'), "Highlights_Implicit")

        # 缺失日志
        miss = pd.DataFrame({'missing_paths': exp["missing"] + imp["missing"]})
        if len(miss): safe_to_excel(xw, miss, "Missing_Logs")

    print(f"\n✅ 已写出整合文件：{OUT_XLSX}")

if __name__ == "__main__":
    main()
