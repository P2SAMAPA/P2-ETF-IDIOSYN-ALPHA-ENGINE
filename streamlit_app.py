"""
Streamlit Dashboard — P2-ETF-IDIOSYN-ALPHA-ENGINE
Dynamic Beta Hedging & Idiosyncratic Alpha
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from huggingface_hub import hf_hub_download

from us_calendar import USMarketCalendar

st.set_page_config(
    page_title="P2Quant Idiosyncratic Alpha",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.main-header{font-size:2.1rem;font-weight:600;color:#0D2B55;margin-bottom:0}
.sub-header{font-size:.92rem;color:#888;margin-bottom:.6rem}

/* Hero cards — three distinct themes */
.hero-idio{background:linear-gradient(135deg,#1a5fa8,#2C5282);border-radius:14px;padding:1.2rem 1.4rem;color:white;margin-bottom:.4rem}
.hero-alpha{background:linear-gradient(135deg,#1a6b3c,#155534);border-radius:14px;padding:1.2rem 1.4rem;color:white;margin-bottom:.4rem}
.hero-conv{background:linear-gradient(135deg,#7b2d8b,#5c1f69);border-radius:14px;padding:1.2rem 1.4rem;color:white;margin-bottom:.4rem}

.hc-label{font-size:.68rem;opacity:.75;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.15rem}
.hc-section{font-size:.75rem;opacity:.8;margin-bottom:.6rem;padding-bottom:.5rem;border-bottom:1px solid rgba(255,255,255,.2)}
.hc-rank{font-size:.65rem;opacity:.7;text-transform:uppercase;letter-spacing:.04em}
.hc-ticker{font-size:1.5rem;font-weight:700;line-height:1.1}
.hc-val{font-size:.82rem;opacity:.85;margin-top:1px}
.hc-sub{font-size:.72rem;opacity:.65;margin-top:2px}
.hc-sep{border-top:1px solid rgba(255,255,255,.15);margin:.5rem 0}

/* Explanation box */
.explain{background:#f0f4fa;border-left:3px solid #1a5fa8;border-radius:0 8px 8px 0;
         padding:.65rem .9rem;font-size:.84rem;color:#333;line-height:1.6;margin-bottom:1rem}
.explain b{color:#0D2B55}

.beta-pill{display:inline-block;font-size:.75rem;padding:2px 8px;border-radius:12px;
           margin-right:4px;background:#f0f4f8;color:#555;border:1px solid #dde3ec}
</style>
""", unsafe_allow_html=True)

HF_RESULTS_REPO = "P2SAMAPA/p2-etf-idiosyn-alpha-engine-results"
HF_RESULTS_FILE = "idiosyn_alpha_results.parquet"

BENCH_COLORS = {"SPY": "#2d7dd2", "AGG": "#1D9E75", "GLD": "#BA7517"}


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_results():
    try:
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=HF_RESULTS_FILE,
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        df["run_date"] = pd.to_datetime(df["run_date"])
        return df
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return None


def get_latest(df, universe):
    sub = df[df["universe"] == universe]
    if sub.empty:
        return sub
    return sub[sub["run_date"] == sub["run_date"].max()]


# ── Sidebar ───────────────────────────────────────────────────────────────────

cal     = USMarketCalendar()
next_td = cal.next_trading_day()
is_today = cal.is_trading_day()

df = load_results()
latest_run = df["run_date"].max() if df is not None and not df.empty else None
n_dates    = df["run_date"].nunique() if df is not None and not df.empty else 0

st.sidebar.markdown("## ⚡ P2Quant Idiosyn Alpha")
st.sidebar.divider()
st.sidebar.markdown("### 📅 Market calendar")
st.sidebar.markdown(f"**Last run:** {latest_run.strftime('%a %d %b %Y') if latest_run else '—'}")
st.sidebar.markdown(f"**Next trading day:** {next_td.strftime('%a %d %b %Y')}")
st.sidebar.success("Today is a trading day") if is_today else st.sidebar.info("Market closed today")
st.sidebar.markdown(f"*{n_dates} trading days in history*")
st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Engine parameters")
st.sidebar.markdown("- **Beta model:** DCC-GARCH(1,1) → OLS fallback")
st.sidebar.markdown("- **Benchmarks:** SPY · AGG · GLD")
st.sidebar.markdown("- **OLS windows:** 63d + 126d")
st.sidebar.markdown("- **DCC lookback:** 252d")
st.sidebar.markdown("- **Conviction:** top 5 in both rankings")
st.sidebar.divider()
st.sidebar.markdown("[GitHub](https://github.com/P2SAMAPA/P2-ETF-IDIOSYN-ALPHA-ENGINE) · [HF Dataset](https://huggingface.co/datasets/P2SAMAPA/p2-etf-idiosyn-alpha-engine-results)")

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">⚡ P2Quant Idiosyncratic Alpha</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dynamic Beta Hedging — ranking ETFs by the return that cannot be explained by SPY · AGG · GLD exposure</div>', unsafe_allow_html=True)

st.markdown("""
<div class="explain">
<b>Three signals, one engine:</b>
&nbsp;<b>Idiosyncratic return (ε)</b> — today's residual after stripping market, duration and commodity beta. High ε = genuine outperformance right now.
&nbsp;·&nbsp;<b>Jensen's alpha</b> — rolling OLS intercept annualised over 63d + 126d. Persistent positive alpha = structural edge independent of factor exposure.
&nbsp;·&nbsp;<b>Conviction</b> — ETFs in the top 5 of <i>both</i> rankings simultaneously. Strongest signal: an ETF generating persistent alpha AND showing idiosyncratic outperformance today.
</div>
""", unsafe_allow_html=True)

if df is None or df.empty:
    st.warning("No results yet. The engine runs daily at 22:30 UTC.")
    st.stop()

# ── Hero section — three columns ──────────────────────────────────────────────

def hero_block(rows: pd.DataFrame, theme: str, section_label: str,
               rank_col: str, val_col: str, val_label: str, val_fmt: str) -> str:
    html = f'<div class="hero-{theme}">'
    html += f'<div class="hc-section">{section_label}</div>'
    for _, row in rows.head(3).iterrows():
        rank  = int(row.get(rank_col, 0)) if not pd.isna(row.get(rank_col, np.nan)) else "—"
        val   = row.get(val_col, np.nan)
        bspy  = row.get("beta_SPY", np.nan)
        bagg  = row.get("beta_AGG", np.nan)
        bgld  = row.get("beta_GLD", np.nan)
        val_str = (val_fmt.format(val) if not pd.isna(val) else "—")
        betas_str = ""
        if not pd.isna(bspy): betas_str += f"β_SPY {bspy:.2f} "
        if not pd.isna(bagg): betas_str += f"β_AGG {bagg:.2f} "
        if not pd.isna(bgld): betas_str += f"β_GLD {bgld:.2f}"
        html += f"""
        <div class="hc-rank">#{rank}</div>
        <div class="hc-ticker">{row['ticker']}</div>
        <div class="hc-val">{val_label}: {val_str}</div>
        <div class="hc-sub">{betas_str}</div>
        <div class="hc-sep"></div>"""
    html += "</div>"
    return html


combined_latest = get_latest(df, "combined")

col_idio, col_alpha, col_conv = st.columns(3)

with col_idio:
    top_idio = combined_latest.nsmallest(3, "idio_rank") if not combined_latest.empty else pd.DataFrame()
    st.markdown(hero_block(top_idio, "idio", "🔵 Today's idiosyncratic return",
                           "idio_rank", "idio_return", "ε today", "{:+.4f}"), unsafe_allow_html=True)

with col_alpha:
    top_alpha = combined_latest.nsmallest(3, "alpha_rank") if not combined_latest.empty else pd.DataFrame()
    st.markdown(hero_block(top_alpha, "alpha", "🟢 Persistent Jensen's alpha",
                           "alpha_rank", "jensen_alpha_combined", "α p.a.", "{:+.2%}"), unsafe_allow_html=True)

with col_conv:
    conv_df = combined_latest[combined_latest["conviction"] == True].sort_values("conviction_rank") \
              if not combined_latest.empty else pd.DataFrame()
    if conv_df.empty:
        st.markdown('<div class="hero-conv"><div class="hc-section">⭐ Conviction — top 5 in both</div>'
                    '<div class="hc-val" style="opacity:.7">No conviction ETFs today</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(hero_block(conv_df, "conv", "⭐ Conviction — top 5 in both",
                               "conviction_rank", "jensen_alpha_combined", "α p.a.", "{:+.2%}"),
                    unsafe_allow_html=True)

st.divider()

# ── Universe tabs ─────────────────────────────────────────────────────────────

tab_comb, tab_eq, tab_fi, tab_history = st.tabs([
    "🌐 Combined", "📈 Equity sectors", "💰 FI / Commodities", "📊 History"
])


def render_tab(latest: pd.DataFrame):
    if latest.empty:
        st.info("No data yet.")
        return

    run_dt = latest["run_date"].iloc[0].strftime("%d %b %Y")
    method = latest["beta_method"].iloc[0] if "beta_method" in latest.columns else "—"
    st.caption(f"Scores as of {run_dt}  ·  Beta method: {method}")

    # ── Two-column chart layout ────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Idiosyncratic return (ε) today")
        s = latest.sort_values("idio_rank")
        colors = ["#1D9E75" if v > 0 else "#d62728" for v in s["idio_return"]]
        fig = go.Figure(go.Bar(
            x=s["idio_return"], y=s["ticker"], orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in s["idio_return"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>ε: %{x:+.5f}<extra></extra>",
        ))
        fig.update_layout(height=max(280, len(s)*34), yaxis=dict(autorange="reversed"),
                          margin=dict(l=10,r=60,t=10,b=10), xaxis_title="Idiosyncratic return",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Jensen's alpha (annualised, combined)")
        s2 = latest.sort_values("alpha_rank")
        colors2 = ["#1D9E75" if v > 0 else "#d62728"
                   for v in s2["jensen_alpha_combined"].fillna(0)]
        fig2 = go.Figure(go.Bar(
            x=s2["jensen_alpha_combined"], y=s2["ticker"], orientation="h",
            marker_color=colors2,
            text=[f"{v:+.1%}" if not pd.isna(v) else "" for v in s2["jensen_alpha_combined"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>α: %{x:+.3%}<extra></extra>",
        ))
        fig2.update_layout(height=max(280, len(s2)*34), yaxis=dict(autorange="reversed"),
                           margin=dict(l=10,r=70,t=10,b=10), xaxis_title="Jensen's alpha (p.a.)",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Beta heatmap ──────────────────────────────────────────────────────────
    st.markdown("#### Dynamic beta exposures — SPY · AGG · GLD")
    st.caption("Time-varying betas from DCC-GARCH. Shows each ETF's current systematic exposure to the three orthogonal risk factors.")
    beta_cols = ["beta_SPY", "beta_AGG", "beta_GLD"]
    avail = [c for c in beta_cols if c in latest.columns]
    if avail:
        heat = latest.set_index("ticker")[avail].copy()
        heat.columns = [c.replace("beta_", "") for c in avail]
        fig3 = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
            colorscale="RdBu", zmid=0,
            text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in heat.values],
            texttemplate="%{text}", textfont={"size": 10},
            hovertemplate="<b>%{y}</b> β_%{x}: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Beta", thickness=12),
        ))
        fig3.update_layout(height=max(280, len(heat)*34+60),
                           margin=dict(l=10,r=10,t=10,b=20),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Conviction ETFs highlight ─────────────────────────────────────────────
    conv = latest[latest["conviction"] == True].sort_values("conviction_rank")
    if not conv.empty:
        st.markdown("#### ⭐ Conviction ETFs — top 5 in both rankings")
        cols = st.columns(min(5, len(conv)))
        for i, (_, row) in enumerate(conv.iterrows()):
            with cols[i]:
                alpha_str = f"{row['jensen_alpha_combined']:+.1%}" \
                    if not pd.isna(row["jensen_alpha_combined"]) else "—"
                idio_str  = f"{row['idio_return']:+.4f}" \
                    if not pd.isna(row["idio_return"]) else "—"
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#7b2d8b,#5c1f69);border-radius:10px;
                            padding:.9rem;color:white;text-align:center">
                  <div style="font-size:.65rem;opacity:.7">#{int(row['conviction_rank']) if not pd.isna(row['conviction_rank']) else '?'}</div>
                  <div style="font-size:1.4rem;font-weight:700">{row['ticker']}</div>
                  <div style="font-size:.78rem;opacity:.85">α {alpha_str}</div>
                  <div style="font-size:.72rem;opacity:.7">ε {idio_str}</div>
                </div>""", unsafe_allow_html=True)

    # ── Full table ────────────────────────────────────────────────────────────
    with st.expander("Full rankings table"):
        show = ["ticker","idio_rank","idio_return","idio_zscore",
                "alpha_rank","jensen_alpha_combined","jensen_alpha_63d","jensen_alpha_126d",
                "conviction","beta_SPY","beta_AGG","beta_GLD","r_squared_63d","beta_method"]
        show = [c for c in show if c in latest.columns]
        st.dataframe(
            latest[show].sort_values("idio_rank").style.format({
                "idio_return":             "{:+.5f}",
                "idio_zscore":             "{:.3f}",
                "jensen_alpha_combined":   "{:+.2%}",
                "jensen_alpha_63d":        "{:+.2%}",
                "jensen_alpha_126d":       "{:+.2%}",
                "beta_SPY":                "{:.3f}",
                "beta_AGG":                "{:.3f}",
                "beta_GLD":                "{:.3f}",
                "r_squared_63d":           "{:.3f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )


with tab_comb:
    render_tab(get_latest(df, "combined"))

with tab_eq:
    render_tab(get_latest(df, "equity"))

with tab_fi:
    render_tab(get_latest(df, "fi"))

# ── History tab ───────────────────────────────────────────────────────────────

with tab_history:
    hist = df[df["universe"] == "combined"].copy()
    if hist.empty:
        st.info("No history yet.")
    else:
        st.markdown("#### Jensen's alpha over time — combined universe")
        tickers = sorted(hist["ticker"].unique())
        sel = st.multiselect("Select ETFs", options=tickers,
                              default=tickers[:6] if len(tickers) >= 6 else tickers)
        if sel:
            pivot = hist[hist["ticker"].isin(sel)].pivot(
                index="run_date", columns="ticker", values="jensen_alpha_combined")
            fig_h = go.Figure()
            for t in pivot.columns:
                fig_h.add_trace(go.Scatter(
                    x=pivot.index, y=pivot[t], name=t, mode="lines",
                    hovertemplate=f"<b>{t}</b><br>%{{x|%d %b %Y}}<br>α: %{{y:+.2%}}<extra></extra>"))
            fig_h.update_layout(
                height=380, xaxis_title="Date", yaxis_title="Jensen's alpha (p.a.)",
                hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=10,r=10,t=40,b=20))
            fig_h.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("#### Idiosyncratic return history")
        if sel:
            pivot2 = hist[hist["ticker"].isin(sel)].pivot(
                index="run_date", columns="ticker", values="idio_return")
            fig_h2 = go.Figure()
            for t in pivot2.columns:
                fig_h2.add_trace(go.Scatter(
                    x=pivot2.index, y=pivot2[t], name=t, mode="lines",
                    hovertemplate=f"<b>{t}</b><br>%{{x|%d %b %Y}}<br>ε: %{{y:+.5f}}<extra></extra>"))
            fig_h2.update_layout(
                height=340, xaxis_title="Date", yaxis_title="Idiosyncratic return",
                hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=10,r=10,t=40,b=20))
            fig_h2.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(fig_h2, use_container_width=True)

st.divider()
st.caption("P2Quant Idiosyncratic Alpha Engine · P2SAMAPA · DCC-GARCH betas: SPY · AGG · GLD")
