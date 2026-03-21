import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatsBench · Leaderboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("StatsBench · Global Leaderboard")
st.caption("Test Cricket · Bayesian career rankings")
st.divider()

# ── Position group ────────────────────────────────────────────────────────────
def pos_group(p):
    if pd.isna(p):
        return np.nan
    p = int(p)
    if p <= 2:  return 'opener'
    if p <= 5: return 'upper middle'
    if p <= 7:  return 'lower middle'
    return 'tail'  # 8-11

# ── Load & enrich ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_enrich(path: str, block_size: int = 6, k: float = 0.2) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    for col in ['year', 'runs', 'balls', 'strike_rate', 'batting_position', 'inns', 'is_out']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    seasons_ordered = sorted(df['season'].dropna().unique())
    season_rank = {s: i for i, s in enumerate(seasons_ordered)}
    df['season_rank'] = df['season'].map(season_rank)
    df['era_block'] = (df['season_rank'] // block_size) * block_size
    df['pos_group'] = df['batting_position'].apply(pos_group)

    # ── Home / Away ──
    NEUTRAL_MAP = {'United Arab Emirates': 'Pakistan'}
    def get_home_away(row):
        home_country = NEUTRAL_MAP.get(row['country'], row['country'])
        return 'Home' if row['team_bat'] == home_country else 'Away'
    df['home_away'] = df.apply(get_home_away, axis=1)

    top7 = df[df['batting_position'] <= 7]

    # ── Helper: compute z-score weights and adj_runs for any baseline ──
    def z_adjust(df_full, base_col):
        mu  = df_full[base_col].mean()
        std = df_full[base_col].std()
        df_full['_z'] = (df_full[base_col] - mu) / std
        df_full['_w'] = np.exp(-k * df_full['_z'])
        adj = df_full['runs'] * df_full['_w']
        df_full.drop(columns=['_z', '_w'], inplace=True)
        return adj

    # ── Baseline 1: ERA ──
    era = (
        top7.groupby('era_block')
        .apply(lambda g: g['runs'].sum() / g['is_out'].sum())
        .reset_index(name='base_era')
    )
    df = df.merge(era, on='era_block', how='left')
    df['adj_runs_era'] = z_adjust(df, 'base_era')

    # ── Baseline 2: ERA-COUNTRY ──
    era_ctry = (
        top7.groupby(['era_block', 'country'])
        .apply(lambda g: g['runs'].sum() / g['is_out'].sum())
        .reset_index(name='base_era_country')
    )
    df = df.merge(era_ctry, on=['era_block', 'country'], how='left')
    df['adj_runs_era_country'] = z_adjust(df, 'base_era_country')

    # ── Baseline 3: ERA-POSITION ──
    era_pos = (
        top7.groupby(['era_block', 'pos_group'])
        .apply(lambda g: g['runs'].sum() / g['is_out'].sum())
        .reset_index(name='base_era_pos')
    )
    df = df.merge(era_pos, on=['era_block', 'pos_group'], how='left')

    # Z-score within each pos_group separately
    pos_stats = (
        df.groupby('pos_group')['base_era_pos']
        .agg(mu='mean', std='std')
        .reset_index()
    )
    df = df.merge(pos_stats, on='pos_group', how='left')
    df['_z_pos'] = (df['base_era_pos'] - df['mu']) / df['std']
    df['adj_runs_era_pos'] = df['runs'] * np.exp(-k * df['_z_pos'])
    df.drop(columns=['_z_pos', 'mu', 'std'], inplace=True)

    # ── Baseline 4: ERA-OPPOSITION ──
    era_opp = (
        top7.groupby(['era_block', 'team_bowl'])
        .apply(lambda g: g['runs'].sum() / g['is_out'].sum())
        .reset_index(name='base_era_opp')
    )
    df = df.merge(era_opp, on=['era_block', 'team_bowl'], how='left')
    df['adj_runs_era_opp'] = z_adjust(df, 'base_era_opp')

    return df,season_rank

# ── Posterior average computation ─────────────────────────────────────────────
@st.cache_data
def compute_leaderboard(df: pd.DataFrame, season_rank: dict,
                         alpha: float = 20, min_inns: int = 20) -> pd.DataFrame:
    top7     = df[df['batting_position'] <= 7].copy()
    global_fallback = df['adj_runs_era'].sum() / df['is_out'].sum()

    rows = []
    for batter, g in df.groupby('bat'):
        inns   = len(g)
        if inns < min_inns:
            continue

        runs   = g['runs'].fillna(0)
        is_out = g['is_out'].fillna(0).astype(int)
        outs   = int(is_out.sum())
        if outs == 0:
            continue

        # ── Standard ave ──
        std_ave = round(runs.sum() / outs, 2)

        # ── All adjusted aves ──
        adj_ave      = round(g['adj_runs_era'].sum()         / outs, 2)
        ctry_ave     = round(g['adj_runs_era_country'].sum() / outs, 2)
        pos_ave      = round(g['adj_runs_era_pos'].sum()     / outs, 2)
        opp_ave      = round(g['adj_runs_era_opp'].sum()     / outs, 2)

        # ── SR (valid balls only) ──
        valid        = g[g['balls'].notna()]
        total_balls  = int(valid['balls'].sum())
        sr           = round(valid['runs'].sum() / total_balls * 100, 1) if total_balls > 0 else float('nan')

        # ── 100s / 50s ──
        hundreds     = int((runs >= 100).sum())
        fifties      = int(((runs >= 50) & (runs < 100)).sum())

        # ── Modal position group ──
        modal_pos = g['pos_group'].mode()
        if modal_pos.empty:
            continue
        modal_pos = modal_pos.iloc[0]

        # ── Career season range ──
        batter_seasons = g['season'].dropna().unique()
        career_ranks   = [season_rank[s] for s in batter_seasons if s in season_rank]
        if not career_ranks:
            continue
        min_rank = min(career_ranks)
        max_rank = max(career_ranks)

        # ── Prior: all top-7 at modal_pos during career span, excl. batter ──
        prior_pool = top7[
            (top7['pos_group'] == modal_pos) &
            (top7['season'].map(season_rank) >= min_rank) &
            (top7['season'].map(season_rank) <= max_rank) &
            (top7['bat'] != batter)
        ]

        # And global fallback:
        global_fallback = df['runs'].sum() / df['is_out'].sum()

        prior_outs = prior_pool['is_out'].sum()
        mu_prior = (prior_pool['runs'].sum() / prior_outs
            if prior_outs > 0 else global_fallback)

        

        # ── Bayesian posterior ──
        posterior = round(
            (g['adj_runs_era_opp'].sum() + alpha * mu_prior) / (outs + alpha), 2
        )

        rows.append({
            'Batter':       batter,
            'Modal pos':    modal_pos,
            'Inns':         inns,
            'NO':           inns - outs,
            'Runs':         int(runs.sum()),
            'Avg':          std_ave,
            'Era-ave':      adj_ave,
            'Ctry-ave':     ctry_ave,
            'Pos-ave':      pos_ave,
            'Opp-ave':      opp_ave,
            'Posterior':    posterior,
            'Prior':        round(mu_prior, 2),
            'SR':           sr,
            '100s':         hundreds,
            '50s':          fifties,
        })

    return (pd.DataFrame(rows)
            .sort_values('Posterior', ascending=False)
            .reset_index(drop=True))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    df_raw, season_rank = load_and_enrich(
        'test_batting_innings.csv'
    )
    st.success(f"{len(df_raw):,} innings loaded")

    st.markdown("---")
    st.markdown("### ⚙️ Parameters")
    min_inns = st.number_input("Minimum innings", min_value=1, value=20, step=1)
    alpha    = st.number_input("Prior strength (α)", min_value=1, value=20, step=1,
                                help="Higher = more shrinkage toward prior for short careers")

    st.markdown("---")
    run_query = st.button("⚡ Build Leaderboard")

# ── Main ──────────────────────────────────────────────────────────────────────
if run_query:
    with st.spinner("Computing posteriors..."):
        tbl = compute_leaderboard(df_raw, season_rank,
                                  alpha=int(alpha), min_inns=int(min_inns))

    st.markdown(f"**{len(tbl)} batters** with ≥ {int(min_inns)} innings · sorted by Posterior ave · α = {int(alpha)}")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    csv_out = tbl.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download leaderboard CSV", data=csv_out,
                       file_name="statsbench_leaderboard.csv", mime="text/csv")
else:
    st.info("Set parameters in the sidebar, then hit **Build Leaderboard**.")