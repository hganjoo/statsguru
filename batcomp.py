import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatsBench · Batter Comparison",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("StatsBench · Batter Comparison")
st.caption("One row per batter — aggregated across all matching innings")
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

# ── Load & enrich (shared with sguru.py) ─────────────────────────────────────
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

    # ── % of team runs ──
    if 'team_innings_runs' in df.columns:
        df['team_runs_pct'] = df['runs'] / df['team_innings_runs'] * 100

    # ── Match factor ──
    # ── Match factor ──
    top8 = df[df['batting_position'] <= 8][['p_match', 'runs', 'is_out']].copy()
    match_agg = (
        top8.groupby('p_match')
        .agg(top8_runs=('runs', 'sum'), top8_outs=('is_out', 'sum'))
        .reset_index()
    )
    match_agg['match_ave'] = match_agg['top8_runs'] / match_agg['top8_outs'].replace(0, np.nan)
    df = df.merge(match_agg[['p_match', 'match_ave']], on='p_match', how='left')
    df['match_factor'] = df['runs'] / df['match_ave']
    df.drop(columns=['match_ave'], inplace=True)

    return df
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    df_raw = load_and_enrich('test_batting_innings.csv')

    st.markdown("---")
    st.markdown("### 🔍 Filters")
    st.caption("Filters apply to all innings used to compute each batter's stats.")

    opps_all = sorted(df_raw['team_bowl'].dropna().unique().tolist())
    sel_opps = st.multiselect("Opposition", options=opps_all)

    col1, col2 = st.columns(2)
    with col1:
        countries_all = sorted(df_raw['country'].dropna().unique().tolist())
        sel_countries = st.multiselect("Country", options=countries_all)
    with col2:
        grounds_all = sorted(df_raw['ground'].dropna().unique().tolist())
        sel_grounds = st.multiselect("Ground", options=grounds_all)

    yr_min = int(df_raw['year'].min())
    yr_max = int(df_raw['year'].max())
    sel_years = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    inns_options = ["1st", "2nd", "3rd", "4th"]
    sel_inns = st.multiselect("Innings", options=inns_options)

    col3, col4 = st.columns(2)
    with col3:
        pos_min = st.number_input("Bat pos (min)", min_value=1, max_value=11, value=1)
    with col4:
        pos_max = st.number_input("Bat pos (max)", min_value=1, max_value=11, value=11)

    col5, col6 = st.columns(2)
    with col5:
        filter_captain = st.checkbox("As captain")
    with col6:
        filter_keeper = st.checkbox("As keeper")
    col7, col8 = st.columns(2)
    with col7:
        entry_runs_min = st.number_input("Entry runs (min)", min_value=0, value=0, step=1)
        entry_runs_max = st.number_input("Entry runs (max)", min_value=0, value=900, step=1)
    with col8:
        entry_wkts_min = st.number_input("Entry wkts (min)", min_value=0, max_value=9, value=0, step=1)
        entry_wkts_max = st.number_input("Entry wkts (max)", min_value=0, max_value=9, value=9, step=1)
    sel_home_away = st.multiselect("Home / Away", options=["Home", "Away"])

    result_options = ["Won", "Lost", "Draw / No result"]
    sel_result = st.multiselect("Match result", options=result_options)

    min_runs = st.number_input("Minimum runs (to appear)", min_value=0, value=0, step=1)
    min_inns = st.number_input("Minimum innings (to appear)", min_value=1, value=10, step=1)

    st.markdown("---")
    run_query = st.button("⚡ Run Comparison")

# ── Filter logic ──────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    mask &= df['runs'].notna()

    if sel_opps:
        mask &= df['team_bowl'].isin(sel_opps)
    if sel_countries:
        mask &= df['country'].isin(sel_countries)
    if sel_grounds:
        mask &= df['ground'].isin(sel_grounds)
    if 'year' in df.columns:
        mask &= df['year'].between(sel_years[0], sel_years[1])
    if sel_inns:
        inns_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        mask &= df['inns'].isin([inns_map[i] for i in sel_inns])
    if 'batting_position' in df.columns:
        mask &= df['batting_position'].between(pos_min, pos_max)
    if 'entry_runs' in df.columns and (entry_runs_min > 0 or entry_runs_max < 900):
        mask &= df['entry_runs'].between(entry_runs_min, entry_runs_max)
    if 'entry_wkts' in df.columns and (entry_wkts_min > 0 or entry_wkts_max < 9):
        mask &= df['entry_wkts'].between(entry_wkts_min, entry_wkts_max)
    if sel_home_away and 'home_away' in df.columns:
        mask &= df['home_away'].isin(sel_home_away)
    if filter_captain and filter_keeper:
        mask &= df['player_role_type'].isin(['C', 'CWK'])
        mask &= df['player_role_type'].isin(['WK', 'CWK'])
        # simplifies to:
        mask &= df['player_role_type'] == 'CWK'
    elif filter_captain:
        mask &= df['player_role_type'].isin(['C', 'CWK'])
    elif filter_keeper:
        mask &= df['player_role_type'].isin(['WK', 'CWK'])
    if sel_result:
        result_mask = pd.Series([False] * len(df), index=df.index)
        if "Won" in sel_result:
            result_mask |= (df['winner'] == df['team_bat'])
        if "Lost" in sel_result:
            result_mask |= (df['winner'] != df['team_bat']) & df['winner'].notna() & (df['winner'] != '')
        if "Draw / No result" in sel_result:
            result_mask |= df['winner'].isna() | (df['winner'] == '')
        mask &= result_mask
    

    return df[mask].copy()

# ── Aggregation per batter ────────────────────────────────────────────────────
def safe_ave(runs_sum, outs):
    return round(runs_sum / outs, 2) if outs > 0 else float('nan')

def build_comparison(df: pd.DataFrame, min_inns: int) -> pd.DataFrame:
    rows = []
    for batter, g in df.groupby('bat'):
        runs   = g['runs'].fillna(0)
        is_out = g['is_out'].fillna(0).astype(int)
        balls  = g['balls'].fillna(0)
        outs   = int(is_out.sum())
        inns   = len(g)

        if inns < min_inns:
            continue
        if runs < min_runs:
            continue
        valid_balls = g[g['balls'].notna()]
        total_runs  = int(runs.sum())
        total_balls = int(valid_balls['balls'].sum())
        sr = round(valid_balls['runs'].sum() / total_balls * 100, 1) if total_balls > 0 else float('nan')

        total_runs  = int(runs.sum())
        total_balls = int(g['balls'].fillna(0).sum())

        rows.append({
            'Batter':        batter,
            'Inns':          inns,
            'NO':            inns - outs,
            'Runs':          total_runs,
            'HS':            int(runs.max()),
            'Avg':           safe_ave(total_runs, outs),
            'Era-ave':       safe_ave(g['adj_runs_era'].sum(),         outs),
            'Ctry-ave':      safe_ave(g['adj_runs_era_country'].sum(), outs),
            'Pos-ave':       safe_ave(g['adj_runs_era_pos'].sum(),     outs),
            'Opp-ave':       safe_ave(g['adj_runs_era_opp'].sum(),     outs),
            'SR':            sr,
            'Team%':         round(g['team_runs_pct'].mean(), 1) if 'team_runs_pct' in g.columns else float('nan'),
            'Match factor':  round(g['match_factor'].mean(),  2) if 'match_factor'  in g.columns else float('nan'),
            '100s':          int((runs >= 100).sum()),
            '50s':           int(((runs >= 50) & (runs < 100)).sum()),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values('Runs', ascending=False).reset_index(drop=True)
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
if run_query:
    df_filtered = apply_filters(df_raw)
    n_innings   = len(df_filtered)
    n_batters   = df_filtered['bat'].nunique()

    col_a, col_b = st.columns(2)
    col_a.metric("Innings matched", f"{n_innings:,}")
    col_b.metric("Batters", f"{n_batters:,}")

    if n_innings == 0:
        st.info("No innings match your filters — try relaxing the criteria.")
    else:
        tbl = build_comparison(df_filtered, int(min_inns))

        if tbl.empty:
            st.info(f"No batter has at least {int(min_inns)} innings under these filters.")
        else:
            st.markdown(f"**{len(tbl)} batters** with ≥ {int(min_inns)} innings — sorted by runs")
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            csv_out = tbl.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download comparison CSV", data=csv_out,
                               file_name="statsbench_comparison.csv", mime="text/csv")
else:
    st.info("Set your filters in the sidebar, then hit **Run Comparison**.")