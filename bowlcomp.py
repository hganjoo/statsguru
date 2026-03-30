import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatsBench · Bowler Comparison",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("StatsBench · Bowler Comparison")
st.caption("One row per bowler — aggregated across all matching innings")
st.divider()

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_bowling(path: str, block_size: int = 6) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ['year', 'balls', 'runs_conceded', 'wickets', 'maidens',
                'economy', 'runs_per_ball', 'dots', 'overs', 'inns']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['ave']  = np.where(df['wickets'] > 0, df['runs_conceded'] / df['wickets'], np.nan)
    df['sr']   = np.where(df['wickets'] > 0, df['balls'] / df['wickets'], np.nan)
    df['econ'] = df['runs_per_ball'] * 6

    NEUTRAL_MAP = {'United Arab Emirates': 'Pakistan'}
    def get_home_away(row):
        home_country = NEUTRAL_MAP.get(row['country'], row['country'])
        return 'Home' if row['team_bowl'] == home_country else 'Away'
    df['home_away'] = df.apply(get_home_away, axis=1)

    def getbtype(s):
        if pd.isna(s): return None
        return 'pace' if 'f' in str(s) else 'spin'
    df['bkind'] = df['bowling_kind'].apply(getbtype)

    seasons_ordered = sorted(df['season'].dropna().unique())
    season_rank = {s: i for i, s in enumerate(seasons_ordered)}
    df['season_rank'] = df['season'].map(season_rank)
    df['era_block'] = (df['season_rank'] // block_size) * block_size

    baseg   = df.groupby(['era_block', 'bkind', 'country'])
    baseav  = (baseg['runs_conceded'].sum() / baseg['wickets'].sum()).reset_index(name='baseav')
    basesr  = (baseg['balls'].sum()         / baseg['wickets'].sum()).reset_index(name='basesr')
    basewpi = (baseg['wickets'].sum()        / baseg['wickets'].count()).reset_index(name='basewpi')

    base = baseav.merge(basesr,  on=['era_block', 'bkind', 'country'])
    base = base.merge(basewpi, on=['era_block', 'bkind', 'country'])
    df   = df.merge(base, on=['era_block', 'bkind', 'country'], how='left')

    matchstats = df.groupby(['p_match', 'bkind']).agg(
        {'balls': 'sum', 'wickets': 'sum', 'runs_conceded': 'sum'}
    ).reset_index()
    matchstats['matchav_inv'] = matchstats['wickets'] / matchstats['runs_conceded']
    matchstats['matchsr_inv'] = matchstats['wickets'] / matchstats['balls']
    df = df.merge(matchstats[['p_match', 'bkind', 'matchav_inv', 'matchsr_inv']],
                  on=['p_match', 'bkind'])

    return df

@st.cache_data
def load_batting_matched(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ['runs', 'is_out', 'batting_position', 'inns']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe(n, d, decimals=2):
    return round(n / d, decimals) if d > 0 and not np.isnan(d) else float('nan')

def safe_diff(val, base):
    if any(pd.isna(x) or x == 0 for x in [val, base]):
        return float('nan')
    return round((val / base) ** -1, 3)

def safe_wpi_diff(basewpi, wpi):
    if any(pd.isna(x) or x == 0 for x in [basewpi, wpi]):
        return float('nan')
    return round(basewpi / wpi, 3)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    df_raw     = load_bowling('test_bowling_innings.csv')
    df_batting = load_batting_matched('batting_matched.csv')

    st.markdown("---")
    st.markdown("### 🔍 Filters")
    st.caption("Filters apply to all innings used to compute each bowler's stats.")

    opps_all = sorted(df_raw['team_bat'].dropna().unique().tolist())
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

    sel_home_away = st.multiselect("Home / Away", options=["Home", "Away"])

    result_options = ["Won", "Lost", "Draw / No result"] if 'winner' in df_raw.columns else []
    sel_result = st.multiselect("Match result", options=result_options)

    col3, col4 = st.columns(2)
    with col3:
        filter_captain = st.checkbox("As captain")
    with col4:
        sel_bkind = st.multiselect("Kind", options=["pace", "spin"])

    min_inns = st.number_input("Minimum innings (to appear)", min_value=1, value=20, step=1)
    min_wkts = st.number_input("Minimum wickets (to appear)", min_value=0, value=50, step=1)
    min_wpt = st.number_input("Minimum wickets per innings (to appear)",min_value=1,value=1.5)

    st.markdown("---")
    run_query = st.button("⚡ Run Comparison")

# ── Filter logic ──────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    if sel_opps:
        mask &= df['team_bat'].isin(sel_opps)
    if sel_countries:
        mask &= df['country'].isin(sel_countries)
    if sel_grounds:
        mask &= df['ground'].isin(sel_grounds)
    if 'year' in df.columns:
        mask &= df['year'].between(sel_years[0], sel_years[1])
    if sel_inns and 'inns' in df.columns:
        inns_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        mask &= df['inns'].isin([inns_map[i] for i in sel_inns])
    if sel_home_away and 'home_away' in df.columns:
        mask &= df['home_away'].isin(sel_home_away)
    if sel_bkind and 'bkind' in df.columns:
        mask &= df['bkind'].isin(sel_bkind)
    if sel_result and 'winner' in df.columns and 'team_bowl' in df.columns:
        result_mask = pd.Series([False] * len(df), index=df.index)
        if "Won" in sel_result:
            result_mask |= (df['winner'] == df['team_bowl'])
        if "Lost" in sel_result:
            result_mask |= (df['winner'] != df['team_bowl']) & df['winner'].notna() & (df['winner'] != '')
        if "Draw / No result" in sel_result:
            result_mask |= df['winner'].isna() | (df['winner'] == '')
        mask &= result_mask
    if filter_captain and 'player_role_type' in df.columns:
        mask &= df['player_role_type'].isin(['C', 'CWK'])
    return df[mask].copy()

# ── Victim lookup ─────────────────────────────────────────────────────────────
@st.cache_data
def prepare_victims(batting: pd.DataFrame) -> pd.DataFrame:
    """Pre-filter batting to only dismissed rows with a named bowler."""
    return batting[
        batting['dismissal_bowler_filled'].notna() &
        batting['is_out'].fillna(0).astype(int).eq(1)
    ][['p_match', 'inns', 'dismissal_bowler_filled',
       'batting_position', 'dismissal_type_short']].copy()

def get_victims_for_bowler(dismissed: pd.DataFrame, bowl_name: str,
                            match_inns_set: set) -> pd.DataFrame:
    """
    Get dismissed batters for a specific bowler name,
    restricted to (p_match, inns) combinations in match_inns_set.
    """
    bv = dismissed[dismissed['dismissal_bowler_filled'] == bowl_name].copy()
    if bv.empty:
        return bv
    bv['_key'] = list(zip(bv['p_match'], bv['inns']))
    return bv[bv['_key'].isin(match_inns_set)].drop(columns=['_key'])

# ── Build comparison table ────────────────────────────────────────────────────
def build_comparison(df: pd.DataFrame, dismissed: pd.DataFrame,
                     min_inns: int, min_wkts: int) -> pd.DataFrame:
    rows = []
    for bowler_id, g in df.groupby('p_bowl'):
        inns    = len(g)
        runs    = int(g['runs_conceded'].fillna(0).sum())
        balls   = int(g['balls'].fillna(0).sum())
        wickets = int(g['wickets'].fillna(0).sum())

        if inns < min_inns or wickets < min_wkts or (1.0*wickets/inns) < min_wpt :
            continue

        name    = g['bowl'].iloc[0]
        maidens = int(g['maidens'].fillna(0).sum())

        ave  = safe(runs,    wickets)
        sr   = safe(balls,   wickets)
        econ = round(runs / balls * 6, 2) if balls > 0 else float('nan')
        wpi  = safe(wickets, inns)

        fwi = int((g['wickets'].fillna(0) >= 5).sum())
        twm = int((g.groupby('p_match')['wickets'].sum() >= 10).sum()) \
              if 'p_match' in g.columns else 0

        baseav  = g['baseav'].mean()  if 'baseav'  in g.columns else float('nan')
        basesr  = g['basesr'].mean()  if 'basesr'  in g.columns else float('nan')
        basewpi = g['basewpi'].mean() if 'basewpi' in g.columns else float('nan')

        # Match factor ave and SR (no inversion — higher = better)
        g_runs  = g['runs_conceded'].fillna(0).values
        g_balls = g['balls'].fillna(0).values
        g_wkts  = g['wickets'].fillna(0).values

        if 'matchav_inv' in g.columns:
            wkts_per_run   = np.where(g_runs  > 0, g_wkts / g_runs,  np.nan)
            mf_ave_ratios  = wkts_per_run / g['matchav_inv'].values
            mf_ave_mean    = np.nanmean(mf_ave_ratios)
            mf_ave = round(mf_ave_mean, 2) if not np.isnan(mf_ave_mean) else float('nan')
        else:
            mf_ave = float('nan')

        if 'matchsr_inv' in g.columns:
            wkts_per_ball  = np.where(g_balls > 0, g_wkts / g_balls, np.nan)
            mf_sr_ratios   = wkts_per_ball / g['matchsr_inv'].values
            mf_sr_mean     = np.nanmean(mf_sr_ratios)
            mf_sr = round(mf_sr_mean, 2) if not np.isnan(mf_sr_mean) else float('nan')
        else:
            mf_sr = float('nan')

        # Victim stats — restrict to (p_match, inns) in this bowler's filtered spells
        match_inns_set = set(zip(g['p_match'], g['inns']))
        bv = get_victims_for_bowler(dismissed, name, match_inns_set)
        n_victims = len(bv)

        top7_pct = round(
            (bv['batting_position'] <= 7).sum() / n_victims * 100, 1
        ) if n_victims > 0 else float('nan')

        clean_types = ['bowled', 'lbw', 'caught keeper']
        clean_pct = round(
            bv['dismissal_type_short'].isin(clean_types).sum() / n_victims * 100, 1
        ) if n_victims > 0 else float('nan')

        rows.append({
            'Bowler':   name,
            'Inns':     inns,
            'Balls':    balls,
            'Runs':     runs,
            'Wkts':     wickets,
            'Ave':      ave,
            'SR':       sr,
            'Econ':     econ,
            'Wkts/Inn': wpi,
            'Ave diff': safe_diff(ave, baseav),
            'SR diff':  safe_diff(sr,  basesr),
            'WPI diff': safe_wpi_diff(wpi, basewpi),
            'MF Ave':   mf_ave,
            'MF SR':    mf_sr,
            '5WI':      fwi,
            '10WM':     twm,
            'Top-7 %':  top7_pct,
            'Clean %':  clean_pct,
        })

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
            .sort_values('Wkts', ascending=False)
            .reset_index(drop=True))

# ── Main ──────────────────────────────────────────────────────────────────────
if run_query:
    df_filtered = apply_filters(df_raw)
    n_innings   = len(df_filtered)
    n_bowlers   = df_filtered['p_bowl'].nunique()

    col_a, col_b = st.columns(2)
    col_a.metric("Innings matched", f"{n_innings:,}")
    col_b.metric("Bowlers",         f"{n_bowlers:,}")

    if n_innings == 0:
        st.info("No innings match your filters — try relaxing the criteria.")
    else:
        dismissed = prepare_victims(df_batting)
        tbl = build_comparison(df_filtered, dismissed, int(min_inns), int(min_wkts))

        if tbl.empty:
            st.info(f"No bowler meets the minimum thresholds ({int(min_inns)} innings, {int(min_wkts)} wickets).")
        else:
            st.markdown(f"**{len(tbl)} bowlers** · ≥ {int(min_inns)} innings · ≥ {int(min_wkts)} wickets · sorted by wickets")
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            csv_out = tbl.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download comparison CSV", data=csv_out,
                               file_name="statsbench_bowlcomp.csv", mime="text/csv")
else:
    st.info("Set your filters in the sidebar, then hit **Run Comparison**.")