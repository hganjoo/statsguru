import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatsBench · Bowling",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("StatsBench · Bowling")
st.caption("Test Cricket · Innings Query Engine")
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

    # ── Home / Away ──
    NEUTRAL_MAP = {'United Arab Emirates': 'Pakistan'}
    def get_home_away(row):
        home_country = NEUTRAL_MAP.get(row['country'], row['country'])
        return 'Home' if row['team_bowl'] == home_country else 'Away'
    df['home_away'] = df.apply(get_home_away, axis=1)

    # ── Bowling kind ──
    def getbtype(s):
        if pd.isna(s): return None
        return 'pace' if 'f' in str(s) else 'spin'
    df['bkind'] = df['bowling_kind'].apply(getbtype)

    # ── Era block ──
    seasons_ordered = sorted(df['season'].dropna().unique())
    season_rank = {s: i for i, s in enumerate(seasons_ordered)}
    df['season_rank'] = df['season'].map(season_rank)
    df['era_block'] = (df['season_rank'] // block_size) * block_size

    # ── Baselines: ave, sr, wpi grouped by era_block + bkind + country ──
    baseg   = df.groupby(['era_block', 'bkind', 'country'])
    baseav  = (baseg['runs_conceded'].sum() / baseg['wickets'].sum()).reset_index(name='baseav')
    basesr  = (baseg['balls'].sum()         / baseg['wickets'].sum()).reset_index(name='basesr')
    basewpi = (baseg['wickets'].sum()        / baseg['wickets'].count()).reset_index(name='basewpi')

    base = baseav.merge(basesr,  on=['era_block', 'bkind', 'country'])
    base = base.merge(basewpi, on=['era_block', 'bkind', 'country'])
    df   = df.merge(base, on=['era_block', 'bkind', 'country'], how='left')

    return df

@st.cache_data
def load_batting_matched(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ['runs', 'is_out', 'batting_position', 'inns']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    df_raw     = load_bowling('test_bowling_innings.csv')
    df_batting = load_batting_matched('batting_matched.csv')

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    bowlers_all = sorted(df_raw['bowl'].dropna().unique().tolist()) if 'bowl' in df_raw.columns else []
    sel_bowlers = st.multiselect("Bowler", options=bowlers_all)

    opps_all = sorted(df_raw['team_bat'].dropna().unique().tolist()) if 'team_bat' in df_raw.columns else []
    sel_opps = st.multiselect("Opposition", options=opps_all)

    col1, col2 = st.columns(2)
    with col1:
        countries_all = sorted(df_raw['country'].dropna().unique().tolist()) if 'country' in df_raw.columns else []
        sel_countries = st.multiselect("Country", options=countries_all)
    with col2:
        grounds_all = sorted(df_raw['ground'].dropna().unique().tolist()) if 'ground' in df_raw.columns else []
        sel_grounds = st.multiselect("Ground", options=grounds_all)

    yr_min = int(df_raw['year'].min())
    yr_max = int(df_raw['year'].max())
    sel_years = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    inns_options = ["1st", "2nd", "3rd", "4th"]
    sel_inns = st.multiselect("Innings", options=inns_options)

    sel_home_away = st.multiselect("Home / Away", options=["Home", "Away"])

    result_options = ["Won", "Lost", "Draw / No result"] if 'winner' in df_raw.columns else []
    sel_result = st.multiselect("Match result", options=result_options)

    filter_captain = st.checkbox("As captain")
    min_balls = st.number_input("Minimum balls bowled", min_value=0, value=0, step=6)

    st.markdown("---")
    run_query = st.button("⚡ Run Query")

# ── Filter logic ──────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    if sel_bowlers and 'bowl' in df.columns:
        mask &= df['bowl'].isin(sel_bowlers)
    if sel_opps and 'team_bat' in df.columns:
        mask &= df['team_bat'].isin(sel_opps)
    if sel_countries and 'country' in df.columns:
        mask &= df['country'].isin(sel_countries)
    if sel_grounds and 'ground' in df.columns:
        mask &= df['ground'].isin(sel_grounds)
    if 'year' in df.columns:
        mask &= df['year'].between(sel_years[0], sel_years[1])
    if sel_inns and 'inns' in df.columns:
        inns_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        mask &= df['inns'].isin([inns_map[i] for i in sel_inns])
    if sel_home_away and 'home_away' in df.columns:
        mask &= df['home_away'].isin(sel_home_away)
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
    if min_balls > 0 and 'balls' in df.columns:
        mask &= df['balls'] >= min_balls
    return df[mask].copy()

# ── Victim distribution helpers ───────────────────────────────────────────────
def get_victims(bowling_df: pd.DataFrame, batting: pd.DataFrame) -> pd.DataFrame:
    dismissed = batting[
        batting['dismissal_bowler_filled'].notna() &
        batting['is_out'].fillna(0).astype(int).eq(1)
    ][['p_match', 'inns', 'dismissal_bowler_filled',
       'dismissal_type_short', 'batting_position']].copy()
    bowl_keys = (bowling_df[['p_match', 'inns', 'bowl']]
                 .drop_duplicates()
                 .rename(columns={'bowl': 'dismissal_bowler_filled'}))
    return dismissed.merge(bowl_keys, on=['p_match', 'inns', 'dismissal_bowler_filled'], how='inner')

def dismissal_type_table(victims: pd.DataFrame) -> pd.DataFrame:
    if victims.empty or 'dismissal_type_short' not in victims.columns:
        return pd.DataFrame()
    counts = victims['dismissal_type_short'].value_counts().reset_index()
    counts.columns = ['Dismissal type', 'Count']
    counts['%'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)
    return counts

def batting_pos_table(victims: pd.DataFrame) -> pd.DataFrame:
    if victims.empty or 'batting_position' not in victims.columns:
        return pd.DataFrame()
    counts = victims['batting_position'].value_counts().sort_index().reset_index()
    counts.columns = ['Batting pos', 'Count']
    counts['%'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)
    return counts

# ── Bowling aggregation helpers ───────────────────────────────────────────────
def safe(n, d, decimals=2):
    """Safe division, rounded."""
    return round(n / d, decimals) if d > 0 and not np.isnan(d) else float('nan')

def safe_diff(val, base):
    """Compute (val/base)^-1, returns nan if either is nan/zero."""
    if any(pd.isna(x) or x == 0 for x in [val, base]):
        return float('nan')
    return round((val / base) ** -1, 3)

def agg_bowling(g: pd.DataFrame):
    """Aggregate a group into bowling stats + diffs. Returns a dict."""
    runs    = g['runs_conceded'].fillna(0)
    balls   = g['balls'].fillna(0)
    wickets = g['wickets'].fillna(0)
    maidens = g['maidens'].fillna(0)

    total_runs    = int(runs.sum())
    total_balls   = int(balls.sum())
    total_wickets = int(wickets.sum())
    total_maidens = int(maidens.sum())
    inns          = len(g)

    ave  = safe(total_runs,  total_wickets)
    sr   = safe(total_balls, total_wickets)
    econ = round(total_runs / total_balls * 6, 2) if total_balls > 0 else float('nan')
    wpi  = safe(total_wickets, inns)

    # Baseline means for this group
    baseav  = g['baseav'].mean()  if 'baseav'  in g.columns else float('nan')
    basesr  = g['basesr'].mean()  if 'basesr'  in g.columns else float('nan')
    basewpi = g['basewpi'].mean() if 'basewpi' in g.columns else float('nan')

    return {
        'inns':     inns,
        'balls':    total_balls,
        'runs':     total_runs,
        'wickets':  total_wickets,
        'maidens':  total_maidens,
        'ave':      ave,
        'sr':       sr,
        'econ':     econ,
        'wpi':      wpi,
        'ave_diff': safe_diff(ave,  baseav),
        'sr_diff':  safe_diff(sr,   basesr),
        'wpi_diff': safe_diff(basewpi,  wpi),
        '5wi':      int((wickets >= 5).sum()),
        '10wm':     int((g.groupby('p_match')['wickets'].sum() >= 10).sum())
                    if 'p_match' in g.columns else 0,
    }

def show_summary_strip(df):
    s = agg_bowling(df)

    # Row 1: raw stats
    row1 = {
        "Innings":  s['inns'],
        "Balls":    s['balls'],
        "Runs":     s['runs'],
        "Wickets":  s['wickets'],
        "Ave":      s['ave'],
        "SR":       s['sr'],
        "Econ":     s['econ'],
        "Wkts/Inn": s['wpi'],
        "5WI":      s['5wi'],
        "10WM":     s['10wm'],
    }
    cols = st.columns(len(row1))
    for col, (label, val) in zip(cols, row1.items()):
        col.metric(label, val)

    # Row 2: scaled diffs
    row2 = {
        "Ave diff":  s['ave_diff'],
        "SR diff":   s['sr_diff'],
        "WPI diff":  s['wpi_diff'],
    }
    st.caption("Scaled vs era/kind/country baseline — higher is better")
    cols2 = st.columns(len(row2))
    for col, (label, val) in zip(cols2, row2.items()):
        col.metric(label, val if not pd.isna(val) else "—")

def agg_stats_grouped(df: pd.DataFrame, group_col: str, group_label: str,
                       sort_by_label: bool = False) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for grp_val, g in df.groupby(group_col):
        s = agg_bowling(g)
        rows.append({
            group_label:  grp_val,
            'Inns':       s['inns'],
            'Balls':      s['balls'],
            'Runs':       s['runs'],
            'Wkts':       s['wickets'],
            'Ave':        s['ave'],
            'SR':         s['sr'],
            'Econ':       s['econ'],
            'Wkts/Inn':   s['wpi'],
            'Ave diff':   s['ave_diff'],
            'SR diff':    s['sr_diff'],
            'WPI diff':   s['wpi_diff'],
            '5WI':        s['5wi'],
            '10WM':       s['10wm'],
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(group_label if sort_by_label else 'Wkts',
                                    ascending=sort_by_label)
    return result

# ── Main ──────────────────────────────────────────────────────────────────────
if run_query:
    df_filtered = apply_filters(df_raw)
    n = len(df_filtered)
    st.metric("Innings matched", f"{n:,}")

    if n == 0:
        st.info("No innings match your filters — try relaxing the criteria.")
    else:
        show_summary_strip(df_filtered)
        st.divider()

        tab_inns, tab_career, tab_avg = st.tabs(["📋 Innings list", "📊 Career summary", "📈 Averages"])

        victims = get_victims(df_filtered, df_batting)

        # ── Tab 1: Innings list ───────────────────────────────────────────────
        with tab_inns:
            d_tbl = dismissal_type_table(victims)
            p_tbl = batting_pos_table(victims)

            if not d_tbl.empty:
                st.markdown("**Wickets by dismissal type**")
                st.dataframe(d_tbl, use_container_width=True, hide_index=True)
            if not p_tbl.empty:
                st.markdown("**Wickets by batting position**")
                st.dataframe(p_tbl, use_container_width=True, hide_index=True)
            st.write("")

            DISPLAY_COLS = [
                'bowl', 'team_bowl', 'team_bat', 'date', 'ground', 'country',
                'inns', 'overs', 'balls', 'maidens', 'runs_conceded', 'wickets',
                'ave', 'sr', 'econ', 'dots',
                'fours_conceded', 'sixes_conceded', 'wides', 'noballs',
                'winner', 'season', 'home_away', 'bkind',
            ]
            col_rename = {
                'bowl': 'Bowler', 'team_bowl': 'Team', 'team_bat': 'Opposition',
                'date': 'Date', 'ground': 'Ground', 'country': 'Country',
                'inns': 'Inns', 'overs': 'Overs', 'balls': 'Balls',
                'maidens': 'Mdns', 'runs_conceded': 'Runs', 'wickets': 'Wkts',
                'ave': 'Ave', 'sr': 'SR', 'econ': 'Econ', 'dots': 'Dots',
                'fours_conceded': '4s', 'sixes_conceded': '6s',
                'wides': 'Wides', 'noballs': 'NBs',
                'winner': 'Winner', 'season': 'Season',
                'home_away': 'H/A', 'bkind': 'Kind',
            }
            display_cols = [c for c in DISPLAY_COLS if c in df_filtered.columns]
            df_show = df_filtered[display_cols].rename(columns=col_rename)
            df_show = df_show.sort_values('Date', ascending=False) if 'Date' in df_show.columns else df_show
            for c in ['Ave', 'SR', 'Econ']:
                if c in df_show.columns:
                    df_show[c] = df_show[c].round(2)
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            csv_out = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download filtered CSV", data=csv_out,
                               file_name="statsbench_bowling_filtered.csv", mime="text/csv")

        # ── Tab 2: Career summary ─────────────────────────────────────────────
        with tab_career:
            if 'winner' in df_filtered.columns and 'team_bowl' in df_filtered.columns:
                df_filtered = df_filtered.copy()
                def result_label(row):
                    if pd.isna(row['winner']) or row['winner'] == '':
                        return 'Draw / NR'
                    return 'Won' if row['winner'] == row['team_bowl'] else 'Lost'
                df_filtered['match_result_label'] = df_filtered.apply(result_label, axis=1)

            GROUPINGS = {
                "By opposition":     ('team_bat',           'Opposition', False),
                "By country":        ('country',            'Country',    False),
                "By innings number": ('inns',               'Inns no',    True),
                "By season":         ('season',             'Season',     True),
                "By match result":   ('match_result_label', 'Result',     False),
                "By home/away":      ('home_away',          'H/A',        False),
            }

            for section_title, (col, label, sort_lbl) in GROUPINGS.items():
                tbl = agg_stats_grouped(df_filtered, col, label, sort_by_label=sort_lbl)
                if tbl.empty:
                    continue
                st.subheader(section_title)
                st.dataframe(tbl, use_container_width=True, hide_index=True)
                st.write("")

        # ── Tab 3: Averages over time ─────────────────────────────────────────
        with tab_avg:
            import altair as alt

            df_avg = df_filtered.copy()
            if 'date' in df_avg.columns:
                df_avg['date'] = pd.to_datetime(df_avg['date'], errors='coerce')
                df_avg = df_avg.sort_values('date').reset_index(drop=True)
            else:
                df_avg = df_avg.reset_index(drop=True)

            df_avg['inns_num']  = df_avg.index + 1
            df_avg['cum_runs']  = df_avg['runs_conceded'].fillna(0).cumsum()
            df_avg['cum_balls'] = df_avg['balls'].fillna(0).cumsum()
            df_avg['cum_wkts']  = df_avg['wickets'].fillna(0).cumsum()
            df_avg['cum_ave']   = df_avg['cum_runs']  / df_avg['cum_wkts'].replace(0, np.nan)
            df_avg['cum_sr']    = df_avg['cum_balls'] / df_avg['cum_wkts'].replace(0, np.nan)
            df_avg['cum_econ']  = df_avg['cum_runs']  / df_avg['cum_balls'].replace(0, np.nan) * 6

            WINDOW = 10
            df_avg['roll_runs']  = df_avg['runs_conceded'].fillna(0).rolling(WINDOW).sum()
            df_avg['roll_balls'] = df_avg['balls'].fillna(0).rolling(WINDOW).sum()
            df_avg['roll_wkts']  = df_avg['wickets'].fillna(0).rolling(WINDOW).sum()
            df_avg['roll_ave']   = df_avg['roll_runs'] / df_avg['roll_wkts'].replace(0, np.nan)

            base = alt.Chart(df_avg).encode(x=alt.X('inns_num:Q', title='Innings number'))
            tt   = ([alt.Tooltip('date:T', title='Date')] if 'date' in df_avg.columns else [])

            def line(field, color, dash, label):
                return base.mark_line(color=color, strokeDash=dash).encode(
                    y=alt.Y(f'{field}:Q', title='Average'),
                    tooltip=[alt.Tooltip('inns_num:Q', title='Inns'),
                             alt.Tooltip(f'{field}:Q', title=label, format='.2f')] + tt
                )

            combined = alt.layer(
                line('cum_ave',  '#1f77b4', [],     'Cum ave'),
                line('roll_ave', '#ff7f0e', [4, 2], f'Roll({WINDOW}) ave'),
            ).resolve_scale(y='shared').properties(
                width='container', height=420,
                title='Blue: cumulative ave  |  Orange: 10-inns rolling ave'
            )
            st.altair_chart(combined, use_container_width=True)

            col_sr, col_econ = st.columns(2)
            with col_sr:
                st.altair_chart(
                    alt.Chart(df_avg).mark_line(color='#2ca02c').encode(
                        x=alt.X('inns_num:Q', title='Innings'),
                        y=alt.Y('cum_sr:Q', title='Strike Rate'),
                        tooltip=[alt.Tooltip('inns_num:Q', title='Inns'),
                                 alt.Tooltip('cum_sr:Q', title='Cum SR', format='.1f')] + tt
                    ).properties(width='container', height=250, title='Cumulative SR'),
                    use_container_width=True
                )
            with col_econ:
                st.altair_chart(
                    alt.Chart(df_avg).mark_line(color='#9467bd').encode(
                        x=alt.X('inns_num:Q', title='Innings'),
                        y=alt.Y('cum_econ:Q', title='Economy'),
                        tooltip=[alt.Tooltip('inns_num:Q', title='Inns'),
                                 alt.Tooltip('cum_econ:Q', title='Cum Econ', format='.2f')] + tt
                    ).properties(width='container', height=250, title='Cumulative Economy'),
                    use_container_width=True
                )

            with st.expander("Show data table"):
                tbl_cols = ['inns_num'] + (['date'] if 'date' in df_avg.columns else []) + \
                           (['bowl'] if 'bowl' in df_avg.columns else []) + \
                           ['runs_conceded', 'wickets', 'cum_ave', 'roll_ave', 'cum_sr', 'cum_econ']
                tbl_rename = {
                    'inns_num': 'Inns', 'date': 'Date', 'bowl': 'Bowler',
                    'runs_conceded': 'Runs', 'wickets': 'Wkts',
                    'cum_ave': 'Cum ave', 'roll_ave': f'Roll({WINDOW}) ave',
                    'cum_sr': 'Cum SR', 'cum_econ': 'Cum Econ',
                }
                df_tbl = df_avg[[c for c in tbl_cols if c in df_avg.columns]].rename(columns=tbl_rename)
                st.dataframe(df_tbl.round(2), use_container_width=True, hide_index=True)

else:
    st.info("Configure your filters in the sidebar, then hit **Run Query**.")