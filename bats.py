import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatsBench · Test Cricket",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("StatsBench")
st.caption("Test Cricket · Innings Query Engine")
st.divider()

# ── Position group helper ─────────────────────────────────────────────────────
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

    players_all = sorted(df_raw['bat'].dropna().unique().tolist()) if 'bat' in df_raw.columns else []
    sel_players = st.multiselect("Player", options=players_all)

    opps_all = sorted(df_raw['team_bowl'].dropna().unique().tolist()) if 'team_bowl' in df_raw.columns else []
    sel_opps = st.multiselect("Opposition", options=opps_all)

    col1, col2 = st.columns(2)
    with col1:
        countries_all = sorted(df_raw['country'].dropna().unique().tolist()) if 'country' in df_raw.columns else []
        sel_countries = st.multiselect("Country", options=countries_all)
    with col2:
        grounds_all = sorted(df_raw['ground'].dropna().unique().tolist()) if 'ground' in df_raw.columns else []
        sel_grounds = st.multiselect("Ground", options=grounds_all)

    if 'year' in df_raw.columns:
        yr_min = int(df_raw['year'].min())
        yr_max = int(df_raw['year'].max())
        sel_years = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
    else:
        sel_years = None

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

    result_options = ["Won", "Lost", "Draw / No result"] if 'winner' in df_raw.columns else []
    sel_result = st.multiselect("Match result", options=result_options)

    min_runs = st.number_input("Minimum runs", min_value=0, value=0, step=1)

    st.markdown("---")
    run_query = st.button("⚡ Run Query")

# ── Query logic ───────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)

    if sel_players and 'bat' in df.columns:
        mask &= df['bat'].isin(sel_players)
    if sel_opps and 'team_bowl' in df.columns:
        mask &= df['team_bowl'].isin(sel_opps)
    if sel_countries and 'country' in df.columns:
        mask &= df['country'].isin(sel_countries)
    if sel_grounds and 'ground' in df.columns:
        mask &= df['ground'].isin(sel_grounds)
    if sel_years and 'year' in df.columns:
        mask &= df['year'].between(sel_years[0], sel_years[1])
    if sel_inns and 'inns' in df.columns:
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
    if sel_result and 'winner' in df.columns and 'team_bat' in df.columns:
        result_mask = pd.Series([False] * len(df), index=df.index)
        if "Won" in sel_result:
            result_mask |= (df['winner'] == df['team_bat'])
        if "Lost" in sel_result:
            result_mask |= (df['winner'] != df['team_bat']) & df['winner'].notna() & (df['winner'] != '')
        if "Draw / No result" in sel_result:
            result_mask |= df['winner'].isna() | (df['winner'] == '')
        mask &= result_mask
    if 'runs' in df.columns:
        mask &= df['runs'] >= min_runs

    return df[mask].copy()

# ── Aggregation helpers ───────────────────────────────────────────────────────
def calc_aves(runs_sum, adj_era, adj_ctry, adj_pos, adj_opp, outs):
    def safe(n): return round(n / outs, 2) if outs > 0 and not pd.isna(n) else float('nan')
    return safe(runs_sum), safe(adj_era), safe(adj_ctry), safe(adj_pos), safe(adj_opp)

def agg_stats(df: pd.DataFrame, group_col: str, group_label: str, sort_by_label: bool = False) -> pd.DataFrame:
    if group_col not in df.columns or 'runs' not in df.columns:
        return pd.DataFrame()

    rows = []
    for grp_val, g in df.groupby(group_col):
        runs   = g['runs'].fillna(0)
        is_out = g['is_out'].fillna(0).astype(int)
        outs   = int(is_out.sum())
        inns   = len(g)
        

        ave, ave_era, ave_ctry, ave_pos, ave_opp = calc_aves(
            runs.sum(),
            g['adj_runs_era'].sum()         if 'adj_runs_era'         in g.columns else float('nan'),
            g['adj_runs_era_country'].sum() if 'adj_runs_era_country' in g.columns else float('nan'),
            g['adj_runs_era_pos'].sum()     if 'adj_runs_era_pos'     in g.columns else float('nan'),
            g['adj_runs_era_opp'].sum()     if 'adj_runs_era_opp'     in g.columns else float('nan'),
            outs,
        )
        team_pct    = round(g['team_runs_pct'].mean(), 1)  if 'team_runs_pct' in g.columns else float('nan')
        match_fac   = round(g['match_factor'].mean(),  2)  if 'match_factor'  in g.columns else float('nan')

        rows.append({
            group_label:    grp_val,
            'Inns':         inns,
            'NO':           inns - outs,
            'Runs':         int(runs.sum()),
            'HS':           int(runs.max()) if len(runs) else 0,
            'Avg':          ave,
            'Era-ave':      ave_era,
            'Ctry-ave':     ave_ctry,
            'Pos-ave':      ave_pos,
            'Opp-ave':      ave_opp,
            'Team%':        team_pct,
            'Match factor': match_fac,
            '100s':         int((runs >= 100).sum()),
            '50s':          int(((runs >= 50) & (runs < 100)).sum()),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(group_label if sort_by_label else 'Runs',
                                    ascending=sort_by_label)
    return result

# ── Summary strip ─────────────────────────────────────────────────────────────
def show_summary_strip(df):
    if 'runs' not in df.columns:
        return
    runs   = df['runs'].fillna(0)
    is_out = df['is_out'].fillna(0).astype(int)
    outs   = int(is_out.sum())

    ave, ave_era, ave_ctry, ave_pos, ave_opp = calc_aves(
        runs.sum(),
        df['adj_runs_era'].sum()         if 'adj_runs_era'         in df.columns else float('nan'),
        df['adj_runs_era_country'].sum() if 'adj_runs_era_country' in df.columns else float('nan'),
        df['adj_runs_era_pos'].sum()     if 'adj_runs_era_pos'     in df.columns else float('nan'),
        df['adj_runs_era_opp'].sum()     if 'adj_runs_era_opp'     in df.columns else float('nan'),
        outs,
    )

    stats = {
        "Innings":      len(df),
        "NO":           len(df) - outs,
        "Runs":         int(runs.sum()),
        "Avg":          ave,
        "Era-ave":      ave_era,
        "Ctry-ave":     ave_ctry,
        "Pos-ave":      ave_pos,
        "Opp-ave":      ave_opp,
        "Team%":        round(df['team_runs_pct'].mean(), 1) if 'team_runs_pct' in df.columns else float('nan'),
        "Match factor": round(df['match_factor'].mean(),  2) if 'match_factor'  in df.columns else float('nan'),
        "HS":           int(runs.max()) if len(runs) else 0,
        "100s":         int((runs >= 100).sum()),
        "50s":          int(((runs >= 50) & (runs < 100)).sum()),
    }
    cols = st.columns(len(stats))
    for col, (label, val) in zip(cols, stats.items()):
        col.metric(label, val)

# ── Main area ─────────────────────────────────────────────────────────────────
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

        # ── Tab 1: Innings list ───────────────────────────────────────────────
        with tab_inns:
            DISPLAY_COLS = [
                'bat', 'team_bat', 'team_bowl', 'date', 'ground', 'country','home_away',
                'inns', 'batting_position', 'runs', 'balls', 'minutes',
                'fours', 'sixes', 'strike_rate', 'is_out', 'dismissal_type_short',
                'dismissal_bowler', 'winner', 'season',
                'adj_runs_era', 'adj_runs_era_country', 'adj_runs_era_pos', 'adj_runs_era_opp',
                'team_runs_pct', 'match_factor',
            ]
            col_rename = {
                'bat': 'Batter', 'team_bat': 'Team', 'team_bowl': 'Opposition','home_away': 'H/A',
                'date': 'Date', 'ground': 'Ground', 'country': 'Country',
                'inns': 'Inns', 'batting_position': 'Pos', 'runs': 'Runs',
                'balls': 'Balls', 'minutes': 'Mins', 'fours': '4s', 'sixes': '6s',
                'strike_rate': 'SR', 'is_out': 'Out', 'dismissal_type_short': 'How out',
                'dismissal_bowler': 'Bowler', 'winner': 'Winner', 'season': 'Season',
                'adj_runs_era': 'Adj(era)', 'adj_runs_era_country': 'Adj(ctry)',
                'adj_runs_era_pos': 'Adj(pos)', 'adj_runs_era_opp': 'Adj(opp)',
                'team_runs_pct': 'Team%', 'match_factor': 'Match factor',
            }
            display_cols = [c for c in DISPLAY_COLS if c in df_filtered.columns]
            df_show = df_filtered[display_cols].rename(columns=col_rename)
            df_show = df_show.sort_values('Date', ascending=False) if 'Date' in df_show.columns else df_show
            for c in ['Adj(era)', 'Adj(ctry)', 'Adj(pos)', 'Adj(opp)']:
                if c in df_show.columns:
                    df_show[c] = df_show[c].round(1)
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            csv_out = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download filtered CSV", data=csv_out,
                               file_name="statsbench_filtered.csv", mime="text/csv")

        # ── Tab 2: Career summary ─────────────────────────────────────────────
        with tab_career:
            if 'winner' in df_filtered.columns and 'team_bat' in df_filtered.columns:
                df_filtered = df_filtered.copy()
                def result_label(row):
                    if pd.isna(row['winner']) or row['winner'] == '':
                        return 'Draw / NR'
                    return 'Won' if row['winner'] == row['team_bat'] else 'Lost'
                df_filtered['match_result_label'] = df_filtered.apply(result_label, axis=1)

            GROUPINGS = {
                "By opposition":       ('team_bowl',          'Opposition', False),
                "By country":          ('country',            'Country',    False),
                #"By ground":           ('ground',             'Ground',     False),
                "By home/away": ('home_away', 'Home/Away', False),
                "By batting position": ('batting_position',   'Bat pos',    True),
                "By innings number":   ('inns',               'Inns no',    True),
                "By season":           ('season',             'Season',     True),
                "By match result":     ('match_result_label', 'Result',     False),
            }

            for section_title, (col, label, sort_lbl) in GROUPINGS.items():
                tbl = agg_stats(df_filtered, col, label, sort_by_label=sort_lbl)
                if tbl.empty:
                    continue
                st.subheader(section_title)
                st.dataframe(tbl, use_container_width=True, hide_index=True)
                st.write("")

        # ── Tab 3: Averages over time ─────────────────────────────────────────
        with tab_avg:
            if 'runs' not in df_filtered.columns:
                st.info("Runs column required for this chart.")
            else:
                import altair as alt

                df_avg = df_filtered.copy()
                if 'date' in df_avg.columns:
                    df_avg['date'] = pd.to_datetime(df_avg['date'], errors='coerce')
                    df_avg = df_avg.sort_values('date').reset_index(drop=True)
                else:
                    df_avg = df_avg.reset_index(drop=True)

                df_avg['is_out_num'] = df_avg['is_out'].fillna(0).astype(int)
                df_avg['inns_num']   = df_avg.index + 1
                df_avg['cum_outs']   = df_avg['is_out_num'].cumsum()

                # Standard cumulative & rolling
                df_avg['cum_runs'] = df_avg['runs'].cumsum()
                df_avg['cum_avg']  = df_avg['cum_runs'] / df_avg['cum_outs'].replace(0, np.nan)

                WINDOW = 10
                df_avg['roll_runs'] = df_avg['runs'].rolling(WINDOW).sum()
                df_avg['roll_outs'] = df_avg['is_out_num'].rolling(WINDOW).sum()
                df_avg['roll_avg']  = df_avg['roll_runs'] / df_avg['roll_outs'].replace(0, np.nan)

                # Cumulative adjusted averages
                df_avg['cum_era_ave']  = df_avg['adj_runs_era'].cumsum()         / df_avg['cum_outs'].replace(0, np.nan)
                df_avg['cum_ctry_ave'] = df_avg['adj_runs_era_country'].cumsum() / df_avg['cum_outs'].replace(0, np.nan)
                df_avg['cum_pos_ave']  = df_avg['adj_runs_era_pos'].cumsum()     / df_avg['cum_outs'].replace(0, np.nan)
                df_avg['cum_opp_ave']  = df_avg['adj_runs_era_opp'].cumsum()     / df_avg['cum_outs'].replace(0, np.nan)

                base = alt.Chart(df_avg).encode(x=alt.X('inns_num:Q', title='Innings number'))
                tt   = ([alt.Tooltip('date:T', title='Date')] if 'date' in df_avg.columns else [])

                def line(field, color, dash, label):
                    return base.mark_line(color=color, strokeDash=dash).encode(
                        y=alt.Y(f'{field}:Q', title='Average'),
                        tooltip=[alt.Tooltip('inns_num:Q', title='Inns'),
                                 alt.Tooltip(f'{field}:Q', title=label, format='.2f')] + tt
                    )

                combined = alt.layer(
                    line('cum_avg',      '#1f77b4', [],      'Cum avg'),
                    line('roll_avg',     '#ff7f0e', [4, 2],  f'Roll({WINDOW})'),
                    line('cum_era_ave',  '#2ca02c', [2, 2],  'Era-ave'),
                    line('cum_ctry_ave', '#9467bd', [6, 2],  'Ctry-ave'),
                    line('cum_pos_ave',  '#d62728', [3, 3],  'Pos-ave'),
                    line('cum_opp_ave',  '#8c564b', [5, 3],  'Opp-ave'),
                ).resolve_scale(y='shared').properties(
                    width='container', height=420,
                    title='Blue: cum avg  |  Orange: 10-inns rolling  |  Green: era-ave  |  Purple: ctry-ave  |  Red: pos-ave  |  Brown: opp-ave'
                )

                st.altair_chart(combined, use_container_width=True)

                with st.expander("Show data table"):
                    tbl_cols = ['inns_num'] + (['date'] if 'date' in df_avg.columns else []) + \
                               (['bat'] if 'bat' in df_avg.columns else []) + \
                               ['runs', 'cum_avg', 'roll_avg', 'cum_era_ave', 'cum_ctry_ave', 'cum_pos_ave', 'cum_opp_ave']
                    tbl_rename = {
                        'inns_num': 'Inns', 'date': 'Date', 'bat': 'Batter', 'runs': 'Runs',
                        'cum_avg': 'Cum avg', 'roll_avg': f'Roll({WINDOW})',
                        'cum_era_ave': 'Era-ave', 'cum_ctry_ave': 'Ctry-ave',
                        'cum_pos_ave': 'Pos-ave', 'cum_opp_ave': 'Opp-ave',
                    }
                    df_tbl = df_avg[[c for c in tbl_cols if c in df_avg.columns]].rename(columns=tbl_rename)
                    st.dataframe(df_tbl.round(2), use_container_width=True, hide_index=True)

else:
    st.info("Configure your filters in the sidebar, then hit **Run Query**.")