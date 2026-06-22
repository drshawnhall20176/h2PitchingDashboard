import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import pytz

st.set_page_config(page_title="H2 Sports Pitching Engine (V6.1 Predictive)", layout="wide")


# --- 1. UTILITIES ---
def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default


@st.cache_data(ttl=300)
def fetch_json(url, params=None):
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}


# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_league_statcast_map():
    year = datetime.now().year
    data_adv = fetch_json(
        f"https://statsapi.mlb.com/api/v1/stats?stats=statcastAdvanced&group=pitching&playerPool=all&season={year}&gameType=R&limit=5000")
    data_exp = fetch_json(
        f"https://statsapi.mlb.com/api/v1/stats?stats=expectedStatistics&group=pitching&playerPool=all&season={year}&gameType=R&limit=5000")

    mapping = {}
    if 'stats' in data_adv:
        for group in data_adv['stats']:
            for split in group['splits']:
                p_id = str(split['player']['id'])
                if p_id not in mapping: mapping[p_id] = {}
                mapping[p_id]["BRL%"] = round(float(split['stat'].get('barrelPercentage', 0)), 1)
    if 'stats' in data_exp:
        for group in data_exp['stats']:
            for split in group['splits']:
                p_id = str(split['player']['id'])
                if p_id not in mapping: mapping[p_id] = {}
                mapping[p_id].update({
                    "xERA": float(split['stat'].get('estimatedEra', 0)),
                    "FIP": float(split['stat'].get('fip', 0))
                })
    return mapping


@st.cache_data(ttl=3600)
def get_live_probables():
    today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": today, "hydrate": "probablePitcher"}
    data = fetch_json(url, params=params)
    starters = []
    if 'dates' in data and data['dates']:
        for game in data['dates'][0]['games']:
            for side in ['away', 'home']:
                p = game['teams'][side].get('probablePitcher')
                if p:
                    starters.append({
                        "Pitcher": p['fullName'].strip(),
                        "Pitcher_ID": str(p['id']),
                        "Team": game['teams'][side]['team']['name'],
                        "Opponent": game['teams']['home' if side == 'away' else 'away']['team']['name']
                    })
    return pd.DataFrame(starters)


@st.cache_data(ttl=3600)
def get_pitcher_baseline_data(pitcher_ids):
    if not pitcher_ids: return {}
    year = datetime.now().year
    url = "https://statsapi.mlb.com/api/v1/people"
    params = {"personIds": ",".join(pitcher_ids),
              "hydrate": f"stats(group=[pitching],type=[season],season={year},gameType=R)"}
    data = fetch_json(url, params=params)
    stats_dict = {}
    if data and 'people' in data:
        for p in data['people']:
            p_id = str(p['id'])
            if 'stats' in p and p['stats']:
                s = p['stats'][0]['splits'][0]['stat']
                ip_val = safe_float(str(s.get('inningsPitched', "0")).replace('.1', '.333').replace('.2', '.666'))
                so = float(s.get('strikeOuts', 0))
                bb = float(s.get('baseOnBalls', 0))
                hr = float(s.get('homeRuns', 0))
                g = max(int(s.get('gamesPlayed', 1)), 1)

                manual_fip = ((13 * hr) + (3 * bb) - (2 * so)) / max(ip_val, 1) + 3.2

                stats_dict[p_id] = {
                    "ERA": safe_float(s.get('era')),
                    "WHIP": safe_float(s.get('whip')),
                    "K%": round((so / max(int(s.get('battersFaced', 1)), 1)) * 100, 1),
                    "IP/G": round(ip_val / g, 1),
                    "Manual_FIP": round(manual_fip, 2)
                }
    return stats_dict


# --- 3. ENGINE EXECUTION ---
st.title("🛡️ Pitching Engine (V6.1 Predictive)")
probables = get_live_probables()

if not probables.empty:
    with st.spinner("Compiling Telemetry..."):
        baseline = get_pitcher_baseline_data(probables['Pitcher_ID'].tolist())
        statcast_map = get_league_statcast_map()

        results = []
        for _, row in probables.iterrows():
            p_id = row['Pitcher_ID']
            s = baseline.get(p_id, {"ERA": 0.0, "K%": 0.0, "WHIP": 0.0, "IP/G": 0.0, "Manual_FIP": 0.0})
            sc = statcast_map.get(p_id, {"xERA": s.get('ERA', 0), "BRL%": 0.0, "FIP": s.get('Manual_FIP', 0)})

            active_fip = sc.get('FIP') if sc.get('FIP', 0) > 0 else s.get('Manual_FIP', 0)
            reg_idx = round(s.get('ERA', 0) - active_fip, 2)

            so_projection = round((s.get('K%', 0) / 100) * 28, 1)
            outs_projection = round(min(20.0, (s.get('IP/G', 0) * 3) * (1 / (1 + (s.get('WHIP', 1.25) - 1.25) * 0.2))),
                                    1)

            so_score = round((s.get('K%', 0) * 2.5) + (s.get('IP/G', 0) * 4.0), 1)
            outs_score = round((s.get('IP/G', 0) * 20.0) - (s.get('WHIP', 0) * 10.0), 1)
            fade_score = round((reg_idx * 20.0) + (sc.get('BRL%', 0) * 3.0), 1)

            results.append({**row, **s, **sc, "Reg_Idx": reg_idx, "SO Score": so_score, "Outs Score": outs_score,
                            "Fade Score": fade_score, "Proj SO": so_projection, "Proj Outs": outs_projection})

        df = pd.DataFrame(results)

        # Rankings logic
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("🔥 Strikeout Targets")
            st.dataframe(df.nlargest(5, "SO Score")[['Pitcher', 'Opponent', 'Proj SO', 'SO Score']].style.format(
                {"Proj SO": "{:.1f}", "SO Score": "{:.1f}"}), hide_index=True)
        with c2:
            st.subheader("🛡️ Outs Recorded")
            st.dataframe(df.nlargest(5, "Outs Score")[['Pitcher', 'Opponent', 'Proj Outs', 'Outs Score']].style.format(
                {"Proj Outs": "{:.1f}", "Outs Score": "{:.1f}"}), hide_index=True)
        with c3:
            st.subheader("🚨 Fade Targets (Regress)")
            st.dataframe(df.nlargest(5, "Fade Score")[['Pitcher', 'Opponent', 'Reg_Idx', 'Fade Score']].style.format(
                {"Reg_Idx": "{:.2f}", "Fade Score": "{:.1f}"}), hide_index=True)

        st.markdown("### 📋 Discord Master Export")
        st.code(f"**🎯 H2 SPORTS MASTER PITCHING PROPS**\n\n🔥 **Strikeout Ticket:**\n" + "\n".join(
            [f"{i + 1}. {r['Pitcher']} (Proj: {r['Proj SO']})" for i, r in
             enumerate(df.nlargest(5, "SO Score").to_dict('records'))]) +
                "\n\n🛡️ **Efficiency Ticket (Outs):**\n" + "\n".join(
            [f"{i + 1}. {r['Pitcher']} (Proj: {r['Proj Outs']})" for i, r in
             enumerate(df.nlargest(5, "Outs Score").to_dict('records'))]) +
                "\n\n🚨 **Fade Targets (High Regress):**\n" + "\n".join(
            [f"{i + 1}. {r['Pitcher']} (vs {r['Opponent']})" for i, r in
             enumerate(df.nlargest(5, "Fade Score").to_dict('records'))]), language="markdown")

        st.dataframe(df.drop(columns=['SO Score', 'Outs Score', 'Fade Score']), use_container_width=True,
                     hide_index=True)

        # --- STRATEGIST'S GUIDE ---
        with st.expander("💡 Strategist's Guide: How to Use This Data"):
            st.markdown("""
            **1. Interpreting Fade Targets (Regress)**
            *   **The Logic:** A positive `Reg_Idx` indicates a pitcher's actual ERA is significantly higher than their `FIP` (Fielding Independent Pitching). This means they are "lucky" (or benefiting from elite defense) and are prime candidates for regression—meaning their performance is likely to worsen in their next start.
            *   **The Play:** Use these pitchers for "Fade" spots in props (e.g., Under on Strikeouts, Over on Runs Allowed).
            *   **High Confidence:** Look for a high `Reg_Idx` combined with a high `BRL%` (Barrel Percentage). A pitcher who is both "lucky" and getting hit hard is the highest-value fade on the board.

            **2. Predictive Projections (Fair Value)**
            *   **`Proj SO`:** This is your "Fair Value" line. If the sportsbook sets a line at 4.5 but your `Proj SO` is 6.5, you have identified a strong "Over" opportunity.
            *   **`Proj Outs`:** We have applied a 20.0-out cap (6.2 innings) to ensure projections remain realistic. Look for discrepancies between this and the sportsbook's "Outs" line to find your edge.

            **3. Workflow Best Practice**
            *   Always cross-reference the `Reg_Idx` with the `Opponent`. Even a "Regress" candidate becomes risky if they are facing a weak, low-contact offense.
            """)
else:
    st.info("Waiting for API data...")