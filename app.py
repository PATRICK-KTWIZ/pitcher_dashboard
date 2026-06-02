import os
import re
import hashlib
import json as _json
import pandas as pd
import numpy as np
from io import BytesIO
import requests

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

# ════════════════════════════════════════════════════════════
# 1. 설정
# ════════════════════════════════════════════════════════════
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OWNER    = "Henryjeon1"
REPO     = "ktdata"
TAG_NAME = "KoreaBaseballOrganization"

LEAGUE_FILES = {
    "KBO":   "KoreaBaseballOrganization.parquet",
    "NPB":   "NPB.parquet",
    "AAA":   "AAA.parquet",
    "Minor": "Minor.parquet",
}
LEAGUE_LABELS = {
    "KBO":   "KBO",
    "NPB":   "NPB",
    "AAA":   "AAA",
    "Minor": "KBO_Minor",
}
CACHE_DIR = "/tmp"
MIN_YEAR  = 2024

NAME_COL_MAP = {
    "KBO":   "NAME_pitcher",
    "Minor": "NAME_pitcher",
    "NPB":   "pitname",
    "AAA":   "pitname",
}

OUT_EVENTS = [
    "field_out", "strikeout", "grounded_into_double_play",
    "double_play", "force_out", "sac_fly", "sac_bunt",
    "fielders_choice_out", "strikeout_double_play",
    "other_out", "triple_play",
]

def get_name_col(league: str, df: pd.DataFrame) -> str:
    preferred = NAME_COL_MAP.get(league, "pitname")
    if preferred in df.columns:
        return preferred
    if "pitname" in df.columns:
        return "pitname"
    return None

# ════════════════════════════════════════════════════════════
# 2. 서버 사이드 DataFrame 캐시
# ════════════════════════════════════════════════════════════
_df_store: dict = {}

def _make_key(*args) -> str:
    raw = _json.dumps([str(a) for a in args], ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()

def _save(key: str, df: pd.DataFrame) -> str:
    _df_store[key] = df
    return key

def _load(key) -> pd.DataFrame:
    if not key:
        return pd.DataFrame()
    return _df_store.get(key, pd.DataFrame())

# ════════════════════════════════════════════════════════════
# 3. 데이터 로드
# ════════════════════════════════════════════════════════════
def load_league_data(league_name, min_year=MIN_YEAR):
    file_name  = LEAGUE_FILES[league_name]
    cache_path = os.path.join(CACHE_DIR, file_name)
    if os.path.exists(cache_path):
        print(f"[{league_name}] 로컬 캐시에서 로드 중...")
        df_tmp = pd.read_parquet(cache_path)
        return df_tmp[df_tmp["game_year"] >= min_year] if min_year else df_tmp
    print(f"[{league_name}] GitHub Release에서 다운로드 중...")
    release_url = (f"https://api.github.com/repos/{OWNER}/{REPO}"
                    f"/releases/tags/{TAG_NAME}")
    headers  = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    response = requests.get(release_url, headers=headers)
    if response.status_code != 200:
        print(f"Release 접근 실패: {response.status_code}")
        return pd.DataFrame()
    assets       = response.json().get("assets", [])
    target_asset = next((a for a in assets if a["name"] == file_name), None)
    if not target_asset:
        print(f"{file_name} 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    with requests.Session() as session:
        session.headers.update({
            "Accept": "application/octet-stream",
            "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
        })
        res = session.get(target_asset["url"])
        if res.status_code != 200:
            print(f"다운로드 실패: {res.status_code}")
            return pd.DataFrame()
        try:
            df_tmp = pd.read_parquet(BytesIO(res.content))
            df_tmp.to_parquet(cache_path, index=False)
            print(f"[{league_name}] 성공 ({df_tmp.shape[0]:,}행)")
            return df_tmp[df_tmp["game_year"] >= min_year] if min_year else df_tmp
        except Exception as e:
            print(f"파싱 실패: {e}")
            return pd.DataFrame()

league_cache: dict = {}
league_cache["KBO"] = load_league_data("KBO")

# ════════════════════════════════════════════════════════════
# 3-B. 리그 랭크 캐시
# ════════════════════════════════════════════════════════════
league_rank_cache: dict = {}

def build_league_rank_cache(league: str):
    df = league_cache.get(league, pd.DataFrame())
    if df.empty:
        return
    id_col = "pitcher" if "pitcher" in df.columns else get_name_col(league, df)
    if id_col is None:
        return
    fb_types       = ["4-Seam Fastball", "2-Seam Fastball", "Cutter"]
    result_by_year = {}
    for year, yr_df in df.groupby("game_year"):
        pa_df = yr_df[yr_df["events"].notna()].copy()
        pa_df["is_out"] = pa_df["events"].isin(OUT_EVENTS)
        pa_df["is_bb"]  = pa_df["events"] == "walk"
        pa_df["is_k"]   = pa_df["events"].isin(["strikeout", "strikeout_double_play"])
        pa_stats = pa_df.groupby(id_col).agg(
            outs=("is_out", "sum"),
            bb  =("is_bb",  "sum"),
            k   =("is_k",   "sum"),
        )
        pa_stats["ip"] = pa_stats["outs"] / 3
        if "hit" in yr_df.columns:
            hit_stats = yr_df.groupby(id_col)["hit"].sum().rename("hit")
            pa_stats  = pa_stats.join(hit_stats, how="left")
            pa_stats["hit"] = pa_stats["hit"].fillna(0)
        else:
            pa_stats["hit"] = 0
        fb_df = yr_df[yr_df["pitch_name"].isin(fb_types)]
        if not fb_df.empty and "rel_speed(km)" in fb_df.columns:
            spd_stats = fb_df.groupby(id_col)["rel_speed(km)"].mean().rename("spd")
            pa_stats  = pa_stats.join(spd_stats, how="left")
        else:
            pa_stats["spd"] = np.nan
        pa_stats = pa_stats[pa_stats["ip"] >= 20].copy()
        pa_stats["BB9"]  = (pa_stats["bb"]  / pa_stats["ip"] * 9).round(2)
        pa_stats["K9"]   = (pa_stats["k"]   / pa_stats["ip"] * 9).round(2)
        pa_stats["WHIP"] = ((pa_stats["hit"] + pa_stats["bb"]) / pa_stats["ip"]).round(2)
        pa_stats["SPD"]  = pa_stats["spd"].round(1)
        result_by_year[int(year)] = pa_stats[["BB9", "K9", "WHIP", "SPD"]].copy()
    league_rank_cache[league] = result_by_year
    print(f"[{league}] 랭크 캐시 완료: {list(result_by_year.keys())}")

build_league_rank_cache("KBO")

# ════════════════════════════════════════════════════════════
# 4. 투수 드롭다운 옵션 생성
# ════════════════════════════════════════════════════════════
THROWS_CANDIDATES = ["p_throws", "throws", "pitch_hand", "pitcher_throws"]

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_pitcher_options(df: pd.DataFrame, league: str = "KBO"):
    name_col    = get_name_col(league, df)
    throws_col  = find_col(df, THROWS_CANDIDATES)
    team_col    = "pitcherteam" if "pitcherteam" in df.columns else None
    latest_year = df["game_year"].max()
    if name_col is None:
        return []
    id_col = "pitcher" if "pitcher" in df.columns else name_col
    agg = {name_col: "first"}
    if throws_col: agg[throws_col] = lambda x: x.mode()[0] if len(x) > 0 else ""
    if team_col:   agg[team_col]   = lambda x: x.mode()[0] if len(x) > 0 else ""
    grp = df.groupby(id_col).agg(agg).reset_index()
    if team_col:
        latest_team = (
            df[df["game_year"] == latest_year]
            .groupby(id_col)[team_col]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else "")
            .rename("latest_team")
        )
        grp = grp.merge(latest_team, on=id_col, how="left")
        grp[team_col] = grp["latest_team"].fillna(grp[team_col])
        grp.drop(columns=["latest_team"], inplace=True)
    options = []
    for _, row in grp.iterrows():
        name  = row[name_col] if pd.notna(row[name_col]) else str(row[id_col])
        parts = []
        if throws_col and throws_col in row and pd.notna(row[throws_col]):
            parts.append(str(row[throws_col]))
        if team_col and team_col in row and pd.notna(row[team_col]):
            parts.append(str(row[team_col]))
        label = f"{name} ({', '.join(parts)})" if parts else name
        options.append({"label": label, "value": str(row[id_col])})
    options.sort(key=lambda x: x["label"])
    return options

def get_pitcher_name(df: pd.DataFrame, pitcher_id: str, league: str) -> str:
    name_col = get_name_col(league, df)
    if name_col is None:
        return pitcher_id
    id_col = "pitcher" if "pitcher" in df.columns else name_col
    sub = df[df[id_col].astype(str) == str(pitcher_id)]
    if sub.empty:
        return pitcher_id
    val = sub[name_col].iloc[0]
    return val if pd.notna(val) else pitcher_id

def get_pitcher_throws(df: pd.DataFrame, pitcher_id: str, league: str) -> str:
    throws_col = find_col(df, THROWS_CANDIDATES)   # build_pitcher_options와 동일
    if throws_col is None:
        return "R"
    id_col = "pitcher" if "pitcher" in df.columns else get_name_col(league, df)
    if id_col is None:
        return "R"
    sub = df[df[id_col].astype(str) == str(pitcher_id)]
    if sub.empty:
        return "R"
    val = sub[throws_col].mode()
    return str(val.iloc[0]).strip().upper() \
           if len(val) > 0 and str(val.iloc[0]).strip().upper() in ("R", "L") \
           else "R"


# ════════════════════════════════════════════════════════════
# 5. 스탯 계산
# ════════════════════════════════════════════════════════════
def calc_pitcher_stats(lg_df):
    if lg_df.empty:
        return pd.DataFrame()
    lg_df = lg_df.copy()
    lg_df["is_swinging_strike"] = lg_df["description"].isin(
        ["swinging_strike","swinging_strike_blocked","foul_tip"])
    lg_df["is_called_strike"]   = lg_df["description"] == "called_strike"
    lg_df["is_inplay"]          = lg_df["description"].isin(
        ["hit_into_play","hit_into_play_no_out","hit_into_play_score"])
    pa_df = lg_df[lg_df["events"].notna()].copy()
    pa_df["is_K"]   = pa_df["events"].isin(["strikeout","strikeout_double_play"])
    pa_df["is_BB"]  = pa_df["events"].isin(["walk","intent_walk"])
    pa_df["is_HR"]  = pa_df["events"] == "home_run"
    pa_df["is_HBP"] = pa_df["events"] == "hit_by_pitch"
    pa_df["is_out"] = pa_df["events"].isin(OUT_EVENTS)
    pitch_stats = lg_df.groupby(["game_year","pitcher","pitname"]).agg(
        total_pitch=("pitch_number","count"),
        swstr      =("is_swinging_strike","sum"),
        called_str =("is_called_strike","sum"),
    ).reset_index()
    pa_stats = pa_df.groupby(["game_year","pitcher","pitname"]).agg(
        PA  =("pa",    "sum"),
        K   =("is_K",  "sum"),
        BB  =("is_BB", "sum"),
        HR  =("is_HR", "sum"),
        HBP =("is_HBP","sum"),
        outs=("is_out","sum"),
    ).reset_index()
    inplay_df = lg_df[lg_df["is_inplay"]].copy()
    agg_dict  = {"inplay_count": ("is_inplay","sum")}
    if "exit_velocity" in inplay_df.columns: agg_dict["avg_EV"]     = ("exit_velocity","mean")
    if "launch_angleX" in inplay_df.columns: agg_dict["avg_LA"]     = ("launch_angleX","mean")
    if "plus_lsa4"     in inplay_df.columns: agg_dict["lsa4_count"] = ("plus_lsa4","sum")
    if "hit"           in inplay_df.columns: agg_dict["hit_count"]  = ("hit","sum")
    inplay_stats = inplay_df.groupby(
        ["game_year","pitcher","pitname"]).agg(**agg_dict).reset_index()
    if "lsa4_count" in inplay_stats.columns:
        inplay_stats["LSA4+%"] = (
            inplay_stats["lsa4_count"] / inplay_stats["inplay_count"] * 100).round(1)
    if "hit_count" in inplay_stats.columns:
        inplay_stats["InPlay_Hit%"] = (
            inplay_stats["hit_count"] / inplay_stats["inplay_count"] * 100).round(1)
    result = pitch_stats.merge(pa_stats,    on=["game_year","pitcher","pitname"], how="left")
    result = result.merge(inplay_stats,     on=["game_year","pitcher","pitname"], how="left")
    result["IP"]     = result["outs"] / 3
    result["K%"]     = (result["K"]  / result["PA"] * 100).round(1)
    result["BB%"]    = (result["BB"] / result["PA"] * 100).round(1)
    result["SwStr%"] = (result["swstr"] / result["total_pitch"] * 100).round(1)
    result["CSW%"]   = (
        (result["called_str"] + result["swstr"]) / result["total_pitch"] * 100).round(1)
    if "avg_EV" in result.columns: result["avg_EV"] = result["avg_EV"].round(1)
    if "avg_LA" in result.columns: result["avg_LA"] = result["avg_LA"].round(1)
    return result

# ════════════════════════════════════════════════════════════
# 6. 차트 상수
# ════════════════════════════════════════════════════════════
PITCH_ORDER = [
    "4-Seam Fastball","2-Seam Fastball","Cutter","Slider",
    "Curveball","Changeup","Split-Finger","Sweeper",
]
COLOR_MAP = {
    "4-Seam Fastball":"red",  "2-Seam Fastball":"pink",
    "Cutter":"purple",        "Slider":"blue",
    "Changeup":"green",       "Curveball":"orange",
    "Split-Finger":"brown",   "Sweeper":"yellow",
}
YEAR_COLORS = ["#E63946","#2196F3","#FF9800","#4CAF50","#9C27B0","#00BCD4"]

# ════════════════════════════════════════════════════════════
# 7-A. runner 분류 헬퍼
# ════════════════════════════════════════════════════════════
def split_by_runner(df: pd.DataFrame):
    if "runner" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    r = df["runner"].fillna("0").astype(str).str.strip()
    return df[r == "0"].copy(), df[r != "0"].copy()

# ════════════════════════════════════════════════════════════
# 7-B. 주요 스탯 헬퍼
# ════════════════════════════════════════════════════════════
def _calc_stats_row(df_src, label_val, label_name="시즌"):
    total_pitch = len(df_src)
    if total_pitch == 0:
        return None
    pa_df = df_src[df_src["events"].notna()]

    def col_sum(col):
        return int(df_src[col].sum()) if col in df_src.columns else 0
    def col_nansum(col):
        return df_src[col].sum() if col in df_src.columns else np.nan

    hit    = col_sum("hit");    ab     = col_sum("ab")
    single = col_sum("single"); double = col_sum("double")
    triple = col_sum("triple"); hr     = col_sum("home_run")
    sf     = col_sum("sac_fly")
    bb     = int((pa_df["events"] == "walk").sum())
    hbp    = int((pa_df["events"] == "hit_by_pitch").sum())
    k      = int(pa_df["events"].isin(["strikeout","strikeout_double_play"]).sum())
    outs   = int(pa_df["events"].isin(OUT_EVENTS).sum())
    ip     = outs / 3 if outs > 0 else 0
    swstr  = int(df_src["description"].isin(
        ["swinging_strike","swinging_strike_blocked","foul_tip"]).sum())
    cstr   = int((df_src["description"] == "called_strike").sum())
    z_in    = col_nansum("z_in");    z_swing = col_nansum("z_swing")
    z_con   = col_nansum("z_con");   z_out   = col_nansum("z_out")
    o_swing = col_nansum("o_swing"); o_con   = col_nansum("o_con")
    swing   = col_nansum("swing");   whiff   = col_nansum("whiff")
    if "f_pitch" in df_src.columns and "S" in df_src.columns:
        fp      = df_src[df_src["f_pitch"] == 1]
        f_s_pct = (fp["S"].sum() / len(fp) * 100) if len(fp) > 0 else np.nan
    else:
        f_s_pct = np.nan
    inplay   = df_src[df_src["description"].isin(
        ["hit_into_play","hit_into_play_no_out","hit_into_play_score"])]
    inplay_n = len(inplay)
    avg_ev   = inplay["exit_velocity"].mean() if "exit_velocity" in df_src.columns else np.nan
    avg_la   = inplay["launch_angleX"].mean()  if "launch_angleX" in df_src.columns else np.nan
    lsa4_cnt = int(inplay["plus_lsa4"].sum()) if "plus_lsa4" in inplay.columns else None
    lsa4_pct = (round(lsa4_cnt / inplay_n * 100, 1)
                if (lsa4_cnt is not None and inplay_n > 0) else np.nan)

    def sp(num, den):
        return round(float(num)/float(den)*100, 1) \
                if (pd.notna(num) and pd.notna(den) and den > 0) else np.nan

    avg     = round(hit/ab, 3)                             if ab > 0 else np.nan
    obp_den = ab + bb + hbp + sf
    obp     = round((hit+bb+hbp)/obp_den, 3)               if obp_den > 0 else np.nan
    slg     = round((single+2*double+3*triple+4*hr)/ab, 3) if ab > 0 else np.nan
    ops     = round(obp+slg, 3) if (pd.notna(obp) and pd.notna(slg)) else np.nan
    bb9     = round(bb/ip*9, 2) if ip > 0 else np.nan
    k9      = round(k /ip*9, 2) if ip > 0 else np.nan
    csw_pct = round((cstr+swstr)/total_pitch*100, 1)
    whip    = round((bb+hit)/ip, 2) if ip > 0 else np.nan

    def fmt(v, fs=None):
        if pd.isna(v): return "-"
        return fs.format(v) if fs else v

    return {
        label_name:  label_val,
        "타구속도":  fmt(avg_ev, "{:.1f}"),
        "발사각도":  fmt(avg_la, "{:.1f}"),
        "타율":      fmt(avg,    "{:.3f}"),
        "출루율":    fmt(obp,    "{:.3f}"),
        "장타율":    fmt(slg,    "{:.3f}"),
        "OPS":       fmt(ops,    "{:.3f}"),
        "LSA4+%":    fmt(lsa4_pct, "{:.1f}"),
        "BB/9":      fmt(bb9,    "{:.2f}"),
        "K/9":       fmt(k9,     "{:.2f}"),
        "CSW%":      csw_pct,
        "WHIP":      fmt(whip,   "{:.2f}"),
        "Z%":        fmt(sp(z_in,    total_pitch)),
        "Z_SW%":     fmt(sp(z_swing, z_in)),
        "Z_CON%":    fmt(sp(z_con,   z_swing)),
        "O%":        fmt(sp(z_out,   total_pitch)),
        "O_SW%":     fmt(sp(o_swing, z_out)),
        "O_CON%":    fmt(sp(o_con,   o_swing)),
        "F_Str%":    fmt(f_s_pct, "{:.1f}"),
        "SW%":       fmt(sp(swing,  total_pitch)),
        "WHIFF%":    fmt(sp(whiff,  swing)),
    }

LEAGUE_AVG_ROW = {
    "시즌":    "리그평균",
    "타구속도": "136.8",
    "발사각도": "9.8",
    "타율":    "0.261",
    "출루율":  "0.336",
    "장타율":  "0.387",
    "OPS":     "0.723",
    "LSA4+%":  "0.331",
    "BB/9":    "3.61",
    "K/9":     "7.77",
    "CSW%":    "28.1",
    "WHIP":    "1.41",
    "Z%":      "51.0",
    "Z_SW%":   "63.9",
    "Z_CON%":  "85.4",
    "O%":      "48.7",
    "O_SW%":   "26.2",
    "O_CON%":  "59.1",
    "F_Str%":  "59.4",
    "SW%":     "45.6",
    "WHIFF%":  "22.0",
}

# 스플릿별 리그 평균 (우투수/좌투수 × 우타/좌타/무주자/유주자)
LEAGUE_AVG_SPLIT = {
    "R": {  # 우투수
        "R":   {"시즌":"리그평균","타구속도":"136.6","발사각도":"11.2","타율":"0.253","출루율":"0.327","장타율":"0.383","OPS":"0.710","LSA4+%":"0.338","BB/9":"3.20","K/9":"7.98","CSW%":"28.7","WHIP":"1.31","Z%":"51.0","Z_SW%":"64.5","Z_CON%":"84.8","O%":"49.0","O_SW%":"27.3","O_CON%":"56.8","F_Str%":"27.6","SW%":"46.3","WHIFF%":"23.3"},
        "L":   {"시즌":"리그평균","타구속도":"137.6","발사각도":"9.1", "타율":"0.269","출루율":"0.346","장타율":"0.400","OPS":"0.746","LSA4+%":"0.331","BB/9":"3.82","K/9":"7.53","CSW%":"27.1","WHIP":"1.47","Z%":"51.5","Z_SW%":"64.7","Z_CON%":"85.8","O%":"48.5","O_SW%":"25.3","O_CON%":"61.0","F_Str%":"27.7","SW%":"45.6","WHIFF%":"20.9"},
        "무주자":{"시즌":"리그평균","타구속도":"138.0","발사각도":"10.5","타율":"0.251","출루율":"0.322","장타율":"0.381","OPS":"0.703","LSA4+%":"0.337","BB/9":"3.29","K/9":"8.44","CSW%":"29.2","WHIP":"1.39","Z%":"52.6","Z_SW%":"62.4","Z_CON%":"85.6","O%":"47.4","O_SW%":"25.7","O_CON%":"58.7","F_Str%":"24.3","SW%":"45.0","WHIFF%":"21.7"},
        "유주자":{"시즌":"리그평균","타구속도":"136.1","발사각도":"9.6", "타율":"0.273","출루율":"0.354","장타율":"0.404","OPS":"0.758","LSA4+%":"0.332","BB/9":"3.74","K/9":"7.03","CSW%":"26.5","WHIP":"1.40","Z%":"49.8","Z_SW%":"67.3","Z_CON%":"85.0","O%":"50.2","O_SW%":"26.9","O_CON%":"59.0","F_Str%":"31.4","SW%":"47.0","WHIFF%":"22.5"},
    },
    "L": {  # 좌투수
        "R":   {"시즌":"리그평균","타구속도":"138.5","발사각도":"11.2","타율":"0.260","출루율":"0.344","장타율":"0.391","OPS":"0.735","LSA4+%":"0.340","BB/9":"4.13","K/9":"7.80","CSW%":"28.2","WHIP":"1.45","Z%":"50.2","Z_SW%":"63.8","Z_CON%":"84.9","O%":"49.8","O_SW%":"26.8","O_CON%":"59.7","F_Str%":"25.2","SW%":"45.4","WHIFF%":"22.5"},
        "L":   {"시즌":"리그평균","타구속도":"133.1","발사각도":"6.0", "타율":"0.266","출루율":"0.341","장타율":"0.369","OPS":"0.710","LSA4+%":"0.301","BB/9":"3.58","K/9":"7.77","CSW%":"29.0","WHIP":"1.43","Z%":"52.7","Z_SW%":"60.5","Z_CON%":"86.4","O%":"47.3","O_SW%":"25.1","O_CON%":"59.4","F_Str%":"26.4","SW%":"43.8","WHIFF%":"20.9"},
        "무주자":{"시즌":"리그평균","타구속도":"136.2","발사각도":"8.8", "타율":"0.259","출루율":"0.336","장타율":"0.375","OPS":"0.711","LSA4+%":"0.328","BB/9":"3.86","K/9":"8.52","CSW%":"29.6","WHIP":"1.50","Z%":"52.3","Z_SW%":"59.7","Z_CON%":"86.3","O%":"47.7","O_SW%":"25.5","O_CON%":"59.7","F_Str%":"21.6","SW%":"43.4","WHIFF%":"21.1"},
        "유주자":{"시즌":"리그평균","타구속도":"135.8","발사각도":"8.9", "타율":"0.267","출루율":"0.350","장타율":"0.388","OPS":"0.738","LSA4+%":"0.316","BB/9":"3.91","K/9":"7.04","CSW%":"27.3","WHIP":"1.38","Z%":"50.2","Z_SW%":"65.4","Z_CON%":"84.7","O%":"49.8","O_SW%":"26.7","O_CON%":"59.4","F_Str%":"30.2","SW%":"46.1","WHIFF%":"22.6"},
    },
}


def make_key_stats_table(df, stand=None, runner=None, pitcher_throws="R"):
    """
    stand          : None(전체) / "L" / "R"
    runner         : None(전체) / "무주자" / "유주자"
    pitcher_throws : "R" / "L"
    """
    src = df
    if stand is not None:
        src = src[src["stand"] == stand]
    if runner == "무주자":
        src, _ = split_by_runner(src)
    elif runner == "유주자":
        _, src = split_by_runner(src)

    if src.empty:
        return pd.DataFrame()

    ys     = sorted(src["game_year"].unique())
    latest = ys[-1]
    prev   = ys[-2] if len(ys) >= 2 else None
    rows   = [r for yr in [y for y in [latest, prev] if y is not None]
                for r in [_calc_stats_row(src[src["game_year"] == yr], int(yr), "시즌")]
                if r]
    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows)

    # ── 리그 평균 행 선택 ──
    if stand is None and runner is None:
        # 통합 테이블 → 투수 유형 무관
        avg_src = LEAGUE_AVG_ROW
    elif runner is not None:
        # 주자 상황별 (무주자 / 유주자) → 투수 유형 적용
        avg_src = LEAGUE_AVG_SPLIT.get(pitcher_throws, {}).get(runner, LEAGUE_AVG_ROW)
    else:
        # 타자 유형별 (L / R) → 투수 유형 적용
        avg_src = LEAGUE_AVG_SPLIT.get(pitcher_throws, {}).get(stand, LEAGUE_AVG_ROW)

    league_row = {col: avg_src.get(col, "-") for col in df_result.columns}
    df_league  = pd.DataFrame([league_row])

    return pd.concat([df_league, df_result], ignore_index=True)


def make_pitch_stats_table(df, year, stand=None):
    df_yr = df[df["game_year"] == year]
    if stand is not None and stand != "ALL":
        df_yr = df_yr[df_yr["stand"] == stand]
    if df_yr.empty: return pd.DataFrame()
    rows = []
    for pname in PITCH_ORDER:
        df_p = df_yr[df_yr["pitch_name"] == pname]
        if df_p.empty: continue
        row = _calc_stats_row(df_p, pname, "구종")
        if row:
            row["투구수"] = len(df_p)
            rows.append(row)
    if not rows: return pd.DataFrame()
    df_r  = pd.DataFrame(rows)
    front = ["구종", "투구수"]
    rest  = [c for c in df_r.columns if c not in front]
    return df_r[front + rest]

def make_dashboard_stats_table(df: pd.DataFrame, base_df: pd.DataFrame,
                                is_date_filtered: bool):
    src   = base_df if (base_df is not None and not base_df.empty) else df
    years = sorted(src["game_year"].unique(), reverse=True)[:3]
    rows  = []
    if is_date_filtered and not df.empty:
        row_filtered = _calc_stats_row(df, "📌 필터적용", "시즌")
        if row_filtered:
            rows.append(row_filtered)
    for yr in years:
        row = _calc_stats_row(src[src["game_year"] == yr], int(yr), "시즌")
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows)

    # ── 리그 평균 행 맨 위에 삽입 ──
    league_row = {col: LEAGUE_AVG_ROW.get(col, "-") for col in df_result.columns}
    df_league  = pd.DataFrame([league_row])

    return pd.concat([df_league, df_result], ignore_index=True)

# ════════════════════════════════════════════════════════════
# 7-C. 주자 상황별 스탯 테이블 블록
# ════════════════════════════════════════════════════════════
def make_runner_stats_section(df: pd.DataFrame, latest_year, dcard_style,
                               id_suffix="", pitcher_throws="R"):   # ← 추가
    if "runner" not in df.columns:
        return html.Div("⚠ runner 컬럼이 없습니다.",
                        style={"color":"#888","padding":"10px","fontSize":"13px"})
    no_runner, yes_runner = split_by_runner(df)

    def _runner_block(src, label, emoji, color, id_label, runner_label):   # ← runner_label 추가
        if src.empty:
            return dbc.Card(dbc.CardBody(
                html.Div(f"{emoji} {label} 데이터 없음",
                        style={"color":"#888","fontSize":"13px"})
            ), className="mb-3", style=dcard_style)
        pitch_cnt = len(src)
        pa_cnt    = src["events"].notna().sum()
        uid = f"{id_label}{id_suffix}"
        return dbc.Card(dbc.CardBody([
            html.H6([
                html.Span(f"{emoji} {label}",
                        style={"color": color, "fontWeight":"700"}),
                html.Span(f"  (투구 {pitch_cnt:,}개 / PA {pa_cnt:,})",
                        style={"fontSize":"12px","color":"#888","marginLeft":"8px"}),
            ], style={"marginBottom":"8px"}),
            html.P("📊 주요 스탯",
                    style={"fontWeight":"600","fontSize":"12px",
                        "color":"#555","marginBottom":"4px"}),
            make_stat_table(
                make_key_stats_table(
                    src,
                    runner=runner_label,           # ← 추가
                    pitcher_throws=pitcher_throws  # ← 추가
                ),
                f"runner-key-{uid}"),
            html.P(f"🎯 구종별 스탯 ({latest_year})",
                    style={"fontWeight":"600","fontSize":"12px",
                        "color":"#555","marginTop":"12px","marginBottom":"4px"}),
            make_stat_table(make_pitch_stats_table(src, latest_year),
                            f"runner-pitch-{uid}"),
        ]), className="mb-3", style=dcard_style)

    return html.Div([
        dbc.Row([
            dbc.Col(_runner_block(no_runner,  "무주자","⬜","#2c7be5","no",  "무주자"), md=6),  # ← 추가
            dbc.Col(_runner_block(yes_runner, "유주자","🟩","#e55c2c","yes", "유주자"), md=6),  # ← 추가
        ], className="mb-2"),
    ])


# ════════════════════════════════════════════════════════════
# 7-D~F. 주자 상황 차트
# ════════════════════════════════════════════════════════════
def make_runner_usage_chart(df, pitcher_name):
    if "runner" not in df.columns:
        return go.Figure()
    no_runner, yes_runner = split_by_runner(df)
    latest_year = df["game_year"].max()

    def get_perc(src):
        if src.empty: return pd.Series(dtype=float)
        cnt = src["pitch_name"].value_counts()
        return (cnt / cnt.sum() * 100).round(1)

    p_no  = get_perc(no_runner)
    p_yes = get_perc(yes_runner)
    available = [p for p in PITCH_ORDER if p in set(p_no.index)|set(p_yes.index)]
    for p in available:
        if p not in p_no.index:  p_no[p]  = 0.0
        if p not in p_yes.index: p_yes[p] = 0.0
    y_base = np.arange(len(available))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_base+0.12, x=[-p_no.get(p,0) for p in available],
        orientation="h", name="무주자", marker_color="steelblue", width=0.35,
        text=[f"{p_no.get(p,0):.1f}%" if p_no.get(p,0)>0 else "" for p in available],
        textposition="outside"))
    fig.add_trace(go.Bar(
        y=y_base-0.12, x=[p_yes.get(p,0) for p in available],
        orientation="h", name="유주자", marker_color="tomato", width=0.35,
        text=[f"{p_yes.get(p,0):.1f}%" if p_yes.get(p,0)>0 else "" for p in available],
        textposition="outside"))
    fig.update_layout(
        title=dict(text=f"{pitcher_name}  무주자 vs 유주자 구종 사용 비율  ({latest_year})",
                    x=0.5, font=dict(size=13)),
        barmode="overlay", plot_bgcolor="white",
        xaxis=dict(tickmode="array", tickvals=[-60,-40,-20,0,20,40,60],
                    ticktext=["60%","40%","20%","0","20%","40%","60%"],
                    range=[-75,75], showgrid=True,
                    gridcolor="rgba(230,230,230,0.7)",
                    zeroline=True, zerolinecolor="gray", zerolinewidth=1),
        yaxis=dict(tickmode="array", tickvals=y_base, ticktext=available,
                    autorange="reversed", range=[-0.7, len(available)-0.3]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, yanchor="top"),
        margin=dict(l=100,r=30,t=70,b=60), autosize=True,
    )
    fig.add_annotation(x=-10, y=1.05, text="[무주자]", showarrow=False,
        xref="x", yref="paper", xanchor="right", yanchor="bottom",
        font=dict(color="steelblue", size=13))
    fig.add_annotation(x=10,  y=1.05, text="[유주자]", showarrow=False,
        xref="x", yref="paper", xanchor="left", yanchor="bottom",
        font=dict(color="tomato", size=13))
    return fig

def make_runner_usage_by_stand_chart(df, pitcher_name, runner_label="무주자"):
    if df.empty or "stand" not in df.columns:
        return go.Figure()
    latest_year = df["game_year"].max()

    def get_stand_perc(src, stand):
        sub = src[src["stand"] == stand]
        if sub.empty: return pd.Series(dtype=float)
        cnt = sub["pitch_name"].value_counts()
        return (cnt / cnt.sum() * 100).round(1)

    p_L = get_stand_perc(df, "L")
    p_R = get_stand_perc(df, "R")
    available = [p for p in PITCH_ORDER if p in set(p_L.index) | set(p_R.index)]
    for p in available:
        if p not in p_L.index: p_L[p] = 0.0
        if p not in p_R.index: p_R[p] = 0.0
    y_base = np.arange(len(available))
    cnt_L  = len(df[df["stand"] == "L"])
    cnt_R  = len(df[df["stand"] == "R"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_base+0.10, x=[-p_L.get(p,0) for p in available],
        orientation="h", name=f"좌타자 ({cnt_L}구)",
        marker_color="#3BBFB0", width=0.30,
        text=[f"{p_L.get(p,0):.1f}%" if p_L.get(p,0)>0 else "" for p in available],
        textposition="outside"))
    fig.add_trace(go.Bar(
        y=y_base-0.10, x=[p_R.get(p,0) for p in available],
        orientation="h", name=f"우타자 ({cnt_R}구)",
        marker_color="#e63946", width=0.30,
        text=[f"{p_R.get(p,0):.1f}%" if p_R.get(p,0)>0 else "" for p in available],
        textposition="outside"))
    fig.update_layout(
        title=dict(
            text=f"{pitcher_name}  {runner_label} — 좌/우타자별 구종 사용 비율  ({latest_year})",
            x=0.5, font=dict(size=12)),
        barmode="overlay", plot_bgcolor="white", autosize=True,
        xaxis=dict(tickmode="array", tickvals=[-60,-40,-20,0,20,40,60],
                    ticktext=["60%","40%","20%","0","20%","40%","60%"],
                    range=[-75,75], showgrid=True,
                    gridcolor="rgba(230,230,230,0.7)",
                    zeroline=True, zerolinecolor="gray", zerolinewidth=1),
        yaxis=dict(tickmode="array", tickvals=y_base, ticktext=available,
                    autorange="reversed", range=[-0.7, len(available)-0.3]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, yanchor="top"),
        margin=dict(l=100, r=30, t=60, b=60))
    fig.add_annotation(x=-10, y=1.06, text="[좌타자]", showarrow=False,
        xref="x", yref="paper", xanchor="right", yanchor="bottom",
        font=dict(color="#3BBFB0", size=12))
    fig.add_annotation(x=10,  y=1.06, text="[우타자]", showarrow=False,
        xref="x", yref="paper", xanchor="left", yanchor="bottom",
        font=dict(color="#e63946", size=12))
    return fig

COUNT_GROUPS = {
    "First Pitch (0-0)":             ["0-0"],
    "Two Strike (0-2/1-2/2-2/3-2)": ["0-2","1-2","2-2","3-2"],
    "Hitter's Count (1-0/2-0/2-1)": ["1-0","2-0","2-1"],
}
COUNT_SLUG = {
    "First Pitch (0-0)":             "fp",
    "Two Strike (0-2/1-2/2-2/3-2)": "ts",
    "Hitter's Count (1-0/2-0/2-1)": "hc",
}

def make_runner_count_usage_chart(df, pitcher_name, count_label, count_values):
    if "runner" not in df.columns or "count" not in df.columns:
        return go.Figure()
    src = df[df["count"].isin(count_values)].copy()
    if src.empty: return go.Figure()
    no_runner, yes_runner = split_by_runner(src)
    latest_year = df["game_year"].max()

    def get_perc(sub):
        if sub.empty: return pd.Series(dtype=float)
        cnt = sub["pitch_name"].value_counts()
        return (cnt / cnt.sum() * 100).round(1)

    p_no  = get_perc(no_runner)
    p_yes = get_perc(yes_runner)
    available = [p for p in PITCH_ORDER if p in set(p_no.index) | set(p_yes.index)]
    for p in available:
        if p not in p_no.index:  p_no[p]  = 0.0
        if p not in p_yes.index: p_yes[p] = 0.0
    y_base = np.arange(len(available))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_base+0.12, x=[-p_no.get(p,0) for p in available],
        orientation="h", name=f"무주자 ({len(no_runner)}구)",
        marker_color="steelblue", width=0.35,
        text=[f"{p_no.get(p,0):.1f}%" if p_no.get(p,0)>0 else "" for p in available],
        textposition="outside"))
    fig.add_trace(go.Bar(
        y=y_base-0.12, x=[p_yes.get(p,0) for p in available],
        orientation="h", name=f"유주자 ({len(yes_runner)}구)",
        marker_color="tomato", width=0.35,
        text=[f"{p_yes.get(p,0):.1f}%" if p_yes.get(p,0)>0 else "" for p in available],
        textposition="outside"))
    fig.update_layout(
        title=dict(text=f"{pitcher_name}  [{count_label}]  무주자 vs 유주자  ({latest_year})",
                    x=0.5, font=dict(size=12)),
        barmode="overlay", plot_bgcolor="white", autosize=True,
        xaxis=dict(tickmode="array", tickvals=[-60,-40,-20,0,20,40,60],
                    ticktext=["60%","40%","20%","0","20%","40%","60%"],
                    range=[-75,75], showgrid=True,
                    gridcolor="rgba(230,230,230,0.7)",
                    zeroline=True, zerolinecolor="gray", zerolinewidth=1),
        yaxis=dict(tickmode="array", tickvals=y_base, ticktext=available,
                    autorange="reversed", range=[-0.7, len(available)-0.3]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, yanchor="top"),
        margin=dict(l=100, r=30, t=60, b=60))
    fig.add_annotation(x=-10, y=1.06, text="[무주자]", showarrow=False,
        xref="x", yref="paper", xanchor="right", yanchor="bottom",
        font=dict(color="steelblue", size=12))
    fig.add_annotation(x=10,  y=1.06, text="[유주자]", showarrow=False,
        xref="x", yref="paper", xanchor="left", yanchor="bottom",
        font=dict(color="tomato", size=12))
    return fig

def make_runner_count_stand_chart(df, pitcher_name, count_values, runner_label, count_label):
    if df.empty or "stand" not in df.columns or "count" not in df.columns:
        return go.Figure()
    src = df[df["count"].isin(count_values)].copy()
    if src.empty: return go.Figure()
    latest_year = src["game_year"].max()

    def get_stand_perc(stand):
        sub = src[src["stand"] == stand]
        if sub.empty: return pd.Series(dtype=float)
        cnt = sub["pitch_name"].value_counts()
        return (cnt / cnt.sum() * 100).round(1)

    p_L = get_stand_perc("L")
    p_R = get_stand_perc("R")
    available = [p for p in PITCH_ORDER if p in set(p_L.index) | set(p_R.index)]
    for p in available:
        if p not in p_L.index: p_L[p] = 0.0
        if p not in p_R.index: p_R[p] = 0.0
    y_base = np.arange(len(available))
    cnt_L  = len(src[src["stand"] == "L"])
    cnt_R  = len(src[src["stand"] == "R"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_base+0.10, x=[-p_L.get(p,0) for p in available],
        orientation="h", name=f"좌타자 ({cnt_L}구)",
        marker_color="#3BBFB0", width=0.30,
        text=[f"{p_L.get(p,0):.1f}%" if p_L.get(p,0)>0 else "" for p in available],
        textposition="outside"))
    fig.add_trace(go.Bar(
        y=y_base-0.10, x=[p_R.get(p,0) for p in available],
        orientation="h", name=f"우타자 ({cnt_R}구)",
        marker_color="#e63946", width=0.30,
        text=[f"{p_R.get(p,0):.1f}%" if p_R.get(p,0)>0 else "" for p in available],
        textposition="outside"))
    fig.update_layout(
        title=dict(
            text=f"{pitcher_name}  [{count_label}]  {runner_label} — 좌/우타자별  ({latest_year})",
            x=0.5, font=dict(size=11)),
        barmode="overlay", plot_bgcolor="white", autosize=True,
        xaxis=dict(tickmode="array", tickvals=[-60,-40,-20,0,20,40,60],
                    ticktext=["60%","40%","20%","0","20%","40%","60%"],
                    range=[-75,75], showgrid=True,
                    gridcolor="rgba(230,230,230,0.7)",
                    zeroline=True, zerolinecolor="gray", zerolinewidth=1),
        yaxis=dict(tickmode="array", tickvals=y_base, ticktext=available,
                    autorange="reversed", range=[-0.7, len(available)-0.3]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, yanchor="top"),
        margin=dict(l=100, r=30, t=55, b=60))
    fig.add_annotation(x=-10, y=1.06, text="[좌타자]", showarrow=False,
        xref="x", yref="paper", xanchor="right", yanchor="bottom",
        font=dict(color="#3BBFB0", size=11))
    fig.add_annotation(x=10,  y=1.06, text="[우타자]", showarrow=False,
        xref="x", yref="paper", xanchor="left", yanchor="bottom",
        font=dict(color="#e63946", size=11))
    return fig

# ════════════════════════════════════════════════════════════
# 8-A. 무브먼트 차트
# ════════════════════════════════════════════════════════════
def make_movement_chart(filtered_df, pitcher_name, is_date_filtered=False, base_df=None):
    years       = sorted(filtered_df["game_year"].unique())
    latest_year = years[-1]
    fig = go.Figure()

    if is_date_filtered and base_df is not None and not base_df.empty:
        # 필터 적용 시: base_df 해당 연도 → 회색, filtered_df → 색상
        base_year_df = base_df[base_df["game_year"] == latest_year]
        for pn in PITCH_ORDER:
            td = base_year_df[base_year_df["pitch_name"] == pn]
            if len(td) > 0:
                fig.add_trace(go.Scattergl(
                    x=td["hor_break"], y=td["ver_break"], mode="markers",
                    marker=dict(size=14, color="rgba(200,200,200,0.25)"),
                    name=f"{pn} ({latest_year} 전체)",
                    customdata=td[["rel_speed(km)","pitch_name","game_date",
                                    "batname","events","description"]].values,
                    hovertemplate=("Pitch: %{customdata[1]}<br>Date: %{customdata[2]}<br>"
                                    "Speed: %{customdata[0]:.1f}<br>Batter: %{customdata[3]}<br>"
                                    "Description: %{customdata[5]}<extra></extra>"),
                    showlegend=False))
        for pn in PITCH_ORDER:
            td = filtered_df[filtered_df["pitch_name"] == pn]
            if len(td) > 0:
                fig.add_trace(go.Scattergl(
                    x=td["hor_break"], y=td["ver_break"], mode="markers",
                    marker=dict(size=14, color=COLOR_MAP.get(pn,"black"), opacity=0.6),
                    name=pn,
                    customdata=td[["rel_speed(km)","pitch_name","game_date",
                                    "batname","events","description"]].values,
                    hovertemplate=("Pitch: %{customdata[1]}<br>Date: %{customdata[2]}<br>"
                                    "Speed: %{customdata[0]:.1f}<br>Batter: %{customdata[3]}<br>"
                                    "Description: %{customdata[5]}<extra></extra>"),
                    showlegend=True))
        title_text = f"{pitcher_name} 구종별 무브먼트  ({latest_year} 전체 회색 / 필터 적용 색상)"

    elif not is_date_filtered:
        # 기존 로직: 이전연도 회색 + 최근연도 색상
        prev_years = [y for y in years if y != latest_year]
        prev_label = (str(prev_years[0]) if len(prev_years)==1
                    else f"{prev_years[0]}~{prev_years[-1]}" if prev_years else "")
        recent_df = filtered_df[filtered_df["game_year"] == latest_year]
        past_df   = filtered_df[filtered_df["game_year"] != latest_year]
        for pn in PITCH_ORDER:
            td = past_df[past_df["pitch_name"] == pn]
            if len(td) > 0:
                fig.add_trace(go.Scattergl(
                    x=td["hor_break"], y=td["ver_break"], mode="markers",
                    marker=dict(size=14, color="rgba(200,200,200,0.25)"),
                    name=f"{pn} (past)",
                    customdata=td[["rel_speed(km)","pitch_name","game_date",
                                    "batname","events","description"]].values,
                    hovertemplate=("Pitch: %{customdata[1]}<br>Date: %{customdata[2]}<br>"
                                    "Speed: %{customdata[0]:.1f}<br>Batter: %{customdata[3]}<br>"
                                    "Description: %{customdata[5]}<extra></extra>"),
                    showlegend=False))
        for pn in PITCH_ORDER:
            td = recent_df[recent_df["pitch_name"] == pn]
            if len(td) > 0:
                fig.add_trace(go.Scattergl(
                    x=td["hor_break"], y=td["ver_break"], mode="markers",
                    marker=dict(size=14, color=COLOR_MAP.get(pn,"black"), opacity=0.5),
                    name=pn,
                    customdata=td[["rel_speed(km)","pitch_name","game_date",
                                    "batname","events","description"]].values,
                    hovertemplate=("Pitch: %{customdata[1]}<br>Date: %{customdata[2]}<br>"
                                    "Speed: %{customdata[0]:.1f}<br>Batter: %{customdata[3]}<br>"
                                    "Description: %{customdata[5]}<extra></extra>"),
                    showlegend=True))
        title_text = f"{pitcher_name} 구종별 무브먼트  ({prev_label} 회색 / {latest_year} 색상)"

    else:
        # is_date_filtered=True이지만 base_df 없는 경우 fallback
        for pn in PITCH_ORDER:
            td = filtered_df[filtered_df["pitch_name"] == pn]
            if len(td) > 0:
                fig.add_trace(go.Scattergl(
                    x=td["hor_break"], y=td["ver_break"], mode="markers",
                    marker=dict(size=14, color=COLOR_MAP.get(pn,"black"), opacity=0.6),
                    name=pn,
                    customdata=td[["rel_speed(km)","pitch_name","game_date",
                                    "batname","events","description"]].values,
                    hovertemplate=("Pitch: %{customdata[1]}<br>Date: %{customdata[2]}<br>"
                                    "Speed: %{customdata[0]:.1f}<br>Batter: %{customdata[3]}<br>"
                                    "Description: %{customdata[5]}<extra></extra>"),
                    showlegend=True))
        title_text = f"{pitcher_name} 구종별 무브먼트  (필터 적용 / {latest_year})"

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title_text, font=dict(size=13), x=0.5),
        legend=dict(orientation="v", x=1.02, xanchor="left",
                    y=0.5, yanchor="middle", font=dict(size=11)),
        xaxis=dict(
            range=[-80, 80], mirror=True,
            title=dict(text="Horizontal Break", font=dict(size=11)),
            tickfont=dict(size=10),
            tickmode="linear", tick0=0, dtick=20,
            scaleanchor="y",   # ← 추가: y축과 스케일 연동
            scaleratio=1,      # ← 추가: 1:1 비율
            constrain="domain" # ← 추가: 범위 고정 보장
        ),
        yaxis=dict(
            range=[-80, 80], mirror=True,
            title=dict(text="Vertical Break", font=dict(size=11)),
            tickfont=dict(size=10),
            tickmode="linear", tick0=0, dtick=20,
            constrain="domain" # ← 추가
        ),
        autosize=True,
        margin=dict(l=50, r=130, t=60, b=50)
    )
    return fig



# ════════════════════════════════════════════════════════════
# 8-B. 구종 사용 비율 차트
# ════════════════════════════════════════════════════════════
def make_pitch_usage_chart(filtered_df, pitcher_name, base_df=None, is_date_filtered=False):
    if filtered_df.empty:
        return go.Figure()
    years       = sorted(filtered_df["game_year"].unique())
    latest_year = years[-1]

    def get_split_perc(df_):
        if df_.empty: return pd.DataFrame()
        c = df_.groupby(["stand","pitch_name"]).size().unstack(fill_value=0)
        return c.div(c.sum(axis=1), axis=0) * 100

    if not is_date_filtered:
        prev_year = years[-2] if len(years) >= 2 else years[-1]
        perc_r = get_split_perc(filtered_df[filtered_df["game_year"] == latest_year])
        perc_p = get_split_perc(filtered_df[filtered_df["game_year"] == prev_year])
        label_color = f"{latest_year}"
        label_gray  = f"{prev_year}"
    else:
        perc_r = get_split_perc(filtered_df)
        if base_df is not None and not base_df.empty:
            perc_p = get_split_perc(base_df[base_df["game_year"] == latest_year])
        else:
            perc_p = pd.DataFrame()
        label_color = "필터 적용"
        label_gray  = f"{latest_year} 전체"

    available = [p for p in PITCH_ORDER
                if p in set(perc_r.columns if not perc_r.empty else []) |
                    set(perc_p.columns if not perc_p.empty else [])]
    for p in available:
        if not perc_r.empty and p not in perc_r.columns: perc_r[p] = 0.0
        if not perc_p.empty and p not in perc_p.columns: perc_p[p] = 0.0

    y_base   = np.arange(len(available))
    y_prev   = y_base + 0.07
    y_recent = y_base - 0.07

    def gv(perc, side, sign=1):
        if perc is None or perc.empty or side not in perc.index:
            return [0]*len(available)
        return [sign*perc.loc[side,p] if p in perc.columns else 0 for p in available]

    fig = go.Figure()
    if not perc_p.empty:
        fig.add_trace(go.Bar(y=y_prev, x=gv(perc_p,"L",-1), orientation="h",
            name=f"{label_gray} 좌타자", marker_color="lightgray",
            marker_opacity=0.8, width=0.60))
        fig.add_trace(go.Bar(y=y_prev, x=gv(perc_p,"R", 1), orientation="h",
            name=f"{label_gray} 우타자", marker_color="lightgray",
            marker_opacity=0.8, width=0.60))
    fig.add_trace(go.Bar(y=y_recent, x=gv(perc_r,"L",-1), orientation="h",
        name=f"{label_color} 좌타자", marker_color="#3BBFB0", width=0.60,
        text=[f"{abs(v):.1f}%" if v!=0 else "" for v in gv(perc_r,"L",-1)],
        textposition="outside"))
    fig.add_trace(go.Bar(y=y_recent, x=gv(perc_r,"R", 1), orientation="h",
        name=f"{label_color} 우타자", marker_color="#e63946", width=0.60,
        text=[f"{v:.1f}%" if v!=0 else "" for v in gv(perc_r,"R", 1)],
        textposition="outside"))
    fig.update_layout(
        title=f"좌/우타자별 구종 사용 비율  ({label_gray} 회색 / {label_color} 색상)",
        barmode="overlay", xaxis_title="구종 사용 비율 (%)",
        plot_bgcolor="white", autosize=True,
        margin=dict(l=80,r=20,t=85,b=40),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15, yanchor="top"))
    fig.update_xaxes(tickmode="array", tickvals=[-75,-50,-25,0,25,50,75],
                    range=[-70,70], showgrid=True, gridcolor="rgba(230,230,230,0.7)",
                    zeroline=True, zerolinecolor="gray", zerolinewidth=1)
    fig.update_yaxes(tickmode="array", tickvals=y_base, ticktext=available,
                    autorange="reversed", range=[-0.7, len(available)-0.3])
    fig.add_annotation(x=-13, y=1.04, text="[좌타자]", showarrow=False,
        xref="x", yref="paper", xanchor="right", yanchor="bottom",
        font=dict(color="#3BBFB0", size=15))
    fig.add_annotation(x=13,  y=1.04, text="[우타자]", showarrow=False,
        xref="x", yref="paper", xanchor="left", yanchor="bottom",
        font=dict(color="#e63946", size=15))
    return fig

# ════════════════════════════════════════════════════════════
# 8-C. 트래킹 테이블
# ════════════════════════════════════════════════════════════
def make_tracking_table(df, year=None, is_date_filtered=False, base_df=None):
    def axis_to_clock(v):
        if pd.isnull(v): return np.nan
        v         = v % 360
        total_min = (v / 360) * 720
        hour      = int(total_min // 60) + 6
        minute    = int(round(total_min % 60))
        if hour > 12: hour -= 12
        if hour == 0: hour  = 12
        return f"{hour}:{minute:02d}"

    def _build_table(src):
        if src.empty: return pd.DataFrame()
        grp   = src.groupby("pitch_name")
        total = len(src)
        s     = pd.DataFrame()
        cnt   = grp["pitch_name"].count()
        s["투구비율"]     = (cnt/total*100).round(1).astype(str)+"% ("+cnt.astype(str)+")"
        s["평균구속"]     = grp["rel_speed(km)"].mean().round(1)
        s["최고구속"]     = grp["rel_speed(km)"].max().round(1)
        s["회전수"]       = grp["release_spin_rate"].mean().round() \
                            if "release_spin_rate" in src.columns else np.nan
        s["회전방향"]     = grp["release_spin_axis"].mean().apply(axis_to_clock) \
                            if "release_spin_axis" in src.columns else np.nan
        s["수직무브먼트"] = grp["ver_break"].mean().round(1)
        s["수평무브먼트"] = grp["hor_break"].mean().round(1)
        s["릴리스 높이"]  = grp["rel_height"].mean().round(2) \
                            if "rel_height" in src.columns else np.nan
        s["릴리스 좌우"]  = grp["rel_side"].mean().round(2) \
                            if "rel_side"   in src.columns else np.nan
        s["익스텐션"]     = grp["extension"].mean().round(2) \
                            if "extension"  in src.columns else np.nan
        s = s.reindex(PITCH_ORDER).dropna(how="all").reset_index()
        s.rename(columns={"pitch_name": "구종"}, inplace=True)
        return s

    if not is_date_filtered:
        if year is None: year = df["game_year"].max()
        return _build_table(df[df["game_year"] == year])
    else:
        t_filtered = _build_table(df)
        if base_df is not None and not base_df.empty:
            base_year = base_df["game_year"].max()
            t_base    = _build_table(base_df[base_df["game_year"] == base_year])
            if not t_base.empty and not t_filtered.empty:
                num_cols = [c for c in t_filtered.columns if c != "구종"]
                t_base_renamed = t_base.rename(
                    columns={c: f"{c}({base_year})" for c in num_cols})
                merged = t_filtered.merge(
                    t_base_renamed[["구종"] + [f"{c}({base_year})" for c in num_cols]],
                    on="구종", how="left")
                return merged
        return t_filtered

# ════════════════════════════════════════════════════════════
# 8-D. 카운트별 차트
# ════════════════════════════════════════════════════════════
def make_count_chart(filtered_df, pitcher_name, orientation="landscape"):
    years       = sorted(filtered_df["game_year"].unique())
    latest_year = years[-1]
    prev_year   = years[-2] if len(years) >= 2 else years[-1]
    count_groups = {
        "First Pitch (0-0)":             ["0-0"],
        "Two Strike (0-2/1-2/2-2/3-2)": ["0-2","1-2","2-2","3-2"],
        "Hitter's Count (1-0/2-0/2-1)": ["1-0","2-0","2-1"],
    }
    available = [p for p in PITCH_ORDER if p in filtered_df["pitch_name"].unique()]
    y_base    = np.arange(len(available))
    y_prev    = y_base + 0.07
    y_recent  = y_base - 0.07
    if orientation == "portrait":
        n_rows, n_cols = 3, 1
        fig_height     = max(220, len(available)*38)*3 + 180
        margin         = dict(l=80,r=30,t=80,b=60)
        title_fs       = 14; ann_fs = 11
        h_sp, v_sp     = 0.05, 0.12
    else:
        n_rows, n_cols = 1, 3
        fig_height     = None
        margin         = dict(l=80,r=30,t=110,b=50)
        title_fs       = 17; ann_fs = 13
        h_sp, v_sp     = 0.03, 0.05
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=list(count_groups.keys()),
                        horizontal_spacing=h_sp, vertical_spacing=v_sp)
    recent_df = filtered_df[filtered_df["game_year"] == latest_year]
    prev_df   = filtered_df[filtered_df["game_year"] == prev_year]
    for idx, (_, counts) in enumerate(count_groups.items(), start=1):
        r = idx if orientation=="portrait" else 1
        c = 1   if orientation=="portrait" else idx
        def gsp(df_):
            ct = df_.groupby(["stand","pitch_name"]).size().unstack(fill_value=0)
            return ct.div(ct.sum(axis=1), axis=0) * 100
        pr = gsp(recent_df[recent_df["count"].isin(counts)])
        pp = gsp(prev_df[prev_df["count"].isin(counts)])
        for p in available:
            if p not in pr.columns: pr[p] = 0.0
            if p not in pp.columns: pp[p] = 0.0
        def gv(perc, side, sign=1):
            return [sign*perc.loc[side,p] if side in perc.index else 0 for p in available]
        sl = (idx == 1)
        fig.add_trace(go.Bar(y=y_prev,   x=gv(pp,"L",-1), orientation="h",
            name=f"{prev_year} 좌타자", marker_color="lightgray",
            marker_opacity=0.8, width=0.40, showlegend=sl, legendgroup="prev_L"),
            row=r, col=c)
        fig.add_trace(go.Bar(y=y_prev,   x=gv(pp,"R", 1), orientation="h",
            name=f"{prev_year} 우타자", marker_color="lightgray",
            marker_opacity=0.8, width=0.40, showlegend=sl, legendgroup="prev_R"),
            row=r, col=c)
        fig.add_trace(go.Bar(y=y_recent, x=gv(pr,"L",-1), orientation="h",
            name=f"{latest_year} 좌타자", marker_color="#3BBFB0", width=0.40,
            text=[f"{abs(v):.1f}%" if v!=0 else "" for v in gv(pr,"L",-1)],
            textposition="outside", showlegend=sl, legendgroup="L"),
            row=r, col=c)
        fig.add_trace(go.Bar(y=y_recent, x=gv(pr,"R", 1), orientation="h",
            name=f"{latest_year} 우타자", marker_color="#e63946", width=0.40,
            text=[f"{v:.1f}%" if v!=0 else "" for v in gv(pr,"R", 1)],
            textposition="outside", showlegend=sl, legendgroup="R"),
            row=r, col=c)
        xref = f"x{'' if idx==1 else idx}"
        yref = (f"y{'' if idx==1 else idx} domain"
                if orientation=="portrait" else "paper")
        fig.add_annotation(x=-13, y=1.06, text="[좌타자]", showarrow=False,
            xref=xref, yref=yref, xanchor="right", yanchor="bottom",
            font=dict(color="#3BBFB0", size=ann_fs))
        fig.add_annotation(x=13,  y=1.06, text="[우타자]", showarrow=False,
            xref=xref, yref=yref, xanchor="left", yanchor="bottom",
            font=dict(color="#e63946", size=ann_fs))
    ax_cfg = dict(tickmode="array", tickvals=[-75,-50,-25,0,25,50,75],
                range=[-80,80], showgrid=True, gridcolor="rgba(230,230,230,0.5)",
                zeroline=True, zerolinecolor="gray", zerolinewidth=1)
    y_cfg  = dict(tickmode="array", tickvals=y_base, ticktext=available,
                autorange="reversed", range=[-0.7, len(available)-0.3])
    fig.update_xaxes(**ax_cfg)
    if orientation == "portrait":
        for i in range(1,4):
            fig.update_yaxes(**y_cfg, showticklabels=True, row=i, col=1)
    else:
        fig.update_yaxes(**y_cfg, showticklabels=True,  row=1, col=1)
        fig.update_yaxes(**y_cfg, showticklabels=False, row=1, col=2)
        fig.update_yaxes(**y_cfg, showticklabels=False, row=1, col=3)
    kw = dict(
        title=dict(text=f"카운트별 좌/우타자 구종 사용 비율  "
                        f"({prev_year} 회색 / {latest_year} 색상)",
                    font=dict(size=title_fs), x=0.5),
        barmode="overlay", plot_bgcolor="white", autosize=True, bargap=0.3,
        margin=margin,
        legend=dict(orientation="h", x=0.5, xanchor="center",
                    y=-0.06 if orientation=="portrait" else -0.12,
                    yanchor="top", font=dict(size=13)))
    if fig_height: kw["height"] = fig_height
    fig.update_layout(**kw)
    return fig

# ════════════════════════════════════════════════════════════
# 8-E. 리그 비교 스캐터 차트
# ════════════════════════════════════════════════════════════
def make_scatter_chart(pitcher_stats_df, player_stats_df,
                        player_stats_df_left, player_stats_df_right,
                        x_col, y_col, x_title, y_title, title_text,
                        x_range=None, y_range=None):
    yal = player_stats_df_left.copy();  yal["stand"] = "L"
    yar = player_stats_df_right.copy(); yar["stand"] = "R"
    ysa = pd.concat([yal, yar], ignore_index=True)
    years       = sorted(player_stats_df["game_year"].unique())
    year_colors = {y: c for y, c in zip(years, YEAR_COLORS)}
    ss = {"L":"circle","R":"square"}
    sl = {"L":"좌타","R":"우타"}

    def _range(col, df):
        vals = df[col].dropna().tolist()
        if not vals: return [0, 100]
        pad = (max(vals) - min(vals)) * 0.08 or 1
        return [min(vals) - pad, max(vals) + pad]

    if x_range is None: x_range = _range(x_col, pitcher_stats_df)
    if y_range is None: y_range = _range(y_col, pitcher_stats_df)
    mx = pitcher_stats_df[x_col].mean()
    my = pitcher_stats_df[y_col].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pitcher_stats_df[x_col], y=pitcher_stats_df[y_col],
        mode="markers", marker=dict(size=8, color="rgba(180,180,180,0.35)"),
        text=pitcher_stats_df.apply(
            lambda r: f"{r['pitname']} ({int(r['game_year'])})", axis=1),
        hovertemplate=(f"%{{text}}<br>{x_col}: %{{x:.1f}}<br>"
                        f"{y_col}: %{{y:.1f}}<extra></extra>"),
        showlegend=False))
    fig.add_vline(x=mx, line=dict(color="crimson", width=1.5, dash="dot"),
                annotation=dict(text=f"{x_col} 평균<br>{mx:.1f}",
                                font=dict(size=10, color="crimson"),
                                bgcolor="rgba(255,255,255,0.7)",
                                showarrow=False, xanchor="left", yanchor="top"),
                annotation_position="top right")
    fig.add_hline(y=my, line=dict(color="crimson", width=1.5, dash="dot"),
                annotation=dict(text=f"{y_col} 평균  {my:.1f}",
                                font=dict(size=10, color="crimson"),
                                bgcolor="rgba(255,255,255,0.7)",
                                showarrow=False, xanchor="right", yanchor="bottom"),
                annotation_position="bottom right")
    for _, row in player_stats_df.iterrows():
        yr = int(row["game_year"])
        if x_col not in row or y_col not in row: continue
        fig.add_trace(go.Scatter(
            x=[row[x_col]], y=[row[y_col]], mode="markers+text",
            marker=dict(size=22, symbol="star",
                        color=year_colors.get(yr, "black"),
                        line=dict(width=1.5, color="white")),
            text=[f"{row['pitname']}<br>{yr}"], textposition="top center",
            textfont=dict(size=10, color=year_colors.get(yr, "black")),
            hovertemplate=(f"{row['pitname']} ({yr})<br>"
                            f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}"
                            f"<extra></extra>"),
            showlegend=False))
    for _, row in ysa.iterrows():
        yr = int(row["game_year"]); stand = row["stand"]
        if stand not in ss or x_col not in row or y_col not in row: continue
        fig.add_trace(go.Scatter(
            x=[row[x_col]], y=[row[y_col]], mode="markers+text",
            marker=dict(size=15, symbol=ss[stand],
                        color=year_colors.get(yr, "black"), opacity=0.85,
                        line=dict(width=1.5, color="white")),
            text=[f"{row['pitname']}<br>{yr} {sl[stand]}"],
            textposition="top center",
            textfont=dict(size=9, color=year_colors.get(yr, "black")),
            hovertemplate=(f"{row['pitname']} ({yr}) {sl[stand]}<br>"
                            f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.1f}}"
                            f"<extra></extra>"),
            showlegend=False))
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=16)),
        xaxis=dict(title=x_title, showgrid=True,
                    gridcolor="rgba(220,220,220,0.6)", range=x_range),
        yaxis=dict(title=y_title, showgrid=True,
                    gridcolor="rgba(220,220,220,0.6)", range=y_range),
        plot_bgcolor="white", showlegend=False, autosize=True,
        margin=dict(l=70, r=70, t=100, b=70))
    return fig

# ════════════════════════════════════════════════════════════
# 8-F. 대시보드 계산 함수
# ════════════════════════════════════════════════════════════
def calc_dashboard_metrics(df: pd.DataFrame, league_df: pd.DataFrame,
                            pitcher_id: str, league: str) -> dict:
    if df.empty:
        return {}
    latest_year = df["game_year"].max()
    prev_year   = latest_year - 1
    df_cur  = df[df["game_year"] == latest_year]
    df_prev = df[df["game_year"] == prev_year] \
            if prev_year in df["game_year"].values else pd.DataFrame()
    result  = {}
    fb_types = ["4-Seam Fastball", "2-Seam Fastball", "Cutter"]

    fb_df = df_cur[df_cur["pitch_name"].isin(fb_types)]
    result["avg_fb_speed"] = round(fb_df["rel_speed(km)"].mean(), 1) \
        if (not fb_df.empty and "rel_speed(km)" in fb_df.columns) else None
    if not df_prev.empty:
        fb_prev  = df_prev[df_prev["pitch_name"].isin(fb_types)]
        prev_spd = fb_prev["rel_speed(km)"].mean() \
            if (not fb_prev.empty and "rel_speed(km)" in fb_prev.columns) else None
        result["avg_fb_speed_diff"] = round(result["avg_fb_speed"] - prev_spd, 1) \
            if (result["avg_fb_speed"] and prev_spd) else None
    else:
        result["avg_fb_speed_diff"] = None

    pitch_cnt   = df_cur["pitch_name"].value_counts() if not df_cur.empty else pd.Series()
    total_pitch = len(df_cur)
    if len(pitch_cnt) > 0:
        result["main_pitch"]         = pitch_cnt.index[0]
        result["main_pitch_pct"]     = round(pitch_cnt.iloc[0] / total_pitch * 100, 1)
        result["main_pitch_2nd"]     = pitch_cnt.index[1] if len(pitch_cnt) > 1 else "-"
        result["main_pitch_2nd_pct"] = round(pitch_cnt.iloc[1] / total_pitch * 100, 1) \
                                        if len(pitch_cnt) > 1 else 0
    else:
        result.update({"main_pitch": "-", "main_pitch_pct": 0,
                        "main_pitch_2nd": "-", "main_pitch_2nd_pct": 0})

    if "count" in df_cur.columns:
        fp_df = df_cur[df_cur["count"] == "0-0"]
        if not fp_df.empty:
            fp_cnt = fp_df["pitch_name"].value_counts(); fp_tot = len(fp_df)
            result["fp_main"]     = fp_cnt.index[0]
            result["fp_main_pct"] = round(fp_cnt.iloc[0] / fp_tot * 100, 1)
            result["fp_2nd"]      = fp_cnt.index[1] if len(fp_cnt) > 1 else ""
            result["fp_2nd_pct"]  = round(fp_cnt.iloc[1] / fp_tot * 100, 1) \
                                    if len(fp_cnt) > 1 else 0
        else:
            result.update({"fp_main": "-", "fp_main_pct": 0,
                            "fp_2nd": "", "fp_2nd_pct": 0})
    else:
        result.update({"fp_main": "-", "fp_main_pct": 0,
                        "fp_2nd": "", "fp_2nd_pct": 0})

    two_strike = ["0-2", "1-2", "2-2", "3-2"]
    if "count" in df_cur.columns:
        ts_df = df_cur[df_cur["count"].isin(two_strike)]
        if not ts_df.empty:
            ts_cnt = ts_df["pitch_name"].value_counts(); ts_tot = len(ts_df)
            result["ts_main"]     = ts_cnt.index[0]
            result["ts_main_pct"] = round(ts_cnt.iloc[0] / ts_tot * 100, 1)
            result["ts_2nd"]      = ts_cnt.index[1] if len(ts_cnt) > 1 else ""
            result["ts_2nd_pct"]  = round(ts_cnt.iloc[1] / ts_tot * 100, 1) \
                                    if len(ts_cnt) > 1 else 0
        else:
            result.update({"ts_main": "-", "ts_main_pct": 0,
                            "ts_2nd": "", "ts_2nd_pct": 0})
    else:
        result.update({"ts_main": "-", "ts_main_pct": 0,
                        "ts_2nd": "", "ts_2nd_pct": 0})

    swstr_by_pitch = {}
    for pname in df_cur["pitch_name"].unique():
        sub = df_cur[df_cur["pitch_name"] == pname]
        if len(sub) < 10: continue
        sw = sub["description"].isin(
            ["swinging_strike", "swinging_strike_blocked", "foul_tip"]).sum()
        swstr_by_pitch[pname] = round(sw / len(sub) * 100, 1)
    if swstr_by_pitch:
        sorted_sw             = sorted(swstr_by_pitch.items(), key=lambda x: -x[1])
        result["sw_main"]     = sorted_sw[0][0]
        result["sw_main_pct"] = sorted_sw[0][1]
        result["sw_2nd"]      = sorted_sw[1][0] if len(sorted_sw) > 1 else ""
        result["sw_2nd_pct"]  = sorted_sw[1][1] if len(sorted_sw) > 1 else 0
    else:
        result.update({"sw_main": "-", "sw_main_pct": 0,
                        "sw_2nd": "", "sw_2nd_pct": 0})

    pa_df = df_cur[df_cur["events"].notna()]
    bb    = int((pa_df["events"] == "walk").sum())
    k     = int(pa_df["events"].isin(["strikeout", "strikeout_double_play"]).sum())
    outs  = int(pa_df["events"].isin(OUT_EVENTS).sum())
    ip    = outs / 3 if outs > 0 else 0
    hit_w = int(df_cur["hit"].sum()) if "hit" in df_cur.columns else 0
    result["bb9"]  = round(bb / ip * 9, 2) if ip > 0 else None
    result["k9"]   = round(k  / ip * 9, 2) if ip > 0 else None
    result["whip"] = round((hit_w + bb) / ip, 2) if ip > 0 else None

    if not df_prev.empty:
        pa_p   = df_prev[df_prev["events"].notna()]
        bb_p   = int((pa_p["events"] == "walk").sum())
        k_p    = int(pa_p["events"].isin(["strikeout", "strikeout_double_play"]).sum())
        outs_p = int(pa_p["events"].isin(OUT_EVENTS).sum())
        ip_p   = outs_p / 3 if outs_p > 0 else 0
        hit_p  = int(df_prev["hit"].sum()) if "hit" in df_prev.columns else 0  # ← 추가
        whip_p = round((hit_p + bb_p) / ip_p, 2) if ip_p > 0 else None        # ← 추가
        bb9_p  = round(bb_p / ip_p * 9, 2) if ip_p > 0 else None
        k9_p   = round(k_p  / ip_p * 9, 2) if ip_p > 0 else None
        result["bb9_diff"]  = round(result["bb9"]  - bb9_p,  2) \
                             if (result["bb9"]  and bb9_p)  else None
        result["k9_diff"]   = round(result["k9"]   - k9_p,   2) \
                             if (result["k9"]   and k9_p)   else None
        result["whip_diff"] = round(result["whip"] - whip_p, 2) \
                             if (result["whip"] and whip_p) else None  # ← 추가
    else:
        result["bb9_diff"]  = None
        result["k9_diff"]   = None
        result["whip_diff"] = None  # ← 추가

    rank_data = league_rank_cache.get(league, {}).get(int(latest_year))

    def pct_rank(val, col, higher_is_better=True):
        if val is None or rank_data is None or col not in rank_data.columns:
            return None
        s = rank_data[col].dropna()
        if s.empty: return None
        if higher_is_better:
            pct = (s < val).sum() / len(s) * 100
        else:
            pct = (s > val).sum() / len(s) * 100
        return round(100 - pct, 0)

    result["bb9_rank"]      = pct_rank(result["bb9"],          "BB9",  higher_is_better=False)
    result["k9_rank"]       = pct_rank(result["k9"],           "K9",   higher_is_better=True)
    result["whip_rank"]     = pct_rank(result["whip"],         "WHIP", higher_is_better=False)
    result["fb_speed_rank"] = pct_rank(result["avg_fb_speed"], "SPD",  higher_is_better=True)
    return result

# ════════════════════════════════════════════════════════════
# 9. 로케이션 차트
# ════════════════════════════════════════════════════════════
LOC_COLORS = {
    "hit_into_play_no_out":    "rgba(255,72,120,1)",
    "hit_into_play_score":     "rgba(255,72,120,1)",
    "hit_into_play":           "rgba(255,72,120,1)",
    "called_strike":           "rgba(67,89,119,0.5)",
    "foul":                    "rgba(67,89,119,0.5)",
    "swinging_strike":         "rgba(67,89,119,0.5)",
    "foul_tip":                "rgba(67,89,119,0.5)",
    "swinging_strike_blocked": "rgba(67,89,119,0.5)",
    "ball":                    "rgba(140,86,75,0.5)",
    "hit_by_pitch":            "rgba(140,86,75,0.5)",
    "BallIntentional":         "rgba(140,86,75,0.5)",
    "pitchout":                "rgba(140,86,75,0.5)",
}

# ★ 3그룹 정의
LOC_GROUPS = {
    "strike": {
        "label": "🔵 스트라이크",
        "color": "#1a3a5c",
        "descs": ["called_strike", "swinging_strike",
                  "swinging_strike_blocked", "foul_tip", "foul"],
    },
    "ball": {
        "label": "🟤 볼",
        "color": "#7a4f3a",
        "descs": ["ball", "hit_by_pitch", "BallIntentional", "pitchout"],
    },
    "inplay": {
        "label": "🔴 인플레이",
        "color": "#cc1a4a",
        "descs": ["hit_into_play", "hit_into_play_no_out", "hit_into_play_score"],
    },
}

LOC_SYMBOLS = {
    "4-Seam Fastball":"circle",    "2-Seam Fastball":"triangle-down",
    "Cutter":"triangle-se",        "Slider":"triangle-right",
    "Sweeper":"cross",             "Curveball":"triangle-up",
    "Changeup":"diamond",          "Split-Finger":"square",
}
STAND_ORDER = ["R", "L"]

def _add_zone_overlays(fig, row, col):
    fig.add_trace(go.Scatter(
        x=[-0.12,  0.12,  0.12, -0.12, -0.12],
        y=[ 0.59,  0.59,  0.91,  0.91,  0.59],
        mode="lines", line=dict(color="red", width=2),
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=[-0.23,  0.23,  0.23, -0.23, -0.23],
        y=[ 0.45,  0.45,  1.05,  1.05,  0.45],
        mode="lines", line=dict(color="rgba(108,122,137,0.9)", width=2),
        showlegend=False, hoverinfo="skip",
    ), row=row, col=col)

def _loc_cell_size(n_cols):
    cell_h = 280 if n_cols <= 4 else 240
    cell_w = int(cell_h * (0.90 / 0.98))
    return cell_w, cell_h

def make_location_fig(filtered_df, pitcher_name, year_filter=None):
    src = filtered_df.copy()
    if year_filter is not None:
        src = src[src["game_year"] == year_filter]
    if src.empty: return go.Figure()
    used_pitches = [p for p in PITCH_ORDER if p in src["pitch_name"].unique()]
    if not used_pitches: return go.Figure()
    n_cols = len(used_pitches)
    n_rows = 2
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=used_pitches + [""]*n_cols,
                        vertical_spacing=0.06, horizontal_spacing=0.02)
    x_range = [-0.50, 0.50]
    y_range = [ 0.15, 1.35]
    for c_idx, pitch_name in enumerate(used_pitches, start=1):
        pitch_df = src[src["pitch_name"] == pitch_name]
        symbol   = LOC_SYMBOLS.get(pitch_name, "circle")
        for r_idx, stand in enumerate(STAND_ORDER, start=1):
            sub = pitch_df[pitch_df["stand"] == stand]
            for desc, grp in sub.groupby("description"):
                color      = LOC_COLORS.get(desc, "rgba(180,180,180,0.4)")
                hover_cols = [c for c in ["game_date","batname","rel_speed(km)",
                                        "events","exit_velocity","launch_angleX"]
                            if c in grp.columns]
                customdata = grp[hover_cols].values if hover_cols else None
                hover_tmpl = (
                    "Date: %{customdata[0]}<br>Batter: %{customdata[1]}<br>"
                    "Speed: %{customdata[2]:.1f}<br>Events: %{customdata[3]}<br>"
                    "EV: %{customdata[4]}<extra></extra>"
                ) if customdata is not None else "<extra></extra>"
                fig.add_trace(go.Scattergl(
                    x=grp["plate_x"], y=grp["plate_z"], mode="markers",
                    marker=dict(size=7, color=color, symbol=symbol,
                                line=dict(width=0.3, color="rgba(0,0,0,0.15)")),
                    name=desc, customdata=customdata,
                    hovertemplate=hover_tmpl, showlegend=False,
                ), row=r_idx, col=c_idx)
            _add_zone_overlays(fig, r_idx, c_idx)
    for r_idx in range(1, n_rows + 1):
        for c_idx in range(1, n_cols + 1):
            ai    = (r_idx - 1) * n_cols + c_idx
            x_key = "xaxis" if ai == 1 else f"xaxis{ai}"
            y_key = "yaxis" if ai == 1 else f"yaxis{ai}"
            fig.update_layout(**{
                x_key: dict(range=x_range, showgrid=False, zeroline=False,
                            showticklabels=False, fixedrange=True,
                            showline=True, linewidth=1,
                            linecolor="rgba(108,122,137,0.9)", mirror=True),
                y_key: dict(range=y_range, showgrid=False, zeroline=False,
                            showticklabels=False, fixedrange=True,
                            showline=True, linewidth=1,
                            linecolor="rgba(108,122,137,0.9)", mirror=True),
            })
    fig.add_annotation(x=-0.02, y=0.78, text="우타자", showarrow=False,
                        xref="paper", yref="paper", textangle=-90,
                        font=dict(size=11, color="#2c3e50"))
    fig.add_annotation(x=-0.02, y=0.22, text="좌타자", showarrow=False,
                        xref="paper", yref="paper", textangle=-90,
                        font=dict(size=11, color="#2c3e50"))
    cell_w, cell_h = _loc_cell_size(n_cols)
    fig_w  = cell_w * n_cols + 20 * (n_cols - 1) + 60
    fig_h  = cell_h * n_rows + 80 * (n_rows - 1) + 120
    suffix = f" ({int(year_filter)} 시즌)" if year_filter else " (전체 시즌)"
    fig.update_layout(
        title=dict(text=f"{pitcher_name} 구종별 로케이션{suffix}",
                    x=0.5, font=dict(size=13)),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        autosize=False, width=fig_w, height=fig_h,
        margin=dict(l=40, r=20, t=60, b=20))
    for ann in fig.layout.annotations:
        ann.font.size = 10
    fig.layout.annotations[-2].font.size = 11
    fig.layout.annotations[-1].font.size = 11
    return fig

def make_result_location_chart(filtered_df, pitcher_name,
                                year_filter=None, pitch_filter=None):
    src = filtered_df.copy()
    if year_filter  is not None: src = src[src["game_year"]  == year_filter]
    if pitch_filter is not None: src = src[src["pitch_name"] == pitch_filter]
    if src.empty: return go.Figure()
    frames = []
    cs = src[src["description"] == "called_strike"].copy()
    cs["swingmap"] = "스트라이크(콜)"; frames.append(cs)
    if "whiff" in src.columns:
        wh = src[src["whiff"] == 1].copy()
        wh["swingmap"] = "헛스윙"; frames.append(wh)
    bl = src[src["type"] == "B"].copy()
    bl["swingmap"] = "볼"; frames.append(bl)
    if "foul" in src.columns:
        fo = src[src["foul"] == 1].copy()
        fo["swingmap"] = "파울"; frames.append(fo)
    if "launch_speed_angle" in src.columns:
        hh = src[src["launch_speed_angle"] >= 4].copy()
        hh["swingmap"] = "강한타구(LSA4+)"; frames.append(hh)
        wk = src[src["launch_speed_angle"] <= 3].copy()
        wk["swingmap"] = "약한타구(LSA3-)"; frames.append(wk)
    if not frames: return go.Figure()
    swdf = pd.concat(frames, ignore_index=True)
    SWING_COLORS = {
        "스트라이크(콜)":  "rgba(24,85,144,0.6)",
        "헛스윙":          "rgba(210,160,0,0.9)",
        "볼":              "rgba(108,122,137,0.7)",
        "파울":            "rgba(241,106,227,0.5)",
        "강한타구(LSA4+)": "rgba(255,105,97,1)",
        "약한타구(LSA3-)": "rgba(140,86,75,0.6)",
    }
    SWING_ORDER = ["스트라이크(콜)","볼","파울","헛스윙","강한타구(LSA4+)","약한타구(LSA3-)"]
    avail_sw    = [s for s in SWING_ORDER if s in swdf["swingmap"].unique()]
    n_cols = len(avail_sw)
    n_rows = len(STAND_ORDER)
    subplot_titles = [
        f"{sw}  {st}({len(swdf[(swdf['stand']==st)&(swdf['swingmap']==sw)])})"
        for st in STAND_ORDER for sw in avail_sw
    ]
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0.01, vertical_spacing=0.12)
    x_range = [-0.45, 0.45]
    y_range = [ 0.27, 1.25]
    for r_idx, stand in enumerate(STAND_ORDER, start=1):
        for c_idx, sw in enumerate(avail_sw, start=1):
            sub   = swdf[(swdf["stand"] == stand) & (swdf["swingmap"] == sw)]
            color = SWING_COLORS.get(sw, "rgba(180,180,180,0.5)")
            for pitch, grp in sub.groupby("pitch_name"):
                symbol     = LOC_SYMBOLS.get(pitch, "circle")
                hover_cols = [c for c in ["game_date","rel_speed(km)","events",
                                        "exit_velocity","launch_speed_angle",
                                        "launch_angle"] if c in grp.columns]
                customdata = grp[hover_cols].values if hover_cols else None
                hover_tmpl = (
                    f"{pitch}<br>Date: %{{customdata[0]}}<br>"
                    f"Speed: %{{customdata[1]:.1f}}<br>"
                    f"Events: %{{customdata[2]}}<br>"
                    f"EV: %{{customdata[3]}}<br>"
                    f"LSA: %{{customdata[4]}}<extra></extra>"
                ) if customdata is not None else "<extra></extra>"
                fig.add_trace(go.Scattergl(
                    x=grp["plate_x"], y=grp["plate_z"], mode="markers",
                    marker=dict(size=9, color=color, symbol=symbol,
                                line=dict(width=0.3, color="rgba(0,0,0,0.2)")),
                    name=pitch, customdata=customdata,
                    hovertemplate=hover_tmpl, showlegend=False,
                ), row=r_idx, col=c_idx)
            _add_zone_overlays(fig, r_idx, c_idx)
    for r_idx in range(1, n_rows + 1):
        for c_idx in range(1, n_cols + 1):
            ai    = (r_idx - 1) * n_cols + c_idx
            x_key = "xaxis" if ai == 1 else f"xaxis{ai}"
            y_key = "yaxis" if ai == 1 else f"yaxis{ai}"
            fig.update_layout(**{
                x_key: dict(range=x_range, showgrid=False, zeroline=False,
                            showticklabels=False, fixedrange=True,
                            showline=True, linewidth=1,
                            linecolor="rgba(108,122,137,0.9)", mirror=True),
                y_key: dict(range=y_range, showgrid=False, zeroline=False,
                            showticklabels=False, fixedrange=True,
                            showline=True, linewidth=1,
                            linecolor="rgba(108,122,137,0.9)", mirror=True),
            })
    fig.add_annotation(x=-0.02, y=0.78, text="우타자", showarrow=False,
                        xref="paper", yref="paper", textangle=-90,
                        font=dict(size=11, color="#2c3e50"))
    fig.add_annotation(x=-0.02, y=0.22, text="좌타자", showarrow=False,
                        xref="paper", yref="paper", textangle=-90,
                        font=dict(size=11, color="#2c3e50"))
    suffix = f"  ({int(year_filter)} 시즌)" if year_filter else "  (전체 시즌)"
    if pitch_filter: suffix += f"  [{pitch_filter}]"
    cell_w, cell_h = _loc_cell_size(n_cols)
    fig_w  = cell_w * n_cols + 20 * (n_cols - 1) + 60
    fig_h  = cell_h * n_rows + 80 * (n_rows - 1) + 120
    fig.update_layout(
        title=dict(text=f"{pitcher_name} 결과별 로케이션{suffix}",
                    x=0.5, font=dict(size=13)),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        autosize=False, width=fig_w, height=fig_h,
        margin=dict(l=60, r=20, t=80, b=40))
    for ann in fig.layout.annotations:
        ann.font.size = 9
    fig.layout.annotations[-2].font.size = 11
    fig.layout.annotations[-1].font.size = 11
    return fig

# ════════════════════════════════════════════════════════════
# 10. Dash 앱 초기화 & 스타일 상수
# ════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>kt wiz Pitcher Report</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            * { font-family: "Noto Sans KR", sans-serif; box-sizing: border-box; }
            body { background: linear-gradient(135deg, #f0f4f8 0%, #e8edf5 100%); min-height: 100vh; }
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 3px; }
            ::-webkit-scrollbar-thumb { background: #c0c8d4; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #e63946; }
            #tab-wrapper {
                background: #ffffff; border-radius: 10px; padding: 6px 8px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.07); margin-bottom: 20px;
                display: flex; flex-wrap: wrap; gap: 4px;
            }
            .nav-tabs { border-bottom: none !important; display: flex !important;
                flex-wrap: wrap !important; gap: 4px !important; width: 100% !important; }
            .nav-tabs .nav-link {
                font-size: 13px !important; font-weight: 600 !important;
                color: #6b7280 !important; padding: 8px 16px !important;
                border-radius: 7px !important; border: none !important;
                background: transparent !important; transition: all 0.2s ease !important;
                white-space: nowrap !important; margin-bottom: 0 !important; }
            .nav-tabs .nav-link:hover { background: #f3f4f6 !important; color: #374151 !important; border: none !important; }
            .nav-tabs .nav-link.active {
                color: #ffffff !important;
                background: linear-gradient(135deg, #e8453c 0%, #c0392b 100%) !important;
                border: none !important; box-shadow: 0 2px 8px rgba(232,69,60,0.35) !important; }
            .dash-clipboard:hover .copy-icon-span {
                color: #3498db !important;
                transition: color 0.15s ease; }
            .dash-metric-card {
                background: linear-gradient(135deg, #ffffff, #f8fafc);
                border: 1px solid #e0e6ed; border-radius: 12px;
                padding: 14px 16px; transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
            .dash-metric-card:hover { transform: translateY(-3px);
                box-shadow: 0 8px 24px rgba(230,57,70,0.12);
                border-color: rgba(230,57,70,0.3); }
            .dash-metric-title { font-size: 10px; font-weight: 700; color: #8a9bb0;
                letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
            .dash-metric-value { font-size: 22px; font-weight: 800; color: #1a2535; line-height: 1.2; }
            .dash-metric-sub { font-size: 11px; color: #6b7a8d; margin-top: 2px; }
            .dash-metric-diff-up   { color: #e63946; font-size: 11px; font-weight: 700; }
            .dash-metric-diff-down { color: #2196F3; font-size: 11px; font-weight: 700; }
            .dash-metric-rank      { color: #ff9800; font-size: 11px; font-weight: 700; }
            .dash-spreadsheet-container td,
            .dash-spreadsheet-container th {
                user-select: text !important;
                -webkit-user-select: text !important; }
            .card { transition: transform 0.2s ease, box-shadow 0.2s ease !important;
                border-radius: 10px !important; }
            .card:hover { transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(0,0,0,0.10) !important; }
            @media (max-width: 900px) {
                #sidebar { left: -260px !important; }
                #page-content { margin-left: 0 !important; padding: 12px !important; }
                .nav-tabs .nav-link { font-size: 11px !important; padding: 8px 10px !important; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "260px", "padding": "20px 16px",
    "background": "linear-gradient(180deg, #4a4e5a 0%, #2e3038 100%)",
    "overflowY": "auto", "zIndex": 999,
    "transition": "left 0.3s ease",
    "boxShadow": "3px 0 15px rgba(0,0,0,0.25)",
}
SIDEBAR_HIDDEN_STYLE = {**SIDEBAR_STYLE, "left": "-260px"}
CONTENT_STYLE = {
    "marginLeft": "270px", "padding": "20px 24px",
    "backgroundColor": "transparent", "minHeight": "100vh",
    "transition": "margin-left 0.3s ease",
}
CONTENT_FULL_STYLE = {**CONTENT_STYLE, "marginLeft": "0px"}
SECTION_STYLE = {
    "color": "#a0b4c8", "fontSize": "10px", "fontWeight": "700",
    "letterSpacing": "1.2px", "textTransform": "uppercase",
    "marginTop": "16px", "marginBottom": "6px",
}
TOGGLE_BTN_STYLE_OPEN = {
    "position": "fixed", "top": "10px", "left": "268px",
    "zIndex": 1100, "width": "28px", "height": "28px",
    "background": "linear-gradient(135deg, #4a4e5a, #2e3038)",
    "color": "white", "border": "none",
    "borderRadius": "0 6px 6px 0", "cursor": "pointer",
    "fontSize": "14px", "display": "flex",
    "alignItems": "center", "justifyContent": "center",
    "transition": "all 0.3s ease",
    "boxShadow": "2px 0 8px rgba(0,0,0,0.3)",
}
TOGGLE_BTN_STYLE_CLOSED = {**TOGGLE_BTN_STYLE_OPEN, "left": "8px"}
LOADING_MSG_STYLE = {
    "fontSize": "11px", "padding": "6px 10px",
    "borderRadius": "8px", "marginTop": "8px",
    "textAlign": "center", "fontWeight": "600",
    "transition": "all 0.3s ease",
}
_SIDEBAR_INPUT = {
    "width": "100%", "fontSize": "12px",
    "padding": "6px 10px", "borderRadius": "6px",
    "border": "1px solid rgba(255,255,255,0.15)",
    "backgroundColor": "rgba(255,255,255,0.08)",
    "color": "#e0e8f0", "boxSizing": "border-box",
    "transition": "border-color 0.2s ease",
}

def expander_btn_style(bg="#f0f4f8", border="#d0d8e4", color="#2c3e50"):
    return {
        "width": "100%", "textAlign": "left",
        "padding": "9px 14px", "marginBottom": "6px",
        "backgroundColor": bg, "border": f"1px solid {border}",
        "borderRadius": "8px", "cursor": "pointer",
        "fontSize": "13px", "fontWeight": "600", "color": color,
        "transition": "all 0.2s ease",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    }

EXPANDER_BTN_STYLE              = expander_btn_style("#f0f4f8", "#d0d8e4", "#2c3e50")
EXPANDER_BTN_RUNNER_STYLE       = expander_btn_style("#eaf4ec", "#b2d8b8", "#1a5c2a")
EXPANDER_BTN_CMP_STYLE          = expander_btn_style("#fdf6e3", "#e0c97f", "#7a5c00")
EXPANDER_BTN_COUNT_RUNNER_STYLE = expander_btn_style("#f0eafa", "#c9b0e8", "#5a3e8a")
# ★ 신규: 전체기간 비교 버튼 스타일 (파란 계열)
EXPANDER_BTN_BASE_STYLE         = expander_btn_style("#e8f4fd", "#90caf9", "#1565c0")

# ════════════════════════════════════════════════════════════
# 11. 사이드바 & 테이블 헬퍼
# ════════════════════════════════════════════════════════════
def make_sidebar():
    return html.Div([
        html.Div([
            html.Div([
                html.Span("KT WIZ BASEBALL CLUB", style={
                    "fontSize": "17px", "fontWeight": "800",
                    "color": "#e63946", "letterSpacing": "0.5px",
                    "lineHeight": "1.3", "display": "block"}),
                html.Span("투수 분석 리포트", style={
                    "fontSize": "12px", "fontWeight": "600",
                    "color": "#a0b4c8", "letterSpacing": "1px",
                    "textTransform": "uppercase", "display": "block"}),
            ], style={"borderLeft": "3px solid #e63946",
                    "paddingLeft": "8px", "marginBottom": "6px"}),
            html.Hr(style={"borderColor": "rgba(255,255,255,0.12)", "margin": "8px 0"}),
            html.Div(
                html.Span("Developed by 전략데이터팀", style={
                    "fontSize": "11px", "color": "#a0b4c8",
                    "background": "rgba(230,57,70,0.12)",
                    "border": "1px solid rgba(230,57,70,0.25)",
                    "borderRadius": "3px", "padding": "1px 6px",
                    "letterSpacing": "0.5px"}),
                style={"textAlign": "right", "marginTop": "4px"}),
            html.Hr(style={"borderColor": "rgba(255,255,255,0.12)", "margin": "8px 0"}),
        ]),
        html.Div("League", style=SECTION_STYLE),
        dcc.Dropdown(
            id="league-dropdown",
            options=[{"label": v, "value": k} for k, v in LEAGUE_LABELS.items()],
            value="KBO", clearable=False, style={"fontSize": "13px"}),
        html.Div(id="league-loading-msg", children=[
            html.Div(
                ["✅ 데이터 로딩 완료.", html.Br(), "선수를 선택해 주세요."],
                style={**LOADING_MSG_STYLE,
                        "backgroundColor": "#d4edda", "color": "#155724",
                        "border": "1px solid #c3e6cb"})]),
        html.Div("Pitcher", style=SECTION_STYLE),
        dcc.Loading(
            id="pitcher-dropdown-loading", type="circle", color="#e63946",
            children=dcc.Dropdown(
                id="pitcher-dropdown",
                options=build_pitcher_options(league_cache["KBO"], "KBO"),
                value=None, placeholder="투수 선택...",
                style={"fontSize": "13px"}, disabled=False)),
        html.Div("Season", style=SECTION_STYLE),
        dcc.Checklist(
            id="year-checklist", options=[], value=[],
            labelStyle={
                "display": "inline-block", "color": "#cfd8e3",
                "fontSize": "12px", "marginRight": "6px", "marginBottom": "4px",
                "padding": "3px 8px",
                "backgroundColor": "rgba(255,255,255,0.08)",
                "borderRadius": "12px", "cursor": "pointer"}),
        html.Div("Date Filter", style=SECTION_STYLE),
        dcc.Input(
            id="date-start-picker", type="text",
            placeholder="시작일 (YYYY-MM-DD)", debounce=True,
            style={**_SIDEBAR_INPUT, "marginBottom": "6px"}),
        dcc.Input(
            id="date-end-picker", type="text",
            placeholder="종료일 (YYYY-MM-DD)", debounce=True,
            style=_SIDEBAR_INPUT),
        html.Div(id="date-range-info",
                style={"color": "#7eb3d8", "fontSize": "11px", "marginTop": "4px"}),
        html.Button(
            "↺  일자 초기화", id="date-reset-btn", n_clicks=0,
            style={
                "width": "100%", "marginTop": "8px", "padding": "6px",
                "background": "rgba(230,57,70,0.15)", "color": "#f0a0a8",
                "border": "1px solid rgba(230,57,70,0.3)",
                "borderRadius": "6px", "fontSize": "12px",
                "cursor": "pointer", "fontWeight": "600",
                "transition": "all 0.2s ease"}),
        html.Hr(style={"borderColor": "rgba(255,255,255,0.12)", "margin": "16px 0"}),
        html.Div("Min IP (League)", style=SECTION_STYLE),
        dcc.Input(
            id="min-ip-input", type="number", value=30, min=0, step=5,
            style=_SIDEBAR_INPUT),
    ], id="sidebar", style=SIDEBAR_STYLE)

def make_stat_table(df, table_id, font_size="12px", fix_first_col=True):
    if df is None or df.empty:
        return html.Div("데이터 없음",
                        style={"color": "#aaa", "padding": "12px",
                               "fontSize": "13px", "textAlign": "center"})

    tsv_string = df.to_csv(sep="\t", index=False)
    hidden_id  = f"_tsv_{table_id}"

    table = dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        fixed_columns={"headers": True, "data": 1} if fix_first_col else {},
        style_table={
            "overflowX": "auto", "minWidth": "100%",
            "borderRadius": "8px", "border": "1px solid #dee2e6",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"},
        style_header={
            "background": "linear-gradient(135deg, #4a4e5a, #2e3038)",
            "color": "white", "fontWeight": "700", "fontSize": font_size,
            "textAlign": "center", "padding": "8px 10px",
            "borderBottom": "2px solid #e63946"},
        style_cell={
            "textAlign": "center", "fontSize": font_size,
            "padding": "6px 10px", "border": "1px solid #e8ecf0",
            "fontFamily": "'Noto Sans KR', sans-serif", "color": "#2c3e50",
            "minWidth": "60px", "maxWidth": "120px",
            "overflow": "hidden", "textOverflow": "ellipsis",
            "userSelect": "text", "WebkitUserSelect": "text",
        },
        style_cell_conditional=[{
            "if": {"column_id": df.columns[0]},
            "backgroundColor": "#e8e8e8", "fontWeight": "700",
            "color": "#2c3e50", "borderRight": "2px solid #c0c8d4",
            "position": "sticky", "left": 0, "zIndex": 1,
            "userSelect": "text", "WebkitUserSelect": "text",
        }],
        style_data_conditional=[
            {"if": {"row_index": "odd"},  "backgroundColor": "#f7f9fc"},
            {"if": {"row_index": "even"}, "backgroundColor": "#ffffff"},
            {"if": {"state": "selected"},
             "backgroundColor": "rgba(230,57,70,0.10)",
             "border": "1px solid rgba(230,57,70,0.4)"}
        ]
    )

    return html.Div(
        children=[
            html.Pre(tsv_string, id=hidden_id, style={"display": "none"}),
            # ── 상단 바: 오른쪽 끝에 복사 버튼 ──
            html.Div(
                dcc.Clipboard(
                    target_id=hidden_id,
                    title="클립보드에 복사",
                    style={
                        "cursor": "pointer",
                        "background": "none",
                        "border": "none",
                        "padding": "2px 4px",
                        "fontSize": "17px",
                        "color": "#bbbbbb",
                        "lineHeight": "1",
                        "display": "inline-flex",
                        "alignItems": "center",
                        "transition": "color 0.15s ease",
                    },
                ),
                style={
                    "display": "flex",
                    "justifyContent": "flex-end",   # 오른쪽 정렬
                    "marginBottom": "2px",
                },
            ),
            # ── 테이블 ──
            table,
        ],
        style={"display": "flex", "flexDirection": "column"},
    )


# ════════════════════════════════════════════════════════════
# 12. 앱 레이아웃
# ════════════════════════════════════════════════════════════
app.layout = html.Div([
    html.Button("☰", id="sidebar-toggle-btn", n_clicks=0,
                style=TOGGLE_BTN_STYLE_OPEN, title="사이드바 열기/닫기"),
    make_sidebar(),
    html.Div([
        html.Div(id="main-loading-banner", children=[],
                style={"marginBottom": "8px"}),
        html.Div([
            dbc.Tabs(id="main-tabs", active_tab="tab-dashboard", children=[
                dbc.Tab(label="대시보드",                 tab_id="tab-dashboard"),
                dbc.Tab(label="주요스탯",                 tab_id="tab-overview"),
                dbc.Tab(label="무브먼트 차트",            tab_id="tab-movement"),
                dbc.Tab(label="구종사용비율(타자유형별)", tab_id="tab-usage"),
                dbc.Tab(label="구종사용비율(카운트별)",   tab_id="tab-count"),
                dbc.Tab(label="구종별 로케이션",          tab_id="tab-location"),
                dbc.Tab(label="결과별 로케이션",          tab_id="tab-result-loc"),
                dbc.Tab(label="리그비교",                 tab_id="tab-league"),
            ]),
        ], id="tab-wrapper"),
        html.Div(html.Div(id="tab-content"), id="tab-content-wrapper"),
    ], id="page-content", style=CONTENT_STYLE),
    dcc.Store(id="sidebar-open",       data=True),
    dcc.Store(id="orientation-store",  data="landscape"),
    dcc.Store(id="filtered-store"),
    dcc.Store(id="base-store"),
    dcc.Store(id="league-load-status", data="done"),
], style={"position": "relative"})

# ════════════════════════════════════════════════════════════
# 13-A. 사이드바 토글
# ════════════════════════════════════════════════════════════
app.clientside_callback(
    "function(n, is_open){ return !is_open; }",
    Output("sidebar-open", "data"),
    Input("sidebar-toggle-btn", "n_clicks"),
    State("sidebar-open", "data"),
    prevent_initial_call=True,
)

@app.callback(
    Output("sidebar",            "style"),
    Output("page-content",       "style"),
    Output("sidebar-toggle-btn", "style"),
    Output("sidebar-toggle-btn", "children"),
    Input("sidebar-open",        "data"),
)
def toggle_sidebar_style(is_open):
    if is_open:
        return SIDEBAR_STYLE, CONTENT_STYLE, TOGGLE_BTN_STYLE_OPEN, "✕"
    return SIDEBAR_HIDDEN_STYLE, CONTENT_FULL_STYLE, TOGGLE_BTN_STYLE_CLOSED, "☰"

app.clientside_callback(
    "function(tab){ return window.innerWidth < window.innerHeight "
    "? 'portrait' : 'landscape'; }",
    Output("orientation-store", "data"),
    Input("main-tabs", "active_tab"),
)

app.clientside_callback(
    """
    function(value) {
        if (value) {
            setTimeout(function() {
                var el = document.getElementById('pitcher-dropdown');
                if (el) { var inp = el.querySelector('input');
                        if (inp) inp.blur(); }
                if (document.activeElement) document.activeElement.blur();
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("pitcher-dropdown", "id"),
    Input("pitcher-dropdown",  "value"),
    prevent_initial_call=True,
)

# ════════════════════════════════════════════════════════════
# 13-B. 리그 변경 → 투수 목록 갱신
# ════════════════════════════════════════════════════════════
@app.callback(
    Output("pitcher-dropdown",   "options"),
    Output("pitcher-dropdown",   "value"),
    Output("pitcher-dropdown",   "disabled"),
    Output("league-load-status", "data"),
    Input("league-dropdown",     "value"),
)
def update_pitcher_options(league):
    if league not in league_cache:
        league_cache[league] = load_league_data(league)
    df = league_cache.get(league, pd.DataFrame())
    if df.empty:
        return [], None, False, "error"
    if league not in league_rank_cache:
        build_league_rank_cache(league)
    return build_pitcher_options(df, league), None, False, "done"

@app.callback(
    Output("league-loading-msg",  "children"),
    Output("main-loading-banner", "children"),
    Input("league-load-status",   "data"),
    Input("league-dropdown",      "value"),
)
def update_loading_ui(status, league):
    league_label = LEAGUE_LABELS.get(league, league)
    if status == "error":
        sidebar_msg = html.Div(
            "❌ 데이터 로딩 실패",
            style={**LOADING_MSG_STYLE,
                    "backgroundColor": "#f8d7da", "color": "#721c24",
                    "border": "1px solid #f5c6cb"})
        main_banner = dbc.Alert(
            [html.Strong("❌ 데이터 로딩에 실패했습니다."),
            "  네트워크 상태를 확인해 주세요."],
            color="danger",
            style={"padding": "10px 16px", "fontSize": "14px"})
        return sidebar_msg, main_banner
    sidebar_msg = html.Div(
        [f"✅ {league_label} 로딩 완료.", html.Br(), "선수를 선택해 주세요."],
        style={**LOADING_MSG_STYLE,
                "backgroundColor": "#d4edda", "color": "#155724",
                "border": "1px solid #c3e6cb"})
    return sidebar_msg, []

# ════════════════════════════════════════════════════════════
# 13-C-1. 투수/연도 → year-checklist만 담당
# ════════════════════════════════════════════════════════════
@app.callback(
    Output("year-checklist", "options"),
    Output("year-checklist", "value"),
    Input("pitcher-dropdown", "value"),
    Input("year-checklist",   "value"),
    Input("date-reset-btn",   "n_clicks"),
    State("league-dropdown",  "value"),
    prevent_initial_call=False,
)
def update_year_options(pitcher_id, selected_years, n_clicks, league):
    empty = ([], [])
    if not pitcher_id or league not in league_cache:
        return empty
    df     = league_cache[league]
    id_col = "pitcher" if "pitcher" in df.columns else get_name_col(league, df)
    if id_col is None:
        return empty
    sub = df[df[id_col].astype(str) == str(pitcher_id)]
    if sub.empty:
        return empty

    years = sorted(sub["game_year"].unique(), reverse=True)
    opts  = [{"label": str(y), "value": y} for y in years]

    ctx     = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "date-reset-btn" in trigger or "pitcher-dropdown" in trigger:
        return opts, [years[0]]

    if "year-checklist" in trigger:
        if not selected_years:
            return opts, [years[0]]
        return opts, selected_years

    if not selected_years:
        return opts, [years[0]]

    return opts, selected_years

# ════════════════════════════════════════════════════════════
# 13-C-2. date picker — 투수변경 / 초기화 버튼 시에만 리셋
# ════════════════════════════════════════════════════════════
@app.callback(
    Output("date-start-picker", "value"),
    Output("date-end-picker",   "value"),
    Input("pitcher-dropdown",   "value"),
    Input("date-reset-btn",     "n_clicks"),
    prevent_initial_call=True,
)
def reset_date_range(pitcher_id, n_clicks):
    return None, None

# ════════════════════════════════════════════════════════════
# 13-C-3. 날짜 입력 안내 메시지
# ════════════════════════════════════════════════════════════
@app.callback(
    Output("date-range-info", "children"),
    Input("date-start-picker", "value"),
    Input("date-end-picker",   "value"),
)
def update_date_info(start, end):
    pat  = r"^\d{4}-\d{2}-\d{2}$"
    msgs = []
    if start and str(start).strip():
        if re.match(pat, str(start).strip()):
            msgs.append(f"▶ {start}")
        else:
            return html.Span("⚠ 시작일 형식 오류 (YYYY-MM-DD)", style={"color": "#ff6b6b"})
    if end and str(end).strip():
        if re.match(pat, str(end).strip()):
            msgs.append(f"◀ {end}")
        else:
            return html.Span("⚠ 종료일 형식 오류 (YYYY-MM-DD)", style={"color": "#ff6b6b"})
    return " ~ ".join(msgs) if msgs else "※ 미선택 시 전체 기간 적용"

# ════════════════════════════════════════════════════════════
# 13-D. filtered-store / base-store
# ════════════════════════════════════════════════════════════
def _is_valid_date(v):
    if not v:
        return False
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(v).strip()))


@app.callback(
    Output("filtered-store",   "data"),
    Input("pitcher-dropdown",  "value"),
    Input("year-checklist",    "value"),
    Input("date-start-picker", "value"),
    Input("date-end-picker",   "value"),
    State("league-dropdown",   "value"),
)
def update_filtered_store(pitcher_id, years, date_start, date_end, league):
    print(f"[filtered_store] pitcher={pitcher_id}, years={years}, "
          f"start={date_start}, end={date_end}, league={league}")

    pat = r"^\d{4}-\d{2}-\d{2}$"

    if not pitcher_id or not years or league not in league_cache:
        return None

    raw    = league_cache[league]
    id_col = "pitcher" if "pitcher" in raw.columns else get_name_col(league, raw)
    if id_col is None:
        return None

    sub = raw[raw[id_col].astype(str) == str(pitcher_id)].copy()
    sub = sub[sub["game_year"].isin(years)]

    if sub.empty:
        print("[filtered_store] → None (empty after year filter)")
        return None

    has_date_filter = (
        bool(date_start and re.match(pat, str(date_start).strip())) or
        bool(date_end   and re.match(pat, str(date_end).strip()))
    )

    if has_date_filter:
        sub["game_date"] = pd.to_datetime(sub["game_date"], errors="coerce")
        print(f"[filtered_store] game_date range after year filter: "
              f"{sub['game_date'].min()} ~ {sub['game_date'].max()}")

        if date_start and re.match(pat, str(date_start).strip()):
            sub = sub[sub["game_date"] >= pd.to_datetime(date_start)]
            print(f"[filtered_store] after start filter: {sub.shape}")
        if date_end and re.match(pat, str(date_end).strip()):
            sub = sub[sub["game_date"] <= pd.to_datetime(date_end)]
            print(f"[filtered_store] after end filter: {sub.shape}")

        if sub.empty:
            print("[filtered_store] → None (empty after date filter)")
            return None

        sub = sub.copy()
        sub["game_date"] = sub["game_date"].dt.strftime("%Y-%m-%d")

    key = _make_key(
        pitcher_id, league,
        sorted(years),
        date_start or "", date_end or ""
    )
    print(f"[filtered_store] → saved key={key}, rows={sub.shape[0]}")
    return _save(key, sub)


@app.callback(
    Output("base-store",      "data"),
    Input("pitcher-dropdown", "value"),
    State("league-dropdown",  "value"),
)
def update_base_store(pitcher_id, league):
    """항상 해당 투수의 최근 3년 전체 데이터 저장 (날짜필터·연도필터 완전 무관)."""
    if not pitcher_id or league not in league_cache:
        return None
    raw    = league_cache[league]
    id_col = "pitcher" if "pitcher" in raw.columns else get_name_col(league, raw)
    if id_col is None:
        return None
    sub = raw[raw[id_col].astype(str) == str(pitcher_id)].copy()
    if sub.empty:
        return None

    if "game_date" in sub.columns:
        sub["game_date"] = pd.to_datetime(
            sub["game_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    all_years   = sorted(sub["game_year"].unique(), reverse=True)
    recent_3yrs = all_years[:3]
    sub         = sub[sub["game_year"].isin(recent_3yrs)].copy()
    key = _make_key(pitcher_id, league, sorted(recent_3yrs), "__base3yr__")
    return _save(key, sub)

# ════════════════════════════════════════════════════════════
# 13-D-helper. store 로드 + fallback 직접 필터링
# ════════════════════════════════════════════════════════════
def _load_filtered_df(filtered_key, pitcher_id, years,
                      date_start, date_end, league):
    df = _load(filtered_key)
    if not df.empty:
        return df

    if not pitcher_id or not years or league not in league_cache:
        return pd.DataFrame()

    pat    = r"^\d{4}-\d{2}-\d{2}$"
    raw    = league_cache[league]
    id_col = "pitcher" if "pitcher" in raw.columns else get_name_col(league, raw)
    if id_col is None:
        return pd.DataFrame()

    sub = raw[raw[id_col].astype(str) == str(pitcher_id)].copy()
    sub = sub[sub["game_year"].isin(years)]
    if sub.empty:
        return pd.DataFrame()

    has_date = (
        bool(date_start and re.match(pat, str(date_start).strip())) or
        bool(date_end   and re.match(pat, str(date_end).strip()))
    )
    if has_date:
        sub["game_date"] = pd.to_datetime(sub["game_date"], errors="coerce")
        if date_start and re.match(pat, str(date_start).strip()):
            sub = sub[sub["game_date"] >= pd.to_datetime(date_start)]
        if date_end and re.match(pat, str(date_end).strip()):
            sub = sub[sub["game_date"] <= pd.to_datetime(date_end)]
        if not sub.empty:
            sub = sub.copy()
            sub["game_date"] = sub["game_date"].dt.strftime("%Y-%m-%d")

    return sub


def _load_base_df(base_key, pitcher_id, league):
    df = _load(base_key)
    if not df.empty:
        return df
    if not pitcher_id or league not in league_cache:
        return pd.DataFrame()
    raw    = league_cache[league]
    id_col = "pitcher" if "pitcher" in raw.columns else get_name_col(league, raw)
    if id_col is None:
        return pd.DataFrame()
    sub     = raw[raw[id_col].astype(str) == str(pitcher_id)].copy()
    if sub.empty:
        return pd.DataFrame()

    if "game_date" in sub.columns:
        sub["game_date"] = pd.to_datetime(
            sub["game_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    all_yrs = sorted(sub["game_year"].unique(), reverse=True)
    return sub[sub["game_year"].isin(all_yrs[:3])].copy()

# ════════════════════════════════════════════════════════════
# 13-E. Expander 토글 콜백
# ════════════════════════════════════════════════════════════
@app.callback(
    Output("stand-collapse",     "is_open"),
    Output("stand-collapse-btn", "children"),
    Input("stand-collapse-btn",  "n_clicks"),
    State("stand-collapse",      "is_open"),
    prevent_initial_call=True,
)
def toggle_stand(n, is_open):
    o = not is_open
    return o, ("▲ 타자유형별 스탯 닫기" if o else "▼ 타자유형별 스탯 보기")

@app.callback(
    Output("runner-collapse-ov",     "is_open"),
    Output("runner-collapse-btn-ov", "children"),
    Input("runner-collapse-btn-ov",  "n_clicks"),
    State("runner-collapse-ov",      "is_open"),
    prevent_initial_call=True,
)
def toggle_runner_ov(n, is_open):
    o = not is_open
    return o, ("▲ 주자 상황별 스탯 닫기" if o else "▼ 주자 상황별 스탯 보기")

@app.callback(
    Output("cmp-collapse",     "is_open"),
    Output("cmp-collapse-btn", "children"),
    Input("cmp-collapse-btn",  "n_clicks"),
    State("cmp-collapse",      "is_open"),
    prevent_initial_call=True,
)
def toggle_cmp(n, is_open):
    o = not is_open
    return o, ("▲ 전체기간 비교 닫기" if o else "▼ 전체기간 비교 보기")

@app.callback(
    Output("runner-collapse-us",     "is_open"),
    Output("runner-collapse-btn-us", "children"),
    Input("runner-collapse-btn-us",  "n_clicks"),
    State("runner-collapse-us",      "is_open"),
    prevent_initial_call=True,
)
def toggle_runner_us(n, is_open):
    o = not is_open
    return o, ("▲ 주자 상황별 스탯 닫기" if o else "▼ 주자 상황별 스탯 보기")

@app.callback(
    Output("runner-collapse-ct",     "is_open"),
    Output("runner-collapse-btn-ct", "children"),
    Input("runner-collapse-btn-ct",  "n_clicks"),
    State("runner-collapse-ct",      "is_open"),
    prevent_initial_call=True,
)
def toggle_runner_ct(n, is_open):
    o = not is_open
    return o, ("▲ 카운트별 주자 상황 닫기" if o else "▼ 카운트별 주자 상황 보기")

@app.callback(
    Output("loc-year-collapse",     "is_open"),
    Output("loc-year-collapse-btn", "children"),
    Input("loc-year-collapse-btn",  "n_clicks"),
    State("loc-year-collapse",      "is_open"),
    prevent_initial_call=True,
)
def toggle_loc_year(n, is_open):
    o = not is_open
    return o, ("▲ 연도별 로케이션 닫기" if o else "▼ 연도별 로케이션 보기")

@app.callback(
    Output("res-loc-year-collapse",     "is_open"),
    Output("res-loc-year-collapse-btn", "children"),
    Input("res-loc-year-collapse-btn",  "n_clicks"),
    State("res-loc-year-collapse",      "is_open"),
    prevent_initial_call=True,
)
def toggle_res_loc_year(n, is_open):
    o = not is_open
    return o, ("▲ 연도별 결과 로케이션 닫기" if o
                else "▼ 연도별 결과 로케이션 보기")

# ★ 신규: 무브먼트 전체기간 비교 expander
@app.callback(
    Output("mov-base-collapse",     "is_open"),
    Output("mov-base-collapse-btn", "children"),
    Input("mov-base-collapse-btn",  "n_clicks"),
    State("mov-base-collapse",      "is_open"),
    prevent_initial_call=True,
)
def toggle_mov_base(n, is_open):
    o = not is_open
    return o, ("▲ 전체기간 차트 닫기" if o else "▼ 전체기간 차트 보기")

# ★ 신규: 구종사용비율 전체기간 비교 expander
@app.callback(
    Output("usage-base-collapse",     "is_open"),
    Output("usage-base-collapse-btn", "children"),
    Input("usage-base-collapse-btn",  "n_clicks"),
    State("usage-base-collapse",      "is_open"),
    prevent_initial_call=True,
)
def toggle_usage_base(n, is_open):
    o = not is_open
    return o, ("▲ 전체기간 차트 닫기" if o else "▼ 전체기간 차트 보기")

# ★ 신규: 카운트별 전체기간 비교 expander
@app.callback(
    Output("count-base-collapse",     "is_open"),
    Output("count-base-collapse-btn", "children"),
    Input("count-base-collapse-btn",  "n_clicks"),
    State("count-base-collapse",      "is_open"),
    prevent_initial_call=True,
)
def toggle_count_base(n, is_open):
    o = not is_open
    return o, ("▲ 전체기간 차트 닫기" if o else "▼ 전체기간 차트 보기")

for _slug in ("fp", "ts", "hc"):
    def _make_count_stand_toggle(slug=_slug):
        @app.callback(
            Output(f"count-stand-collapse-{slug}-ct", "is_open"),
            Output(f"count-stand-btn-{slug}-ct",      "children"),
            Input(f"count-stand-btn-{slug}-ct",       "n_clicks"),
            State(f"count-stand-collapse-{slug}-ct",  "is_open"),
            prevent_initial_call=True,
        )
        def _toggle(n, is_open, slug=slug):
            _rev  = {v: k for k, v in COUNT_SLUG.items()}
            label = _rev.get(slug, slug)
            o     = not is_open
            txt   = (f"▲ {label} — 좌/우타자별 닫기"
                    if o else f"▼ {label} — 좌/우타자별 보기")
            return o, txt
        return _toggle
    _make_count_stand_toggle(_slug)

# ════════════════════════════════════════════════════════════
# 13-F. 로케이션 인터랙티브 필터 콜백
# ════════════════════════════════════════════════════════════

# 탭6: 3그룹 필터 → 차트 업데이트
@app.callback(
    Output("loc-filtered-chart-area", "children"),
    Input("loc-desc-filter",   "value"),
    State("filtered-store",    "data"),
    State("league-dropdown",   "value"),
    State("pitcher-dropdown",  "value"),
    State("date-start-picker", "value"),
    State("date-end-picker",   "value"),
    State("year-checklist",    "value"),
    prevent_initial_call=False,
)
def update_loc_chart(selected_groups, filtered_key,
                     league, pitcher_id, date_start, date_end, years):
    if not pitcher_id:
        return html.Div()

    df = _load_filtered_df(filtered_key, pitcher_id, years,
                           date_start, date_end, league)
    if df.empty:
        return html.Div("데이터 없음",
                        style={"color": "#aaa", "padding": "20px"})

    pitcher_name = get_pitcher_name(
        league_cache.get(league, pd.DataFrame()), pitcher_id, league)

    # 그룹 키 → description 목록 변환
    if selected_groups:
        allowed_descs = []
        for gkey in selected_groups:
            allowed_descs.extend(LOC_GROUPS[gkey]["descs"])
        df_filtered = df[df["description"].isin(allowed_descs)].copy()
    else:
        df_filtered = pd.DataFrame()

    if df_filtered.empty:
        return html.Div("선택한 결과에 해당하는 데이터가 없습니다.",
                        style={"color": "#aaa", "padding": "20px",
                               "textAlign": "center", "fontSize": "13px"})

    fig = make_location_fig(df_filtered, pitcher_name)

    dcard_style = {
        "background": "rgba(255,255,255,0.85)",
        "backdropFilter": "blur(8px)",
        "border": "1px solid rgba(220,228,236,0.8)",
        "borderRadius": "10px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
    }
    return dbc.Card(dbc.CardBody(
        dcc.Graph(figure=fig, config={"displayModeBar": True},
                  style={"overflowX": "auto"})
    ), className="mb-3", style=dcard_style)


# 탭7: 구종 필터 → 차트 업데이트
@app.callback(
    Output("res-loc-filtered-chart-area", "children"),
    Input("res-loc-pitch-filter", "value"),
    State("filtered-store",    "data"),
    State("league-dropdown",   "value"),
    State("pitcher-dropdown",  "value"),
    State("date-start-picker", "value"),
    State("date-end-picker",   "value"),
    State("year-checklist",    "value"),
    prevent_initial_call=False,
)
def update_res_loc_chart(pitch_filter, filtered_key,
                         league, pitcher_id, date_start, date_end, years):
    if not pitcher_id:
        return html.Div()

    df = _load_filtered_df(filtered_key, pitcher_id, years,
                           date_start, date_end, league)
    if df.empty:
        return html.Div("데이터 없음",
                        style={"color": "#aaa", "padding": "20px"})

    pitcher_name = get_pitcher_name(
        league_cache.get(league, pd.DataFrame()), pitcher_id, league)

    pitch_val = None if (not pitch_filter or pitch_filter == "__ALL__") else pitch_filter

    fig = make_result_location_chart(df, pitcher_name, pitch_filter=pitch_val)

    dcard_style = {
        "background": "rgba(255,255,255,0.85)",
        "backdropFilter": "blur(8px)",
        "border": "1px solid rgba(220,228,236,0.8)",
        "borderRadius": "10px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
    }
    return dbc.Card(dbc.CardBody(
        dcc.Graph(figure=fig, config={"displayModeBar": True},
                  style={"overflowX": "auto"})
    ), className="mb-3", style=dcard_style)

# ════════════════════════════════════════════════════════════
# 14. render_tab 메인 콜백
# ════════════════════════════════════════════════════════════
@app.callback(
    Output("tab-content",      "children"),
    Input("main-tabs",         "active_tab"),
    Input("filtered-store",    "data"),
    Input("base-store",        "data"),
    State("league-dropdown",   "value"),
    State("pitcher-dropdown",  "value"),
    State("orientation-store", "data"),
    State("min-ip-input",      "value"),
    State("date-start-picker", "value"),
    State("date-end-picker",   "value"),
    State("year-checklist",    "value"),
)
def render_tab(active_tab, filtered_key, base_key,
                league, pitcher_id, orientation,
                min_ip, date_start, date_end, years):

    no_data = html.Div(
        "⚾ 좌측에서 투수를 선택해 주세요.",
        style={"color": "#aaa", "padding": "40px",
                "textAlign": "center", "fontSize": "15px"})

    if not pitcher_id:
        return no_data

    df      = _load_filtered_df(filtered_key, pitcher_id, years,
                                 date_start, date_end, league)
    base_df = _load_base_df(base_key, pitcher_id, league)

    if df.empty:
        return html.Div("데이터가 없습니다.",
                        style={"color": "#aaa", "padding": "40px",
                                "textAlign": "center"})

    pat              = r"^\d{4}-\d{2}-\d{2}$"
    is_date_filtered = bool(
        (date_start and re.match(pat, date_start)) or
        (date_end   and re.match(pat, date_end))
    )

    pitcher_name = get_pitcher_name(
        league_cache.get(league, pd.DataFrame()), pitcher_id, league)
    latest_year  = df["game_year"].max()
    years_in_df  = sorted(df["game_year"].unique())
    years        = years_in_df

    dcard_style = {
        "background":     "rgba(255,255,255,0.85)",
        "backdropFilter": "blur(8px)",
        "border":         "1px solid rgba(220,228,236,0.8)",
        "borderRadius":   "10px",
        "boxShadow":      "0 2px 8px rgba(0,0,0,0.06)",
    }

    # ──────────────────────────────────────────────────────
    # 공통 헬퍼: 전체기간 데이터 소스
    # ──────────────────────────────────────────────────────
    def _get_base_src():
        """base_df 또는 fallback으로 전체기간 데이터 반환"""
        if base_df is not None and not base_df.empty:
            return base_df
        _lg  = league_cache.get(league, pd.DataFrame())
        _id  = "pitcher" if "pitcher" in _lg.columns else get_name_col(league, _lg)
        if _id and not _lg.empty:
            _sub     = _lg[_lg[_id].astype(str) == str(pitcher_id)].copy()
            _all_yrs = sorted(_sub["game_year"].unique(), reverse=True)
            return _sub[_sub["game_year"].isin(_all_yrs[:3])].copy()
        return df

    def _filter_badge(src_label="필터 적용"):
        """날짜 필터 배너 반환"""
        if not is_date_filtered:
            return []
        return [dbc.Alert(
            [html.Strong(f"📌 날짜 필터 적용 중  "),
             f"{date_start or '?'} ~ {date_end or '?'}  |  ",
             f"필터 구수: {len(df):,}구  ",
             html.Span(
                 "(메인 차트/테이블 = 필터 기간 / 아래 expander = 전체기간)",
                 style={"fontSize": "11px", "color": "#856404"})],
            color="warning",
            style={"padding": "6px 14px", "fontSize": "12px",
                    "marginBottom": "10px"})]

    # ──────────────────────────────────────────────────────
    # 탭 1: 대시보드
    # ──────────────────────────────────────────────────────
    if active_tab == "tab-dashboard":

        _base_src    = _get_base_src()
        _base_latest = int(_base_src["game_year"].max())

        # 지표카드는 항상 base_src 기준 (IP 충분해야 랭크 계산 가능)
        metrics = calc_dashboard_metrics(
            _base_src,
            league_cache.get(league, pd.DataFrame()),
            pitcher_id, league)

        def _mini_card(title, value, sub="", sub2="", diff=None, rank=None, icon=""):
            diff_el = html.Span()
            if diff is not None:
                arrow = "▲" if diff > 0 else "▼"
                cls   = "dash-metric-diff-up" if diff > 0 else "dash-metric-diff-down"
                diff_el = html.Span(
                    f"{arrow}{abs(diff):.1f}",
                    className=cls,
                    style={"marginRight": "3px", "fontSize": "10px"})
            rank_el = html.Span()
            if rank is not None:
                rank_el = html.Span(
                    f"상위{rank:.0f}%",
                    className="dash-metric-rank",
                    style={"fontSize": "10px"})
            has_sub2 = bool(sub2)
            return html.Div([
                html.Div(f"{icon} {title}", style={
                    "fontSize": "12px", "fontWeight": "700",
                    "color": "#8a9bb0", "letterSpacing": "0.6px",
                    "textTransform": "uppercase", "whiteSpace": "nowrap",
                    "overflow": "hidden", "textOverflow": "ellipsis"}),
                html.Div(str(value), style={
                    "fontSize": "18px", "fontWeight": "800",
                    "color": "#1a2535", "lineHeight": "1.4",
                    "whiteSpace": "nowrap", "overflow": "hidden",
                    "textOverflow": "ellipsis"}),
                html.Div(sub, style={
                    "fontSize": "12px", "color": "#6b7a8d",
                    "lineHeight": "1.2", "minHeight": "0px",
                    "whiteSpace": "nowrap", "overflow": "hidden",
                    "textOverflow": "ellipsis"}),
                html.Div(sub2, style={
                    "fontSize": "11px", "color": "#9aabb8",
                    "minHeight": "9px", "whiteSpace": "nowrap",
                    "overflow": "hidden", "textOverflow": "ellipsis",
                }) if has_sub2 else html.Div(style={"minHeight": "0px"}),
                html.Div([diff_el, rank_el], style={
                    "height": "10px", "display": "flex", "alignItems": "center"}),
            ], style={
                "background":     "linear-gradient(135deg,#ffffff,#f8fafc)",
                "border":         "1px solid #e0e6ed",
                "borderRadius":   "8px",
                "padding":        "8px 11px",
                "boxShadow":      "0 2px 6px rgba(0,0,0,0.05)",
                "transition":     "all 0.2s ease",
                "height":         "96px",
                "overflow":       "hidden",
                "display":        "flex",
                "flexDirection":  "column",
                "gap":            "0px",
                "justifyContent": "space-between",
            })

        def _fmt_speed(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return f"{v} km/h"

        def _fmt_val(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            return str(v)

        top_cards = dbc.Row([
            dbc.Col(_mini_card("평균구속",
                _fmt_speed(metrics.get("avg_fb_speed")),
                sub="FB계열(4S/2S/CT)",
                diff=metrics.get("avg_fb_speed_diff"),
                rank=metrics.get("fb_speed_rank"), icon="🔥"),
                xs=3, sm=3, md=True, className="mb-2 px-1"),
            dbc.Col(_mini_card("BB/9",
                _fmt_val(metrics.get("bb9")),
                sub="볼넷/9이닝",
                diff=metrics.get("bb9_diff"),
                rank=metrics.get("bb9_rank"), icon="🚶"),
                xs=3, sm=3, md=True, className="mb-2 px-1"),
            dbc.Col(_mini_card("K/9",
                _fmt_val(metrics.get("k9")),
                sub="삼진/9이닝",
                diff=metrics.get("k9_diff"),
                rank=metrics.get("k9_rank"), icon="⚡"),
                xs=3, sm=3, md=True, className="mb-2 px-1"),
            dbc.Col(_mini_card("WHIP",
                _fmt_val(metrics.get("whip")),
                sub="(안타+볼넷)/이닝",
                diff=metrics.get("whip_diff"), 
                rank=metrics.get("whip_rank"), icon="📉"),
                xs=3, sm=3, md=True, className="mb-2 px-1"),
            dbc.Col(_mini_card("주구종",
                metrics.get("main_pitch", "-"),
                sub=f"{metrics.get('main_pitch_pct',0):.1f}%",
                sub2=f"2nd: {metrics.get('main_pitch_2nd','-')} "
                    f"{metrics.get('main_pitch_2nd_pct',0):.1f}%",
                icon="🎯"),
                xs=3, sm=3, md=True, className="mb-2 px-1"),
            dbc.Col(_mini_card("초구(0-0)",
                metrics.get("fp_main", "-"),
                sub=f"{metrics.get('fp_main_pct',0):.1f}%",
                sub2=f"2nd: {metrics.get('fp_2nd','')} "
                    f"{metrics.get('fp_2nd_pct',0):.1f}%"
                    if metrics.get("fp_2nd") else "",
                icon="1️⃣"),
                xs=3, sm=3, md=True, className="mb-2 px-1"),
            dbc.Col(_mini_card("2S결정구",
                metrics.get("ts_main", "-"),
                sub=f"{metrics.get('ts_main_pct',0):.1f}%",
                sub2=f"2nd: {metrics.get('ts_2nd','')} "
                    f"{metrics.get('ts_2nd_pct',0):.1f}%"
                    if metrics.get("ts_2nd") else "",
                icon="2️⃣"),
                xs=3, sm=3, md=True, className="mb-2 px-1"),
            dbc.Col(_mini_card("헛스윙",
                metrics.get("sw_main", "-"),
                sub=f"SwStr {metrics.get('sw_main_pct',0):.1f}%",
                sub2=f"2nd: {metrics.get('sw_2nd','')} "
                    f"{metrics.get('sw_2nd_pct',0):.1f}%"
                    if metrics.get("sw_2nd") else "",
                icon="💨"),
                xs=3, sm=3, md=True, className="mb-2 px-1"),
        ], className="mb-3 g-1")

        # ★ 수정: 주요스탯 테이블 — df(필터) 기준, 전체기간은 expander
        stats_df = make_dashboard_stats_table(df, _base_src, is_date_filtered)

        stats_df_slim = stats_df

        # ★ 수정: 메인 차트는 df(필터) 기준
        usage_fig = make_pitch_usage_chart(
            df, pitcher_name,
            base_df=_base_src if is_date_filtered else None,
            is_date_filtered=is_date_filtered)
        usage_fig.update_layout(
            title=None, autosize=True,
            margin=dict(l=90, r=20, t=5, b=80))

        mov_fig = make_movement_chart(
            df, pitcher_name,
            is_date_filtered=is_date_filtered,
            base_df=_base_src if is_date_filtered else None)

        
        mov_fig.update_layout(
            title=None, autosize=True,
            margin=dict(l=50, r=110, t=5, b=45))

        tracking_df = make_tracking_table(
            df, year=int(latest_year),
            is_date_filtered=is_date_filtered,
            base_df=_base_src if is_date_filtered else None)

        left_col = html.Div([
            dbc.Card(dbc.CardBody([
                html.P("📊 주요 스탯",
                        style={"fontWeight": "700", "color": "#2c3e50",
                            "fontSize": "12px", "marginBottom": "8px"}),
                make_stat_table(stats_df_slim, "dash-stats-table", font_size="11px"),
            ]), className="mb-3",
                style={**dcard_style, "overflowX": "auto"}),
            dbc.Card(dbc.CardBody([
                html.P("📊 구종사용비율",
                        style={"fontWeight": "700", "color": "#2c3e50",
                            "fontSize": "12px", "marginBottom": "4px"}),
                html.Div(
                    dcc.Graph(figure=usage_fig,
                            config={"displayModeBar": False},
                            style={"width": "100%", "height": "420px"}),
                    style={"width": "100%", "minWidth": "320px",
                            "maxWidth": "620px", "margin": "0 auto",
                            "display": "flex", "justifyContent": "center",
                            "alignItems": "center"}),
            ], style={"display": "flex", "flexDirection": "column",
                    "height": "100%", "padding": "12px"}),
            style={**dcard_style, "flex": "1", "display": "flex",
                    "flexDirection": "column", "minHeight": "0"}),
        ], style={"display": "flex", "flexDirection": "column", "height": "100%"})

        right_col = html.Div([
            dbc.Card(dbc.CardBody([
                html.P("📐 구종별 무브먼트",
                        style={"fontWeight": "700", "color": "#2c3e50",
                            "fontSize": "12px", "marginBottom": "4px"}),
                html.Div(
                    dcc.Graph(figure=mov_fig,
                            config={"displayModeBar": False},
                            style={"width": "100%", "height": "420px"}),
                    style={"width": "100%", "minWidth": "320px",
                            "maxWidth": "560px", "margin": "0 auto",
                            "display": "flex", "justifyContent": "center",
                            "alignItems": "center"}),
            ]), className="mb-3",
                style={**dcard_style, "flex": "0 0 auto"}),
            dbc.Card(dbc.CardBody([
                html.P("📡 구종별 트래킹",
                        style={"fontWeight": "700", "color": "#2c3e50",
                            "fontSize": "12px", "marginBottom": "8px"}),
                make_stat_table(tracking_df, "dash-tracking-table", font_size="11px"),
            ]), style={**dcard_style, "flex": "1", "display": "flex",
                        "flexDirection": "column", "overflowX": "auto"}),
        ], style={"display": "flex", "flexDirection": "column", "height": "100%"})

        return html.Div([
            html.H5([
                html.Span("⚾ ", style={"color": "#e63946"}),
                html.Span(f"{pitcher_name}",
                        style={"fontWeight": "800", "color": "#1a2535"}),
                html.Span(f"  {latest_year} 시즌 대시보드",
                        style={"fontWeight": "400", "color": "#6b7a8d",
                                "fontSize": "14px"}),
            ], style={"marginBottom": "10px", "paddingBottom": "8px",
                    "borderBottom": "2px solid #f0f4f8"}),
            *_filter_badge(),
            top_cards,
            dbc.Row([
                dbc.Col(left_col,  md=6, className="pe-2",
                        style={"display": "flex", "flexDirection": "column"}),
                dbc.Col(right_col, md=6, className="ps-2",
                        style={"display": "flex", "flexDirection": "column"}),
            ], style={"alignItems": "stretch"}, className="g-0"),
        ])

    # ──────────────────────────────────────────────────────
    # 탭 2: 주요스탯
    # ──────────────────────────────────────────────────────
    elif active_tab == "tab-overview":
        # ── 투수 투구 손 판별 (한 번만) ──
        p_throws = get_pitcher_throws(df, pitcher_id, league)

        key_df   = make_key_stats_table(df)                              # 통합 → LEAGUE_AVG_ROW
        pitch_df = make_pitch_stats_table(df, latest_year)
        key_L    = make_key_stats_table(df, stand="L", pitcher_throws=p_throws)
        key_R    = make_key_stats_table(df, stand="R", pitcher_throws=p_throws)
        pitch_L  = make_pitch_stats_table(df, latest_year, stand="L")
        pitch_R  = make_pitch_stats_table(df, latest_year, stand="R")

        stand_section = dbc.Collapse([
            dbc.Row([
                dbc.Col([
                    html.P("좌타자 (L)", style={"fontWeight":"700",
                            "color":"#2c7be5","marginBottom":"4px"}),
                    make_stat_table(key_L,   "ov-key-L"),
                    html.Br(),
                    make_stat_table(pitch_L, "ov-pitch-L"),
                ], md=6),
                dbc.Col([
                    html.P("우타자 (R)", style={"fontWeight":"700",
                            "color":"#e55c2c","marginBottom":"4px"}),
                    make_stat_table(key_R,   "ov-key-R"),
                    html.Br(),
                    make_stat_table(pitch_R, "ov-pitch-R"),
                ], md=6),
            ])
        ], id="stand-collapse", is_open=False)

        runner_section = dbc.Collapse(
            make_runner_stats_section(
                df, latest_year, dcard_style,
                id_suffix="-ov",
                pitcher_throws=p_throws),          # ← 추가
            id="runner-collapse-ov", is_open=False)

        # ★ cmp_section — 항상 표시 (is_date_filtered 조건 제거)
        _base_src = _get_base_src()
        cmp_section_content = []
        if _base_src is not None and not _base_src.empty:
            base_key_df   = make_key_stats_table(_base_src)             # 통합 → LEAGUE_AVG_ROW
            base_pitch_df = make_pitch_stats_table(
                _base_src, int(_base_src["game_year"].max()))
            cmp_section_content = [
                html.P("📅 전체기간 (날짜필터 없음)",
                        style={"fontWeight":"700","color":"#7a5c00",
                            "marginBottom":"6px"}),
                make_stat_table(base_key_df,   "ov-cmp-key"),
                html.Br(),
                make_stat_table(base_pitch_df, "ov-cmp-pitch"),
            ]
        cmp_section = dbc.Collapse(
            cmp_section_content, id="cmp-collapse", is_open=False)

        return html.Div([
            html.H6(f"📊 {pitcher_name} — 주요 스탯",
                    style={"fontWeight":"700","marginBottom":"12px"}),
            *_filter_badge(),
            dbc.Card(dbc.CardBody([
                make_stat_table(key_df,   "ov-key-all"),
                html.Br(),
                make_stat_table(pitch_df, "ov-pitch-all"),
            ]), className="mb-3", style=dcard_style),
            html.Button("▼ 타자유형별 스탯 보기",
                        id="stand-collapse-btn", n_clicks=0,
                        style=EXPANDER_BTN_STYLE),
            stand_section,
            html.Button("▼ 주자 상황별 스탯 보기",
                        id="runner-collapse-btn-ov", n_clicks=0,
                        style=EXPANDER_BTN_RUNNER_STYLE),
            runner_section,
            html.Button("▼ 전체기간 비교 보기",
                        id="cmp-collapse-btn", n_clicks=0,
                        style=EXPANDER_BTN_CMP_STYLE),
            cmp_section,
        ])


    # ──────────────────────────────────────────────────────
    # 탭 3: 무브먼트 차트
    # ──────────────────────────────────────────────────────
    elif active_tab == "tab-movement":
        # ★ 수정: 메인 차트 = df(필터) 기준
        mov_fig = make_movement_chart(df, pitcher_name, is_date_filtered=is_date_filtered)

        # ★ 수정: 트래킹 = df(필터) 기준
        tracking_df = make_tracking_table(
            df, year=int(latest_year),
            is_date_filtered=is_date_filtered,
            base_df=_get_base_src() if is_date_filtered else None)

        # ★ 전체기간 expander용 차트/트래킹
        _base_src       = _get_base_src()
        _base_latest    = int(_base_src["game_year"].max())
        base_mov_fig    = make_movement_chart(_base_src, pitcher_name, is_date_filtered=False)
        base_tracking   = make_tracking_table(
            _base_src, year=_base_latest,
            is_date_filtered=False, base_df=None)

        base_content = [
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(
                    html.Div(
                        dcc.Graph(
                            figure=mov_fig,
                            config={"displayModeBar": True},
                            style={"width": "100%", "height": "650px"}),
                        style={
                            "position": "relative",
                            "width": "100%",
                            "paddingBottom": "100%",   # 1:1 비율 유지
                        },
                        id="mov-chart-square-wrapper"
                    )
                ), style=dcard_style), md=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("📡 구종별 트래킹 (전체기간)",
                            style={"fontWeight": "700", "marginBottom": "8px",
                                "color": "#2c3e50"}),
                    make_stat_table(base_tracking, "mov-base-tracking-table", font_size="11px"),
                ]), style=dcard_style), md=5),
            ]),
        ]

        return html.Div([
            html.H6(f"📐 {pitcher_name} — 무브먼트 차트",
                    style={"fontWeight": "700", "marginBottom": "12px"}),
            *_filter_badge(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=mov_fig,
                            config={"displayModeBar": True},
                            style={"width": "100%", "height": "650px"})),
                    style=dcard_style), md=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("📡 구종별 트래킹",
                            style={"fontWeight": "700", "marginBottom": "8px",
                                "color": "#2c3e50"}),
                    make_stat_table(tracking_df, "mov-tracking-table", font_size="11px"),
                ]), style=dcard_style), md=5),
            ]),
            html.Div(style={"marginTop": "12px"}),
            html.Button("▼ 전체기간 차트 보기",
                        id="mov-base-collapse-btn", n_clicks=0,
                        style=EXPANDER_BTN_BASE_STYLE),
            dbc.Collapse(base_content, id="mov-base-collapse", is_open=False),
        ])

    # ──────────────────────────────────────────────────────
    # 탭 4: 구종사용비율 (타자유형별)
    # ──────────────────────────────────────────────────────
    elif active_tab == "tab-usage":
        # ★ 수정: 메인 차트 = df(필터) 기준
        _base_src = _get_base_src()
        usage_fig = make_pitch_usage_chart(
            df, pitcher_name,
            base_df=_base_src if is_date_filtered else None,
            is_date_filtered=is_date_filtered)

        no_runner_df, yes_runner_df = split_by_runner(df)
        runner_section = dbc.Collapse([
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=make_runner_usage_chart(df, pitcher_name),
                        config={"displayModeBar": False},
                        style={"height": "380px"})),
                className="mb-3", style=dcard_style),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(
                    dcc.Graph(
                        figure=make_runner_usage_by_stand_chart(
                            no_runner_df, pitcher_name, runner_label="무주자"),
                        config={"displayModeBar": False},
                        style={"height": "340px"})),
                    style=dcard_style), md=6),
                dbc.Col(dbc.Card(dbc.CardBody(
                    dcc.Graph(
                        figure=make_runner_usage_by_stand_chart(
                            yes_runner_df, pitcher_name, runner_label="유주자"),
                        config={"displayModeBar": False},
                        style={"height": "340px"})),
                    style=dcard_style), md=6),
            ]),
        ], id="runner-collapse-us", is_open=False)

        # ★ 전체기간 expander용 차트
        base_usage_fig = make_pitch_usage_chart(
            _base_src, pitcher_name, base_df=None, is_date_filtered=False)

        base_content = [
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=base_usage_fig,
                        config={"displayModeBar": False},
                        style={"height": "420px"})),
                className="mb-3", style=dcard_style),
        ]

        return html.Div([
            html.H6(f"📊 {pitcher_name} — 구종사용비율 (타자유형별)",
                    style={"fontWeight": "700", "marginBottom": "12px"}),
            *_filter_badge(),
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=usage_fig,
                        config={"displayModeBar": False},
                        style={"height": "420px"})),
                className="mb-3", style=dcard_style),
            html.Button("▼ 주자 상황별 스탯 보기",
                        id="runner-collapse-btn-us", n_clicks=0,
                        style=EXPANDER_BTN_RUNNER_STYLE),
            runner_section,
            html.Button("▼ 전체기간 차트 보기",
                        id="usage-base-collapse-btn", n_clicks=0,
                        style=EXPANDER_BTN_BASE_STYLE),
            dbc.Collapse(base_content, id="usage-base-collapse", is_open=False),
        ])

    # ──────────────────────────────────────────────────────
    # 탭 5: 구종사용비율 (카운트별)
    # ──────────────────────────────────────────────────────
    elif active_tab == "tab-count":
        # ★ 수정: 메인 차트 = df(필터) 기준
        count_fig = make_count_chart(df, pitcher_name, orientation or "landscape")
        no_runner_df, yes_runner_df = split_by_runner(df)

        count_runner_charts = []
        for c_label, c_vals in COUNT_GROUPS.items():
            slug = COUNT_SLUG[c_label]
            count_runner_charts.append(html.Div([
                html.P(f"📌 {c_label}",
                        style={"fontWeight": "700", "color": "#5a3e8a",
                            "marginBottom": "6px"}),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody(
                        dcc.Graph(
                            figure=make_runner_count_usage_chart(
                                df, pitcher_name, c_label, c_vals),
                            config={"displayModeBar": False},
                            style={"height": "320px"})),
                        style=dcard_style), md=4),
                    dbc.Col(dbc.Card(dbc.CardBody(
                        dcc.Graph(
                            figure=make_runner_count_stand_chart(
                                no_runner_df, pitcher_name,
                                c_vals, "무주자", c_label),
                            config={"displayModeBar": False},
                            style={"height": "320px"})),
                        style=dcard_style), md=4),
                    dbc.Col(dbc.Card(dbc.CardBody(
                        dcc.Graph(
                            figure=make_runner_count_stand_chart(
                                yes_runner_df, pitcher_name,
                                c_vals, "유주자", c_label),
                            config={"displayModeBar": False},
                            style={"height": "320px"})),
                        style=dcard_style), md=4),
                ], className="mb-3"),
                html.Button(
                    f"▼ {c_label} — 좌/우타자별 보기",
                    id=f"count-stand-btn-{slug}-ct", n_clicks=0,
                    style=EXPANDER_BTN_COUNT_RUNNER_STYLE),
                dbc.Collapse([], id=f"count-stand-collapse-{slug}-ct", is_open=False),
            ], className="mb-4"))

        # ★ 전체기간 expander용 차트
        _base_src      = _get_base_src()
        base_count_fig = make_count_chart(_base_src, pitcher_name, orientation or "landscape")

        base_content = [
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=base_count_fig,
                        config={"displayModeBar": False},
                        style={"height": "460px" if orientation == "portrait"
                                else "380px"})),
                className="mb-3", style=dcard_style),
        ]

        return html.Div([
            html.H6(f"📊 {pitcher_name} — 구종사용비율 (카운트별)",
                    style={"fontWeight": "700", "marginBottom": "12px"}),
            *_filter_badge(),
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=count_fig,
                        config={"displayModeBar": False},
                        style={"height": "460px" if orientation == "portrait"
                                else "380px"})),
                className="mb-3", style=dcard_style),
            html.Button("▼ 카운트별 주자 상황 보기",
                        id="runner-collapse-btn-ct", n_clicks=0,
                        style=EXPANDER_BTN_COUNT_RUNNER_STYLE),
            dbc.Collapse(count_runner_charts, id="runner-collapse-ct", is_open=False),
            html.Button("▼ 전체기간 차트 보기",
                        id="count-base-collapse-btn", n_clicks=0,
                        style=EXPANDER_BTN_BASE_STYLE),
            dbc.Collapse(base_content, id="count-base-collapse", is_open=False),
        ])

    # ──────────────────────────────────────────────────────
    # 탭 6: 구종별 로케이션
    # ──────────────────────────────────────────────────────
    elif active_tab == "tab-location":
        _base_src = _get_base_src()
        exist_descs = df["description"].dropna().unique().tolist()
    
        # 실제 데이터에 존재하는 그룹만 옵션으로
        group_options = []
        for gkey, ginfo in LOC_GROUPS.items():
            if any(d in exist_descs for d in ginfo["descs"]):
                group_options.append({
                    "label": ginfo["label"],
                    "value": gkey,
                })
    
        year_figs = []
        for yr in sorted(_base_src["game_year"].unique(), reverse=True):
            fig_yr = make_location_fig(_base_src, pitcher_name, year_filter=yr)
            year_figs.append(dbc.Card(dbc.CardBody([
                html.P(f"📅 {int(yr)} 시즌",
                       style={"fontWeight": "700", "color": "#2c3e50",
                              "marginBottom": "6px"}),
                dcc.Graph(figure=fig_yr, config={"displayModeBar": False},
                          style={"overflowX": "auto"}),
            ]), className="mb-3", style=dcard_style))
    
        return html.Div([
            html.H6(f"📍 {pitcher_name} — 구종별 로케이션",
                    style={"fontWeight": "700", "marginBottom": "12px"}),
            *_filter_badge(),
    
            # ★ 3그룹 필터 UI
            dbc.Card(dbc.CardBody([
                html.Div([
                    html.Span("🎛 타석결과 필터",
                              style={"fontWeight": "700", "fontSize": "13px",
                                     "color": "#2c3e50", "marginRight": "12px"}),
                    html.Span("(복수 선택 가능)",
                              style={"fontSize": "11px", "color": "#888"}),
                ], style={"marginBottom": "10px"}),
                dcc.Checklist(
                    id="loc-desc-filter",
                    options=group_options,
                    value=[g["value"] for g in group_options],  # 초기: 전체 선택
                    inline=True,
                    labelStyle={
                        "marginRight": "12px",
                        "fontSize": "13px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                        "padding": "5px 14px",
                        "borderRadius": "20px",
                        "border": "1px solid #d0d8e4",
                        "backgroundColor": "#f0f4f8",
                        "display": "inline-block",
                        "transition": "all 0.2s ease",
                    },
                ),
            ]), className="mb-3", style=dcard_style),
    
            # ★ 필터 적용 차트 영역
            html.Div(id="loc-filtered-chart-area"),
    
            html.Button("▼ 연도별 로케이션 보기",
                        id="loc-year-collapse-btn", n_clicks=0,
                        style=EXPANDER_BTN_STYLE),
            dbc.Collapse(year_figs, id="loc-year-collapse", is_open=False),
        ])

    # ──────────────────────────────────────────────────────
    # 탭 7: 결과별 로케이션
    # ──────────────────────────────────────────────────────
    elif active_tab == "tab-result-loc":
        _base_src = _get_base_src()
    
        # 구종 선택지 생성
        exist_pitches  = [p for p in PITCH_ORDER if p in df["pitch_name"].unique()]
        pitch_options  = [{"label": p, "value": p} for p in exist_pitches]
    
        fig_all = make_result_location_chart(df, pitcher_name)
    
        year_figs = []
        for yr in sorted(_base_src["game_year"].unique(), reverse=True):
            fig_yr = make_result_location_chart(_base_src, pitcher_name, year_filter=yr)
            year_figs.append(dbc.Card(dbc.CardBody([
                html.P(f"📅 {int(yr)} 시즌",
                       style={"fontWeight": "700", "color": "#2c3e50", "marginBottom": "6px"}),
                dcc.Graph(figure=fig_yr, config={"displayModeBar": False},
                          style={"overflowX": "auto"}),
            ]), className="mb-3", style=dcard_style))
    
        return html.Div([
            html.H6(f"🎯 {pitcher_name} — 결과별 로케이션",
                    style={"fontWeight": "700", "marginBottom": "12px"}),
            *_filter_badge(),
    
            # ★ 구종 필터 UI
            dbc.Card(dbc.CardBody([
                html.Div([
                    html.Span("🎛 구종 필터",
                              style={"fontWeight": "700", "fontSize": "13px",
                                     "color": "#2c3e50", "marginRight": "12px"}),
                    html.Span("(선택한 구종만 차트에 표시 / 미선택 시 전체)",
                              style={"fontSize": "11px", "color": "#888"}),
                ], style={"marginBottom": "8px"}),
                dcc.RadioItems(
                    id="res-loc-pitch-filter",
                    options=[{"label": "전체", "value": "__ALL__"}] + pitch_options,
                    value="__ALL__",
                    inline=True,
                    labelStyle={
                        "marginRight": "10px", "marginBottom": "6px",
                        "fontSize": "12px", "cursor": "pointer",
                        "padding": "3px 10px",
                        "backgroundColor": "#f0f4f8",
                        "borderRadius": "10px",
                        "border": "1px solid #d0d8e4",
                        "display": "inline-block",
                    },
                ),
            ]), className="mb-3", style=dcard_style),
    
            # ★ 필터 적용 차트 영역
            html.Div(id="res-loc-filtered-chart-area"),
    
            html.Button("▼ 연도별 결과 로케이션 보기",
                        id="res-loc-year-collapse-btn", n_clicks=0,
                        style=EXPANDER_BTN_STYLE),
            dbc.Collapse(year_figs, id="res-loc-year-collapse", is_open=False),
        ])

    # ──────────────────────────────────────────────────────
    # 탭 8: 리그 비교
    # ──────────────────────────────────────────────────────
    elif active_tab == "tab-league":
        lg_df = league_cache.get(league, pd.DataFrame())
        if lg_df.empty:
            return html.Div("리그 데이터 없음",
                            style={"color": "#aaa", "padding": "40px",
                                    "textAlign": "center"})
        min_ip_val = min_ip if min_ip else 30
        lg_stats   = calc_pitcher_stats(lg_df)
        lg_stats   = lg_stats[lg_stats["IP"] >= min_ip_val]

        id_col = "pitcher" if "pitcher" in lg_df.columns \
                else get_name_col(league, lg_df)
        player_stats_all = calc_pitcher_stats(
            lg_df[lg_df[id_col].astype(str) == str(pitcher_id)])
        player_stats_L   = calc_pitcher_stats(
            lg_df[(lg_df[id_col].astype(str) == str(pitcher_id)) &
                (lg_df["stand"] == "L")])
        player_stats_R   = calc_pitcher_stats(
            lg_df[(lg_df[id_col].astype(str) == str(pitcher_id)) &
                (lg_df["stand"] == "R")])

        scatter_pairs = [
            ("K%",    "BB%",   "K%",    "BB%",   f"{pitcher_name} K% vs BB%"),
            ("SwStr%","CSW%",  "SwStr%","CSW%",  f"{pitcher_name} SwStr% vs CSW%"),
        ]
        scatter_cards = []
        for xc, yc, xt, yt, title in scatter_pairs:
            if xc in lg_stats.columns and yc in lg_stats.columns:
                fig = make_scatter_chart(
                    lg_stats, player_stats_all,
                    player_stats_L, player_stats_R,
                    xc, yc, xt, yt, title)
                scatter_cards.append(
                    dbc.Col(dbc.Card(dbc.CardBody(
                        dcc.Graph(figure=fig,
                                config={"displayModeBar": False},
                                style={"height": "420px"})),
                        style=dcard_style), md=6, className="mb-3"))

        return html.Div([
            html.H6(f"📈 {pitcher_name} — 리그 비교  (Min IP: {min_ip_val})",
                    style={"fontWeight": "700", "marginBottom": "12px"}),
            dbc.Row(scatter_cards),
        ])

    return no_data

# ════════════════════════════════════════════════════════════
# 15. 앱 실행
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
    