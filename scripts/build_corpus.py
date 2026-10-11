"""Build the multi-season corpus in ``data/corpus/`` from a local clone of worldfootballR_data.

    git clone https://github.com/JaseZiv/worldfootballR_data
    python scripts/build_corpus.py path/to/worldfootballR_data

Source: https://github.com/JaseZiv/worldfootballR_data (FBref season tables for the big five
leagues, collected while FBref carried StatsBomb data, plus Transfermarkt positions, market
values and transfers, and the FBref-to-Transfermarkt player mapping). The converted tables
are committed to this repository, so this script only needs to be run to rebuild them.

Seasons ending 2018 to 2022 are kept. The 2022-23 rows in the source stop mid-season and come
from a different data provider (no pressures), so they are left out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from footballiq import config                      # noqa: E402
from footballiq.rds import read_rds                # noqa: E402

SEASONS = (2018, 2022)
KEY = ["Season_End_Year", "Squad", "Url"]

# source table -> {source column: corpus column}. Every column is a season total.
COUNTS = {
    "standard": {"MP_Playing": "MP", "Starts_Playing": "Starts", "Min_Playing": "Min", "Gls": "Gls",
                 "Ast": "Ast", "G_minus_PK": "npG", "PK": "PK", "PKatt": "PKatt", "CrdY": "CrdY",
                 "CrdR": "CrdR", "xG_Expected": "xG", "npxG_Expected": "npxG"},
    "shooting": {"Sh_Standard": "Sh", "SoT_Standard": "SoT", "FK_Standard": "ShFK"},
    "passing": {"Cmp_Total": "PassCmp", "Att_Total": "PassAtt", "TotDist_Total": "PassDist",
                "PrgDist_Total": "PassPrgDist", "Cmp_Short": "ShortCmp", "Att_Short": "ShortAtt",
                "Cmp_Medium": "MediumCmp", "Att_Medium": "MediumAtt", "Cmp_Long": "LongCmp",
                "Att_Long": "LongAtt", "xA": "xA", "KP": "KeyPasses", "Final_Third": "PassFinalThird",
                "PPA": "PassPenArea", "CrsPA": "CrossPenArea", "Prog": "PrgPasses"},
    "passing_types": {"Live_Pass": "PassLive", "Dead_Pass": "PassDead", "FK_Pass": "PassFK",
                      "TB_Pass": "ThroughBalls", "Press_Pass": "PassUnderPressure", "Sw_Pass": "Switches",
                      "Crs_Pass": "Crosses", "CK_Pass": "Corners", "Ground_Height": "PassGround",
                      "Low_Height": "PassLow", "High_Height": "PassHigh", "Head_Body": "PassHead",
                      "TI_Body": "ThrowIns", "Off_Outcomes": "PassOffside", "Int_Outcomes": "PassIntercepted",
                      "Blocks_Outcomes": "PassBlocked"},
    "gca": {"SCA_SCA": "SCA", "PassLive_SCA": "SCAPassLive", "PassDead_SCA": "SCAPassDead",
            "Drib_SCA": "SCADribble", "Sh_SCA": "SCAShot", "Fld_SCA": "SCAFouled", "Def_SCA": "SCADefence",
            "GCA_GCA": "GCA"},
    "defense": {"Tkl_Tackles": "Tackles", "TklW_Tackles": "TacklesWon", "Def 3rd_Tackles": "TacklesDef3rd",
                "Mid 3rd_Tackles": "TacklesMid3rd", "Att 3rd_Tackles": "TacklesAtt3rd",
                "Tkl_Vs": "DribblersTackled", "Att_Vs": "DribblersFaced", "Past_Vs": "DribbledPast",
                "Press_Pressures": "Pressures", "Succ_Pressures": "PressuresWon",
                "Def 3rd_Pressures": "PressuresDef3rd", "Mid 3rd_Pressures": "PressuresMid3rd",
                "Att 3rd_Pressures": "PressuresAtt3rd", "Blocks_Blocks": "Blocks", "Sh_Blocks": "ShotsBlocked",
                "Pass_Blocks": "PassesBlocked", "Int": "Interceptions", "Clr": "Clearances", "Err": "Errors"},
    "possession": {"Touches_Touches": "Touches", "Def Pen_Touches": "TouchesDefPen",
                   "Def 3rd_Touches": "TouchesDef3rd", "Mid 3rd_Touches": "TouchesMid3rd",
                   "Att 3rd_Touches": "TouchesAtt3rd", "Att Pen_Touches": "TouchesAttPen",
                   "Succ_Dribbles": "DribblesWon", "Att_Dribbles": "Dribbles", "Megs_Dribbles": "Nutmegs",
                   "Carries_Carries": "Carries", "TotDist_Carries": "CarryDist",
                   "PrgDist_Carries": "CarryPrgDist", "Prog_Carries": "PrgCarries",
                   "Final_Third_Carries": "CarriesFinalThird", "CPA_Carries": "CarriesPenArea",
                   "Mis_Carries": "Miscontrols", "Dis_Carries": "Dispossessed", "Targ_Receiving": "PassTargets",
                   "Rec_Receiving": "PassesReceived", "Prog_Receiving": "PrgPassesReceived"},
    "misc": {"Fls": "Fouls", "Fld": "Fouled", "Off": "Offsides", "PKwon": "PKwon", "PKcon": "PKcon",
             "OG": "OwnGoals", "Recov": "Recoveries", "Won_Aerial": "AerialsWon", "Lost_Aerial": "AerialsLost"},
}
TM_POSITION = {"midfield": "Central Midfield", "Midfield": "Central Midfield", "attack": "Centre-Forward",
               "Attack": "Centre-Forward", "defence": "Centre-Back", "Defence": "Centre-Back",
               "Defender": "Centre-Back"}


def player_id(url: pd.Series) -> pd.Series:
    return url.str.extract(r"/players/([0-9a-f]{8})/")[0]


def build(source: Path) -> None:
    stats_dir = source / "data" / "fb_big5_advanced_season_stats"
    merged = None
    for table, columns in COUNTS.items():
        frame = read_rds(stats_dir / f"big5_player_{table}.rds")
        frame = frame[frame["Season_End_Year"].between(*SEASONS)]
        keep = frame[KEY + list(columns)].rename(columns=columns)
        if table == "standard":
            keep = keep.join(frame[["Player", "Nation", "Pos", "Age", "Born", "Comp"]])
        if table == "shooting":                       # average shot distance is not additive
            keep["ShotDistTotal"] = frame["Dist_Standard"].fillna(0) * frame["Sh_Standard"].fillna(0)
        merged = keep if merged is None else merged.merge(keep, on=KEY, how="left", validate="1:1")
    merged["player_id"] = player_id(merged["Url"])
    numeric = [c for cols in COUNTS.values() for c in cols.values()] + ["ShotDistTotal"]
    merged[numeric] = merged[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # one row per player-season: totals summed over club spells, identity from the main spell
    merged = merged.sort_values("Min", ascending=False)
    grouped = merged.groupby(["Season_End_Year", "player_id"], sort=False)
    main = grouped[["Player", "Nation", "Pos", "Age", "Born", "Comp", "Squad", "Url"]].first()
    totals = grouped[numeric].sum()
    extra = pd.concat([grouped["Squad"].agg(" / ".join).rename("Squads"),
                       grouped.size().rename("n_spells")], axis=1)
    seasons = pd.concat([main, extra, totals], axis=1).reset_index().copy()

    # Transfermarkt: position (season-specific where available), height, foot, market value
    mapping = pd.read_csv(source / "raw-data" / "fbref-tm-player-mapping" / "output" / "fbref_to_tm_mapping.csv",
                          encoding="latin-1")
    mapping["player_id"] = player_id(mapping["UrlFBref"])
    seasons = seasons.merge(mapping[["player_id", "UrlTmarkt", "TmPos"]].drop_duplicates("player_id"),
                            on="player_id", how="left")
    values = read_rds(source / "data" / "tm_player_vals" / "big5_player_vals.rds")
    values["Season_End_Year"] = values["season_start_year"].astype(int) + 1
    values = values.sort_values("player_market_value_euro", ascending=False).drop_duplicates(
        ["Season_End_Year", "player_url"])
    values = values.rename(columns={"player_url": "UrlTmarkt", "player_position": "tm_position_season",
                                    "player_height_mtrs": "height_m", "player_foot": "foot",
                                    "player_market_value_euro": "market_value_eur"})
    seasons = seasons.merge(values[["Season_End_Year", "UrlTmarkt", "tm_position_season", "height_m",
                                    "foot", "market_value_eur"]],
                            on=["Season_End_Year", "UrlTmarkt"], how="left")
    seasons["tm_position"] = seasons["tm_position_season"].fillna(seasons["TmPos"]).replace(TM_POSITION)
    seasons["height_m"] = pd.to_numeric(seasons["height_m"], errors="coerce")
    seasons = seasons.drop(columns=["TmPos", "tm_position_season", "Url"])
    seasons = seasons.rename(columns={"Season_End_Year": "season", "Player": "player", "Nation": "nation",
                                      "Pos": "pos", "Age": "age", "Born": "born", "Comp": "league",
                                      "Squad": "squad", "Squads": "squads", "UrlTmarkt": "tm_url"})
    seasons = seasons.sort_values(["season", "player", "player_id"]).reset_index(drop=True)
    counts = [c for c in numeric if c not in ("xG", "npxG", "xA", "PassDist", "PassPrgDist", "CarryDist",
                                              "CarryPrgDist", "ShotDistTotal")]
    seasons[counts] = seasons[counts].round().astype(int)
    seasons = seasons.round({"xG": 2, "npxG": 2, "xA": 2, "ShotDistTotal": 1})

    # transfers between the big five leagues, mapped to FBref player ids
    transfers = read_rds(source / "data" / "tm_transfers" / "big_5_transfers.rds")
    transfers["season"] = transfers["season"].astype(int) + 1          # season the move is for
    transfers = transfers[transfers["season"].between(SEASONS[0], SEASONS[1] + 1)]
    transfers = transfers.merge(mapping[["UrlTmarkt", "player_id"]].drop_duplicates("UrlTmarkt"),
                                left_on="player_url", right_on="UrlTmarkt", how="inner")
    transfers = transfers[["season", "window", "team_name", "league", "transfer_type", "player_id",
                           "player_name", "player_position", "club_2", "transfer_fee", "is_loan"]]
    transfers = transfers.sort_values(["season", "team_name", "transfer_type", "player_name"])

    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    seasons.to_csv(config.CORPUS_SEASONS, index=False, compression={"method": "gzip", "mtime": 0})
    transfers.to_csv(config.CORPUS_TRANSFERS, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"player-seasons: {len(seasons)} rows, {seasons['player_id'].nunique()} players, "
          f"{len(seasons.columns)} columns")
    print(f"transfers: {len(transfers)} rows")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(Path(sys.argv[1]))
