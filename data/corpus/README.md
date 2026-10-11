# Multi-season corpus

| File | Rows | Content |
|---|---|---|
| `player_seasons.csv.gz` | 13,229 | One row per player and season (big five leagues, seasons ending 2018 to 2022): season totals for shooting, passing, pass types, shot creation, defending, pressing, possession and carrying, plus Transfermarkt position, height, foot and market value |
| `transfers.csv.gz` | 21,182 | Arrivals and departures of big-five clubs, seasons 2018 to 2023, for players that can be linked to the table above |

`player_id` is the FBref player id. It is the same in every season, so a player can be
followed over time without matching names.

## Source and how it was built

Both files are converted from [worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data)
by `scripts/build_corpus.py`:

- `data/fb_big5_advanced_season_stats/big5_player_*.rds`: FBref season tables, collected while
  FBref carried StatsBomb data
- `raw-data/fbref-tm-player-mapping/output/fbref_to_tm_mapping.csv`: FBref to Transfermarkt player mapping
- `data/tm_player_vals/big5_player_vals.rds` and `data/tm_transfers/big_5_transfers.rds`: Transfermarkt

The underlying statistics belong to their providers (FBref / StatsBomb, Transfermarkt). They are
included here for research and teaching. Cite the original sources if you publish results.

## Why these seasons

- FBref removed all advanced statistics in January 2026, so they cannot be collected for
  recent seasons any more. This archive predates the removal.
- The 2022-23 rows in the source stop mid-season and come from a different data provider
  (no pressures), so the corpus ends with 2021-22.
