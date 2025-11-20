# utils.py

#--------------------------
# Basketball Ref Scraper
#--------------------------
from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_bref_per_game(
    season: int,
    cache_path: str | None = None,
    use_cache_if_fail: bool = True,
) -> pd.DataFrame:
    """
    Load Basketball-Reference per-game stats for a given season.

    - First tries to scrape via pandas.read_html.
    - If that fails and a cache_path exists and use_cache_if_fail=True,
      it falls back to the cached CSV.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"

    try:
        # Let pandas handle the HTTP request
        dfs = pd.read_html(url, attrs={"id": "per_game_stats"})
        df = dfs[0]

        # Drop repeated header rows
        df = df[df["Rk"] != "Rk"].reset_index(drop=True)

        # Convert numeric-looking columns
        non_numeric = ["Player", "Team", "Pos", "Awards"]
        numeric_cols = [c for c in df.columns if c not in non_numeric]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Make Rk int-like
        df["Rk"] = df["Rk"].astype("Int64")

        # Save cache if requested
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)

        return df

    except Exception as e:
        # Optional fallback to existing CSV
        if cache_path and use_cache_if_fail and Path(cache_path).exists():
            print(
                f"Warning: failed to scrape {url} ({e}). "
                f"Falling back to cached file: {cache_path}"
            )
            return pd.read_csv(cache_path)

        # If no cache, re-raise
        raise



# -------------------------------------------------------------------
# Player Index + Weighted Merge Utilities for roundBall
# -------------------------------------------------------------------
#from __future__ import annotations
import re
from pathlib import Path
import unicodedata  # add this near the top of utils.py if not already there
from typing import Iterable, List, Dict, Optional

import pandas as pd


# =========================
# Name normalization
# =========================
def normalize_name(s: str) -> str:
    """
    Normalize player names to improve cross-file joins:
      - remove parenthetical junk like (OKC - PG)
      - strip trailing separators like " - ", " — ", " | " and anything after
      - strip trailing TEAM/POS tokens
      - remove accent marks (Jokić -> Jokic)
      - lowercase, trim, punctuation cleanup
    """
    import re as _re
    import unicodedata
    import pandas as pd

    if pd.isna(s):
        return ""
    s = str(s).strip()

    # Remove anything inside parentheses
    s = _re.sub(r"\(.*?\)", "", s)
    # Remove anything after " - ", " | ", etc.
    s = _re.sub(r"\s[-–—]\s.*$", "", s)
    s = _re.sub(r"\s\|\s.*$", "", s)
    # Remove trailing TEAM/POS tokens
    s = _re.sub(
        r"(?:\s(?:OKC|LAL|LAC|PHX|PHI|BOS|NYK|BKN|DAL|DEN|MIN|NOP|MEM|MIA|MIL|TOR|CHI|CLE|ATL|CHA|WAS|ORL|SAS|HOU|UTA|POR|GSW|SAC|DET)"
        r"|\s(?:PG|SG|SF|PF|C|G|F))+$",
        "",
        s,
        flags=_re.IGNORECASE
    )

    # 💡 NEW: normalize accents (Jokić → Jokic)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Lowercase and punctuation cleanup
    s = s.lower()
    s = _re.sub(r"[.,\u2019']", "", s)
    s = s.replace("-", " ")
    s = _re.sub(r"\b(jr|sr|iii|ii)\b", "", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s


# =========================
# Player index loading
# =========================
def load_player_index(utils_dir: Path | str = "utils",
                      index_filename: str = "playerIndex.csv") -> pd.DataFrame:
    """
    Load player index CSV (columns must be: INDEX, PLAYER) and
    add PLAYER_NORM for joining.
    """
    utils_dir = Path(utils_dir)
    index_df = pd.read_csv(utils_dir / index_filename)

    required = {"INDEX", "PLAYER"}
    missing = required - set(index_df.columns)
    if missing:
        raise ValueError(f"playerIndex.csv missing columns: {missing}")

    index_df["PLAYER_NORM"] = index_df["PLAYER"].apply(normalize_name)

    # Warn on duplicate normalized names (can indicate ambiguous mapping)
    dups = index_df["PLAYER_NORM"].duplicated(keep=False)
    if dups.any():
        print("⚠️ Warning: duplicate normalized names found in playerIndex.csv. "
              "Consider disambiguating these rows:")
        print(index_df.loc[dups, ["INDEX", "PLAYER", "PLAYER_NORM"]]
                      .sort_values("PLAYER_NORM")
                      .to_string(index=False))

    return index_df


# =========================
# Column helpers
# =========================
def detect_player_column(df: pd.DataFrame) -> str:
    """
    Try to find the player-name column in a projection dataframe.
    """
    candidates = ["PLAYER", "Player", "player", "Name", "NAME"]
    for c in candidates:
        if c in df.columns:
            return c
    # If nothing obvious, fallback to first text-like column
    textlike = [c for c in df.columns if df[c].dtype == "object"]
    if textlike:
        return textlike[0]
    raise ValueError("Could not detect player-name column.")


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Safely convert columns to numeric, leaving non-existing columns alone.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# Enrichment (add INDEX)
# =========================
def enrich_with_index(
    in_path: Path | str,
    index_df: pd.DataFrame,
    player_col: Optional[str] = None,
    source_name: Optional[str] = None,
    out_path: Optional[Path | str] = None,
    write_missing_report: bool = True,
    alias_map: Optional[Dict[str, str]] = None,   # 👈 NEW: alias support
) -> Path:
    """
    Adds an INDEX column to a projection CSV by joining on normalized player name.

    Args:
        in_path:         path to the projection CSV.
        index_df:        dataframe from load_player_index().
        player_col:      column name with player names; auto-detected if None.
        source_name:     optional source label added as a 'SOURCE' column.
        out_path:        where to write the enriched CSV (defaults to same filename
                         under data/enriched/).
        write_missing_report: write a CSV listing unmatched players.
        alias_map:       dict of name aliases (e.g., {"alex sarr":"alexander sarr"}).
                         Keys/values may be raw or normalized; we normalize both.

    Returns:
        Path to the enriched CSV written to disk.
    """
    in_path = Path(in_path)
    df = pd.read_csv(in_path)

    if player_col is None:
        player_col = detect_player_column(df)

    # Drop rows with missing player names to avoid bogus matches/reports
    df = df[df[player_col].notna()].copy()

    # Normalize player names
    df["PLAYER_NORM"] = df[player_col].apply(normalize_name)

    # Apply alias map (normalize keys/values first)
    if alias_map:
        norm_alias = {normalize_name(k): normalize_name(v) for k, v in alias_map.items()}
        df["PLAYER_NORM"] = df["PLAYER_NORM"].replace(norm_alias)

    # Join to index on normalized name
    df = df.merge(
        index_df[["INDEX", "PLAYER_NORM"]],
        on="PLAYER_NORM",
        how="left",
        validate="m:1"
    )

    if source_name is not None:
        df["SOURCE"] = source_name

    # Report missing matches
    missing_count = df["INDEX"].isna().sum()
    if missing_count:
        print(f"⚠️ {in_path.name}: {missing_count} players did not match INDEX.")
        if write_missing_report:
            misses = (
                df.loc[df["INDEX"].isna(), [player_col, "PLAYER_NORM"]]
                  .drop_duplicates()
                  .sort_values(player_col)
            )
            report_dir = in_path.parent / "enriched"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / f"{in_path.stem}_missing_matches.csv").write_text(
                misses.to_csv(index=False)
            )

    # Decide output path
    if out_path is None:
        out_dir = in_path.parent / "enriched"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / in_path.name
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop helper
    df.drop(columns=["PLAYER_NORM"], inplace=True)

    df.to_csv(out_path, index=False)
    print(f"✅ Indexed → {out_path}")
    return out_path


# =========================
# Weighted merge
# =========================
def weighted_merge(
    enriched_paths: Iterable[Path | str],
    weights: Dict[str, float],
    metric_cols: List[str],
    index_df: pd.DataFrame,
    out_path: Path | str = "data/merged/projections_weighted.csv",
) -> Path:
    """
    Merge multiple enriched CSVs (must contain INDEX and SOURCE) and compute
    weighted averages across `metric_cols` per INDEX.

    Args:
        enriched_paths: list of CSVs that already include INDEX (and optionally SOURCE).
        weights:        mapping {source_name -> weight}. Should sum to ~1.0.
        metric_cols:    numeric columns to average (only those present will be used).
        index_df:       dataframe from load_player_index() to attach canonical PLAYER.
        out_path:       where to write the final merged CSV.

    Returns:
        Path to the final merged CSV.
    """
    frames = []
    for p in enriched_paths:
        t = pd.read_csv(p)
        if "INDEX" not in t.columns:
            raise ValueError(f"{Path(p).name} is missing required column: INDEX")
        if "SOURCE" not in t.columns:
            # If SOURCE missing, try to infer from filename
            t["SOURCE"] = Path(p).stem
        frames.append(t)

    stack = pd.concat(frames, ignore_index=True)

    # Map weights to rows
    stack["__w"] = stack["SOURCE"].map(weights).fillna(0.0)

    # Keep only metrics that actually exist
    metrics = [c for c in metric_cols if c in stack.columns]
    if not metrics:
        raise ValueError("None of the specified metric_cols are present in the inputs.")

    # Coerce metrics to numeric
    stack = coerce_numeric(stack, metrics)

    # Weighted average per INDEX
    def _wavg(group: pd.DataFrame) -> pd.Series:
        out = {}
        w = group["__w"].fillna(0.0)
        for c in metrics:
            x = group[c]
            mask = x.notna() & w.notna()
            ww = w[mask]
            if ww.sum() > 0:
                out[c] = (x[mask] * ww).sum() / ww.sum()
            else:
                out[c] = pd.NA
        return pd.Series(out)

    weighted = (
        stack.groupby("INDEX", dropna=False)
             .apply(_wavg)
             .reset_index()
    )

    # Attach canonical PLAYER from index
    labels = index_df[["INDEX", "PLAYER"]].drop_duplicates()
    weighted = weighted.merge(labels, on="INDEX", how="left")

    # Reorder cols: INDEX, PLAYER, metrics...
    weighted = weighted[["INDEX", "PLAYER"] + metrics]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weighted.to_csv(out_path, index=False)
    print(f"🎯 Weighted projections → {out_path}")
    return out_path


# =========================
# Orchestration helper
# =========================
def enrich_many(
    in_paths: Iterable[Path | str],
    index_df: pd.DataFrame,
    source_names: Optional[Iterable[str]] = None,
    out_dir: Optional[Path | str] = None
) -> List[Path]:
    """
    Convenience to enrich multiple projection files in one call.

    Args:
        in_paths:     list of input CSVs.
        index_df:     dataframe from load_player_index().
        source_names: optional list of same length for 'SOURCE' labels.
        out_dir:      optional directory to write outputs into.

    Returns:
        List of output paths.
    """
    outputs = []
    in_paths = list(in_paths)
    if source_names is None:
        source_names = [None] * len(in_paths)
    else:
        source_names = list(source_names)
        if len(source_names) != len(in_paths):
            raise ValueError("source_names must match length of in_paths.")

    for src, path in zip(source_names, in_paths):
        if out_dir is None:
            out_path = None
        else:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = Path(out_dir) / Path(path).name

        outputs.append(
            enrich_with_index(
                in_path=path,
                index_df=index_df,
                source_name=src,
                out_path=out_path
            )
        )
    return outputs


from typing import Dict, Iterable, Optional

# === Historical stat weights (unitless multipliers) ===
STAT_WEIGHTS: Dict[str, float] = {
    "PTS": 0.7457,
    "REB": 0.5556,
    "AST": 0.5194,
    "STL": 0.5722,
    "BLK": 0.4644,
    "3PM": 0.4316,
    "FG%": 0.4245,
    "FT%": 0.2991,
    "TO":  0.3760,
}

# Common alias map so code is resilient to different source headers
_STAT_ALIASES: Dict[str, str] = {
    # Threes
    "3P": "3PM", "3PM": "3PM", "3PTM": "3PM", "3-PT MADE": "3PM",
    # Field goal %
    "FG%": "FG%", "FG_PCT": "FG%", "FIELD_GOAL_PCT": "FG%",
    # Free throw %
    "FT%": "FT%", "FT_PCT": "FT%", "FREE_THROW_PCT": "FT%",
    # Turnovers
    "TO": "TO", "TOV": "TO", "TURNOVERS": "TO",
    # Others (identity)
    "PTS": "PTS", "REB": "REB", "REBounds": "REB",
    "AST": "AST", "ASSISTS": "AST",
    "STL": "STL", "STEALS": "STL",
    "BLK": "BLK", "BLOCKS": "BLK",
}

def resolve_stat_name(name: str) -> str:
    """
    Normalize a column/stat label to our canonical keys used in STAT_WEIGHTS.
    Returns the upper-cased input if no alias match is found (so you can detect missing weights).
    """
    key = name.strip().upper()
    return _STAT_ALIASES.get(key, key)

def get_stat_weights(include: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """
    Return a dict of {stat: weight}. If `include` is provided, only those stats (after alias resolution) are returned.
    Unknown stats are ignored.
    """
    if include is None:
        return dict(STAT_WEIGHTS)

    out: Dict[str, float] = {}
    for raw in include:
        stat = resolve_stat_name(raw)
        if stat in STAT_WEIGHTS:
            out[stat] = STAT_WEIGHTS[stat]
    return out
