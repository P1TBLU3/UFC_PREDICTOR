"""Scrape fight metadata, totals, and round-level UFCStats data.

Existing outputs are checkpointed so interrupted runs can resume safely.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ufcstats"
FIGHTS_PATH = DATA_DIR / "fights.csv"
DETAILS_PATH = DATA_DIR / "fight_details.csv"
TOTALS_PATH = DATA_DIR / "fight_totals.csv"
ROUNDS_PATH = DATA_DIR / "round_stats.csv"

DETAIL_COLUMNS = [
    "fight_id", "event_id", "event_name", "event_date", "fight_url",
    "fighter_1_id", "fighter_1_name", "fighter_1_result",
    "fighter_2_id", "fighter_2_name", "fighter_2_result", "winner_id",
    "bout_type", "method", "finish_round", "finish_time",
    "finish_time_seconds", "time_format", "scheduled_rounds",
    "total_fight_time_seconds", "referee", "details",
    "stats_available", "rounds_with_stats", "scraped_at_utc",
]

STAT_COLUMNS = [
    "fight_id", "event_id", "event_name", "event_date", "fight_url",
    "fighter_id", "fighter_name", "fighter_result", "fighter_order",
    "opponent_id", "opponent_name", "knockdowns",
    "sig_str_landed", "sig_str_attempted", "sig_str_pct",
    "total_str_landed", "total_str_attempted",
    "takedowns_landed", "takedowns_attempted", "takedown_pct",
    "submission_attempts", "reversals", "control_time", "control_seconds",
    "head_landed", "head_attempted", "body_landed", "body_attempted",
    "leg_landed", "leg_attempted", "distance_landed", "distance_attempted",
    "clinch_landed", "clinch_attempted", "ground_landed", "ground_attempted",
]

TOTAL_COLUMNS = STAT_COLUMNS.copy()
ROUND_COLUMNS = STAT_COLUMNS[:11] + ["round"] + STAT_COLUMNS[11:]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fight details and round-level statistics from fights.csv."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument(
        "--refresh-no-stats-days", type=int, default=30,
        help="Retry completed fights without stats from the past N days (default: 30).",
    )
    parser.add_argument(
        "--refresh-last-days", type=int, default=0,
        help="Force recheck of all completed fights from the past N days (default: off).",
    )
    parser.add_argument(
        "--refresh-all-incomplete", action="store_true",
        help="Retry every completed fight missing statistics, including older fights.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with an error if any selected fight could not be refreshed.",
    )
    return parser.parse_args()


def clean_text(value: str) -> str:
    return " ".join(value.split())


def extract_id(url: str) -> str:
    path = urlparse(url.strip()).path.rstrip("/")
    return path.split("/")[-1] if path else ""


def parse_int(value: str) -> int | None:
    value = clean_text(value).replace(",", "")
    if value in {"", "---", "--", "N/A"}:
        return None
    match = re.search(r"-?\d+", value)
    return int(match.group()) if match else None


def parse_percentage(value: str) -> float | None:
    value = clean_text(value)
    if value in {"", "---", "--", "N/A"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def parse_of(value: str) -> tuple[int | None, int | None]:
    value = clean_text(value)
    match = re.search(r"(\d+)\s+of\s+(\d+)", value, flags=re.IGNORECASE)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_clock_seconds(value: str) -> int | None:
    match = re.fullmatch(r"(\d+):(\d{2})", clean_text(value))
    if not match or int(match.group(2)) >= 60:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", clean_text(value).lower()).strip("_")


def split_labeled_text(value: str) -> tuple[str, str]:
    value = clean_text(value)
    if ":" not in value:
        return "", value
    label, result = value.split(":", 1)
    return normalize_label(label), clean_text(result)


def parse_scheduled_rounds(time_format: str) -> int | None:
    match = re.search(r"(\d+)\s*Rnd", time_format, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\(([^)]+)\)", time_format)
    return len(match.group(1).split("-")) if match else None


def parse_round_durations(time_format: str) -> list[int]:
    match = re.search(r"\(([^)]+)\)", time_format)
    if not match:
        return []
    values = [value.strip() for value in match.group(1).split("-")]
    if not all(value.isdigit() for value in values):
        return []
    return [int(value) * 60 for value in values]


def calculate_total_fight_time(
    finish_round: int | None,
    finish_time_seconds: int | None,
    time_format: str,
) -> int | None:
    if finish_round is None or finish_time_seconds is None:
        return None
    durations = parse_round_durations(time_format)
    if durations and finish_round <= len(durations):
        return sum(durations[: finish_round - 1]) + finish_time_seconds
    return (finish_round - 1) * 300 + finish_time_seconds


def load_fights() -> pd.DataFrame:
    if not FIGHTS_PATH.exists():
        raise FileNotFoundError(f"File not found: {FIGHTS_PATH}")
    fights = pd.read_csv(FIGHTS_PATH, dtype=str, keep_default_na=False)
    required = {
        "fight_id", "event_id", "event_name", "event_date", "fight_url",
        "fighter_1_id", "fighter_1_name", "fighter_1_result",
        "fighter_2_id", "fighter_2_name", "fighter_2_result", "winner_id",
    }
    missing = required - set(fights.columns)
    if missing:
        raise ValueError(f"Missing columns in fights.csv: {sorted(missing)}")
    fights = fights.drop_duplicates("fight_id", keep="last").reset_index(drop=True)
    if (~fights["fight_url"].str.contains("/fight-details/", regex=False)).any():
        raise ValueError("fights.csv contains invalid fight_url values.")
    return fights


def load_existing(path: Path, columns: list[str], restart: bool) -> pd.DataFrame:
    if restart or not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        dataframe = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    missing = set(columns) - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{path.name} does not have the expected structure. "
            f"Use --restart. Missing: {sorted(missing)}"
        )
    return dataframe[columns].copy()


def load_rendered_fight(page: Page, fight_url: str) -> BeautifulSoup:
    """Load a fight and wait for its rendered statistics to stabilize."""

    response = page.goto(
        fight_url.strip(), wait_until="commit", timeout=120_000
    )
    if response is None:
        raise RuntimeError(f"No response was received: {fight_url}")
    if response.status >= 400:
        raise RuntimeError(f"HTTP {response.status}: {fight_url}")

    fighters = page.locator("a.b-fight-details__person-link")
    try:
        fighters.first.wait_for(state="attached", timeout=60_000)
    except PlaywrightTimeoutError as error:
        raise RuntimeError("The fighters did not appear.") from error

    tables = page.locator("table.b-fight-details__table")
    stat_cells = page.locator("p.b-fight-details__table-text")

    # UFCStats may render table shells before populating their cells.
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15_000)
    except PlaywrightTimeoutError:
        pass

    previous_count: int | None = None
    stable_checks = 0

    for attempt in range(60):
        current_count = stat_cells.count()

        if current_count == previous_count:
            # Wait longer when no cells appear; the fight may lack data or load slowly.
            if current_count > 0 or attempt >= 16:
                stable_checks += 1
        else:
            stable_checks = 0
            previous_count = current_count

        if stable_checks >= 4:
            break

        page.wait_for_timeout(250)

    page.wait_for_timeout(250)

    print(f"  HTTP status: {response.status}")
    print(f"  Final title: {page.title()}")
    print(f"  Fighters detected: {fighters.count()}")
    print(f"  Statistics tables: {tables.count()}")
    print(f"  Rendered statistic values: {stat_cells.count()}")

    return BeautifulSoup(page.content(), "lxml")


def load_with_retries(
    page: Page, fight_url: str, max_attempts: int
) -> BeautifulSoup:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(f"  Retry {attempt}/{max_attempts}...")
                page.goto("about:blank", wait_until="commit", timeout=10_000)
                page.wait_for_timeout(1_000 * attempt)
            return load_rendered_fight(page, fight_url)
        except Exception as error:
            last_error = error
            print(f"  Attempt {attempt} failed: {error}")
    raise RuntimeError(
        f"The page could not be loaded after {max_attempts} attempts: {last_error}"
    )


def get_page_fighters(soup: BeautifulSoup) -> list[dict[str, str]]:
    links = soup.select("a.b-fight-details__person-link")
    if len(links) < 2:
        links = soup.select(
            ".b-fight-details__persons a[href*='/fighter-details/']"
        )
    fighters: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in links:
        url = link.get("href", "").strip()
        fighter_id = extract_id(url)
        if not fighter_id or fighter_id in seen:
            continue
        seen.add(fighter_id)
        fighters.append(
            {
                "fighter_id": fighter_id,
                "fighter_name": clean_text(link.get_text(" ", strip=True)),
                "fighter_url": url,
            }
        )
        if len(fighters) == 2:
            break
    if len(fighters) != 2:
        raise ValueError("Exactly two fighters could not be identified.")
    return fighters


def validate_page_fighters(
    page_fighters: list[dict[str, str]], source: pd.Series
) -> None:
    page_ids = {fighter["fighter_id"] for fighter in page_fighters}
    source_ids = {str(source["fighter_1_id"]), str(source["fighter_2_id"])}
    if page_ids != source_ids:
        raise ValueError(
            f"Fighters do not match. Page={page_ids}; source={source_ids}"
        )


def extract_metadata(soup: BeautifulSoup) -> dict[str, Any]:
    metadata: dict[str, str] = {}
    for item in soup.select(
        "i.b-fight-details__text-item_first, i.b-fight-details__text-item"
    ):
        label, value = split_labeled_text(item.get_text(" ", strip=True))
        if label and label not in metadata:
            metadata[label] = value

    details = ""
    for paragraph in soup.select("p.b-fight-details__text"):
        label, value = split_labeled_text(paragraph.get_text(" ", strip=True))
        if label == "details":
            details = value
            break

    bout_node = soup.select_one("div.b-fight-details__fight-head")
    bout_type = clean_text(bout_node.get_text(" ", strip=True)) if bout_node else ""
    finish_round = parse_int(metadata.get("round", ""))
    finish_time = metadata.get("time", "")
    finish_time_seconds = parse_clock_seconds(finish_time)
    time_format = metadata.get("time_format", "")

    return {
        "bout_type": bout_type,
        "method": metadata.get("method", ""),
        "finish_round": finish_round,
        "finish_time": finish_time,
        "finish_time_seconds": finish_time_seconds,
        "time_format": time_format,
        "scheduled_rounds": parse_scheduled_rounds(time_format),
        "total_fight_time_seconds": calculate_total_fight_time(
            finish_round, finish_time_seconds, time_format
        ),
        "referee": metadata.get("referee", ""),
        "details": details,
    }


def classify_table(table: Tag) -> str | None:
    headers = " | ".join(
        clean_text(tag.get_text(" ", strip=True)).upper()
        for tag in table.select("thead th, thead td")
    )
    if "TOTAL STR" in headers or "SUB. ATT" in headers or "CTRL" in headers:
        return "totals"
    if "HEAD" in headers and "BODY" in headers and "GROUND" in headers:
        return "significant"
    return None


def extract_stats_rows(
    soup: BeautifulSoup,
) -> dict[str, list[tuple[list[str], list[str]]]]:
    """Extract statistic blocks from UFCStats tables in DOM order.

    A cell may contain both the fight summary and every round, so the parser
    groups the flattened values according to each table's expected width.
    """

    result: dict[str, list[tuple[list[str], list[str]]]] = {
        "totals": [],
        "significant": [],
    }

    fields_per_fighter = {
        "totals": 10,
        "significant": 9,
    }

    for table in soup.select("table.b-fight-details__table"):
        table_type = classify_table(table)

        if table_type is None:
            continue

        values = [
            clean_text(node.get_text(" ", strip=True))
            for node in table.select("p.b-fight-details__table-text")
        ]
        values = [value for value in values if value]

        if not values:
            continue

        block_size = fields_per_fighter[table_type] * 2

        if len(values) % block_size != 0:
            raise ValueError(
                f"The {table_type} contains {len(values)} values; "
                f"expected a multiple of {block_size}."
            )

        for start in range(0, len(values), block_size):
            block = values[start : start + block_size]

            # Values alternate between fighter 1 and fighter 2 within each field.
            fighter_1 = block[0::2]
            fighter_2 = block[1::2]

            result[table_type].append((fighter_1, fighter_2))

    return result


def build_stat_record(
    source: pd.Series,
    fighters: list[dict[str, str]],
    fighter_index: int,
    totals: list[str],
    significant: list[str],
) -> dict[str, Any]:
    if len(totals) < 10 or len(significant) < 9:
        raise ValueError("A statistics row has fewer columns than expected.")

    fighter = fighters[fighter_index]
    opponent = fighters[1 - fighter_index]
    result = (
        str(source["fighter_1_result"])
        if fighter["fighter_id"] == str(source["fighter_1_id"])
        else str(source["fighter_2_result"])
    )

    sig_landed, sig_attempted = parse_of(totals[2])
    sig2_landed, sig2_attempted = parse_of(significant[1])
    if (
        sig_landed is not None
        and sig2_landed is not None
        and (sig_landed, sig_attempted) != (sig2_landed, sig2_attempted)
    ):
        raise ValueError("SIG. STR. differs between the two tables.")

    total_landed, total_attempted = parse_of(totals[4])
    td_landed, td_attempted = parse_of(totals[5])
    head_landed, head_attempted = parse_of(significant[3])
    body_landed, body_attempted = parse_of(significant[4])
    leg_landed, leg_attempted = parse_of(significant[5])
    distance_landed, distance_attempted = parse_of(significant[6])
    clinch_landed, clinch_attempted = parse_of(significant[7])
    ground_landed, ground_attempted = parse_of(significant[8])
    control_time = clean_text(totals[9])

    return {
        "fight_id": str(source["fight_id"]),
        "event_id": str(source["event_id"]),
        "event_name": str(source["event_name"]),
        "event_date": str(source["event_date"]),
        "fight_url": str(source["fight_url"]),
        "fighter_id": fighter["fighter_id"],
        "fighter_name": fighter["fighter_name"],
        "fighter_result": result,
        "fighter_order": str(fighter_index + 1),
        "opponent_id": opponent["fighter_id"],
        "opponent_name": opponent["fighter_name"],
        "knockdowns": parse_int(totals[1]),
        "sig_str_landed": sig_landed,
        "sig_str_attempted": sig_attempted,
        "sig_str_pct": parse_percentage(totals[3]),
        "total_str_landed": total_landed,
        "total_str_attempted": total_attempted,
        "takedowns_landed": td_landed,
        "takedowns_attempted": td_attempted,
        "takedown_pct": parse_percentage(totals[6]),
        "submission_attempts": parse_int(totals[7]),
        "reversals": parse_int(totals[8]),
        "control_time": control_time,
        "control_seconds": parse_clock_seconds(control_time),
        "head_landed": head_landed,
        "head_attempted": head_attempted,
        "body_landed": body_landed,
        "body_attempted": body_attempted,
        "leg_landed": leg_landed,
        "leg_attempted": leg_attempted,
        "distance_landed": distance_landed,
        "distance_attempted": distance_attempted,
        "clinch_landed": clinch_landed,
        "clinch_attempted": clinch_attempted,
        "ground_landed": ground_landed,
        "ground_attempted": ground_attempted,
    }


def _sum_optional(records: list[dict[str, Any]], field: str) -> int | None:
    """Sum a numeric field, returning None when every value is missing."""

    values = [record.get(field) for record in records]
    numeric_values = [int(value) for value in values if value is not None]
    return sum(numeric_values) if numeric_values else None


def _percentage_from_counts(
    landed: int | None,
    attempted: int | None,
) -> float | None:
    """Calculate a percentage from landed and attempted counts."""

    if landed is None or attempted is None:
        return None
    if attempted == 0:
        return 0.0
    return round((landed / attempted) * 100, 2)


def _format_clock(seconds: int | None) -> str:
    """Convert seconds to UFCStats' M:SS representation."""

    if seconds is None:
        return ""
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}"


def aggregate_round_records(
    round_stats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build fight totals by aggregating the round-level statistics.
    """

    if not round_stats:
        return []

    count_fields = [
        "knockdowns",
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "takedowns_landed",
        "takedowns_attempted",
        "submission_attempts",
        "reversals",
        "control_seconds",
        "head_landed",
        "head_attempted",
        "body_landed",
        "body_attempted",
        "leg_landed",
        "leg_attempted",
        "distance_landed",
        "distance_attempted",
        "clinch_landed",
        "clinch_attempted",
        "ground_landed",
        "ground_attempted",
    ]

    totals: list[dict[str, Any]] = []

    for fighter_order in ("1", "2"):
        fighter_rounds = [
            record
            for record in round_stats
            if str(record["fighter_order"]) == fighter_order
        ]

        if not fighter_rounds:
            raise ValueError(
                f"No round statistics found for fighter_order={fighter_order}."
            )

        total = {
            key: value
            for key, value in fighter_rounds[0].items()
            if key != "round"
        }

        for field in count_fields:
            total[field] = _sum_optional(fighter_rounds, field)

        total["sig_str_pct"] = _percentage_from_counts(
            total["sig_str_landed"],
            total["sig_str_attempted"],
        )
        total["takedown_pct"] = _percentage_from_counts(
            total["takedowns_landed"],
            total["takedowns_attempted"],
        )
        total["control_time"] = _format_clock(total["control_seconds"])

        totals.append(total)

    return totals


def parse_detailed_stats(
    soup: BeautifulSoup,
    source: pd.Series,
    fighters: list[dict[str, str]],
    expected_finish_round: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Parse UFCStats tables with or without a separate summary block.
    """

    rows = extract_stats_rows(soup)
    totals_rows = rows["totals"]
    significant_rows = rows["significant"]

    if not totals_rows and not significant_rows:
        return [], [], 0
    if not totals_rows or not significant_rows:
        raise ValueError("Only one of the two statistics tables appeared.")
    if len(totals_rows) != len(significant_rows):
        raise ValueError(
            "The tables contain different block counts: "
            f"{len(totals_rows)} and {len(significant_rows)}."
        )

    block_count = len(totals_rows)
    has_summary_block = False

    if expected_finish_round is not None and expected_finish_round > 0:
        if block_count == expected_finish_round:
            # Current layout: one block per round.
            has_summary_block = False
        elif block_count == expected_finish_round + 1:
            # Historical layout: summary followed by one block per round.
            has_summary_block = True
        else:
            raise ValueError(
                "The statistic block count does not match the final round: "
                f"blocks={block_count}, finish_round={expected_finish_round}. "
                "Expected finish_round or finish_round + 1."
            )
    else:
        # Without a reliable final round, retain every block and aggregate the totals.
        has_summary_block = False

    round_start = 1 if has_summary_block else 0
    number_of_rounds = block_count - round_start

    if number_of_rounds <= 0:
        raise ValueError("No round-level blocks were found.")

    round_stats: list[dict[str, Any]] = []

    for round_index in range(number_of_rounds):
        source_index = round_index + round_start

        for fighter_index in range(2):
            record = build_stat_record(
                source,
                fighters,
                fighter_index,
                totals_rows[source_index][fighter_index],
                significant_rows[source_index][fighter_index],
            )
            record["round"] = round_index + 1
            round_stats.append(record)

    if has_summary_block:
        fight_totals = [
            build_stat_record(
                source,
                fighters,
                fighter_index,
                totals_rows[0][fighter_index],
                significant_rows[0][fighter_index],
            )
            for fighter_index in range(2)
        ]
    else:
        fight_totals = aggregate_round_records(round_stats)

    return fight_totals, round_stats, number_of_rounds

def build_detail_record(
    source: pd.Series,
    metadata: dict[str, Any],
    stats_available: bool,
    rounds_with_stats: int,
) -> dict[str, Any]:
    return {
        "fight_id": str(source["fight_id"]),
        "event_id": str(source["event_id"]),
        "event_name": str(source["event_name"]),
        "event_date": str(source["event_date"]),
        "fight_url": str(source["fight_url"]),
        "fighter_1_id": str(source["fighter_1_id"]),
        "fighter_1_name": str(source["fighter_1_name"]),
        "fighter_1_result": str(source["fighter_1_result"]),
        "fighter_2_id": str(source["fighter_2_id"]),
        "fighter_2_name": str(source["fighter_2_name"]),
        "fighter_2_result": str(source["fighter_2_result"]),
        "winner_id": str(source["winner_id"]),
        **metadata,
        "stats_available": stats_available,
        "rounds_with_stats": rounds_with_stats,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def validate_landed_attempted(dataframe: pd.DataFrame) -> None:
    pairs = [
        ("sig_str_landed", "sig_str_attempted"),
        ("total_str_landed", "total_str_attempted"),
        ("takedowns_landed", "takedowns_attempted"),
        ("head_landed", "head_attempted"),
        ("body_landed", "body_attempted"),
        ("leg_landed", "leg_attempted"),
        ("distance_landed", "distance_attempted"),
        ("clinch_landed", "clinch_attempted"),
        ("ground_landed", "ground_attempted"),
    ]
    for landed_column, attempted_column in pairs:
        landed = pd.to_numeric(dataframe[landed_column], errors="coerce")
        attempted = pd.to_numeric(dataframe[attempted_column], errors="coerce")
        if (landed.notna() & attempted.notna() & (landed > attempted)).any():
            raise ValueError(f"{landed_column} exceeds {attempted_column}.")


def validate_parsed(
    detail: dict[str, Any],
    totals: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
) -> None:
    if not detail["fight_id"]:
        raise ValueError("fight_id is empty.")
    if not detail["stats_available"]:
        if totals or rounds:
            raise ValueError("The fight is marked as having no statistics but contains rows.")
        return
    if len(totals) != 2 or not rounds:
        raise ValueError("A fight with statistics must contain two totals and round rows.")

    totals_df = pd.DataFrame(totals)
    rounds_df = pd.DataFrame(rounds)
    expected = {detail["fighter_1_id"], detail["fighter_2_id"]}
    if set(totals_df["fighter_id"]) != expected:
        raise ValueError("The fighters in the totals do not match.")
    if rounds_df.duplicated(["fight_id", "fighter_id", "round"]).any():
        raise ValueError("Duplicate round rows found.")
    if not rounds_df.groupby("round")["fighter_id"].nunique().eq(2).all():
        raise ValueError("Each round must contain two fighters.")
    validate_landed_attempted(totals_df)
    validate_landed_attempted(rounds_df)


def write_atomic(dataframe: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    dataframe[columns].to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def is_filled(value: object) -> bool:
    """Return False for the empty strings used in the raw CSV checkpoints."""

    return not pd.isna(value) and str(value).strip().lower() not in {
        "", "nan", "none", "<na>",
    }


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def has_fight_result(row: pd.Series) -> bool:
    return is_filled(row["fighter_1_result"]) and is_filled(row["fighter_2_result"])


def select_fights_for_processing(
    fights: pd.DataFrame,
    details: pd.DataFrame,
    totals: pd.DataFrame,
    rounds: pd.DataFrame,
    *,
    today: date,
    refresh_no_stats_days: int,
    refresh_last_days: int,
    refresh_all_incomplete: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Choose new or outdated fights, without re-scraping all historical UFC data."""

    indexed_details = details.drop_duplicates("fight_id", keep="last").set_index("fight_id")
    totals_count = totals.groupby("fight_id").size().to_dict() if not totals.empty else {}
    rounds_count = rounds.groupby("fight_id").size().to_dict() if not rounds.empty else {}
    selected_indices: list[int] = []
    reasons: dict[str, str] = {}

    for index, source in fights.iterrows():
        fight_id = str(source["fight_id"])
        if fight_id not in indexed_details.index:
            reason = "new_fight"
        else:
            old = indexed_details.loc[fight_id]
            reason = ""
            # In particular, keep the same fighter IDs and results as the
            # newly refreshed fight-card data, even if UFCStats reorders names.
            for field in (
                "fighter_1_id", "fighter_2_id", "fighter_1_result",
                "fighter_2_result", "winner_id",
            ):
                if str(source[field]).strip() != str(old[field]).strip():
                    reason = "fight_card_changed"
                    break

            completed = has_fight_result(source)
            parsed_date = pd.to_datetime(source["event_date"], errors="coerce")
            age = (today - parsed_date.date()).days if pd.notna(parsed_date) else None
            if completed and not reason:
                if not is_filled(old["total_fight_time_seconds"]):
                    reason = "missing_duration"
                elif is_true(old["stats_available"]) and (
                    totals_count.get(fight_id, 0) != 2 or rounds_count.get(fight_id, 0) == 0
                ):
                    reason = "incomplete_stat_rows"
                elif not is_true(old["stats_available"]) and (
                    refresh_all_incomplete or (
                        age is not None and 0 <= age <= refresh_no_stats_days
                    )
                ):
                    reason = "missing_stats"
                elif (
                    refresh_last_days > 0
                    and age is not None
                    and 0 <= age <= refresh_last_days
                ):
                    reason = "recent_recheck"

        if reason:
            selected_indices.append(index)
            reasons[fight_id] = reason

    return fights.loc[selected_indices].copy(), reasons


def merge_and_save(
    existing_details: pd.DataFrame,
    existing_totals: pd.DataFrame,
    existing_rounds: pd.DataFrame,
    new_details: list[dict[str, Any]],
    new_totals: list[dict[str, Any]],
    new_rounds: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace complete fight blocks so old round records cannot survive a refresh."""

    refreshed_ids = {str(record["fight_id"]) for record in new_details}
    previous = (
        existing_details.drop_duplicates("fight_id", keep="last").set_index("fight_id")
    )
    for record in new_details:
        fight_id = str(record["fight_id"])
        if (
            fight_id in previous.index
            and is_true(previous.loc[fight_id, "stats_available"])
            and not is_true(record["stats_available"])
        ):
            raise ValueError(
                f"Refusing to overwrite previously available statistics for {fight_id}."
            )

    details = pd.concat(
        [
            existing_details.loc[~existing_details["fight_id"].isin(refreshed_ids)],
            pd.DataFrame(new_details, columns=DETAIL_COLUMNS),
        ],
        ignore_index=True,
    ).drop_duplicates("fight_id", keep="last").reset_index(drop=True)

    totals = pd.concat(
        [
            existing_totals.loc[~existing_totals["fight_id"].isin(refreshed_ids)],
            pd.DataFrame(new_totals, columns=TOTAL_COLUMNS),
        ],
        ignore_index=True,
    ).drop_duplicates(["fight_id", "fighter_id"], keep="last").reset_index(drop=True)

    rounds = pd.concat(
        [
            existing_rounds.loc[~existing_rounds["fight_id"].isin(refreshed_ids)],
            pd.DataFrame(new_rounds, columns=ROUND_COLUMNS),
        ],
        ignore_index=True,
    ).drop_duplicates(["fight_id", "fighter_id", "round"], keep="last").reset_index(drop=True)

    # Work on in-memory copies first; only write after all checks pass.
    write_atomic(details, DETAILS_PATH, DETAIL_COLUMNS)
    write_atomic(totals, TOTALS_PATH, TOTAL_COLUMNS)
    write_atomic(rounds, ROUNDS_PATH, ROUND_COLUMNS)
    return details, totals, rounds


def initialize_empty_outputs() -> None:
    write_atomic(pd.DataFrame(columns=DETAIL_COLUMNS), DETAILS_PATH, DETAIL_COLUMNS)
    write_atomic(pd.DataFrame(columns=TOTAL_COLUMNS), TOTALS_PATH, TOTAL_COLUMNS)
    write_atomic(pd.DataFrame(columns=ROUND_COLUMNS), ROUNDS_PATH, ROUND_COLUMNS)


def main() -> None:
    args = parse_arguments()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    if args.max_attempts <= 0 or args.checkpoint_every <= 0:
        raise ValueError("Attempts and checkpoint intervals must be greater than zero.")

    fights = load_fights()
    if args.restart:
        initialize_empty_outputs()

    existing_details = load_existing(DETAILS_PATH, DETAIL_COLUMNS, False)
    existing_totals = load_existing(TOTALS_PATH, TOTAL_COLUMNS, False)
    existing_rounds = load_existing(ROUNDS_PATH, ROUND_COLUMNS, False)

    if args.refresh_no_stats_days < 0 or args.refresh_last_days < 0:
        raise ValueError("Refresh windows must be non-negative.")

    pending, refresh_reasons = select_fights_for_processing(
        fights, existing_details, existing_totals, existing_rounds,
        today=datetime.now(timezone.utc).date(),
        refresh_no_stats_days=args.refresh_no_stats_days,
        refresh_last_days=args.refresh_last_days,
        refresh_all_incomplete=args.refresh_all_incomplete,
    )
    if args.limit is not None:
        pending = pending.head(args.limit)

    # Old fight-details records are deliberately not erased: they may relate
    # to UFCStats events omitted temporarily from its event listing.
    orphan_ids = set(existing_details["fight_id"]) - set(fights["fight_id"])
    if orphan_ids:
        print(
            f"WARNING: {len(orphan_ids)} old fight-detail records have no "
            "matching fights.csv entry. Not deleting them automatically."
        )

    print(f"Available fights: {len(fights)}")
    print(f"Fights selected for initial scraping or refresh: {len(pending)}")
    if not pending.empty:
        selected_reasons = [refresh_reasons[fid] for fid in pending["fight_id"]]
        print("Refresh reasons:")
        for reason, count in pd.Series(selected_reasons).value_counts().items():
            print(f"  {reason}: {count}")
    print(f"Existing fight details: {len(existing_details)}")
    print(f"Total rows: {len(existing_totals)}")
    print(f"Round rows: {len(existing_rounds)}\n")
    if pending.empty:
        print("No new or incomplete fights require processing.")
        return

    details_buffer: list[dict[str, Any]] = []
    totals_buffer: list[dict[str, Any]] = []
    rounds_buffer: list[dict[str, Any]] = []
    successes = 0
    without_stats = 0
    failures: list[tuple[str, str]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.show_browser)
        context = browser.new_context(locale="en-US")
        page = context.new_page()
        page.set_default_timeout(60_000)
        page.set_default_navigation_timeout(120_000)

        try:
            for position, (_, source) in enumerate(pending.iterrows(), start=1):
                fight_id = str(source["fight_id"])
                name = f"{source['fighter_1_name']} vs. {source['fighter_2_name']}"
                print(
                    f"[{position}/{len(pending)}] Processing: {name} "
                    f"({refresh_reasons[fight_id]})"
                )
                try:
                    soup = load_with_retries(
                        page, str(source["fight_url"]), args.max_attempts
                    )
                    fighters = get_page_fighters(soup)
                    validate_page_fighters(fighters, source)
                    metadata = extract_metadata(soup)
                    if has_fight_result(source) and (
                        metadata.get("finish_round") is None
                        or metadata.get("total_fight_time_seconds") is None
                    ):
                        raise ValueError(
                            "Completed fight still has no valid duration on UFCStats; "
                            "keeping its previous checkpoint for a later retry."
                        )
                    totals, rounds, round_count = parse_detailed_stats(
                        soup,
                        source,
                        fighters,
                        metadata.get("finish_round"),
                    )
                    detail = build_detail_record(
                        source, metadata, bool(totals), round_count
                    )
                    validate_parsed(detail, totals, rounds)

                    # An incomplete rendering must not erase statistics saved
                    # by an earlier successful scrape.
                    prior_detail = existing_details.loc[
                        existing_details["fight_id"] == fight_id
                    ]
                    if (
                        not prior_detail.empty
                        and is_true(prior_detail.iloc[-1]["stats_available"])
                        and not detail["stats_available"]
                    ):
                        raise ValueError(
                            "Statistics disappeared on refresh; preserving existing data."
                        )

                    details_buffer.append(detail)
                    totals_buffer.extend(totals)
                    rounds_buffer.extend(rounds)
                    successes += 1

                    if totals:
                        print(
                            f"  Processed: {round_count} rounds, "
                            f"{len(rounds)} round rows."
                        )
                    else:
                        without_stats += 1
                        print("  Processed: UFCStats provides no detailed statistics.")

                    if successes % args.checkpoint_every == 0:
                        existing_details, existing_totals, existing_rounds = merge_and_save(
                            existing_details, existing_totals, existing_rounds,
                            details_buffer, totals_buffer, rounds_buffer,
                        )
                        details_buffer.clear()
                        totals_buffer.clear()
                        rounds_buffer.clear()
                        print("  Checkpoint saved.")

                except Exception as error:
                    failures.append((fight_id, f"{name}: {error}"))
                    print(f"  Error: {error}")
                    print(f"  Fight URL: {source['fight_url']}")

                print()
                if position < len(pending) and args.delay > 0:
                    time.sleep(args.delay)
        finally:
            context.close()
            browser.close()

    if details_buffer:
        existing_details, existing_totals, existing_rounds = merge_and_save(
            existing_details, existing_totals, existing_rounds,
            details_buffer, totals_buffer, rounds_buffer,
        )

    print("Processing complete.")
    print(f"Fights processed in this run: {successes}")
    print(f"Fights without detailed statistics: {without_stats}")
    print(f"Fight details saved: {len(existing_details)}")
    print(f"Total rows: {len(existing_totals)}")
    print(f"Round rows: {len(existing_rounds)}")
    print(f"File: {DETAILS_PATH}")
    print(f"File: {TOTALS_PATH}")
    print(f"File: {ROUNDS_PATH}")

    if failures:
        print(f"\nFailed fights: {len(failures)}")
        for fight_id, error in failures:
            print(f"- {fight_id}: {error}")
        if args.strict:
            raise RuntimeError(f"Scraper failed to refresh {len(failures)} fights.")


if __name__ == "__main__":
    main()