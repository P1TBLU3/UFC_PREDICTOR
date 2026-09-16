"""Scrape fight cards for the events stored in ``events.csv``.

Upcoming and incomplete cards are refreshed because UFCStats may revise them.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup, Tag
from playwright.sync_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ufcstats"
    / "events.csv"
)

FIGHTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ufcstats"
    / "fights.csv"
)


FIGHT_COLUMNS = [
    "fight_id",
    "event_id",
    "event_name",
    "event_date",
    "event_location",
    "event_url",
    "fight_order",
    "fight_url",
    "fighter_1_id",
    "fighter_1_name",
    "fighter_1_url",
    "fighter_1_result",
    "fighter_2_id",
    "fighter_2_name",
    "fighter_2_url",
    "fighter_2_result",
    "winner_id",
    "fighter_1_knockdowns",
    "fighter_2_knockdowns",
    "fighter_1_sig_strikes",
    "fighter_2_sig_strikes",
    "fighter_1_takedowns",
    "fighter_2_takedowns",
    "fighter_1_submission_attempts",
    "fighter_2_submission_attempts",
    "weight_class",
    "method",
    "method_details",
    "finish_round",
    "finish_time",
]


def parse_arguments() -> argparse.Namespace:
    """Define the command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download fights for the events stored "
            "in events.csv."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of events to process. "
            "Example: --limit 3"
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help=(
            "Delay in seconds between events. "
            "Default: 0.75"
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore the existing fights.csv and "
            "process all events again."
        ),
    )

    parser.add_argument(
        "--show-browser",
        action="store_true",
        help=(
            "Show Chromium while the scraper runs. "
            "It runs in the background by default."
        ),
    )

    return parser.parse_args()


def clean_text(value: str) -> str:
    """Collapse redundant whitespace."""

    return " ".join(value.split())


def extract_id(url: str) -> str:
    """Extract the final identifier from a URL."""

    path = urlparse(url.strip()).path.rstrip("/")

    if not path:
        return ""

    return path.split("/")[-1]


def get_cell_values(cell: Tag) -> list[str]:
    """
    Return the visible values from a UFCStats table cell.
    """

    paragraphs = cell.select("p")

    if paragraphs:
        return [
            clean_text(paragraph.get_text(" ", strip=True))
            for paragraph in paragraphs
            if clean_text(
                paragraph.get_text(" ", strip=True)
            )
        ]

    text = clean_text(
        cell.get_text(" ", strip=True)
    )

    return [text] if text else []


def get_list_value(
    values: list[str],
    position: int,
) -> str:
    """Read a list position without raising IndexError."""

    if 0 <= position < len(values):
        return values[position]

    return ""


def normalize_result(value: str) -> str:
    """Normalize a fighter result to W, L, D, or NC."""

    normalized = clean_text(value).lower()

    mapping = {
        "w": "W",
        "win": "W",
        "l": "L",
        "loss": "L",
        "d": "D",
        "draw": "D",
        "nc": "NC",
        "no contest": "NC",
    }

    return mapping.get(
        normalized,
        normalized.upper(),
    )


def load_events() -> pd.DataFrame:
    """Load and validate the previously extracted events."""

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Event file not found: "
            f"{EVENTS_PATH}"
        )

    events = pd.read_csv(
        EVENTS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "event_id",
        "event_name",
        "event_date",
        "event_location",
        "event_url",
    }

    missing_columns = (
        required_columns
        - set(events.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns in events.csv: "
            f"{sorted(missing_columns)}"
        )

    events = (
        events.drop_duplicates(
            subset="event_id",
            keep="first",
        )
        .reset_index(drop=True)
    )

    invalid_urls = ~events[
        "event_url"
    ].str.contains(
        "/event-details/",
        regex=False,
    )

    if invalid_urls.any():
        raise ValueError(
            "events.csv contains URLs that do not point "
            "a event-details."
        )

    return events


def load_existing_fights(
    restart: bool,
) -> pd.DataFrame:
    """Load fights.csv or create an empty table."""

    if restart or not FIGHTS_PATH.exists():
        return pd.DataFrame(
            columns=FIGHT_COLUMNS
        )

    try:
        existing_fights = pd.read_csv(
            FIGHTS_PATH,
            dtype=str,
            keep_default_na=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=FIGHT_COLUMNS
        )

    missing_columns = (
        set(FIGHT_COLUMNS)
        - set(existing_fights.columns)
    )

    if missing_columns:
        raise ValueError(
            "The existing fights.csv does not have the "
            "expected structure. Run again with "
            "--restart. Missing columns: "
            f"{sorted(missing_columns)}"
        )

    return existing_fights[
        FIGHT_COLUMNS
    ].copy()


def load_rendered_event(
    page: Page,
    event_url: str,
) -> tuple[BeautifulSoup, int]:
    """
    Open an event and return its rendered HTML.
    """

    response = page.goto(
        event_url.strip(),
        wait_until="domcontentloaded",
        timeout=60_000,
    )

    if response is None:
        raise RuntimeError(
            "No response was received while opening: "
            f"{event_url}"
        )

    if response.status >= 400:
        raise RuntimeError(
            "UFCStats returned HTTP "
            f"{response.status}: {event_url}"
        )

    row_locator = page.locator(
        "tr[data-link*='/fight-details/']"
    )

    try:
        row_locator.first.wait_for(
            state="attached",
            timeout=45_000,
        )
    except PlaywrightTimeoutError as error:
        raise RuntimeError(
            "The page responded, but the "
            "fight rows did not appear."
        ) from error

    # Wait until the row count stabilizes.
    previous_count = -1

    for _ in range(10):
        current_count = row_locator.count()

        if (
            current_count > 0
            and current_count == previous_count
        ):
            break

        previous_count = current_count

        page.wait_for_timeout(250)

    browser_row_count = row_locator.count()

    print(
        f"  HTTP status: {response.status}"
    )
    print(
        f"  Final title: {page.title()}"
    )
    print(
        "  Fights loaded in the browser: "
        f"{browser_row_count}"
    )

    rendered_html = page.content()

    return (
        BeautifulSoup(
            rendered_html,
            "lxml",
        ),
        browser_row_count,
    )


def parse_fight_row(
    row: Tag,
    event: pd.Series,
    fight_order: int,
) -> dict[str, str] | None:
    """Convert an event row into a fight record."""

    fight_url = row.get(
        "data-link",
        "",
    ).strip()

    if "/fight-details/" not in fight_url:
        return None

    cells = row.find_all(
        "td",
        recursive=False,
    )

    if len(cells) < 10:
        return None
    
    results = get_cell_values(
        cells[0]
    )

    bout_result = normalize_result(
        get_list_value(results, 0)
    )

    if bout_result == "W":
        fighter_1_result = "W"
        fighter_2_result = "L"

    elif bout_result == "L":
        # Handle the uncommon single-value layout.
        fighter_1_result = "L"
        fighter_2_result = "W"

    elif bout_result == "D":
        fighter_1_result = "D"
        fighter_2_result = "D"

    elif bout_result == "NC":
        fighter_1_result = "NC"
        fighter_2_result = "NC"

    else:
        fighter_1_result = ""
        fighter_2_result = ""

   

    fighter_links = cells[1].select(
        "a[href*='/fighter-details/']"
    )

    if len(fighter_links) < 2:
        return None

    fighter_1_url = fighter_links[0].get(
        "href",
        "",
    ).strip()

    fighter_2_url = fighter_links[1].get(
        "href",
        "",
    ).strip()

    fighter_1_name = clean_text(
        fighter_links[0].get_text(
            " ",
            strip=True,
        )
    )

    fighter_2_name = clean_text(
        fighter_links[1].get_text(
            " ",
            strip=True,
        )
    )

    fighter_1_id = extract_id(
        fighter_1_url
    )

    fighter_2_id = extract_id(
        fighter_2_url
    )

    # Upcoming rows reuse statistic cells for preview text, so do not parse those
    # cells until a result exists.
    is_upcoming_fight = (
        fighter_1_result == ""
        and fighter_2_result == ""
    )

    weight_class_values = get_cell_values(
        cells[6]
    )

    if is_upcoming_fight:
        knockdowns = []
        significant_strikes = []
        takedowns = []
        submission_attempts = []
        method_values = []
        round_values = []
        time_values = []
    else:
        knockdowns = get_cell_values(
            cells[2]
        )

        significant_strikes = get_cell_values(
            cells[3]
        )

        takedowns = get_cell_values(
            cells[4]
        )

        submission_attempts = get_cell_values(
            cells[5]
        )

        method_values = get_cell_values(
            cells[7]
        )

        round_values = get_cell_values(
            cells[8]
        )

        time_values = get_cell_values(
            cells[9]
        )

    winner_id = ""

    if fighter_1_result == "W":
        winner_id = fighter_1_id
    elif fighter_2_result == "W":
        winner_id = fighter_2_id

    return {
        "fight_id": extract_id(fight_url),
        "event_id": str(event["event_id"]),
        "event_name": str(event["event_name"]),
        "event_date": str(event["event_date"]),
        "event_location": str(
            event["event_location"]
        ),
        "event_url": str(event["event_url"]),
        "fight_order": str(fight_order),
        "fight_url": fight_url,

        "fighter_1_id": fighter_1_id,
        "fighter_1_name": fighter_1_name,
        "fighter_1_url": fighter_1_url,
        "fighter_1_result": fighter_1_result,

        "fighter_2_id": fighter_2_id,
        "fighter_2_name": fighter_2_name,
        "fighter_2_url": fighter_2_url,
        "fighter_2_result": fighter_2_result,

        "winner_id": winner_id,

        "fighter_1_knockdowns": get_list_value(
            knockdowns,
            0,
        ),
        "fighter_2_knockdowns": get_list_value(
            knockdowns,
            1,
        ),

        "fighter_1_sig_strikes": get_list_value(
            significant_strikes,
            0,
        ),
        "fighter_2_sig_strikes": get_list_value(
            significant_strikes,
            1,
        ),

        "fighter_1_takedowns": get_list_value(
            takedowns,
            0,
        ),
        "fighter_2_takedowns": get_list_value(
            takedowns,
            1,
        ),

        "fighter_1_submission_attempts": (
            get_list_value(
                submission_attempts,
                0,
            )
        ),
        "fighter_2_submission_attempts": (
            get_list_value(
                submission_attempts,
                1,
            )
        ),

        "weight_class": get_list_value(
            weight_class_values,
            0,
        ),

        "method": get_list_value(
            method_values,
            0,
        ),

        "method_details": " ".join(
            method_values[1:]
        ),

        "finish_round": get_list_value(
            round_values,
            0,
        ),

        "finish_time": get_list_value(
            time_values,
            0,
        ),
    }


def parse_event_fights(
    soup: BeautifulSoup,
    event: pd.Series,
    browser_row_count: int,
) -> list[dict[str, str]]:
    """Extract and validate an event's fights."""

    rows = soup.select(
        "tr[data-link*='/fight-details/']"
    )

    print(
        "  Rows detected in the rendered HTML: "
        f"{len(rows)}"
    )

    if len(rows) != browser_row_count:
        raise RuntimeError(
            "The browser row count does not "
            "match the rendered HTML."
        )

    fights: list[dict[str, str]] = []

    for fight_order, row in enumerate(
        rows,
        start=1,
    ):
        fight = parse_fight_row(
            row=row,
            event=event,
            fight_order=fight_order,
        )

        if fight is None:
            raise RuntimeError(
                "Could not parse row "
                f"for fight number {fight_order}."
            )

        fights.append(fight)

    validate_event_fights(
        fights=fights,
        expected_count=len(rows),
    )

    return fights


def validate_event_fights(
    fights: list[dict[str, str]],
    expected_count: int,
) -> None:
    """Validate that the event was extracted completely."""

    if not fights:
        raise ValueError(
            "No fights were extracted."
        )

    if len(fights) != expected_count:
        raise ValueError(
            "Not all rows were parsed."
        )

    dataframe = pd.DataFrame(
        fights
    )

    required_non_empty = [
        "fight_id",
        "event_id",
        "fight_url",
        "fighter_1_id",
        "fighter_1_name",
        "fighter_2_id",
        "fighter_2_name",
    ]

    for column in required_non_empty:
        empty_values = (
            dataframe[column]
            .fillna("")
            .str.strip()
            .eq("")
        )

        if empty_values.any():
            raise ValueError(
                "Empty values found "
                f"in {column}."
            )

    if dataframe["fight_id"].duplicated().any():
        raise ValueError(
            "The event contains duplicate fight_id values."
        )

    valid_fight_urls = dataframe[
        "fight_url"
    ].str.contains(
        "/fight-details/",
        regex=False,
    )

    if not valid_fight_urls.all():
        raise ValueError(
            "Invalid fight URLs "
            "were found."
        )
    
    valid_result_pairs = {
        ("W", "L"),
        ("L", "W"),
        ("D", "D"),
        ("NC", "NC"),
        ("", "")       # upcoming fight
    }

    for _, fight in dataframe.iterrows():
        result_pair = (
            fight["fighter_1_result"],
            fight["fighter_2_result"],
        )

        if result_pair not in valid_result_pairs:
            raise ValueError(
                "An invalid result combination was found "
                f"for fight {fight['fight_id']}: "
                f"{result_pair}"
            )


def save_fights(
    existing_fights: pd.DataFrame,
    new_fights: list[dict[str, str]],
) -> pd.DataFrame:
    """
    Replace an event's complete snapshot in ``fights.csv``.

    This refreshes changed cards, final results, added fights, and cancellations.
    """

    new_dataframe = pd.DataFrame(
        new_fights,
        columns=FIGHT_COLUMNS,
    )

    if new_dataframe.empty:
        raise ValueError(
            "No new fights to save."
        )

    event_ids = (
        new_dataframe["event_id"]
        .fillna("")
        .astype(str)
        .unique()
        .tolist()
    )

    if len(event_ids) != 1 or not event_ids[0]:
        raise ValueError(
            "save_fights expected exactly one event_id."
        )

    event_id = event_ids[0]

    # Replace the event with the latest complete UFCStats snapshot.
    remaining_fights = existing_fights[
        existing_fights["event_id"]
        .fillna("")
        .astype(str)
        .ne(event_id)
    ].copy()

    complete_dataframe = pd.concat(
        [
            remaining_fights,
            new_dataframe,
        ],
        ignore_index=True,
    )

    complete_dataframe = (
        complete_dataframe
        .drop_duplicates(
            subset="fight_id",
            keep="last",
        )
        .reset_index(drop=True)
    )

    complete_dataframe = complete_dataframe[
        FIGHT_COLUMNS
    ]

    FIGHTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        FIGHTS_PATH.parent
        / "fights.tmp"
    )

    complete_dataframe.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
    )

    temporary_path.replace(
        FIGHTS_PATH
    )

    return complete_dataframe


def main() -> None:
    args = parse_arguments()

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit must be greater than zero."
        )

    if args.delay < 0:
        raise ValueError(
            "--delay cannot be negative."
        )

    events = load_events()

    existing_fights = load_existing_fights(
        restart=args.restart,
    )

    processed_event_ids: set[str] = set()
    unresolved_event_ids: set[str] = set()

    if not existing_fights.empty:
        processed_event_ids = set(
            existing_fights["event_id"]
            .dropna()
            .astype(str)
        )

        # Refresh stored events with unresolved fights until results are published.
        result_1 = (
            existing_fights["fighter_1_result"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        result_2 = (
            existing_fights["fighter_2_result"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        unresolved_mask = (
            result_1.eq("")
            & result_2.eq("")
        )

        unresolved_event_ids = set(
            existing_fights.loc[
                unresolved_mask,
                "event_id",
            ]
            .dropna()
            .astype(str)
        )

    event_dates = pd.to_datetime(
        events["event_date"],
        errors="coerce",
    )

    today = pd.Timestamp.today().normalize()

    new_event_mask = ~events[
        "event_id"
    ].astype(str).isin(
        processed_event_ids
    )

    unresolved_event_mask = events[
        "event_id"
    ].astype(str).isin(
        unresolved_event_ids
    )

    current_or_future_mask = (
        event_dates >= today
    )

    # Process new, unresolved, current, and future events because their cards may change.
    pending_events = events[
        new_event_mask
        | unresolved_event_mask
        | current_or_future_mask
    ].copy()

    if args.limit is not None:
        pending_events = pending_events.head(
            args.limit
        )

    total_pending = len(
        pending_events
    )

    print(
        f"Available events: {len(events)}"
    )
    print(
        f"Pending/refreshable events: {total_pending}"
    )
    print(
        "Previously saved events with "
        f"unresolved fights: {len(unresolved_event_ids)}"
    )
    print(
        "Existing fights: "
        f"{len(existing_fights)}"
    )
    print()

    if total_pending == 0:
        print(
            "No new events or fight cards "
            "need to be refreshed."
        )
        return

    complete_fights = existing_fights

    failed_events: list[
        tuple[str, str]
    ] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.show_browser,
        )

        context = browser.new_context(
            locale="en-US",
        )

        page = context.new_page()

        page.set_default_timeout(
            45_000
        )

        page.set_default_navigation_timeout(
            60_000
        )

        try:
            for position, (_, event) in enumerate(
                pending_events.iterrows(),
                start=1,
            ):
                event_name = str(
                    event["event_name"]
                )

                event_url = str(
                    event["event_url"]
                )

                print(
                    f"[{position}/{total_pending}] "
                    f"Processing: {event_name}"
                )

                try:
                    (
                        soup,
                        browser_row_count,
                    ) = load_rendered_event(
                        page=page,
                        event_url=event_url,
                    )

                    event_fights = parse_event_fights(
                        soup=soup,
                        event=event,
                        browser_row_count=(
                            browser_row_count
                        ),
                    )

                    complete_fights = save_fights(
                        existing_fights=(
                            complete_fights
                        ),
                        new_fights=event_fights,
                    )

                    print(
                        "  Fights extracted "
                        "and saved: "
                        f"{len(event_fights)}"
                    )

                except Exception as error:
                    failed_events.append(
                        (
                            event_name,
                            str(error),
                        )
                    )

                    print(
                        "  Error processing "
                        f"the event: {error}"
                    )

                print()

                if (
                    position < total_pending
                    and args.delay > 0
                ):
                    time.sleep(
                        args.delay
                    )

        finally:
            context.close()
            browser.close()

    print("Processing complete.")
    print(
        "Fights saved: "
        f"{len(complete_fights)}"
    )
    print(
        f"File: {FIGHTS_PATH}"
    )

    if failed_events:
        print()
        print(
            "Failed events: "
            f"{len(failed_events)}"
        )

        for event_name, error in failed_events:
            print(
                f"- {event_name}: {error}"
            )


if __name__ == "__main__":
    main()