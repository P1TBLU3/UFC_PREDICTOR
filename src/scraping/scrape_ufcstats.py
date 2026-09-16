"""Scrape completed and upcoming UFCStats events into ``events.csv``.

Historical events are retained when UFCStats temporarily omits them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


COMPLETED_EVENTS_URL = (
    "http://ufcstats.com/"
    "statistics/events/completed?page=all"
)

UPCOMING_EVENTS_URL = (
    "http://ufcstats.com/"
    "statistics/events/upcoming"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ufcstats"
    / "events.csv"
)


def parse_arguments() -> argparse.Namespace:
    """Define the command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download the list of completed and upcoming "
            "UFCStats events."
        )
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


def extract_event_id(event_url: str) -> str:
    """Extract the event identifier from its URL."""

    path = urlparse(event_url).path.rstrip("/")

    if not path:
        return ""

    return path.split("/")[-1]


def load_rendered_page(
    page: Page,
    url: str,
) -> BeautifulSoup:
    """
    Open a page in Chromium and return its rendered HTML.
    """

    response = page.goto(
        url.strip(),
        wait_until="domcontentloaded",
        timeout=60_000,
    )

    if response is None:
        raise RuntimeError(
            f"No response was received while opening: {url}"
        )

    if response.status >= 400:
        raise RuntimeError(
            f"UFCStats returned HTTP "
            f"{response.status}: {url}"
        )

    event_links = page.locator(
        "a[href*='/event-details/']"
    )

    try:
        event_links.first.wait_for(
            state="attached",
            timeout=45_000,
        )
    except PlaywrightTimeoutError as error:
        raise RuntimeError(
            "The page loaded, but the expected "
            f"event links did not appear: {url}"
        ) from error

    # Allow the DOM to finish updating.
    page.wait_for_timeout(500)

    print(f"HTTP status: {response.status}")
    print(f"Initial/final title: {page.title()}")
    print(
        "Event links detected in the browser: "
        f"{event_links.count()}"
    )

    rendered_html = page.content()

    return BeautifulSoup(
        rendered_html,
        "lxml",
    )


def parse_events(
    soup: BeautifulSoup,
) -> pd.DataFrame:
    """Extract events from the rendered HTML."""

    events: list[dict[str, str]] = []

    rows = soup.select(
        "tr.b-statistics__table-row"
    )

    print(
        "Table rows detected in the HTML: "
        f"{len(rows)}"
    )

    for row in rows:
        event_link = row.select_one(
            "a[href*='/event-details/']"
        )

        event_date = row.select_one(
            "span.b-statistics__date"
        )

        if event_link is None or event_date is None:
            # Skip the table header.
            continue

        event_url = event_link.get(
            "href",
            "",
        ).strip()

        event_name = clean_text(
            event_link.get_text(
                " ",
                strip=True,
            )
        )

        date_text = clean_text(
            event_date.get_text(
                " ",
                strip=True,
            )
        )

        columns = row.find_all(
            "td",
            recursive=False,
        )

        location = ""

        if len(columns) >= 2:
            location = clean_text(
                columns[1].get_text(
                    " ",
                    strip=True,
                )
            )

        event_id = extract_event_id(
            event_url
        )

        if not event_id:
            continue

        events.append(
            {
                "event_id": event_id,
                "event_name": event_name,
                "event_date": date_text,
                "event_location": location,
                "event_url": event_url,
            }
        )

    dataframe = pd.DataFrame(events)

    if dataframe.empty:
        raise RuntimeError(
            "No events could be extracted."
        )

    dataframe["event_date"] = pd.to_datetime(
        dataframe["event_date"],
        format="%B %d, %Y",
        errors="coerce",
    )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset="event_id",
            keep="first",
        )
        .sort_values(
            by="event_date",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return dataframe


def load_existing_events() -> pd.DataFrame:
    """
    Load the previous ``events.csv`` to retain historical events that UFCStats may temporarily omit.
    """

    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    existing = pd.read_csv(
        OUTPUT_PATH,
        dtype={"event_id": "string"},
    )

    if "event_date" in existing.columns:
        existing["event_date"] = pd.to_datetime(
            existing["event_date"],
            errors="coerce",
        )

    return existing


def merge_with_existing_history(
    scraped_events: pd.DataFrame,
    existing_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the current scrape with stored history.

    Fresh data takes precedence. Only missing past events are retained; absent
    future events may have been cancelled or changed.
    """

    if existing_events.empty:
        return scraped_events

    required = {
        "event_id",
        "event_name",
        "event_date",
        "event_location",
        "event_url",
    }

    if not required.issubset(existing_events.columns):
        print(
            "Warning: the previous events.csv does not contain all "
            "expected columns and will not be reused."
        )
        return scraped_events

    today = pd.Timestamp.today().normalize()

    scraped_ids = set(
        scraped_events["event_id"]
        .astype("string")
        .dropna()
        .tolist()
    )

    historical_missing = existing_events[
        (~existing_events["event_id"].astype("string").isin(scraped_ids))
        & (existing_events["event_date"].notna())
        & (existing_events["event_date"] <= today)
    ].copy()

    if not historical_missing.empty:
        print()
        print(
            "Historical events retained from the previous CSV "
            f"because UFCStats omitted them: {len(historical_missing)}"
        )
        print(
            historical_missing[
                ["event_date", "event_name", "event_id"]
            ]
            .sort_values("event_date", ascending=False)
            .head(10)
            .to_string(index=False)
        )

    combined = pd.concat(
        [scraped_events, historical_missing],
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            subset="event_id",
            keep="first",
        )
        .sort_values(
            by="event_date",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return combined


def validate_events(
    events: pd.DataFrame,
) -> None:
    """Validate the extracted dataset."""

    required_columns = [
        "event_id",
        "event_name",
        "event_date",
        "event_location",
        "event_url",
    ]

    missing_columns = (
        set(required_columns)
        - set(events.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if events.empty:
        raise ValueError(
            "The event dataset is empty."
        )

    if events["event_id"].isna().any():
        raise ValueError(
            "Some events have no event_id."
        )

    if events["event_id"].duplicated().any():
        duplicated = events.loc[
            events["event_id"].duplicated(),
            "event_id",
        ].tolist()

        raise ValueError(
            "Duplicate event_id values found: "
            f"{duplicated[:10]}"
        )

    if events["event_name"].fillna("").str.strip().eq("").any():
        raise ValueError(
            "Some events have no name."
        )

    if events["event_date"].isna().any():
        invalid_events = events.loc[
            events["event_date"].isna(),
            "event_name",
        ].tolist()

        raise ValueError(
            "Some dates could not be parsed: "
            f"{invalid_events[:10]}"
        )

    valid_urls = (
        events["event_url"]
        .fillna("")
        .str.contains(
            "/event-details/",
            regex=False,
        )
    )

    if not valid_urls.all():
        invalid_urls = events.loc[
            ~valid_urls,
            "event_url",
        ].tolist()

        raise ValueError(
            "Invalid event URLs found: "
            f"{invalid_urls[:10]}"
        )


def save_events(
    events: pd.DataFrame,
) -> None:
    """
    Write ``events.csv`` atomically through a temporary file.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        OUTPUT_PATH.parent
        / "events.tmp"
    )

    events.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    temporary_path.replace(
        OUTPUT_PATH
    )


def print_summary(
    events: pd.DataFrame,
) -> None:
    """Print a summary of the extracted events."""

    newest_date = events["event_date"].max()
    oldest_date = events["event_date"].min()
    today = pd.Timestamp.today().normalize()

    completed_count = int(
        (events["event_date"] < today).sum()
    )
    today_count = int(
        (events["event_date"] == today).sum()
    )
    upcoming_count = int(
        (events["event_date"] > today).sum()
    )

    print()
    print("Validation completed successfully.")
    print(f"Events extracted/saved: {len(events)}")
    print(
        "Unique identifiers: "
        f"{events['event_id'].nunique()}"
    )
    print(f"Events before today: {completed_count}")
    print(f"Events scheduled for today: {today_count}")
    print(f"Upcoming events: {upcoming_count}")
    print(
        "Most recent event: "
        f"{newest_date:%Y-%m-%d}"
    )
    print(
        "Oldest event: "
        f"{oldest_date:%Y-%m-%d}"
    )
    print(f"File saved to: {OUTPUT_PATH}")

    print()
    print("First events by date:")
    print(
        events[
            [
                "event_date",
                "event_name",
                "event_location",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


def main() -> None:
    args = parse_arguments()

    existing_events = load_existing_events()

    sources = [
        ("COMPLETED EVENTS", COMPLETED_EVENTS_URL),
        ("UPCOMING EVENTS", UPCOMING_EVENTS_URL),
    ]

    scraped_frames: list[pd.DataFrame] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.show_browser,
        )

        context = browser.new_context(
            locale="en-US",
        )

        page = context.new_page()

        try:
            for label, url in sources:
                print()
                print("=" * 70)
                print(label)
                print("=" * 70)
                print(f"URL: {url}")
                print()

                soup = load_rendered_page(
                    page=page,
                    url=url,
                )

                events = parse_events(
                    soup=soup,
                )

                print(
                    f"Events extracted from this page: {len(events)}"
                )

                scraped_frames.append(events)

        finally:
            context.close()
            browser.close()

    scraped_events = pd.concat(
        scraped_frames,
        ignore_index=True,
    )

    scraped_events = (
        scraped_events
        .drop_duplicates(
            subset="event_id",
            keep="first",
        )
        .sort_values(
            by="event_date",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    events = merge_with_existing_history(
        scraped_events=scraped_events,
        existing_events=existing_events,
    )

    validate_events(
        events=events,
    )

    save_events(
        events=events,
    )

    print_summary(
        events=events,
    )


if __name__ == "__main__":
    main()