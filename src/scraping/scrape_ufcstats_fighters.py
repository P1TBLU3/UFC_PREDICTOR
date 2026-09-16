"""Scrape UFCStats profiles for fighters referenced in ``fights.csv``.

Profiles are checkpointed so interrupted runs can resume safely.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGHTS_PATH = PROJECT_ROOT / "data" / "raw" / "ufcstats" / "fights.csv"
FIGHTERS_PATH = PROJECT_ROOT / "data" / "raw" / "ufcstats" / "fighters.csv"

FIGHTER_COLUMNS = [
    "fighter_id", "fighter_name", "fighter_name_source", "nickname", "fighter_url",
    "height_raw", "height_cm", "weight_raw", "weight_lbs", "weight_kg",
    "reach_raw", "reach_inches", "reach_cm", "stance", "date_of_birth",
    "record_raw_current", "wins_current", "losses_current", "draws_current",
    "no_contests_current", "slpm_current", "striking_accuracy_current",
    "sapm_current", "striking_defense_current", "takedown_average_current",
    "takedown_accuracy_current", "takedown_defense_current",
    "submission_average_current", "profile_status", "scraped_at_utc",
]

PROFILE_LABEL_MAPPING = {
    "HEIGHT": "height_raw",
    "WEIGHT": "weight_raw",
    "REACH": "reach_raw",
    "STANCE": "stance",
    "DOB": "date_of_birth_raw",
    "SLPM": "slpm_current",
    "STRACC": "striking_accuracy_current",
    "SAPM": "sapm_current",
    "STRDEF": "striking_defense_current",
    "TDAVG": "takedown_average_current",
    "TDACC": "takedown_accuracy_current",
    "TDDEF": "takedown_defense_current",
    "SUBAVG": "submission_average_current",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download profiles for fighters listed in fights.csv."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--show-browser", action="store_true")
    return parser.parse_args()


def clean_text(value: str) -> str:
    return " ".join(str(value).split())


def extract_id(url: str) -> str:
    path = urlparse(str(url).strip()).path.rstrip("/")
    return path.split("/")[-1] if path else ""


def normalize_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", clean_text(label).upper())


def parse_height_cm(value: str) -> str:
    text = clean_text(value)
    if not text or text == "--":
        return ""
    match = re.search(r"(?P<feet>\d+)\s*'\s*(?P<inches>\d+)\s*(?:\"|in)?", text)
    if match is None:
        return ""
    total_inches = int(match.group("feet")) * 12 + int(match.group("inches"))
    return f"{total_inches * 2.54:.2f}"


def parse_weight(value: str) -> tuple[str, str]:
    text = clean_text(value)
    if not text or text == "--":
        return "", ""
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return "", ""
    pounds = float(match.group(1))
    return f"{pounds:g}", f"{pounds * 0.45359237:.2f}"


def parse_reach(value: str) -> tuple[str, str]:
    text = clean_text(value)
    if not text or text == "--":
        return "", ""
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return "", ""
    inches = float(match.group(1))
    return f"{inches:g}", f"{inches * 2.54:.2f}"


def parse_percentage(value: str) -> str:
    text = clean_text(value)
    if not text or text == "--":
        return ""
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if match is None:
        return ""
    return f"{float(match.group(1)) / 100:.6f}".rstrip("0").rstrip(".")


def parse_decimal(value: str) -> str:
    text = clean_text(value)
    if not text or text == "--":
        return ""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return f"{float(match.group(0)):g}" if match else ""


def parse_date_of_birth(value: str) -> str:
    text = clean_text(value)
    if not text or text == "--":
        return ""
    try:
        parsed = datetime.strptime(text, "%b %d, %Y")
    except ValueError:
        return ""
    return parsed.strftime("%Y-%m-%d")


def parse_record(value: str) -> tuple[str, str, str, str]:
    text = clean_text(value)
    match = re.search(
        r"(?P<wins>\d+)\s*-\s*(?P<losses>\d+)\s*-\s*(?P<draws>\d+)"
        r"(?:\s*\(\s*(?P<nc>\d+)\s*NC\s*\))?",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return "", "", "", ""
    return (
        match.group("wins"),
        match.group("losses"),
        match.group("draws"),
        match.group("nc") or "0",
    )


def load_unique_fighters() -> pd.DataFrame:
    if not FIGHTS_PATH.exists():
        raise FileNotFoundError(f"Fight file not found: {FIGHTS_PATH}")

    fights = pd.read_csv(
        FIGHTS_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    required_columns = {
        "fighter_1_id", "fighter_1_name", "fighter_1_url",
        "fighter_2_id", "fighter_2_name", "fighter_2_url",
    }
    missing_columns = required_columns - set(fights.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns in fights.csv: "
            f"{sorted(missing_columns)}"
        )

    fighter_1 = fights[["fighter_1_id", "fighter_1_name", "fighter_1_url"]].rename(
        columns={
            "fighter_1_id": "fighter_id",
            "fighter_1_name": "fighter_name_source",
            "fighter_1_url": "fighter_url",
        }
    )
    fighter_2 = fights[["fighter_2_id", "fighter_2_name", "fighter_2_url"]].rename(
        columns={
            "fighter_2_id": "fighter_id",
            "fighter_2_name": "fighter_name_source",
            "fighter_2_url": "fighter_url",
        }
    )

    fighters = pd.concat([fighter_1, fighter_2], ignore_index=True)
    fighters = fighters[
        fighters["fighter_id"].str.strip().ne("")
        & fighters["fighter_url"].str.contains("/fighter-details/", regex=False)
    ].copy()

    url_counts = fighters.groupby("fighter_id")["fighter_url"].nunique()
    inconsistent_ids = url_counts[url_counts > 1].index.tolist()
    if inconsistent_ids:
        raise ValueError(
            "Some fighter_id values map to multiple URLs: "
            f"{inconsistent_ids[:10]}"
        )

    return (
        fighters
        .drop_duplicates(subset="fighter_id", keep="first")
        .sort_values(by=["fighter_name_source", "fighter_id"])
        .reset_index(drop=True)
    )


def load_existing_fighters(restart: bool) -> pd.DataFrame:
    if restart or not FIGHTERS_PATH.exists():
        return pd.DataFrame(columns=FIGHTER_COLUMNS)
    try:
        existing = pd.read_csv(
            FIGHTERS_PATH,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=FIGHTER_COLUMNS)

    missing_columns = set(FIGHTER_COLUMNS) - set(existing.columns)
    if missing_columns:
        raise ValueError(
            "The existing fighters.csv does not have the expected structure. "
            "Run with --restart. Missing columns: "
            f"{sorted(missing_columns)}"
        )
    return existing[FIGHTER_COLUMNS].copy()


def load_rendered_profile(page: Page, fighter_url: str) -> BeautifulSoup:
    response = page.goto(
        fighter_url.strip(),
        wait_until="domcontentloaded",
        timeout=60_000,
    )
    if response is None:
        raise RuntimeError(f"No response was received while opening: {fighter_url}")
    if response.status >= 400:
        raise RuntimeError(
            f"UFCStats returned HTTP {response.status}: {fighter_url}"
        )

    title_locator = page.locator("span.b-content__title-highlight")
    try:
        title_locator.first.wait_for(state="attached", timeout=45_000)
    except PlaywrightTimeoutError as error:
        raise RuntimeError(
            "The page responded, but the fighter profile did not appear."
        ) from error

    profile_items = page.locator("li.b-list__box-list-item")
    previous_count = -1
    for _ in range(12):
        current_count = profile_items.count()
        if current_count > 0 and current_count == previous_count:
            break
        previous_count = current_count
        page.wait_for_timeout(250)

    print(f"  HTTP status: {response.status}")
    print(f"  Final title: {page.title()}")
    print(f"  Rendered profile fields: {profile_items.count()}")

    return BeautifulSoup(page.content(), "lxml")


def load_profile_with_retries(
    page: Page,
    fighter_url: str,
    max_attempts: int,
) -> BeautifulSoup:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(f"  Retry {attempt}/{max_attempts}...")
                page.goto("about:blank", wait_until="commit", timeout=10_000)
                page.wait_for_timeout(1_500 * attempt)
            return load_rendered_profile(page, fighter_url)
        except Exception as error:
            last_error = error
            print(f"  Attempt {attempt} failed: {error}")
    raise RuntimeError(
        "The profile could not be loaded after "
        f"{max_attempts} attempts: {last_error}"
    )


def extract_profile_values(soup: BeautifulSoup) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in soup.select("li.b-list__box-list-item"):
        label_node = item.select_one("i.b-list__box-item-title")
        if label_node is None:
            continue
        label = clean_text(label_node.get_text(" ", strip=True))
        complete_text = clean_text(item.get_text(" ", strip=True))
        value = complete_text[len(label):].strip() if complete_text.startswith(label) else complete_text
        destination = PROFILE_LABEL_MAPPING.get(normalize_label(label))
        if destination is not None:
            values[destination] = value
    return values


def parse_fighter_profile(
    soup: BeautifulSoup,
    source_fighter: pd.Series,
) -> dict[str, str]:
    name_node = soup.select_one("span.b-content__title-highlight")
    if name_node is None:
        raise RuntimeError("The fighter name was not found.")
    fighter_name = clean_text(name_node.get_text(" ", strip=True))

    record_node = soup.select_one("span.b-content__title-record")
    record_raw = ""
    if record_node is not None:
        record_raw = clean_text(record_node.get_text(" ", strip=True))
        record_raw = re.sub(r"^Record:\s*", "", record_raw, flags=re.IGNORECASE)

    nickname_node = soup.select_one("p.b-content__Nickname")
    nickname = clean_text(nickname_node.get_text(" ", strip=True)) if nickname_node else ""

    values = extract_profile_values(soup)
    height_raw = values.get("height_raw", "")
    weight_raw = values.get("weight_raw", "")
    reach_raw = values.get("reach_raw", "")
    weight_lbs, weight_kg = parse_weight(weight_raw)
    reach_inches, reach_cm = parse_reach(reach_raw)
    wins, losses, draws, no_contests = parse_record(record_raw)

    source_id = clean_text(source_fighter["fighter_id"])
    source_url = clean_text(source_fighter["fighter_url"])
    if extract_id(source_url) != source_id:
        raise ValueError(
            f"fighter_id does not match the URL: {source_id} / {source_url}"
        )

    result = {
        "fighter_id": source_id,
        "fighter_name": fighter_name,
        "fighter_name_source": clean_text(source_fighter["fighter_name_source"]),
        "nickname": nickname,
        "fighter_url": source_url,
        "height_raw": height_raw,
        "height_cm": parse_height_cm(height_raw),
        "weight_raw": weight_raw,
        "weight_lbs": weight_lbs,
        "weight_kg": weight_kg,
        "reach_raw": reach_raw,
        "reach_inches": reach_inches,
        "reach_cm": reach_cm,
        "stance": values.get("stance", ""),
        "date_of_birth": parse_date_of_birth(values.get("date_of_birth_raw", "")),
        "record_raw_current": record_raw,
        "wins_current": wins,
        "losses_current": losses,
        "draws_current": draws,
        "no_contests_current": no_contests,
        "slpm_current": parse_decimal(values.get("slpm_current", "")),
        "striking_accuracy_current": parse_percentage(values.get("striking_accuracy_current", "")),
        "sapm_current": parse_decimal(values.get("sapm_current", "")),
        "striking_defense_current": parse_percentage(values.get("striking_defense_current", "")),
        "takedown_average_current": parse_decimal(values.get("takedown_average_current", "")),
        "takedown_accuracy_current": parse_percentage(values.get("takedown_accuracy_current", "")),
        "takedown_defense_current": parse_percentage(values.get("takedown_defense_current", "")),
        "submission_average_current": parse_decimal(values.get("submission_average_current", "")),
        "profile_status": "ok",
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    validate_fighter(result)
    return result


def validate_fighter(fighter: dict[str, str]) -> None:
    for column in [
        "fighter_id", "fighter_name", "fighter_url", "profile_status", "scraped_at_utc"
    ]:
        if not clean_text(fighter.get(column, "")):
            raise ValueError(f"Field {column} is empty.")

    if "/fighter-details/" not in fighter["fighter_url"]:
        raise ValueError("The fighter URL is invalid.")

    for percentage_field in [
        "striking_accuracy_current", "striking_defense_current",
        "takedown_accuracy_current", "takedown_defense_current",
    ]:
        value = fighter.get(percentage_field, "")
        if value and not 0 <= float(value) <= 1:
            raise ValueError(f"{percentage_field} is outside [0, 1].")

    if fighter["height_cm"] and not 120 <= float(fighter["height_cm"]) <= 230:
        raise ValueError(f"Implausible height: {fighter['height_cm']}")
    if fighter["reach_cm"] and not 120 <= float(fighter["reach_cm"]) <= 250:
        raise ValueError(f"Implausible reach: {fighter['reach_cm']}")


def save_fighters(fighters: pd.DataFrame) -> None:
    fighters = (
        fighters
        .drop_duplicates(subset="fighter_id", keep="last")
        .sort_values(by=["fighter_name", "fighter_id"])
        .reset_index(drop=True)
    )[FIGHTER_COLUMNS]

    FIGHTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = FIGHTERS_PATH.parent / "fighters.tmp"
    fighters.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(FIGHTERS_PATH)


def main() -> None:
    args = parse_arguments()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be greater than zero.")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be greater than zero.")

    all_fighters = load_unique_fighters()
    existing_fighters = load_existing_fighters(args.restart)
    processed_ids = set(existing_fighters["fighter_id"].astype(str).str.strip())
    pending = all_fighters[~all_fighters["fighter_id"].isin(processed_ids)].copy()
    if args.limit is not None:
        pending = pending.head(args.limit)

    print(f"Available fighters: {len(all_fighters)}")
    print(f"Pending fighters: {len(pending)}")
    print(f"Existing profiles: {len(existing_fighters)}")
    print()

    if pending.empty:
        print("No new fighters to process.")
        return

    complete = existing_fighters.copy()
    processed_in_run = 0
    failed: list[tuple[str, str, str]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.show_browser)
        context = browser.new_context(locale="en-US")
        page = context.new_page()
        page.set_default_timeout(45_000)
        page.set_default_navigation_timeout(60_000)

        try:
            for position, (_, fighter) in enumerate(pending.iterrows(), start=1):
                source_name = clean_text(fighter["fighter_name_source"])
                fighter_id = clean_text(fighter["fighter_id"])
                fighter_url = clean_text(fighter["fighter_url"])
                print(f"[{position}/{len(pending)}] Processing: {source_name}")

                try:
                    soup = load_profile_with_retries(
                        page=page,
                        fighter_url=fighter_url,
                        max_attempts=args.max_attempts,
                    )
                    parsed = parse_fighter_profile(soup, fighter)
                    complete = pd.concat(
                        [complete, pd.DataFrame([parsed], columns=FIGHTER_COLUMNS)],
                        ignore_index=True,
                    )
                    complete = complete.drop_duplicates(
                        subset="fighter_id", keep="last"
                    ).reset_index(drop=True)
                    processed_in_run += 1
                    print(f"  Profile processed: {parsed['fighter_name']}")

                    if processed_in_run % args.checkpoint_every == 0:
                        save_fighters(complete)
                        print("  Checkpoint saved.")

                except Exception as error:
                    failed.append((fighter_id, source_name, str(error)))
                    print(f"  Error: {error}")

                print()
                if position < len(pending) and args.delay > 0:
                    time.sleep(args.delay)
        finally:
            context.close()
            browser.close()

    save_fighters(complete)
    print("Processing complete.")
    print(f"Fighters processed in this run: {processed_in_run}")
    print(f"Profiles saved: {len(complete)}")
    print(f"File: {FIGHTERS_PATH}")

    if failed:
        print()
        print(f"Failed fighters: {len(failed)}")
        for fighter_id, fighter_name, error in failed:
            print(f"- {fighter_id}: {fighter_name}: {error}")


if __name__ == "__main__":
    main()
