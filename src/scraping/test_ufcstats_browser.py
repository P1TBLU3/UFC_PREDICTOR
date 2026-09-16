"""Open one UFCStats event to verify browser access and rendered selectors."""

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


URL = (
    "http://ufcstats.com/"
    "event-details/31e1ea6fe6b682f8"
)


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False,
        )

        page = browser.new_page()

        print("Opening UFCStats...")
        print(f"Requested URL: {URL}")

        try:
            response = page.goto(
                URL,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            print(f"Final URL: {page.url}")

            if response is not None:
                print(
                    f"HTTP status: {response.status}"
                )

            print(
                f"Initial title: {page.title()}"
            )

            # Allow the loading screen to disappear.
            page.wait_for_timeout(5_000)

            rows = page.locator(
                "tr[data-link*='/fight-details/']"
            )

            try:
                rows.first.wait_for(
                    state="attached",
                    timeout=30_000,
                )
            except PlaywrightError:
                pass

            print(
                f"Final title: {page.title()}"
            )
            print(
                "Fights detected: "
                f"{rows.count()}"
            )

            fight_links = page.locator(
                "a[href*='/fight-details/']"
            )

            print(
                "Fight-detail links: "
                f"{fight_links.count()}"
            )

        except PlaywrightError as error:
            print(
                f"Navigation error: {error}"
            )

        input(
            "Press Enter to close the browser..."
        )

        browser.close()


if __name__ == "__main__":
    main()