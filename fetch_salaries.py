"""
Scrapes 2025-26 NBA player salaries from Basketball Reference.
Outputs raw_salaries.csv with columns: player_id, player, team, salary
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

URL = "https://www.basketball-reference.com/contracts/players.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}


def fetch_salaries() -> pd.DataFrame:
    print(f"Fetching: {URL}")
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    print(f"Status: {resp.status_code}  |  Content length: {len(resp.text):,}")

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "player-contracts"})
    if table is None:
        raise RuntimeError("Could not find #player-contracts table. Page structure may have changed.")

    rows = []
    tbody = table.find("tbody")
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        cells = tr.find_all(["th", "td"])
        if len(cells) < 4:
            continue

        # Column layout: Rank | Player | Team | 2025-26 | 2026-27 | ...
        player_cell = cells[1]
        team_cell = cells[2]
        salary_cell = cells[3]  # 2025-26 season salary

        player_link = player_cell.find("a")
        if player_link is None:
            continue

        player_name = player_link.get_text(strip=True)
        href = player_link.get("href", "")
        # href looks like /players/a/adebaba01.html
        player_id = href.split("/")[-1].replace(".html", "") if href else ""

        team = team_cell.get_text(strip=True)
        raw_salary = salary_cell.get_text(strip=True).replace("$", "").replace(",", "")

        try:
            salary = int(raw_salary)
        except ValueError:
            salary = None  # injured reserve / two-way / missing

        rows.append(
            {
                "player_id": player_id,
                "player": player_name,
                "team": team,
                "salary": salary,
            }
        )

    df = pd.DataFrame(rows)
    df = df[df["salary"].notna()].copy()
    df["salary"] = df["salary"].astype(int)
    df["salary_millions"] = (df["salary"] / 1_000_000).round(2)

    print(f"Players scraped: {len(df)}")
    print(df.head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    df = fetch_salaries()
    df.to_csv("raw_salaries.csv", index=False)
    print("\nSaved: raw_salaries.csv")
