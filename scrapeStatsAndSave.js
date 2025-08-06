const puppeteer = require("puppeteer");
const fs = require("fs");
const xlsx = require("xlsx");
const path = require("path");
const { allWeeks } = require("./links"); // links.js içinde 10 sezonluk maç linkleri var

async function extractMatchData(page, url) {
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  try {
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.waitForSelector(".scorebox", { timeout: 20000 });
    await page.waitForSelector("#team_stats_extra", { timeout: 20000 });

    await sleep(10000);
    await page.evaluate(() => window.scrollBy(0, 3500));
    await sleep(3000);

    const data = await page.evaluate(() => {
      const clean = (s) => s?.trim().replace(/\n/g, "") || "";
      const result = {};

      const divs = document.querySelectorAll(".scorebox > div");
      const home = divs[0],
        away = divs[1];
      const homeTeam = clean(home.querySelector("strong a")?.textContent);
      const awayTeam = clean(away.querySelector("strong a")?.textContent);
      result.HomeTeam = homeTeam;
      result.AwayTeam = awayTeam;

      const homeScore = clean(home.querySelector(".score")?.textContent);
      const awayScore = clean(away.querySelector(".score")?.textContent);

      const hGoal = parseInt(homeScore);
      const aGoal = parseInt(awayScore);
      result.HomeGoal = hGoal;
      result.AwayGoal = aGoal;

      result.Result =
        hGoal > aGoal ? "HomeWin" : hGoal < aGoal ? "AwayWin" : "Draw";

      result.hXG = clean(home.querySelector(".score_xg")?.textContent);
      result.aXG = clean(away.querySelector(".score_xg")?.textContent);

      const section = document.querySelector("#team_stats_extra");
      const rawText = section?.innerText || "";
      const lines = rawText.split("\n").map(clean).filter(Boolean);

      const shortNames = ["Leverkusen", "Eint Frankfurt", "Gladbach"];
      const filtered = lines.filter(
        (line) =>
          line !== "" &&
          line !== homeTeam &&
          line !== awayTeam &&
          !shortNames.includes(line)
      );

      for (let i = 0; i < filtered.length; i += 3) {
        const homeVal = filtered[i];
        const statName = filtered[i + 1];
        const awayVal = filtered[i + 2];

        if (statName && homeVal && awayVal) {
          result[`Home${statName}`] = homeVal;
          result[`Away${statName}`] = awayVal;
        }
      }

      const barSection = document.querySelector("#team_stats");
      const barLines =
        barSection?.innerText?.split("\n").map(clean).filter(Boolean) || [];

      for (let i = 0; i < barLines.length; i++) {
        const statName = barLines[i];
        const interest = [
          "Possession",
          "Passing Accuracy",
          "Shots on Target",
          "Saves",
        ];
        if (interest.includes(statName) && barLines[i + 1] && barLines[i + 2]) {
          result[`Home${statName}`] = barLines[i + 1];
          result[`Away${statName}`] = barLines[i + 2];
          i += 2;
        }
      }

      return result;
    });

    console.log("📦 Extracted match data preview:");
    console.log(JSON.stringify(data, null, 2));
    return data;
  } catch (err) {
    console.error("❌ extractMatchData error:", err.message);
    return null;
  }
}

(async () => {
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();

  for (const weekObj of allWeeks) {
    console.log(`📅 Processing week ${weekObj.week} (${weekObj.urls.length} matches)`);
    const allMatchData = [];

    for (const url of weekObj.urls) {
      console.log("⏳ Scraping:", url);
      const matchData = await extractMatchData(page, url);

      if (matchData) {
        allMatchData.push(matchData);
      } else {
        console.log("⚠️ No data extracted from:", url);
      }

      await new Promise((res) =>
        setTimeout(res, 4000 + Math.random() * 4000)
      );
    }

    if (allMatchData.length > 0) {
      const ws = xlsx.utils.json_to_sheet(allMatchData);
      const wb = xlsx.utils.book_new();
      xlsx.utils.book_append_sheet(wb, ws, "Matches");
      const folder = path.join(__dirname, "excels");
      if (!fs.existsSync(folder)) fs.mkdirSync(folder);
      const filePath = path.join(folder, `week${weekObj.week}.xlsx`);
      xlsx.writeFile(wb, filePath);
      console.log(`✅ Saved: ${filePath}`);
    } else {
      console.log(`⚠️ No matches saved for week ${weekObj.week}`);
    }
  }

  await browser.close();
})();
