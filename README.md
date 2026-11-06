# Polymarket sports calibration

Research code for studying the calibration of closed Polymarket sports and
esports markets. The analysis uses saved hourly trade aggregates, settled
outcomes, and event-inferred sport and game labels.

## Repository contents

- `Polymarket_Advanced_Statistics.ipynb` analyses sport calibration, event-level
  bootstrap intervals, logistic regression, time-based validation, multiple
  testing, and ridge shrinkage.
- `Polymarket_Esports_Game_Analysis.ipynb` compares videogames within the
  esports subset and plots game-specific logistic calibration curves.
- `Polymarket_Closed_Sports_Combined_Pipeline.ipynb` builds the local dataset
  and event labels; `Fetch_Cali/` contains the original collection and export
  scripts.
- `plot_event_sport_calibration.py` produces sport calibration plots from the
  existing hourly data.

## Local data and running the notebooks

Datasets, generated figures, notebook outputs, caches, and archives are not
included in Git. Obtain the dataset separately and place it in
`Data/all_closed_sports/`. The two analysis notebooks require the saved hourly
master CSV (`closed_sports_all_data.csv`) and the event files (`events.jsonl`
and `market_events.jsonl`); they do not need to download trades again.

Open Jupyter from the repository root. Run the Advanced Statistics notebook
once before the Esports notebook: the first notebook builds a local analysis
cache that the second notebook uses. The first run can take several minutes.
The notebooks use Python, Jupyter, pandas, NumPy, SciPy, matplotlib, and
scikit-learn; SciPy 1.11 or newer is needed for false-discovery correction.

Git versions of the notebooks contain code and explanations but no saved cell
outputs. Run the cells locally to regenerate tables and figures. The local
`Figures/` folder is ignored by Git and can contain previously saved results.

The analyses use available hourly prices for hours 2–24 before each market's
`endDate`; each market is counted at most once per probability bin. They are
retrospective analyses, not necessarily pre-game forecasts. Final volume,
liquidity, and spread are closing snapshots rather than historical values.
