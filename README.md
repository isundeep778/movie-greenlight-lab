# Greenlight Lab

An interactive, student-built prototype for exploring movie box-office history and stress-testing a film concept. It combines data analysis, a machine-learning baseline, scenario calculations, and an optional market comparison.

> **Important:** This is an educational decision-support prototype, not a reliable studio investment system. Its model uses observational public data and does not establish that a genre, actor, director, or budget change causes a particular result.

## What it can do

- Summarize historical gross and budgets by genre.
- Train a random-forest regression model on historical records.
- Estimate a worldwide box-office range for a film concept using budget, year, genre, director, lead actor, and language.
- Compare budget scenarios and show a simple downside/net calculation based on user-entered marketing and studio-share assumptions.
- Compare markets if the user supplies a separate, sourced country-level revenue file.
- Report an out-of-time validation error using the latest release years as a holdout.

Marketing effectiveness, production costs beyond the entered budget, studio accounting, streaming deals, and country-level performance are not inferred unless appropriate evidence is supplied. Marketing spend is only an input to the scenario arithmetic.

## Run locally

1. Install Python 3.10 or newer.
2. In this folder, create and activate a virtual environment.
3. Install the packages in `requirements.txt`.
4. Start the app with `streamlit run app.py`.
5. In the sidebar, upload the two CSVs from the TMDB 5000 Movie Metadata dataset: `tmdb_5000_movies.csv` and, for cast/director analysis, `tmdb_5000_credits.csv`.

Dataset page: <https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata>. Review the dataset's current terms and attribution requirements before redistribution. This repository does not bundle the dataset.

## Expected data

The movie file needs `title`, `budget`, `revenue`, `release_date`, and `genres`. The optional credits file needs `movie_id`, `cast`, and `crew`. The app joins credits to movies by movie ID and parses the common JSON fields in the TMDB 5000 files. Rows with missing/zero budget, revenue, or invalid release dates are excluded from modeling.

Optional market CSV format, one row per title and market:

```csv
title,market,revenue
Example Film,United States,120000000
Example Film,United Kingdom,25000000
```

The regional file must have the same title spelling as the uploaded movie dataset, comparable currency, and a trustworthy source. Without this file, the app does not rank markets.

## Modeling notes

- Target: log of worldwide revenue.
- Inputs: log budget, release year, primary genre, director, lead billed actor, and original language.
- Estimator: scikit-learn random forest with one-hot encoded categorical inputs.
- Validation: chronological holdout of the latest 20% of release years.
- Range: an empirical band based on the 80th percentile absolute log error on the holdout. This is not a calibrated prediction interval.
- Financial scenario: estimated gross × assumed studio share − production budget − entered marketing budget. This omits distribution fees, participations, taxes, financing, streaming, and ancillary revenue.

## Resume wording (only after you run it and understand the work)

**Greenlight Lab — Movie Box-Office Scenario Analyzer** | Python, Streamlit, pandas, scikit-learn

Built an interactive film-slate analysis prototype using historical movie metadata; trained and evaluated a time-based box-office regression baseline, then added budget/cast/director what-if scenarios and transparent downside assumptions.

Do not claim marketing optimization, true profit prediction, or causal star/director impact: this version does not establish those.
