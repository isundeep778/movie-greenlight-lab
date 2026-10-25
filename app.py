"""Movie Greenlight Lab: exploratory box-office decision support prototype."""

from __future__ import annotations

import json
import math
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Greenlight | Film Slate Lab", page_icon="🎬", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root { --ink:#17221e; --muted:#66736d; --paper:#f5f4ef; --card:#fffefa; --line:#e5e5dc; --green:#246b4b; --lime:#c7ed77; --coral:#ed8061; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background:var(--paper); }
header[data-testid="stHeader"] { background:rgba(245,244,239,.92); }
.block-container { max-width:1360px; padding-top:2.1rem; padding-bottom:4rem; }
h1,h2,h3,h4,[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4 { color:#17221e !important; }
h1 { font-family:'Playfair Display',serif; font-size:3.15rem; letter-spacing:-.04em; margin-bottom:.15rem; }
.eyebrow { font:500 .72rem 'DM Mono',monospace; letter-spacing:.16em; color:var(--green); text-transform:uppercase; }
.lede { color:var(--muted); font-size:1.03rem; max-width:780px; line-height:1.6; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:22px 24px; box-shadow:0 3px 14px rgba(30,45,38,.035); }
.metric-label { font:500 .7rem 'DM Mono',monospace; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); }
.metric-value { font:700 1.65rem 'DM Sans',sans-serif; letter-spacing:-.04em; color:var(--ink); margin-top:8px; }
.metric-note { color:var(--muted); font-size:.8rem; margin-top:5px; }
.tag { display:inline-block; border-radius:99px; padding:5px 10px; background:#e8f1e8; color:var(--green); font:500 .72rem 'DM Mono',monospace; }
.warning { border-left:3px solid var(--coral); padding:11px 14px; color:#64483d; background:#fff4ef; border-radius:0 10px 10px 0; font-size:.86rem; }
.fine { font-size:.78rem; color:var(--muted); line-height:1.55; }
[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); padding:17px 19px; border-radius:15px; }
div[data-testid="stTabs"] button { font-family:'DM Mono',monospace; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def _first_json_names(value, kind=None, limit=3):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        items = json.loads(value)
    except (ValueError, TypeError):
        return []
    names = []
    for item in items if isinstance(items, list) else []:
        if kind and item.get("job") != kind:
            continue
        name = item.get("name")
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _parse_movies(movie_file, credits_file=None):
    movies = pd.read_csv(movie_file)
    movies.columns = [c.strip().lower().replace(" ", "_") for c in movies.columns]
    if credits_file is not None:
        credits = pd.read_csv(credits_file)
        credits.columns = [c.strip().lower().replace(" ", "_") for c in credits.columns]
        key = "movie_id" if "movie_id" in credits.columns else "id"
        movie_key = "id" if "id" in movies.columns else key
        for field in ("cast", "crew"):
            if field in credits.columns and field not in movies.columns:
                movies = movies.merge(credits[[key, field]], left_on=movie_key, right_on=key, how="left")
                if key != movie_key and key in movies.columns:
                    movies = movies.drop(columns=[key])
    aliases = {"release_date": "release_date", "budget": "budget", "revenue": "revenue", "genres": "genres", "title": "title"}
    for required in aliases:
        if required not in movies.columns:
            raise ValueError(f"Missing required column: {required}")
    movies["release_year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    movies["budget"] = pd.to_numeric(movies["budget"], errors="coerce")
    movies["revenue"] = pd.to_numeric(movies["revenue"], errors="coerce")
    movies = movies[(movies.budget > 0) & (movies.revenue > 0) & movies.release_year.notna()].copy()
    movies["genre"] = movies["genres"].apply(lambda v: ", ".join(_first_json_names(v, limit=2)) or "Unknown")
    movies["lead_cast"] = movies.get("cast", pd.Series("", index=movies.index)).apply(lambda v: ", ".join(_first_json_names(v, limit=3)) or "Unknown")
    movies["director"] = movies.get("crew", pd.Series("", index=movies.index)).apply(lambda v: ", ".join(_first_json_names(v, kind="Director", limit=1)) or "Unknown")
    if "original_language" not in movies:
        movies["original_language"] = "Unknown"
    else:
        movies["original_language"] = movies.original_language.fillna("Unknown").astype(str)
    movies["log_budget"] = np.log1p(movies.budget)
    movies["log_revenue"] = np.log1p(movies.revenue)
    movies["primary_genre"] = movies.genre.str.split(",").str[0].str.strip()
    movies["lead_actor"] = movies.lead_cast.str.split(",").str[0].str.strip()
    return movies.reset_index(drop=True)


FEATURES = ["log_budget", "release_year", "primary_genre", "director", "lead_actor", "original_language"]
NUMERIC = ["log_budget", "release_year"]
CATEGORICAL = ["primary_genre", "director", "lead_actor", "original_language"]


def _fit_model(movies):
    ordered = movies.sort_values("release_year")
    cut = max(int(len(ordered) * .8), 1)
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    if len(train) < 30:
        raise ValueError("Need at least 40 usable films with budget, revenue, and release dates.")
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), CATEGORICAL),
    ])
    model = Pipeline([("features", pre), ("forest", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=.8, random_state=42, n_jobs=-1))])
    model.fit(train[FEATURES], train.log_revenue)
    pred = np.expm1(model.predict(test[FEATURES])) if len(test) else np.array([])
    actual = test.revenue.to_numpy()
    mae = mean_absolute_error(actual, pred) if len(test) else float("nan")
    medae = median_absolute_error(actual, pred) if len(test) else float("nan")
    residual = np.abs(np.log1p(actual) - np.log1p(pred)) if len(test) else np.array([.8])
    band = float(np.quantile(residual, .8)) if len(residual) else .8
    model.fit(movies[FEATURES], movies.log_revenue)
    return model, {"mae": mae, "medae": medae, "band": band, "test_n": len(test), "cut_year": int(test.release_year.min()) if len(test) else None}


def _money(value):
    if value is None or not np.isfinite(value):
        return "—"
    if abs(value) >= 1e9:
        return f"${value/1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    return f"${value/1e3:.0f}K"


def _predict(model, budget, year, genre, director, actor, language):
    row = pd.DataFrame([{"log_budget": np.log1p(budget), "release_year": year, "primary_genre": genre, "director": director, "lead_actor": actor, "original_language": language}])
    return float(np.expm1(model.predict(row[FEATURES])[0]))


st.markdown('<div class="eyebrow">Studio strategy • prototype</div>', unsafe_allow_html=True)
st.title("Greenlight Lab")
st.markdown('<div class="lede">Explore historical box-office patterns, compare creative and budget choices, and stress-test a film concept before a greenlight. Every estimate is a scenario, not a promise.</div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.markdown("### Load a movie dataset")
    st.caption("Works with TMDB 5000 movies CSV; add its credits CSV for cast/director comparisons.")
    movie_file = st.file_uploader("Movies CSV", type="csv")
    credits_file = st.file_uploader("Credits CSV (optional)", type="csv")
    st.markdown("---")
    st.markdown("**Expected movie columns**")
    st.caption("title, budget, revenue, release_date, genres. The optional credits file should contain cast, crew, and movie_id.")
    st.markdown("---")
    st.caption("Data note: movie datasets may contain incomplete or estimated budgets and revenues. This prototype does not have studio accounting or campaign data.")

if not movie_file:
    st.markdown('<div class="panel"><div class="eyebrow">Start here</div><h3>Bring historical movie records to explore the slate.</h3><p class="fine">Upload a CSV with movie title, budget, revenue, release date, and genres. The TMDB 5000 movie metadata dataset is a common starting point. Add the matching credits CSV to compare directors and lead cast. This app filters out incomplete financial records and holds out the latest release years to estimate out-of-time error.</p><div class="warning">No dataset is bundled, so the app will not invent movie results. Market-level and marketing effects require separate evidence; the controls below treat marketing and distribution as explicit assumptions.</div></div>', unsafe_allow_html=True)
    st.stop()

try:
    movies = _parse_movies(movie_file, credits_file)
    model, quality = _fit_model(movies)
except Exception as exc:
    st.error(f"Could not prepare this dataset: {exc}")
    st.stop()

st.markdown(f'<span class="tag">{len(movies):,} usable films</span> &nbsp; <span class="tag">Latest-year holdout: {quality["test_n"]:,} films</span>', unsafe_allow_html=True)
overview, simulator, markets, methodology = st.tabs(["Slate overview", "What-if simulator", "Markets & downside", "How to read this"])

with overview:
    a, b, c, d = st.columns(4)
    a.metric("Median worldwide gross", _money(movies.revenue.median()), "in uploaded historical records")
    b.metric("Median production budget", _money(movies.budget.median()), "records with known budget + gross")
    c.metric("Holdout median absolute error", _money(quality["medae"]), f"latest {quality['test_n']} films; revenue scale")
    d.metric("Films analyzed", f"{len(movies):,}", f"years {int(movies.release_year.min())}–{int(movies.release_year.max())}")
    st.write("")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Genre history")
        summary = movies.groupby("primary_genre").agg(films=("title", "count"), median_gross=("revenue", "median"), median_budget=("budget", "median")).query("films >= 8").sort_values("median_gross", ascending=False).head(12)
        if len(summary):
            chart = summary[["median_gross", "median_budget"]].rename(columns={"median_gross": "Median gross", "median_budget": "Median budget"})
            st.bar_chart(chart)
            st.dataframe(summary.style.format({"median_gross": "${:,.0f}", "median_budget": "${:,.0f}"}), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Recent comparable films")
        st.caption("Sorted by release year; records are not audited studio financials.")
        cols = ["title", "release_year", "primary_genre", "budget", "revenue"]
        st.dataframe(movies.sort_values("release_year", ascending=False)[cols].head(12).style.format({"budget": "${:,.0f}", "revenue": "${:,.0f}"}), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

with simulator:
    st.markdown('<div class="fine">The model learns patterns from historical records. Changing cast or director shows association in this dataset, not a causal guarantee that a person changes revenue.</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    genres = sorted(movies.primary_genre.dropna().unique().tolist())
    directors = sorted(set(movies.loc[movies.director != "Unknown", "director"].str.split(", ").str[0]))
    actors = sorted(set(movies.loc[movies.lead_actor != "Unknown", "lead_actor"].str.split(", ").str[0]))
    with p1:
        concept = st.text_input("Working title", "Untitled film")
        genre = st.selectbox("Primary genre", genres, index=0)
        release_year = st.number_input("Release year", min_value=int(movies.release_year.min()), max_value=int(movies.release_year.max()) + 5, value=int(movies.release_year.max()) + 1)
    with p2:
        budget = st.number_input("Production budget (USD)", min_value=1_000_000, max_value=500_000_000, value=50_000_000, step=5_000_000)
        director = st.selectbox("Director", ["Unknown / new"] + directors)
        actor = st.selectbox("Lead actor", ["Unknown / new"] + actors)
    with p3:
        language = st.selectbox("Original language", sorted(movies.original_language.unique().tolist()))
        marketing = st.number_input("Marketing budget assumption (USD)", min_value=0, max_value=500_000_000, value=25_000_000, step=5_000_000)
        studio_share = st.slider("Studio share of worldwide gross assumption", 0.25, 0.65, 0.45, 0.01)
    chosen_director = "Unknown" if director == "Unknown / new" else director
    chosen_actor = "Unknown" if actor == "Unknown / new" else actor
    pred = _predict(model, budget, release_year, genre, chosen_director, chosen_actor, language)
    band = quality["band"]
    low = max(0, float(np.expm1(np.log1p(pred) - band)))
    high = float(np.expm1(np.log1p(pred) + band))
    recoup = pred * studio_share
    net = recoup - budget - marketing
    low_net = low * studio_share - budget - marketing
    high_net = high * studio_share - budget - marketing
    st.write("")
    st.markdown(f"### Scenario: {concept or 'Untitled film'}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated worldwide gross", _money(pred), f"historical range {_money(low)} – {_money(high)}")
    k2.metric("Estimated studio receipts", _money(recoup), f"at {studio_share:.0%} assumed share")
    k3.metric("Simple net after costs", _money(net), f"budget + marketing: {_money(budget + marketing)}")
    k4.metric("Downside net scenario", _money(low_net), "low gross band; before other costs")
    st.markdown('<div class="warning">Gross is not profit. Net here is a rough scenario: gross × assumed studio share − production budget − entered marketing budget. Financing, distribution fees, participations, taxes, streaming, and ancillary revenue are excluded.</div>', unsafe_allow_html=True)
    st.write("")
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        st.markdown("#### Budget sensitivity")
        budget_grid = np.array([budget * .7, budget, budget * 1.3], dtype=float)
        rows = []
        for bdg in budget_grid:
            gross = _predict(model, bdg, release_year, genre, chosen_director, chosen_actor, language)
            rows.append({"Production budget": bdg, "Estimated gross": gross, "Simple net": gross * studio_share - bdg - marketing})
        st.dataframe(pd.DataFrame(rows).style.format({c: "${:,.0f}" for c in ["Production budget", "Estimated gross", "Simple net"]}), use_container_width=True, hide_index=True)
    with sc2:
        st.markdown("#### What the model has seen")
        peers = movies[movies.primary_genre == genre].copy()
        peers["budget_gap"] = (np.log1p(peers.budget) - np.log1p(budget)).abs()
        peers = peers.sort_values("budget_gap").head(5)
        if len(peers):
            st.dataframe(peers[["title", "release_year", "budget", "revenue"]].style.format({"budget": "${:,.0f}", "revenue": "${:,.0f}"}), use_container_width=True, hide_index=True)
        else:
            st.caption("No close examples in this genre were available.")

with markets:
    st.subheader("Market and risk view")
    st.markdown('<div class="warning">The standard TMDB 5000 file contains worldwide revenue, not a reliable country-by-country box-office split or campaign spend. This app will not fabricate market rankings or a marketing return curve.</div>', unsafe_allow_html=True)
    st.write("To add a real market comparison, upload a separate CSV with columns `title`, `market`, and `revenue` (one row per film-market). The table below then compares the selected concept genre against observed country-level outcomes. The dataset should have a defensible source and consistent currency.")
    market_file = st.file_uploader("Optional regional box-office CSV", type="csv", key="market_csv")
    if market_file:
        try:
            regional = pd.read_csv(market_file)
            regional.columns = [c.strip().lower().replace(" ", "_") for c in regional.columns]
            required = {"title", "market", "revenue"}
            if not required.issubset(regional.columns):
                st.error("Regional file needs title, market, and revenue columns.")
            else:
                regional["revenue"] = pd.to_numeric(regional.revenue, errors="coerce")
                joined = regional.merge(movies[["title", "primary_genre"]], on="title", how="inner")
                result = joined[joined.primary_genre == genre].groupby("market").agg(films=("title", "nunique"), median_revenue=("revenue", "median"), total_revenue=("revenue", "sum")).sort_values("median_revenue", ascending=False)
                st.dataframe(result.style.format({"median_revenue": "${:,.0f}", "total_revenue": "${:,.0f}"}), use_container_width=True)
                if result.empty:
                    st.info("No records matched this genre. Check titles and genre coverage.")
        except Exception as exc:
            st.error(f"Could not read the market file: {exc}")
    st.markdown("#### Marketing and external factors")
    st.write("Campaign effectiveness, competition, reviews, release-date clashes, labor disruptions, and macroeconomic conditions are not reliably present in the standard movie metadata. Treat them as scenario notes or add a sourced dataset before modeling them. Avoid interpreting actor/director correlations as causal impact.")

with methodology:
    st.subheader("Model, validation, and limits")
    st.write("**Model.** A random-forest regressor estimates log worldwide revenue from log production budget, release year, primary genre, director, lead actor, and original language. JSON genre/cast/crew fields are parsed from the common TMDB 5000 format.")
    st.write(f"**Validation.** The latest 20% of release years are held out from training; the displayed median absolute error is evaluated on that later-period set ({quality['test_n']} films, starting {quality['cut_year']}). The estimate band is based on the 80th percentile absolute log error on the holdout. This is a rough historical error band, not a calibrated confidence interval.")
    st.write("**Risks.** The dataset is observational and may have missing or estimated financial records, selection bias, inflation effects, and limited representation of newer distribution models. Cast/director history can be confounded by genre, franchise, studio, and budget. Results are not causal, and this prototype is not investment advice.")
    st.write("**Markets.** Country-level comparisons require separate regional revenue records. Marketing budget is entered by the user and used only in a simple net scenario; marketing lift is not learned by the model.")

st.markdown("---")
st.markdown('<div class="fine">Greenlight Lab • student portfolio prototype • built for transparent exploration, not studio financial decisions.</div>', unsafe_allow_html=True)
