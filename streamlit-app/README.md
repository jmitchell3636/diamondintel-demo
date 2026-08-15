# DiamondIntel — Live Demo Build

This is the real DiamondIntel app (`app.py`), unmodified except for the
team/player identity — it's pointed at a fictional team (Brookhaven Bandits)
and four fictional opponents instead of real data, so every page is the
literal real app rendering invented players.

`Data/` holds ~1,300 synthetic pitch-by-pitch rows across 6 games, generated
by `tools/generate_data.py` (pure Python, no external services — see that
file for the simulation logic). `Data/roster.csv` has class year/division for
the Returner Board page.

## Deploy

**Streamlit Community Cloud** (easiest, free):
1. https://share.streamlit.io → "New app"
2. Point it at this repo, branch `main`, main file path `streamlit-app/app.py`
3. Deploy — it reads `streamlit-app/requirements.txt` and
   `streamlit-app/.streamlit/config.toml` automatically.

**Render**: create a Web Service from this repo, set the root directory to
`streamlit-app`, build command `pip install -r requirements.txt`, start
command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Known limitation

This was built without a local Python/pandas environment to test against, so
treat the first deploy as a smoke test — if something errors, share the
traceback and it can be fixed from there.
