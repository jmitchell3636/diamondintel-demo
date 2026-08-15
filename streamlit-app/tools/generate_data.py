"""Synthetic TrackMan-format pitch-by-pitch data generator for the
DiamondIntel demo — fictional teams/players, real schema, real app logic.
Pure stdlib (no pandas/numpy available in this sandbox)."""
import csv, random, os

random.seed(7)

HEADER = open("/tmp/demo_app_build/header.csv").read().strip().split(",")

OUT_DIR = "/tmp/demo_app_build/Data"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Teams ──────────────────────────────────────────────────────────
BRK, CON, DOV, POR, MAN = "BRK_BAN", "CON_RIV", "DOV_ANC", "POR_PRI", "MAN_MIL"
STADIUM = {BRK: "BanditsBallpark", CON: "RiverCatsField", DOV: "AnchorPark",
           POR: "HarborField", MAN: "MillYardPark"}

# ── Rosters: name -> (bats/throws, position/role) ────────────────────
# Hitters: name -> (side, pos)
HITTERS = {
    BRK: [
        ("Callahan, Derek", "Right", "RF"), ("Whitfield, Owen", "Right", "CF"),
        ("Reyes, Julian", "Right", "1B"), ("Pike, Jordan", "Left", "LF"),
        ("Boyd, Marcus", "Right", "C"), ("Alvarez, Sam", "Right", "3B"),
        ("Odom, Casey", "Right", "DH"), ("Nakashima, Kevin", "Right", "SS"),
        ("Lang, Trevor", "Left", "2B"),
    ],
    CON: [
        ("Marsh, Eli", "Left", "CF"), ("Holcomb, Trey", "Right", "SS"),
        ("Sorensen, Blake", "Right", "1B"), ("Ferraro, Nico", "Right", "RF"),
        ("Wexler, Grant", "Right", "3B"), ("Voss, Parker", "Left", "LF"),
        ("Doyle, Hunter", "Right", "C"), ("Pratt, Silas", "Right", "2B"),
        ("Vance, Ronnie", "Left", "DH"),
    ],
    DOV: [
        ("Salas, Diego", "Right", "OF"), ("Ruiz, Danny", "Right", "1B"),
        ("Ott, Cameron", "Right", "SS"), ("Whitman, Tanner", "Left", "3B"),
        ("Cho, Daniel", "Right", "C"), ("Beckett, Ryan", "Right", "2B"),
        ("Farro, Miles", "Left", "LF"), ("Guerra, Adan", "Right", "DH"),
        ("Ives, Colton", "Right", "CF"),
    ],
    POR: [
        ("Ferris, Nate", "Right", "P"), ("Duarte, Lucas", "Right", "SS"),
        ("Boone, Charlie", "Left", "1B"), ("Nagy, Simon", "Right", "3B"),
        ("Ramos, Elias", "Right", "CF"), ("Teague, Brody", "Left", "LF"),
        ("Ashby, Marco", "Right", "C"), ("Kessler, Owen", "Right", "2B"),
        ("Villar, Josue", "Right", "RF"),
    ],
    MAN: [
        ("Sato, Reggie", "Right", "P"), ("Krantz, Dylan", "Right", "3B"),
        ("Ojeda, Mateo", "Right", "SS"), ("Sturgis, Ben", "Left", "1B"),
        ("Palmieri, Nico", "Right", "CF"), ("Wu, Ethan", "Left", "LF"),
        ("Damico, Gio", "Right", "C"), ("Reddick, Cole", "Right", "2B"),
        ("Bianchi, Andre", "Right", "RF"),
    ],
}

# Pitchers: name -> (throws, role, mix{pitchtype: (usage, velo, ivb, hb, spin)})
PITCHERS = {
    BRK: {
        "Brooks, Tyler": ("Right", "SP", {
            "Four-Seam": (0.58, 93.4, 16.8, 7.2, 2250),
            "Slider": (0.26, 83.6, 2.1, -4.8, 2450),
            "Changeup": (0.16, 85.2, 9.4, 11.0, 1750),
        }),
        "Bennett, Cole": ("Left", "SP", {
            "Four-Seam": (0.55, 91.2, 14.5, -6.0, 2180),
            "Curveball": (0.25, 76.4, -9.0, -3.0, 2650),
            "Changeup": (0.20, 83.1, 7.5, -9.5, 1700),
        }),
        "Frost, Adam": ("Right", "SP", {
            "Four-Seam": (0.50, 90.5, 13.0, 8.0, 2150),
            "Sinker": (0.25, 90.0, 6.0, 14.0, 2100),
            "Slider": (0.25, 81.0, 1.5, -3.5, 2350),
        }),
        "Delacruz, Ray": ("Right", "RP", {
            "Four-Seam": (0.60, 95.5, 17.5, 8.0, 2350),
            "Slider": (0.40, 85.0, 2.0, -5.5, 2500),
        }),
        "Ito, Mason": ("Left", "RP", {
            "Four-Seam": (0.55, 89.8, 15.0, -7.0, 2100),
            "Changeup": (0.45, 81.5, 8.0, -10.0, 1650),
        }),
        "Sharpe, Devon": ("Right", "RP", {
            "Four-Seam": (0.65, 91.0, 13.5, 7.5, 2080),
            "Curveball": (0.35, 77.0, -8.0, -2.0, 2500),
        }),
    },
    CON: {
        "Delgado, Marcus": ("Right", "SP", {
            "Four-Seam": (0.55, 94.1, 15.5, 7.8, 2280),
            "Curveball": (0.30, 78.3, -8.5, -2.5, 2600),
            "Changeup": (0.15, 84.7, 8.0, 10.5, 1720),
        }),
        "Dunmore, Chris": ("Right", "RP", {
            "Four-Seam": (0.62, 92.0, 14.0, 7.0, 2200),
            "Slider": (0.38, 82.5, 1.8, -4.5, 2400),
        }),
    },
    DOV: {
        "Halstrom, Owen": ("Left", "SP", {
            "Four-Seam": (0.52, 89.6, 13.8, -6.5, 2100),
            "Slider": (0.28, 80.5, 1.2, 3.5, 2300),
            "Changeup": (0.20, 81.8, 7.0, -9.0, 1680),
        }),
    },
    POR: {
        "Ferris, Nate": ("Right", "SP", {
            "Four-Seam": (0.54, 91.8, 14.9, 7.4, 2210),
            "Slider": (0.30, 82.9, 1.9, -4.6, 2420),
            "Changeup": (0.16, 84.0, 8.2, 10.2, 1700),
        }),
    },
    MAN: {
        "Sato, Reggie": ("Right", "SP", {
            "Four-Seam": (0.56, 91.0, 14.2, 7.6, 2190),
            "Curveball": (0.24, 77.6, -8.2, -2.7, 2550),
            "Changeup": (0.20, 83.5, 7.8, 10.0, 1690),
        }),
    },
}

CATCHERS = {BRK: ["Boyd, Marcus", "Trager, Will"], CON: ["Doyle, Hunter"],
            DOV: ["Cho, Daniel"], POR: ["Ashby, Marco"], MAN: ["Damico, Gio"]}

HIT_TYPE_BY_ANGLE = lambda a: ("GroundBall" if a < 10 else
                               "LineDrive" if a < 25 else
                               "FlyBall" if a < 50 else "Popup")

def clamp(x, lo, hi): return max(lo, min(hi, x))

def sim_pitch(pitcher_throws, ptype, base):
    usage, velo, ivb, hb, spin = base
    return dict(
        TaggedPitchType=ptype,
        RelSpeed=round(velo + random.gauss(0, 0.9), 1),
        InducedVertBreak=round(ivb + random.gauss(0, 1.3), 1),
        HorzBreak=round(hb + random.gauss(0, 1.3), 1),
        SpinRate=round(spin + random.gauss(0, 60)),
        RelHeight=round(5.8 + random.gauss(0, 0.15), 2) if pitcher_throws == "Right" else round(5.9 + random.gauss(0, 0.15), 2),
        RelSide=round((1.9 if pitcher_throws == "Right" else -1.9) + random.gauss(0, 0.1), 2),
        Extension=round(6.2 + random.gauss(0, 0.2), 2),
    )

def pick_pitch_type(mix):
    r, cum = random.random(), 0.0
    for pt, base in mix.items():
        cum += base[0]
        if r <= cum:
            return pt, base
    pt = list(mix.keys())[-1]
    return pt, mix[pt]

def sim_location(ahead_count):
    # ahead_count True = pitcher's advantage, tends to work more to edges
    sx = random.gauss(0, 0.62 if not ahead_count else 0.78)
    sy = random.gauss(2.5, 0.62 if not ahead_count else 0.78)
    return round(clamp(sx, -2.1, 2.1), 2), round(clamp(sy, 0.6, 4.6), 2)

def in_zone(sx, sy):
    return abs(sx) <= 0.83 and 1.755 <= sy <= 3.378

def sim_pa(game, half, inning, top_bot, pa_idx, outs_before,
           pitcher_name, pthrows, pmix, batter_name, bside, catcher, rows,
           pitcher_team, batter_team, home_team, away_team, stadium, date, game_id):
    balls = strikes = 0
    pitch_idx = 0
    outcome = None  # "K","BB","HBP","OUT","1B","2B","3B","HR"
    outs_on_play = 0
    runs = 0
    while True:
        pitch_idx += 1
        ptype, base = pick_pitch_type(pmix)
        phys = sim_pitch(pthrows, ptype, base)
        ahead = strikes > balls
        sx, sy = sim_location(ahead)
        zone = in_zone(sx, sy)

        row = dict(
            Date=date, PAofInning=pa_idx, PitchofPA=pitch_idx,
            Pitcher=pitcher_name, PitcherThrows=pthrows, PitcherTeam=pitcher_team,
            Batter=batter_name, BatterSide=bside, BatterTeam=batter_team,
            Inning=inning, **{"Top/Bottom": top_bot}, Outs=outs_before,
            Balls=balls, Strikes=strikes,
            PlateLocSide=sx, PlateLocHeight=sy,
            HomeTeam=home_team, AwayTeam=away_team, Stadium=stadium, GameID=game_id,
            Catcher=catcher, CatcherThrows="Right", CatcherTeam=pitcher_team,
            **phys,
        )
        row["TaggedPitchType"] = phys["TaggedPitchType"]
        row["AutoPitchType"] = phys["TaggedPitchType"]

        # Decide swing
        swing_prob = 0.68 if zone else 0.28
        swings = random.random() < swing_prob

        if not swings:
            if zone:
                row["PitchCall"] = "StrikeCalled"; strikes += 1
            else:
                row["PitchCall"] = "BallCalled"; balls += 1
        else:
            contact_prob = 0.72 if zone else 0.52
            contact = random.random() < contact_prob
            if not contact:
                row["PitchCall"] = "StrikeSwinging"; strikes += 1
            else:
                foul_prob = 0.42
                if random.random() < foul_prob:
                    row["PitchCall"] = "FoulBallNotFieldable"
                    if strikes < 2:
                        strikes += 1
                else:
                    row["PitchCall"] = "InPlay"

        # HBP: rare, only on a ball pitch very close to batter (simplify: small chance each ball)
        if row["PitchCall"] == "BallCalled" and random.random() < 0.012:
            row["PitchCall"] = "HitByPitch"
            outcome = "HBP"

        if strikes >= 3 and row["PitchCall"] in ("StrikeCalled", "StrikeSwinging"):
            row["KorBB"] = "Strikeout"
            outcome = "K"
        elif balls >= 4 and row["PitchCall"] == "BallCalled":
            row["KorBB"] = "Walk"
            outcome = "BB"

        if row["PitchCall"] == "InPlay":
            ev = round(clamp(random.gauss(84, 9), 55, 112), 1)
            angle = round(clamp(random.gauss(14, 16), -25, 65), 1)
            row["ExitSpeed"] = ev
            row["Angle"] = angle
            httype = HIT_TYPE_BY_ANGLE(angle)
            row["TaggedHitType"] = httype
            row["AutoHitType"] = httype
            # outcome from EV/angle quality
            quality = (ev - 70) / 40 + (1 - abs(angle - 16) / 40)
            r = random.random()
            if angle > 15 and ev > 95 and r < 0.30:
                res = "HomeRun"; outcome = "HR"
            elif quality > 0.9 and r < 0.35:
                res = "Triple" if r < 0.05 else "Double"; outcome = "2B" if res == "Double" else "3B"
            elif quality > 0.55 and r < 0.55:
                res = "Single"; outcome = "1B"
            else:
                res = "Out"; outcome = "OUT"
                row["OutsOnPlay"] = 1
                outs_on_play = 1
            row["PlayResult"] = res
            if outcome == "HR":
                runs = 1
            elif outcome in ("1B", "2B", "3B") and random.random() < {"1B": 0.10, "2B": 0.32, "3B": 0.60}[outcome]:
                runs = 1

        rows.append(row)
        if outcome:
            if outcome == "K":
                outs_on_play = 1
            return outcome, outs_on_play, runs

def full_row(partial, header):
    r = {h: "" for h in header}
    r.update(partial)
    return r

def simulate_game(game_id, date, home_team, away_team, stadium,
                   home_pitchers, away_pitchers, home_hitters, away_hitters, innings=7):
    rows = []
    home_lineup = [h[0] for h in HITTERS[home_team]]
    away_lineup = [h[0] for h in HITTERS[away_team]]
    home_sides = {h[0]: h[1] for h in HITTERS[home_team]}
    away_sides = {h[0]: h[1] for h in HITTERS[away_team]}
    home_idx = away_idx = 0

    home_sp = list(home_pitchers.items())[0]
    away_sp = list(away_pitchers.items())[0]
    home_bullpen = list(home_pitchers.items())[1:]
    away_bullpen = list(away_pitchers.items())[1:]

    for inning in range(1, innings + 1):
        for top_bot, batting_team, lineup, sides, def_team, pitcher_pool, sp, bullpen in (
            ("Top", away_team, away_lineup, away_sides, home_team, home_pitchers, home_sp, home_bullpen),
            ("Bottom", home_team, home_lineup, home_sides, away_team, away_pitchers, away_sp, home_bullpen if False else away_bullpen),
        ):
            outs = 0
            pa_idx = 0
            batter_ptr = home_idx if batting_team == home_team else away_idx
            pname, (pthrows, prole, pmix) = sp if inning <= 4 or not bullpen else random.choice(bullpen)
            catcher = CATCHERS[def_team][0]
            while outs < 3:
                pa_idx += 1
                batter = lineup[batter_ptr % len(lineup)]
                batter_ptr += 1
                bside = sides[batter]
                outcome, outs_on_play, runs = sim_pa(
                    None, None, inning, top_bot, pa_idx, outs,
                    pname, pthrows, pmix, batter, bside, catcher, rows,
                    def_team, batting_team, home_team, away_team, stadium, date, game_id)
                outs += outs_on_play
            if batting_team == home_team:
                home_idx = batter_ptr
            else:
                away_idx = batter_ptr
    return rows

# ── Schedule ───────────────────────────────────────────────────────
SCHEDULE = [
    ("20260602", BRK, CON), ("20260605", BRK, DOV), ("20260608", CON, BRK),
    ("20260611", BRK, POR), ("20260614", BRK, MAN), ("20260617", BRK, CON),
]

all_rows = []
gid = 1
for date, home, away in SCHEDULE:
    game_id = f"{date}-{STADIUM[home]}-1"
    rows = simulate_game(game_id, date, home, away, STADIUM[home],
                          PITCHERS[home], PITCHERS[away],
                          HITTERS[home], HITTERS[away])
    fname = f"{OUT_DIR}/{game_id}.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(full_row(r, HEADER))
    print(fname, len(rows), "pitches")
    all_rows.extend(rows)
    gid += 1

print("TOTAL PITCHES:", len(all_rows))

# ── quick sanity tallies (no pandas) ──
from collections import Counter, defaultdict
bf = Counter(); k = Counter(); bb = Counter(); h = Counter(); ab = Counter()
for r in all_rows:
    if r.get("PitchofPA") == 1:
        bf[r["Batter"]] += 1
    if r.get("KorBB") == "Strikeout":
        k[r["Batter"]] += 1
    if r.get("KorBB") == "Walk":
        bb[r["Batter"]] += 1
    pr = r.get("PlayResult")
    if pr in ("Single", "Double", "Triple", "HomeRun"):
        h[r["Batter"]] += 1
    if r.get("PitchCall") == "InPlay" or r.get("KorBB") == "Strikeout":
        ab[r["Batter"]] += 1

print("\nSample hitter lines (H/AB, BB, K):")
for name in ["Callahan, Derek", "Whitfield, Owen", "Sorensen, Blake", "Salas, Diego"]:
    a = ab[name]; hh = h[name]
    avg = hh / a if a else 0
    print(f"  {name:22s} {hh}/{a} = {avg:.3f}   BB={bb[name]} K={k[name]}")
