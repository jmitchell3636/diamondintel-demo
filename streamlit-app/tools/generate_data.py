"""Synthetic TrackMan-format pitch-by-pitch data generator — trimmed to
exactly two fictional teams with five players each, for the DiamondIntel demo.
Pure stdlib (no pandas/numpy needed)."""

import math
import csv
import random
import os

random.seed(7)

HEADER = ("PitchNo,Date,Time,PAofInning,PitchofPA,Pitcher,PitcherId,PitcherThrows,PitcherTeam,"
"Batter,BatterId,BatterSide,BatterTeam,PitcherSet,Inning,Top/Bottom,Outs,Balls,Strikes,"
"TaggedPitchType,AutoPitchType,PitchCall,KorBB,TaggedHitType,PlayResult,OutsOnPlay,RunsScored,"
"Notes,RelSpeed,VertRelAngle,HorzRelAngle,SpinRate,SpinAxis,Tilt,RelHeight,RelSide,Extension,"
"VertBreak,InducedVertBreak,HorzBreak,PlateLocHeight,PlateLocSide,ZoneSpeed,VertApprAngle,"
"HorzApprAngle,ZoneTime,ExitSpeed,Angle,Direction,HitSpinRate,PositionAt110X,PositionAt110Y,"
"PositionAt110Z,Distance,LastTrackedDistance,Bearing,HangTime,pfxx,pfxz,x0,y0,z0,vx0,vy0,vz0,"
"ax0,ay0,az0,HomeTeam,AwayTeam,Stadium,Level,League,GameID,PitchUID,EffectiveVelo,MaxHeight,"
"MeasuredDuration,SpeedDrop,PitchLastMeasuredX,PitchLastMeasuredY,PitchLastMeasuredZ,"
"ContactPositionX,ContactPositionY,ContactPositionZ,GameUID,UTCDate,UTCTime,LocalDateTime,"
"UTCDateTime,AutoHitType,System,HomeTeamForeignID,AwayTeamForeignID,GameForeignID,Catcher,"
"CatcherId,CatcherThrows,CatcherTeam,PlayID,PitchTrajectoryXc0,PitchTrajectoryXc1,"
"PitchTrajectoryXc2,PitchTrajectoryYc0,PitchTrajectoryYc1,PitchTrajectoryYc2,"
"PitchTrajectoryZc0,PitchTrajectoryZc1,PitchTrajectoryZc2,HitSpinAxis,HitTrajectoryXc0,"
"HitTrajectoryXc1,HitTrajectoryXc2,HitTrajectoryXc3,HitTrajectoryXc4,HitTrajectoryXc5,"
"HitTrajectoryXc6,HitTrajectoryXc7,HitTrajectoryXc8,HitTrajectoryYc0,HitTrajectoryYc1,"
"HitTrajectoryYc2,HitTrajectoryYc3,HitTrajectoryYc4,HitTrajectoryYc5,HitTrajectoryYc6,"
"HitTrajectoryYc7,HitTrajectoryYc8,HitTrajectoryZc0,HitTrajectoryZc1,HitTrajectoryZc2,"
"HitTrajectoryZc3,HitTrajectoryZc4,HitTrajectoryZc5,HitTrajectoryZc6,HitTrajectoryZc7,"
"HitTrajectoryZc8,ThrowSpeed,PopTime,ExchangeTime,TimeToBase,CatchPositionX,CatchPositionY,"
"CatchPositionZ,ThrowPositionX,ThrowPositionY,ThrowPositionZ,BasePositionX,BasePositionY,"
"BasePositionZ,ThrowTrajectoryXc0,ThrowTrajectoryXc1,ThrowTrajectoryXc2,ThrowTrajectoryYc0,"
"ThrowTrajectoryYc1,ThrowTrajectoryYc2,ThrowTrajectoryZc0,ThrowTrajectoryZc1,ThrowTrajectoryZc2,"
"PitchReleaseConfidence,PitchLocationConfidence,PitchMovementConfidence,HitLaunchConfidence,"
"HitLandingConfidence,CatcherThrowCatchConfidence,CatcherThrowReleaseConfidence,"
"CatcherThrowLocationConfidence").split(",")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")

# ── Teams (exactly two) ───────────────────────────────────────────────
BRK, CON = "BRK_BAN", "CON_RIV"
STADIUM = {BRK: "BanditsBallpark", CON: "RiverCatsField"}

# ── Rosters: exactly 5 players per team. Every player bats; the two
# pitchers on each team also hit (small-roster demo league). ──────────
HITTERS = {
    BRK: [
        ("Boyd, Marcus", "Right", "C"),
        ("Reyes, Julian", "Right", "1B"),
        ("Callahan, Derek", "Left", "OF"),
        ("Brooks, Tyler", "Right", "RHP"),
        ("Frost, Adam", "Left", "RHP"),
    ],
    CON: [
        ("Doyle, Hunter", "Right", "C"),
        ("Sorensen, Blake", "Right", "1B"),
        ("Marsh, Eli", "Left", "OF"),
        ("Delgado, Marcus", "Right", "RHP"),
        ("Dunmore, Chris", "Left", "RHP"),
    ],
}

PITCHERS = {
    BRK: {
        "Brooks, Tyler": ("Right", "SP", {
            "Four-Seam": (0.58, 93.4, 16.8, 7.2, 2250),
            "Slider": (0.26, 83.6, 2.1, -4.8, 2450),
            "Changeup": (0.16, 85.2, 9.4, 11.0, 1750),
        }),
        "Frost, Adam": ("Right", "RP", {
            "Four-Seam": (0.50, 90.5, 13.0, 8.0, 2150),
            "Sinker": (0.25, 90.0, 6.0, 14.0, 2100),
            "Slider": (0.25, 81.0, 1.5, -3.5, 2350),
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
}

CATCHERS = {BRK: ["Boyd, Marcus"], CON: ["Doyle, Hunter"]}

HIT_TYPE_BY_ANGLE = lambda a: ("GroundBall" if a < 10 else
                               "LineDrive" if a < 25 else
                               "FlyBall" if a < 50 else "Popup")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sim_pitch(pitcher_throws, ptype, base):
    usage, velo, ivb, hb, spin = base
    ivb_j = ivb + random.gauss(0, 1.3)
    hb_j = hb + random.gauss(0, 1.3)
    spin_axis = (math.degrees(math.atan2(hb_j, ivb_j)) + 360) % 360
    vert_appr = round(clamp(-6.2 - (ivb_j - 15) / 12 + random.gauss(0, 0.5), -11, -1.5), 1)
    horz_appr = round(clamp(hb_j / 6 + random.gauss(0, 0.5), -6, 6), 1)
    return dict(
        TaggedPitchType=ptype,
        RelSpeed=round(velo + random.gauss(0, 0.9), 1),
        InducedVertBreak=round(ivb_j, 1),
        HorzBreak=round(hb_j, 1),
        SpinRate=round(spin + random.gauss(0, 60)),
        SpinAxis=round(spin_axis, 1),
        VertApprAngle=vert_appr,
        HorzApprAngle=horz_appr,
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
    sx = random.gauss(0, 0.62 if not ahead_count else 0.78)
    sy = random.gauss(2.5, 0.62 if not ahead_count else 0.78)
    return round(clamp(sx, -2.1, 2.1), 2), round(clamp(sy, 0.6, 4.6), 2)


def in_zone(sx, sy):
    return abs(sx) <= 0.83 and 1.755 <= sy <= 3.378


def sim_pa(inning, top_bot, pa_idx, outs_before,
           pitcher_name, pthrows, pmix, batter_name, bside, catcher, rows,
           pitcher_team, batter_team, home_team, away_team, stadium, date, game_id):
    balls = strikes = 0
    pitch_idx = 0
    outcome = None
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

            if outcome == "HR":
                distance = clamp(random.gauss(370, 22), 330, 430)
            elif outcome == "3B":
                distance = clamp(random.gauss(345, 20), 290, 410)
            elif outcome == "2B":
                distance = clamp(random.gauss(310, 25), 220, 385)
            elif angle < 10:
                distance = clamp(random.gauss(110, 35), 20, 220)
            elif angle > 40:
                distance = clamp(random.gauss(110, 35), 30, 220)
            else:
                distance = clamp(random.gauss(230, 55), 80, 340)
            row["Distance"] = round(distance, 1)
            row["LastTrackedDistance"] = row["Distance"]

            pull_mean = -14 if bside == "Right" else 14
            direction = clamp(random.gauss(pull_mean, 22), -45, 45)
            row["Direction"] = round(direction, 1)

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
                   home_pitchers, away_pitchers, innings=7):
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
            ("Bottom", home_team, home_lineup, home_sides, away_team, away_pitchers, away_sp, away_bullpen),
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
                    inning, top_bot, pa_idx, outs,
                    pname, pthrows, pmix, batter, bside, catcher, rows,
                    def_team, batting_team, home_team, away_team, stadium, date, game_id)
                outs += outs_on_play
            if batting_team == home_team:
                home_idx = batter_ptr
            else:
                away_idx = batter_ptr
    return rows


# ── Schedule: every game is Brookhaven vs Concord ─────────────────────
SCHEDULE = [
    ("20260602", BRK, CON), ("20260605", CON, BRK), ("20260608", BRK, CON),
    ("20260611", CON, BRK), ("20260614", BRK, CON), ("20260617", CON, BRK),
]

all_rows = []
for date, home, away in SCHEDULE:
    game_id = f"{date}-{STADIUM[home]}-1"
    rows = simulate_game(game_id, date, home, away, STADIUM[home],
                          PITCHERS[home], PITCHERS[away])
    fname = f"{OUT_DIR}/{game_id}.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(full_row(r, HEADER))
    print(fname, len(rows), "pitches")
    all_rows.extend(rows)

print("TOTAL PITCHES:", len(all_rows))

from collections import Counter
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

print("\nHitter lines (H/AB, BB, K):")
for team, players in HITTERS.items():
    for name, _, _ in players:
        a = ab[name]; hh = h[name]
        avg = hh / a if a else 0
        print(f"  {name:22s} {hh}/{a} = {avg:.3f}   BB={bb[name]} K={k[name]}")
