"""
Opponent Positioning vs Nashua Hitters' Spray — DiamondIntel
Pools all *_playerpositioning_FHC.csv files, pairs the opponent's fielder
alignment with where Nashua hitters actually put the ball in play, and
renders a top-down field to reveal exploitable gaps.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


@st.cache_data(ttl=600, show_spinner=False)
def _load_positioning(data_dir: str):
    """Load and concat all player-positioning files in the data dir."""
    files = list(Path(data_dir).glob("*playerpositioning*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    pos = pd.concat(frames, ignore_index=True)
    if "PitchUID" in pos.columns:
        pos = pos.drop_duplicates(subset="PitchUID")
    return pos


def _landing(row):
    """Hit landing (side, dist) from Distance + Direction (deg, +=1B/right)."""
    d = row.get("Distance", np.nan)
    ang = row.get("Direction", np.nan)
    if pd.isna(d) or pd.isna(ang):
        return None, None
    rad = np.radians(ang)
    return float(d * np.sin(rad)), float(d * np.cos(rad))



# Standard defensive alignment (feet, same side/dist coordinate system as the
# canvas below: side is + toward 1B/right, dist is toward CF) — used as the
# fielder overlay when no *_playerpositioning_FHC.csv files are on file yet.
# Depths are typical college-summer-league straight-away starting spots, not
# shaded to any one hitter, since we don't have a real per-play alignment to
# show.
_STANDARD_ALIGNMENT = [
    {"p": "1B", "side": 46.0,  "dist": 66.0},
    {"p": "2B", "side": 26.0,  "dist": 130.0},
    {"p": "SS", "side": -24.0, "dist": 145.0},
    {"p": "3B", "side": -55.0, "dist": 68.0},
    {"p": "LF", "side": -140.0, "dist": 290.0},
    {"p": "CF", "side": 5.0,   "dist": 330.0},
    {"p": "RF", "side": 150.0, "dist": 290.0},
]


def render(df_all, *, MY_TEAM, player_last, DATA_DIR):
    st.title("Opponent Positioning vs Our Spray")
    st.caption(
        "Where our hitters put the ball in play, overlaid with how the "
        "opposing defense actually aligned. Green = hits, gray = outs. "
        "Find the gaps in their alignment."
    )

    pos = _load_positioning(str(DATA_DIR))

    # Our balls in play — this comes straight from the game CSVs, so it's
    # real regardless of whether positioning files exist.
    bip = df_all[(df_all["BatterTeam"] == MY_TEAM) &
                 (df_all["PitchCall"] == "InPlay")].copy()
    if bip.empty:
        st.info("No balls in play found for our hitters yet.")
        return

    using_standard_alignment = pos.empty
    if using_standard_alignment:
        st.info(
            "No `*_playerpositioning_FHC.csv` files on file yet, so the fielder "
            "overlay below is a standard straight-away alignment rather than the "
            "opponent's actual tracked positioning for these plays. Add those "
            "TrackMan files alongside the game CSVs to replace it with the real "
            "thing.",
            icon="ℹ️",
        )
        merged = bip
    else:
        merged = bip.merge(
            pos[pos["BatterTeam"] == MY_TEAM],
            on="PitchUID", suffixes=("", "_pos"), how="inner"
        )
        if merged.empty:
            st.info("No overlap between our balls in play and positioning files yet.")
            return

    # ── Hitter filter (per-hitter, sorted by sample size) ──────────────
    counts = merged["Batter"].value_counts()
    hitters = counts.index.tolist()  # most balls in play first
    c1, c2 = st.columns([2, 1])
    with c1:
        sel = st.selectbox(
            "Hitter",
            hitters,
            format_func=lambda x: f"{player_last(x)}  ({counts[x]} BIP)",
            key="pos_hitter",
        )
    with c2:
        hand = st.radio("vs Pitcher", ["All", "RHP", "LHP"],
                        horizontal=True, key="pos_hand")

    view = merged[merged["Batter"] == sel].copy()
    if hand == "RHP":
        view = view[view["PitcherThrows"] == "Right"]
    elif hand == "LHP":
        view = view[view["PitcherThrows"] == "Left"]

    if view.empty:
        st.info("No balls in play match this filter.")
        return

    # ── Spray points ───────────────────────────────────────────────────
    sprays = []
    for _, r in view.iterrows():
        sx, sy = _landing(r)
        if sx is None:
            continue
        ev_raw = r.get("ExitSpeed")
        ev = 0.0 if pd.isna(ev_raw) else float(ev_raw)
        sprays.append({
            "x": round(sx, 1), "y": round(sy, 1),
            "res": str(r.get("PlayResult", "")),
            "ev": round(ev),
        })

    # ── Average opponent alignment over this filter ────────────────────
    if using_standard_alignment:
        fielders = list(_STANDARD_ALIGNMENT)
    else:
        fielders = []
        for p in ["1B", "2B", "3B", "SS", "LF", "CF", "RF"]:
            zx, dx = f"{p}_PositionAtReleaseZ", f"{p}_PositionAtReleaseX"
            if zx in view.columns and view[zx].notna().any():
                fielders.append({
                    "p": p,
                    "side": round(float(view[zx].mean()), 1),
                    "dist": round(float(view[dx].mean()), 1),
                })

    if not sprays or not fielders:
        st.info("Not enough data to render the field.")
        return

    # ── Summary metrics ────────────────────────────────────────────────
    n_bip = len(sprays)
    n_hits = sum(1 for s in sprays if s["res"] not in ("Out", "FieldersChoice", "Sacrifice"))
    st.markdown(f"### {player_last(sel)}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Balls in play", n_bip)
    m2.metric("Hits", n_hits)
    m3.metric("BABIP-ish", f"{n_hits / n_bip:.3f}" if n_bip else "—")

    data = json.dumps({"sprays": sprays, "fielders": fielders})
    components.html(_field_html(data), height=470, scrolling=False)


def _field_html(data_json):
    html = """
<div style="font-family:system-ui,sans-serif;">
<div style="position:relative;width:100%;height:460px;background:#0b1a10;border-radius:12px;overflow:hidden;">
  <canvas id="posfield"></canvas>
</div>
<div style="margin-top:8px;font-size:12px;color:#94a3b8;display:flex;gap:16px;flex-wrap:wrap;justify-content:center;">
  <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#ffffff;border:1px solid #888;"></span> hit</span>
  <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#ef4444;"></span> out</span>
  <span><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#f59e0b;"></span> home run</span>
  <span><span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:#1e3a8a;border:1px solid #dbe4ee;"></span> avg fielder</span>
</div>
</div>
<script>
const D=__DATA__;
const canvas=document.getElementById('posfield');const ctx=canvas.getContext('2d');
let W,H;
function resize(){const r=canvas.parentElement.getBoundingClientRect();W=r.width;H=r.height;canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;canvas.style.width=W+'px';canvas.style.height=H+'px';ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);}
resize();
// side (x, +right/1B), dist (y, toward CF). Home near bottom.
function P(side,dist){const cx=W/2,cy=H-30,sc=Math.min(W*0.94,H*1.02)/360;return [cx+side*sc, cy-dist*sc];}
const SC=()=>Math.min(W*0.94,H*1.02)/360;

// Bases (90 ft)
const HOME=[0,0], FIRST=[63.64,63.64], SECOND=[0,127.28], THIRD=[-63.64,63.64];

function arcPts(cx,cy,r,a0,a1,step){const out=[];for(let a=a0;a<=a1+0.001;a+=step){out.push([cx+Math.cos(a)*r, cy+Math.sin(a)*r]);}return out;}

function poly(pts,fill,stroke,lw){ctx.beginPath();pts.forEach((p,i)=>{const q=P(p[0],p[1]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw||1;ctx.stroke();}}

function draw(){
  ctx.clearRect(0,0,W,H);
  // ===== GRASS BASE =====
  ctx.fillStyle='#2e7d39';ctx.fillRect(0,0,W,H);

  // ===== FAIR-TERRITORY GRASS with mowing stripes =====
  // (arc fan from home out to a generic 330ft wall)
  const WALL=330;
  for(let a=-45;a<45;a+=3){
    const r1=a*Math.PI/180,r2=(a+3)*Math.PI/180;
    const stripe=(Math.floor((a+45)/3)%2===0);
    ctx.fillStyle=stripe?'#37923f':'#2f8537';
    ctx.beginPath();const h=P(0,0);ctx.moveTo(h[0],h[1]);
    const p1=P(Math.sin(r1)*WALL,Math.cos(r1)*WALL),p2=P(Math.sin(r2)*WALL,Math.cos(r2)*WALL);
    ctx.lineTo(p1[0],p1[1]);ctx.lineTo(p2[0],p2[1]);ctx.closePath();ctx.fill();
  }

  // ===== OUTFIELD WARNING TRACK =====
  ctx.fillStyle='#c79a5b';
  for(let a=-45;a<45;a+=3){const r1=a*Math.PI/180,r2=(a+3)*Math.PI/180;
    poly([[Math.sin(r1)*(WALL-12),Math.cos(r1)*(WALL-12)],[Math.sin(r1)*WALL,Math.cos(r1)*WALL],[Math.sin(r2)*WALL,Math.cos(r2)*WALL],[Math.sin(r2)*(WALL-12),Math.cos(r2)*(WALL-12)]],'#c79a5b');}

  // ===== INFIELD DIRT (skin) — outer edge is a 95 ft radius arc from the mound =====
  // Mound center 59 ft from home; for a ray from home at angle a, solve distance t
  // to the 95-ft circle around the mound: t^2 - 2*59*cos(a)*t + (59^2-95^2)=0
  ctx.beginPath();const hp0=P(0,2);ctx.moveTo(hp0[0],hp0[1]);
  for(let a=-58;a<=58;a+=2){
    const rad=a*Math.PI/180, ca=Math.cos(rad);
    const t=(2*59*ca + Math.sqrt((2*59*ca)*(2*59*ca) - 4*(59*59-95*95)))/2;
    const q=P(Math.sin(rad)*t, ca*t);
    ctx.lineTo(q[0],q[1]);
  }
  ctx.closePath();ctx.fillStyle='#b06a3c';ctx.fill();

  // ===== INFIELD GRASS — diamond inside the basepaths =====
  poly([[0,20],[60,72],[0,126],[-60,72]],'#3a9a40');

  // ===== BASE PATHS (dirt lanes) =====
  function lane(a,b){const dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy),px=-dy/L*4.5,py=dx/L*4.5;
    poly([[a[0]+px,a[1]+py],[b[0]+px,b[1]+py],[b[0]-px,b[1]-py],[a[0]-px,a[1]-py]],'#b06a3c');}
  lane(HOME,FIRST);lane(FIRST,SECOND);lane(SECOND,THIRD);lane(THIRD,HOME);

  // ===== DIRT CUTOUTS around each base =====
  function cutout(b,r){ctx.fillStyle='#b06a3c';ctx.beginPath();const pts=arcPts(b[0],b[1],r,0,Math.PI*2,0.3);pts.forEach((pt,i)=>{const q=P(pt[0],pt[1]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.closePath();ctx.fill();}
  cutout(FIRST,11);cutout(SECOND,11);cutout(THIRD,11);

  // ===== PITCHER'S MOUND =====
  ctx.fillStyle='#bb7a45';ctx.beginPath();const md=arcPts(0,60.5,9,0,Math.PI*2,0.3);md.forEach((pt,i)=>{const q=P(pt[0],pt[1]);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.closePath();ctx.fill();
  // rubber
  const rb=P(0,60.5);ctx.fillStyle='#f5f5f5';ctx.fillRect(rb[0]-3,rb[1]-1.5,6,3);

  // ===== HOME PLATE dirt circle + boxes =====
  cutout(HOME,13);
  // batter's boxes
  ctx.strokeStyle='rgba(255,255,255,0.55)';ctx.lineWidth=1;
  [-1,1].forEach(s=>{const bx=[[s*2.5,-2.5],[s*6,-2.5],[s*6,5.5],[s*2.5,5.5]];poly(bx,null,'rgba(255,255,255,0.5)',1);});

  // ===== FOUL LINES + BASE LINES (chalk) =====
  ctx.strokeStyle='#ffffff';ctx.lineWidth=2;
  let fl=P(Math.sin(-0.785)*WALL,Math.cos(-0.785)*WALL),fr=P(Math.sin(0.785)*WALL,Math.cos(0.785)*WALL),hpp=P(0,0);
  ctx.beginPath();ctx.moveTo(hpp[0],hpp[1]);ctx.lineTo(fl[0],fl[1]);ctx.stroke();
  ctx.beginPath();ctx.moveTo(hpp[0],hpp[1]);ctx.lineTo(fr[0],fr[1]);ctx.stroke();

  // ===== BASES (white) =====
  [FIRST,SECOND,THIRD].forEach(b=>{const q=P(b[0],b[1]);ctx.save();ctx.translate(q[0],q[1]);ctx.rotate(Math.PI/4);ctx.fillStyle='#fff';ctx.fillRect(-4,-4,8,8);ctx.restore();});
  // home plate (pentagon, point toward catcher)
  const hpts=[[-2,0],[2,0],[2,-2],[0,-4],[-2,-2]];
  poly(hpts,'#ffffff',null,0);

  // ===== OUTFIELD WALL =====
  ctx.strokeStyle='#13324c';ctx.lineWidth=4;ctx.beginPath();
  for(let a=-45;a<=45;a+=2){const rad=a*Math.PI/180;const q=P(Math.sin(rad)*WALL,Math.cos(rad)*WALL);a===-45?ctx.moveTo(q[0],q[1]):ctx.lineTo(q[0],q[1]);}
  ctx.stroke();

  // ===== FIELDER COVERAGE ZONES =====
  D.fielders.forEach(f=>{const p=P(f.side,f.dist);ctx.fillStyle='rgba(59,130,246,0.10)';ctx.beginPath();ctx.arc(p[0],p[1],SC()*52,0,7);ctx.fill();});

  // ===== SPRAY POINTS =====
  D.sprays.forEach(s=>{const p=P(s.x,s.y);const hit=!(s.res==="Out"||s.res==="FieldersChoice"||s.res==="Sacrifice");
    ctx.fillStyle=s.res==="HomeRun"?"#f59e0b":hit?"#ffffff":"#ef4444";
    ctx.beginPath();ctx.arc(p[0],p[1],hit?5:4,0,7);ctx.fill();
    ctx.strokeStyle='rgba(0,0,0,0.45)';ctx.lineWidth=1;ctx.stroke();});

  // ===== FIELDERS on top =====
  D.fielders.forEach(f=>{const p=P(f.side,f.dist);
    ctx.fillStyle='#1e3a8a';ctx.strokeStyle='#dbe4ee';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(p[0],p[1],8,0,7);ctx.fill();ctx.stroke();
    ctx.fillStyle='#fff';ctx.font='bold 8px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(f.p,p[0],p[1]);});
  ctx.textAlign='left';ctx.textBaseline='alphabetic';
}
draw();window.addEventListener('resize',()=>{resize();draw();});
</script>
"""
    return html.replace("__DATA__", data_json)
