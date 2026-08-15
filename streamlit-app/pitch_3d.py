"""
3D Pitch Trajectory Renderer — DiamondIntel
Reconstructs true pitch flight paths from Trackman's 9-parameter physics fit
(x0,y0,z0 / vx0,vy0,vz0 / ax0,ay0,az0) and renders them in interactive 3D.
"""
import numpy as np
import pandas as pd
import json
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go


PITCH_COLORS = {
    "Four-Seam":"#3b82f6","Sinker":"#22c55e","Cutter":"#8b5cf6",
    "Slider":"#ef4444","Sweeper":"#f97316","Curveball":"#f59e0b",
    "Changeup":"#06b6d4","Splitter":"#ec4899","Other":"#94a3b8",
}

PLATE_Y = 17.0 / 12.0  # front of plate at 1.417 ft


MOUND_DIST = 60.5  # rubber to front of plate

FT_PER_MPH = 1.4667
GRAVITY = -32.174  # ft/s^2

# Trackman's raw 9-parameter physics columns (x0/y0/z0, vx0/vy0/vz0,
# ax0/ay0/az0) aren't populated in this demo's TrackMan export — the columns
# exist but every row is blank. The rest of this module (release point,
# plate location, velocity, and the two break numbers) IS populated, and a
# constant-acceleration flight is exactly what those numbers describe, so we
# back out a physically consistent set of 9-parameter values from them
# instead of doing nothing. The derived path lands on the pitch's own real,
# charted plate location and matches its own real release point/speed —
# only the (unmeasured) drag deceleration along y is an assumed constant.
_TRAJ_SOURCE_COLS = ["RelSpeed", "RelHeight", "RelSide", "Extension",
                     "PlateLocHeight", "PlateLocSide", "InducedVertBreak", "HorzBreak"]


def _derive_traj_row(row):
    try:
        relspeed = float(row["RelSpeed"])
        z0 = float(row["RelHeight"])
        x0 = float(row["RelSide"])
        ext = float(row["Extension"])
        plate_h = float(row["PlateLocHeight"])
        plate_s = float(row["PlateLocSide"])
        ivb = float(row["InducedVertBreak"])
        hb = float(row["HorzBreak"])
    except (TypeError, ValueError):
        return [np.nan] * 9

    y0 = MOUND_DIST - ext
    dist = y0 - PLATE_Y
    v0_fps = relspeed * FT_PER_MPH
    if v0_fps <= 0 or dist <= 0:
        return [np.nan] * 9
    vbar = v0_fps * 0.935  # ball loses ~6.5% of its speed to drag over the flight
    t = dist / vbar

    ay0 = 15.5  # typical Trackman drag deceleration along y, ft/s^2
    vy0 = (PLATE_Y - y0 - 0.5 * ay0 * t * t) / t

    magnus_z = 2.0 * (ivb / 12.0) / (t * t)
    az0 = GRAVITY + magnus_z
    vz0 = (plate_h - z0 - 0.5 * az0 * t * t) / t

    ax0 = 2.0 * (hb / 12.0) / (t * t)
    vx0 = (plate_s - x0 - 0.5 * ax0 * t * t) / t

    return [x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0]


def _with_derived_trajectory(df_all, traj_cols):
    """Return a copy of df_all with traj_cols filled in from the standard
    Trackman columns wherever the raw 9-parameter fit is missing. No-op
    (returns df_all unchanged) if the source columns aren't there either."""
    if not all(c in df_all.columns for c in _TRAJ_SOURCE_COLS):
        return df_all
    have_native = all(c in df_all.columns for c in traj_cols) and \
        df_all[traj_cols].notna().any().any()
    if have_native:
        return df_all
    out = df_all.copy()
    derived = out.apply(_derive_traj_row, axis=1, result_type="expand")
    derived.columns = traj_cols
    for c in traj_cols:
        out[c] = derived[c]
    return out


def _reconstruct(row, n_points=50, extension=6.0):
    """
    Reconstruct a pitch's full 3D path with constant-acceleration physics.

    IMPORTANT: Trackman reports its 9-parameter fit at y=50 ft by convention,
    which is NOT the release point. We back-extrapolate (negative time) to the
    true release at (60.5 - extension) ft so the path starts at the hand.
    Validation: extrapolated release height matches Trackman's RelHeight
    column to within ~0.01 ft.
    """
    x0, y0, z0    = row["x0"], row["y0"], row["z0"]
    vx0, vy0, vz0 = row["vx0"], row["vy0"], row["vz0"]
    ax0, ay0, az0 = row["ax0"], row["ay0"], row["az0"]

    a = 0.5 * ay0
    b = vy0
    if a == 0:
        return None

    # Time at true release (before the y=50 reference → negative t)
    ext = extension if (extension is not None and not np.isnan(extension)) else 6.0
    release_y = MOUND_DIST - ext
    c_rel  = y0 - release_y
    disc_r = b**2 - 4*a*c_rel
    if disc_r < 0:
        t_rel = 0.0
    else:
        t_rel = (-b + np.sqrt(disc_r)) / (2*a)
        if t_rel > 0:
            t_rel = (-b - np.sqrt(disc_r)) / (2*a)

    # Time at front of plate
    c_pl   = y0 - PLATE_Y
    disc_p = b**2 - 4*a*c_pl
    if disc_p < 0:
        return None
    t_plate = (-b - np.sqrt(disc_p)) / (2*a)
    if t_plate <= 0:
        t_plate = (-b + np.sqrt(disc_p)) / (2*a)
    if t_plate <= 0:
        return None

    ts = np.linspace(t_rel, t_plate, n_points)
    xs = x0 + vx0*ts + 0.5*ax0*ts**2
    ys = y0 + vy0*ts + 0.5*ay0*ts**2
    zs = z0 + vz0*ts + 0.5*az0*ts**2
    return xs, ys, zs, t_plate


def _draw_strike_zone(fig, show_commit_point=True):
    """Add strike zone, home plate, and pitcher's mound."""
    # Strike zone box at the plate — TrackMan measured: 1.755–3.378 ft
    fig.add_trace(go.Scatter3d(
        x=[-0.83, 0.83, 0.83, -0.83, -0.83],
        y=[PLATE_Y]*5,
        z=[1.755, 1.755, 3.378, 3.378, 1.755], mode="lines",
        line=dict(color="rgba(255,255,255,0.95)", width=6),
        showlegend=False, hoverinfo="skip",
    ))
    # Inner thirds of the zone
    for zz in (2.296, 2.837):
        fig.add_trace(go.Scatter3d(
            x=[-0.83, 0.83], y=[PLATE_Y, PLATE_Y], z=[zz, zz], mode="lines",
            line=dict(color="rgba(255,255,255,0.3)", width=1),
            showlegend=False, hoverinfo="skip"))
    for xx in (-0.277, 0.277):
        fig.add_trace(go.Scatter3d(
            x=[xx, xx], y=[PLATE_Y, PLATE_Y], z=[1.755, 3.378], mode="lines",
            line=dict(color="rgba(255,255,255,0.3)", width=1),
            showlegend=False, hoverinfo="skip"))

    # Home plate pentagon — point faces the catcher (toward -y), flat edge toward the mound
    pw = 0.708
    fig.add_trace(go.Scatter3d(
        x=[-pw, pw, pw, 0.0, -pw, -pw],
        y=[PLATE_Y+0.7, PLATE_Y+0.7, PLATE_Y, PLATE_Y-0.7, PLATE_Y, PLATE_Y+0.7],
        z=[0]*6, mode="lines",
        line=dict(color="rgba(255,255,255,0.5)", width=3),
        showlegend=False, hoverinfo="skip",
    ))

    # Pitcher's mound — circle centered ~57 ft (real mound is 9ft radius,
    # centered slightly in front of the rubber), raised above field level
    mound_y, mound_r = 57.0, 4.5
    theta = np.linspace(0, 2*np.pi, 40)
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta)*mound_r,
        y=mound_y + np.sin(theta)*mound_r,
        z=[0.35]*len(theta), mode="lines",
        line=dict(color="rgba(196,154,98,0.7)", width=4),
        showlegend=False, hoverinfo="skip",
    ))
    # Pitching rubber — 2ft wide at exactly 60.5 ft, 10in above field
    fig.add_trace(go.Scatter3d(
        x=[-1.0, 1.0],
        y=[MOUND_DIST, MOUND_DIST],
        z=[0.83, 0.83], mode="lines",
        line=dict(color="rgba(255,255,255,0.9)", width=8),
        showlegend=False, hoverinfo="skip",
    ))

    # Commit point — the hitter must decide ~24 ft in front of the plate.
    # Pitches that still look alike here (tunnel) are hardest to read.
    # Optional: shown on the main 3D page, hidden in the umpire app.
    if show_commit_point:
        COMMIT_Y = PLATE_Y + 24.0
        cp_x = [-2.5, 2.5, 2.5, -2.5, -2.5]
        cp_z = [0.0, 0.0, 5.0, 5.0, 0.0]
        fig.add_trace(go.Scatter3d(
            x=cp_x, y=[COMMIT_Y]*5, z=cp_z, mode="lines",
            line=dict(color="rgba(255,210,0,0.7)", width=4),
            name="Commit point (~24 ft)", showlegend=True,
            hovertext="Hitter commit point (~24 ft from plate)", hoverinfo="text",
        ))
        fig.add_trace(go.Scatter3d(
            x=[0], y=[COMMIT_Y], z=[5.4], mode="text",
            text=["Commit point (~24 ft)"],
            textfont=dict(color="rgba(255,210,0,0.95)", size=12),
            showlegend=False, hoverinfo="skip",
        ))



PITCH_COLORS_HEX = {
    "Four-Seam":"#3b82f6","Sinker":"#22c55e","Cutter":"#8b5cf6",
    "Slider":"#ef4444","Sweeper":"#f97316","Curveball":"#f59e0b",
    "Changeup":"#06b6d4","Splitter":"#ec4899","Other":"#94a3b8",
}


def _stadium_html(pitch_data):
    """
    Build the interactive Holman Stadium WebGL-style canvas renderer.
    pitch_data: list of {pt, color, velo, x[], y[], z[]} with reconstructed paths.
    """
    data_json = json.dumps(pitch_data)
    # The renderer is a self-contained HTML/canvas/JS scene.
    html = """
<div style="font-family:system-ui,sans-serif;">
<div id="controls" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;justify-content:center;"></div>
<div style="position:relative;width:100%;height:520px;background:#0a0e1a;border-radius:12px;overflow:hidden;">
  <canvas id="view3d"></canvas>
  <div style="position:absolute;top:8px;left:12px;font-size:11px;color:#fff;font-family:monospace;font-weight:600;text-shadow:0 1px 3px rgba(0,0,0,0.7);">HOLMAN STADIUM &middot; NASHUA, NH</div>
  <div style="position:absolute;bottom:8px;right:12px;font-size:10px;color:rgba(255,255,255,0.7);font-family:monospace;text-shadow:0 1px 2px rgba(0,0,0,0.7);">drag &middot; scroll to zoom</div>
  <div style="position:absolute;bottom:8px;left:12px;display:flex;gap:5px;">
    <button id="zin" style="width:30px;height:30px;border:none;border-radius:6px;background:rgba(255,255,255,0.18);color:#fff;font-size:18px;cursor:pointer;">+</button>
    <button id="zout" style="width:30px;height:30px;border:none;border-radius:6px;background:rgba(255,255,255,0.18);color:#fff;font-size:18px;cursor:pointer;">&minus;</button>
    <button id="zreset" style="height:30px;padding:0 10px;border:none;border-radius:6px;background:rgba(255,255,255,0.18);color:#fff;font-size:11px;cursor:pointer;">reset</button>
  </div>
</div>
<div id="legend" style="display:flex;flex-wrap:wrap;gap:14px;margin:10px 0 0;font-size:12px;justify-content:center;color:#cbd5e1;"></div>
</div>
<script>
const PITCHES=__DATA__;
const active={};PITCHES.forEach(p=>active[p.pt]=true);
const canvas=document.getElementById('view3d');const ctx=canvas.getContext('2d');
let W,H;
function resize(){const r=canvas.parentElement.getBoundingClientRect();W=r.width;H=r.height;canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;canvas.style.width=W+'px';canvas.style.height=H+'px';ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);}
const HAZE=[183,214,235];
function mix(hex,depth){const n=parseInt(hex.slice(1),16);let r=(n>>16)&255,g=(n>>8)&255,b=n&255;const f=Math.max(0,Math.min(0.45,(depth-120)/520));return'rgb('+((r+(HAZE[0]-r)*f)|0)+','+((g+(HAZE[1]-g)*f)|0)+','+((b+(HAZE[2]-b)*f)|0)+')';}
let azim=0.6,elev=0.26,D=148;const D_DEF=148,D_MIN=18,D_MAX=460,NEAR=1.5;const TGT=[0,32,2];
let dragging=false,lastX,lastY,CAM;
function camBasis(){const ce=Math.cos(elev),se=Math.sin(elev),ca=Math.cos(azim),sa=Math.sin(azim);const pos=[TGT[0]+D*ce*sa,TGT[1]-D*ce*ca,TGT[2]+D*se];let f=[TGT[0]-pos[0],TGT[1]-pos[1],TGT[2]-pos[2]];const fl=Math.hypot(f[0],f[1],f[2]);f=[f[0]/fl,f[1]/fl,f[2]/fl];let r=[f[1],-f[0],0];const rl=Math.hypot(r[0],r[1],r[2]);r=[r[0]/rl,r[1]/rl,r[2]/rl];let u=[r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]];return{pos,f,r,u};}
const FOC=()=>Math.min(W,900)*1.2;
function toCam(P){const rel=[P[0]-CAM.pos[0],P[1]-CAM.pos[1],P[2]-CAM.pos[2]];return[rel[0]*CAM.r[0]+rel[1]*CAM.r[1]+rel[2]*CAM.r[2],rel[0]*CAM.u[0]+rel[1]*CAM.u[1]+rel[2]*CAM.u[2],rel[0]*CAM.f[0]+rel[1]*CAM.f[1]+rel[2]*CAM.f[2]];}
function toScreen(c){const f=FOC();return[W/2+f*c[0]/c[2],H/2-f*c[1]/c[2]];}
function clipNear(cam){const out=[];for(let i=0;i<cam.length;i++){const A=cam[i],B=cam[(i+1)%cam.length];const inA=A[2]>=NEAR,inB=B[2]>=NEAR;if(inA)out.push(A);if(inA!==inB){const t=(NEAR-A[2])/(B[2]-A[2]);out.push([A[0]+(B[0]-A[0])*t,A[1]+(B[1]-A[1])*t,NEAR]);}}return out;}
function fillPoly(wp,fill,haze){const cam=wp.map(toCam);const cl=clipNear(cam);if(cl.length<3)return;const sc=cl.map(toScreen);const d=cl.reduce((s,c)=>s+c[2],0)/cl.length;ctx.beginPath();sc.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=haze?mix(fill,d):fill;ctx.fill();}
function strokeSeg(a,b,col,lw){const ca=toCam(a),cb=toCam(b);if(ca[2]<NEAR&&cb[2]<NEAR)return;let A=ca,B=cb;if(ca[2]<NEAR){const t=(NEAR-ca[2])/(cb[2]-ca[2]);A=[ca[0]+(cb[0]-ca[0])*t,ca[1]+(cb[1]-ca[1])*t,NEAR];}if(cb[2]<NEAR){const t=(NEAR-cb[2])/(ca[2]-cb[2]);B=[cb[0]+(ca[0]-cb[0])*t,cb[1]+(ca[1]-cb[1])*t,NEAR];}const sa=toScreen(A),sb=toScreen(B);ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.beginPath();ctx.moveTo(sa[0],sa[1]);ctx.lineTo(sb[0],sb[1]);ctx.stroke();}
function dirtPath(A,B,w,col){const dx=B[0]-A[0],dy=B[1]-A[1];const len=Math.hypot(dx,dy);const px=-dy/len*w,py=dx/len*w;fillPoly([[A[0]+px,A[1]+py,0.02],[B[0]+px,B[1]+py,0.02],[B[0]-px,B[1]-py,0.02],[A[0]-px,A[1]-py,0.02]],col);}
function diskGround(cx,cy,r,col){const pts=[];for(let a=0;a<Math.PI*2;a+=0.25)pts.push([cx+Math.cos(a)*r,cy+Math.sin(a)*r,0.02]);fillPoly(pts,col);}
let LAYER=[];
function pushPoly(wp,fill,stroke,lw,haze){const cam=wp.map(toCam);const cl=clipNear(cam);if(cl.length<3)return;const sc=cl.map(toScreen);const d=cl.reduce((s,c)=>s+c[2],0)/cl.length;LAYER.push({d,fn:()=>{ctx.beginPath();sc.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();if(fill){ctx.fillStyle=haze?mix(fill,d):fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw||1;ctx.stroke();}}});}
function wallDist(ang){const t=Math.abs(ang)/45;return 401*(1-t)+((ang<0?307:315))*t;}
function draw(){
  CAM=camBasis();LAYER=[];ctx.clearRect(0,0,W,H);
  let sky=ctx.createLinearGradient(0,0,0,H);sky.addColorStop(0,'#4a8fd0');sky.addColorStop(0.4,'#8fc0e4');sky.addColorStop(0.66,'#c9e1f1');sky.addColorStop(0.8,'#b7d6eb');sky.addColorStop(0.83,'#2c5a40');sky.addColorStop(1,'#21472f');ctx.fillStyle=sky;ctx.fillRect(0,0,W,H);
  const H0=[0,1.4],B1=[63.64,65.0],B2=[0,128.7],B3=[-63.64,65.0];
  fillPoly([[-300,-140,-0.05],[300,-140,-0.05],[300,200,-0.05],[-300,200,-0.05]],'#2c6329',true);
  for(let ang=-45;ang<45;ang+=2.5){const r1=ang*Math.PI/180,r2=(ang+2.5)*Math.PI/180,d1=wallDist(ang),d2=wallDist(ang+2.5);const base=(Math.floor((ang+45)/2.5)%2===0)?'#4aa043':'#3c8d38';fillPoly([[0,1.4,0],[Math.sin(r1)*d1,Math.cos(r1)*d1,0],[Math.sin(r2)*d2,Math.cos(r2)*d2,0]],base,true);}
  fillPoly([[0,4,0.005],[B1[0]-3,B1[1],0.005],[0,B2[1]-4,0.005],[B3[0]+3,B3[1],0.005]],'#54ad4b',true);
  for(let ang=-45;ang<45;ang+=3){const r1=ang*Math.PI/180,r2=(ang+3)*Math.PI/180,d1=wallDist(ang),d2=wallDist(ang+3);fillPoly([[Math.sin(r1)*(d1-11),Math.cos(r1)*(d1-11),0.01],[Math.sin(r1)*d1,Math.cos(r1)*d1,0.01],[Math.sin(r2)*d2,Math.cos(r2)*d2,0.01],[Math.sin(r2)*(d2-11),Math.cos(r2)*(d2-11),0.01]],'#c49a5e',true);}
  fillPoly([[-70,8,0.008],[70,8,0.008],[95,-30,0.008],[-95,-30,0.008]],'rgba(0,0,0,0.13)');
  const clay='#b06a3c',clay2='#bb7444';
  dirtPath(H0,B1,4,clay);dirtPath(B1,B2,4,clay);dirtPath(B2,B3,4,clay);dirtPath(B3,H0,4,clay);
  diskGround(B1[0],B1[1],12,clay2);diskGround(B2[0],B2[1],12,clay2);diskGround(B3[0],B3[1],12,clay2);
  diskGround(0,2,15,clay2);diskGround(0,59,9,clay);
  strokeSeg([0,1.4,0.05],[Math.sin(-0.7854)*312,Math.cos(-0.7854)*312,0.05],'rgba(255,255,255,0.92)',2);
  strokeSeg([0,1.4,0.05],[Math.sin(0.7854)*318,Math.cos(0.7854)*318,0.05],'rgba(255,255,255,0.92)',2);
  [B1,B2,B3].forEach(b=>fillPoly([[b[0]-1,b[1]-1,0.05],[b[0]+1,b[1]-1,0.05],[b[0]+1,b[1]+1,0.05],[b[0]-1,b[1]+1,0.05]],'#fff'));
  fillPoly([[-0.708,1.0,0.05],[0.708,1.0,0.05],[0.708,1.7,0.05],[0,2.4,0.05],[-0.708,1.7,0.05]],'#fff');
  fillPoly([[-1,60.5,0.4],[1,60.5,0.4],[1,60.9,0.4],[-1,60.9,0.4]],'#fff');
  const wp=[];for(let ang=-45;ang<=45;ang+=3){const rad=ang*Math.PI/180;const d=wallDist(ang);wp.push([Math.sin(rad)*d,Math.cos(rad)*d]);}
  for(let i=0;i<wp.length-1;i++){const a=wp[i],b=wp[i+1];const angA=Math.atan2(a[0],a[1])*180/Math.PI;const isLF=angA<-20;const wh=isLF?4:8;const col=isLF?'#8a4b3a':'#16324c';pushPoly([[a[0],a[1],0],[b[0],b[1],0],[b[0],b[1],wh],[a[0],a[1],wh]],col,'rgba(0,0,0,0.2)',1,true);}
  ['42','39','36'].forEach((n,i)=>{const ang=(-38+i*4)*Math.PI/180,d=307;const cam=toCam([Math.sin(ang)*d,Math.cos(ang)*d,2]);if(cam[2]>NEAR){const s=toScreen(cam);const rr=Math.max(4,FOC()*1.3/cam[2]);LAYER.push({d:cam[2]-0.1,fn:()=>{ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(s[0],s[1],rr,0,7);ctx.fill();ctx.fillStyle='#16324c';ctx.font=rr+'px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(n,s[0],s[1]);ctx.textAlign='left';ctx.textBaseline='alphabetic';}});}});
  [[-1,307],[1,315]].forEach(fp=>{const ang=(fp[0]<0?-45:45)*Math.PI/180,d=fp[1];const x=Math.sin(ang)*d,y=Math.cos(ang)*d;const ct=toCam([x,y,28]),cb=toCam([x,y,0]);if(ct[2]>NEAR){const st=toScreen(ct),sb=toScreen(cb);LAYER.push({d:ct[2],fn:()=>{ctx.strokeStyle='#f3d72b';ctx.lineWidth=Math.max(2,FOC()*0.5/ct[2]);ctx.beginPath();ctx.moveTo(sb[0],sb[1]);ctx.lineTo(st[0],st[1]);ctx.stroke();}});}});
  for(let ang=-44;ang<=44;ang+=4){const rad=ang*Math.PI/180;const d=wallDist(ang)+15;const cam=toCam([Math.sin(rad)*d,Math.cos(rad)*d,24]);if(cam[2]>NEAR){const s=toScreen(cam);const rr=Math.max(5,FOC()*9/cam[2]);LAYER.push({d:cam[2],fn:()=>{ctx.fillStyle=mix('#2c5630',cam[2]);ctx.beginPath();ctx.arc(s[0],s[1],rr,0,7);ctx.fill();}});}}
  const cyc=-4,aS=-82,aE=82,steps=48,rIn=22,rOut=64,zTop=22;
  for(let s=0;s<steps;s++){const a1=(aS+(aE-aS)*s/steps)*Math.PI/180,a2=(aS+(aE-aS)*(s+1)/steps)*Math.PI/180;const ix1=Math.sin(a1)*rIn,iy1=-Math.cos(a1)*rIn+cyc,ox1=Math.sin(a1)*rOut,oy1=-Math.cos(a1)*rOut+cyc;const ix2=Math.sin(a2)*rIn,iy2=-Math.cos(a2)*rIn+cyc,ox2=Math.sin(a2)*rOut,oy2=-Math.cos(a2)*rOut+cyc;pushPoly([[ox1,oy1,0],[ox2,oy2,0],[ox2,oy2,zTop],[ox1,oy1,zTop]],'#1b3656','rgba(0,0,0,0.3)',1);pushPoly([[ix1,iy1,2.5],[ix2,iy2,2.5],[ox2,oy2,zTop],[ox1,oy1,zTop]],'#8a97a6',null,0);pushPoly([[ix1,iy1+2,zTop],[ix2,iy2+2,zTop],[ox2,oy2,zTop+3],[ox1,oy1,zTop+3]],'#243b54','rgba(0,0,0,0.3)',1);}
  for(let s=0;s<=12;s++){const a=(-32+64*s/12)*Math.PI/180,r=19;const x=Math.sin(a)*r,y=-Math.cos(a)*r+cyc;strokeSeg([x,y,0],[x,y,8],'rgba(180,200,220,0.22)',1);}
  [[-150,250],[150,250],[-95,-40],[95,-40]].forEach(t=>{const tx=t[0],ty=t[1];strokeSeg([tx,ty,0],[tx,ty,68],'#39414e',3);const cam=toCam([tx,ty,70]);if(cam[2]>NEAR){const s=toScreen(cam);const w=Math.max(6,FOC()*7/cam[2]),h=Math.max(3,FOC()*3/cam[2]);LAYER.push({d:cam[2],fn:()=>{ctx.fillStyle='#2b3340';ctx.fillRect(s[0]-w/2,s[1]-h/2,w,h);ctx.fillStyle='rgba(255,250,210,0.95)';for(let r=0;r<3;r++)for(let cc=0;cc<5;cc++)ctx.fillRect(s[0]-w/2+cc*w/5+1,s[1]-h/2+r*h/3+1,w/5-1.5,h/3-1.5);}});}});
  LAYER.sort((a,b)=>b.d-a.d);LAYER.forEach(o=>o.fn());
  (function(){const zx=0.83,z1=1.755,z2=3.378,zy=1.42;const corners=[[-zx,zy,z1],[zx,zy,z1],[zx,zy,z2],[-zx,zy,z2]];const sc=corners.map(c=>{const cm=toCam(c);return cm[2]>NEAR?toScreen(cm):null;});if(sc.every(p=>p)){ctx.strokeStyle='rgba(255,255,255,0.9)';ctx.lineWidth=1.6;ctx.beginPath();sc.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.stroke();ctx.strokeStyle='rgba(255,255,255,0.3)';ctx.lineWidth=0.8;for(let k=1;k<3;k++){const xx=-zx+2*zx*k/3;const a=toScreen(toCam([xx,zy,z1])),b=toScreen(toCam([xx,zy,z2]));ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();const zz=z1+(z2-z1)*k/3;const c=toScreen(toCam([-zx,zy,zz])),dd=toScreen(toCam([zx,zy,zz]));ctx.beginPath();ctx.moveTo(c[0],c[1]);ctx.lineTo(dd[0],dd[1]);ctx.stroke();}}})();
  PITCHES.filter(p=>active[p.pt]).forEach(p=>{const pts=[];for(let i=0;i<p.x.length;i++){const c=toCam([p.x[i],p.y[i],p.z[i]]);if(c[2]>NEAR)pts.push(toScreen(c));}if(pts.length<2)return;ctx.strokeStyle='rgba(0,0,0,0.4)';ctx.lineWidth=2.4;ctx.lineJoin='round';ctx.beginPath();pts.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.stroke();ctx.strokeStyle=p.color;ctx.lineWidth=1.5;ctx.beginPath();pts.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.stroke();const e=pts[pts.length-1];ctx.fillStyle=p.color;ctx.beginPath();ctx.arc(e[0],e[1],2,0,7);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=0.8;ctx.stroke();});
}
canvas.addEventListener('mousedown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;});
window.addEventListener('mouseup',()=>dragging=false);
window.addEventListener('mousemove',e=>{if(!dragging)return;azim+=(e.clientX-lastX)*0.01;elev+=(e.clientY-lastY)*0.006;elev=Math.max(0.05,Math.min(1.4,elev));lastX=e.clientX;lastY=e.clientY;draw();});
canvas.addEventListener('wheel',e=>{e.preventDefault();D*=(1+Math.sign(e.deltaY)*0.12);D=Math.max(D_MIN,Math.min(D_MAX,D));draw();},{passive:false});
canvas.addEventListener('touchstart',e=>{if(e.touches.length===1){dragging=true;lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;}},{passive:true});
let pinch=null;
canvas.addEventListener('touchmove',e=>{if(e.touches.length===2){const dx=e.touches[0].clientX-e.touches[1].clientX,dy=e.touches[0].clientY-e.touches[1].clientY,dd=Math.hypot(dx,dy);if(pinch)D*=pinch/dd;pinch=dd;D=Math.max(D_MIN,Math.min(D_MAX,D));draw();return;}pinch=null;if(!dragging)return;azim+=(e.touches[0].clientX-lastX)*0.01;elev+=(e.touches[0].clientY-lastY)*0.006;elev=Math.max(0.05,Math.min(1.4,elev));lastX=e.touches[0].clientX;lastY=e.touches[0].clientY;draw();},{passive:true});
canvas.addEventListener('touchend',()=>{dragging=false;pinch=null;});
document.getElementById('zin').onclick=()=>{D=Math.max(D_MIN,D*0.8);draw();};
document.getElementById('zout').onclick=()=>{D=Math.min(D_MAX,D*1.25);draw();};
document.getElementById('zreset').onclick=()=>{D=D_DEF;azim=0.6;elev=0.26;draw();};
const controls=document.getElementById('controls');
PITCHES.forEach(p=>{const b=document.createElement('button');b.textContent=p.pt;b.style.cssText='border:0.5px solid '+p.color+';color:'+p.color+';background:'+p.color+'22;padding:4px 10px;border-radius:8px;font-size:12px;cursor:pointer;';b.onclick=()=>{active[p.pt]=!active[p.pt];b.style.background=active[p.pt]?p.color+'22':'transparent';b.style.opacity=active[p.pt]?'1':'0.4';draw();};controls.appendChild(b);});
const legend=document.getElementById('legend');
PITCHES.forEach(p=>{const d=document.createElement('span');d.style.cssText='display:flex;align-items:center;gap:6px;';d.innerHTML='<span style="width:10px;height:10px;border-radius:2px;background:'+p.color+';"></span><b style="color:#e2e8f0;font-weight:500;">'+p.pt+'</b> '+p.velo+' mph';legend.appendChild(d);});
resize();draw();window.addEventListener('resize',()=>{resize();draw();});
</script>
"""
    return html.replace("__DATA__", data_json)


def render(df_all, *, player_last, MY_TEAM, team_label=None, show_characteristics=True):
    if team_label is None:
        team_label = lambda x: x
    st.title("3D Pitch Trajectories")
    st.caption(
        "True flight paths reconstructed from Trackman's physics data. "
        "See exactly how each pitch moves from release to the plate."
    )
    st.divider()

    # Check trajectory columns exist
    traj_cols = ["x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0"]
    if not all(c in df_all.columns for c in traj_cols):
        st.error("Trajectory data (x0, vy0, etc.) not found in this dataset.")
        return

    # This TrackMan export has the 9-parameter columns but every row is
    # blank — derive them from release point/plate location/velocity/break
    # instead (see _derive_traj_row above) so this page has something to
    # show instead of "no valid trajectory data" for every pitcher.
    _had_native = df_all[traj_cols].notna().any().any()
    df_all = _with_derived_trajectory(df_all, traj_cols)
    if not _had_native and df_all[traj_cols].notna().any().any():
        st.caption(
            "ℹ️ This dataset's raw 9-parameter TrackMan fit wasn't captured, so these paths are "
            "reconstructed from release point, plate location, velocity, and break instead — same "
            "physics, just solved from the numbers TrackMan did record rather than read directly "
            "off the 9-parameter columns."
        )

    # ── Selectors ──────────────────────────────────────────────────────
    # Teams/players hidden from every picker across the app — kept in sync
    # with EXCLUDED_TEAMS/REMOVED_FROM_ROSTER in app.py (this module doesn't
    # import app.py).
    _EXCLUDED_TEAMS = {"KEE_SWA"}
    _REMOVED_FROM_ROSTER = {"jones", "gopal", "collins", "martorano", "piwnicki"}
    def _is_removed(name):
        return isinstance(name, str) and name.split(",")[0].strip().lower() in _REMOVED_FROM_ROSTER
    all_teams = sorted(t for t in df_all["PitcherTeam"].dropna().unique() if t not in _EXCLUDED_TEAMS)
    sorted_teams = ([MY_TEAM] if MY_TEAM in all_teams else []) + \
                   [t for t in all_teams if t != MY_TEAM]

    c1, c2 = st.columns(2)
    with c1:
        sel_team = st.selectbox("Team", sorted_teams,
            format_func=lambda c: team_label(c), key="p3d_team")
    pitchers = sorted(p for p in df_all[df_all["PitcherTeam"] == sel_team]["Pitcher"].dropna().unique()
                      if not _is_removed(p))
    with c2:
        sel_pitcher = st.selectbox("Pitcher",
            [""] + pitchers,
            format_func=lambda x: player_last(x) if x else "Select pitcher…",
            key="p3d_pitcher")

    if not sel_pitcher:
        st.info("Select a pitcher to render their arsenal in 3D.")
        return

    pp = df_all[df_all["Pitcher"] == sel_pitcher].copy()
    pp = pp[pp[traj_cols].notna().all(axis=1)]
    throws = pp["PitcherThrows"].iloc[0] if len(pp) > 0 else "Right"

    if len(pp) == 0:
        st.warning("No valid trajectory data for this pitcher.")
        return

    # ── Filters ────────────────────────────────────────────────────────
    # Build outing options from this pitcher's games (most recent first)
    def _outing_label(gid):
        # GameID looks like 20260529-LeLacheurPark-1
        parts = str(gid).split("-")
        if len(parts) >= 2 and len(parts[0]) == 8:
            d = parts[0]
            date_str = f"{d[4:6]}/{d[6:8]}/{d[0:4]}"
            venue = parts[1]
            # space out camelcase venue
            import re as _re
            venue = _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", venue)
            return f"{date_str} — {venue}"
        return str(gid)

    outings = sorted(pp["GameID"].dropna().unique().tolist(), reverse=True)
    outing_opts = ["All outings"] + outings

    f1, f2, f3 = st.columns(3)
    with f1:
        avail_types = sorted(pp["PitchType"].dropna().unique().tolist())
        sel_types = st.multiselect("Pitch types",
            options=avail_types, default=avail_types, key="p3d_types")
    with f2:
        hand_filter = st.radio("vs Batter hand", ["All","RHH","LHH"],
                               horizontal=True, key="p3d_hand")
    with f3:
        sel_outing = st.selectbox("Outing", outing_opts,
            format_func=lambda g: "All outings" if g == "All outings" else _outing_label(g),
            key="p3d_outing")

    view = pp.copy()
    if sel_types:
        view = view[view["PitchType"].isin(sel_types)]
    if hand_filter == "RHH":
        view = view[view["BatterSide"] == "Right"]
    elif hand_filter == "LHH":
        view = view[view["BatterSide"] == "Left"]
    if sel_outing != "All outings":
        view = view[view["GameID"] == sel_outing]

    mode = st.radio("Display",
        ["Average path per pitch type", "All individual pitches", "Animated flight", "Stadium View (Holman)"],
        horizontal=True, key="p3d_mode")

    if len(view) == 0:
        st.info("No pitches match the filters.")
        return

    # ── STADIUM VIEW (realistic Holman Stadium) ───────────────────────
    if mode == "Stadium View (Holman)":
        stadium_pitches = []
        for pt in sorted(view["PitchType"].dropna().unique()):
            grp = view[view["PitchType"] == pt]
            avg_row = grp[["x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0"]].mean()
            avg_ext = grp["Extension"].mean() if "Extension" in grp.columns else 6.0
            result = _reconstruct(avg_row, extension=avg_ext)
            if result is None:
                continue
            xs, ys, zs, _ = result
            stadium_pitches.append({
                "pt": pt,
                "color": PITCH_COLORS_HEX.get(pt, "#94a3b8"),
                "velo": round(float(grp["RelSpeed"].mean()), 1),
                "x": [round(float(v), 2) for v in xs],
                "y": [round(float(v), 2) for v in ys],
                "z": [round(float(v), 2) for v in zs],
            })
        if not stadium_pitches:
            st.info("No pitches to render.")
            return
        components.html(_stadium_html(stadium_pitches), height=620, scrolling=False)
        st.caption(
            f"{sel_pitcher} · Throws {throws} · Reconstructed from Trackman physics "
            "· Holman Stadium (LF 307 / CF 401 / RF 315)"
        )
        # Characteristics table still shown below
        if show_characteristics:
            st.divider()
            st.markdown("#### Pitch Characteristics")
            rows = []
            for pt in sorted(view["PitchType"].dropna().unique()):
                grp = view[view["PitchType"] == pt]
                rows.append({
                    "Pitch": pt, "Count": len(grp),
                    "Velo": f"{grp['RelSpeed'].mean():.1f}",
                    "IVB": f"{grp['InducedVertBreak'].mean():+.1f}",
                    "HB": f"{grp['HorzBreak'].mean():+.1f}",
                    "Rel Ht": f"{grp['RelHeight'].mean():.2f}",
                    "Ext": f"{grp['Extension'].mean():.1f}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        return

    # ── Zoom control ───────────────────────────────────────────────────
    vc1, vc2 = st.columns([1.2, 2])
    with vc1:
        viewpoint = st.selectbox("Watch from", 
            ["Catcher (default)", "Batter's box (RHH)", "Batter's box (LHH)",
             "Behind the pitcher", "First base side", "Third base side", "Overhead"],
            key="p3d_viewpoint",
            help="Sets the starting camera. You can always drag the plot to rotate "
                 "anywhere — the view stays where you put it, even while the "
                 "animation plays.")
    with vc2:
        zoom = st.slider("Zoom", min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                     key="p3d_zoom",
                     help="Lower = zoomed in, higher = zoomed out")

    # Camera eye per viewpoint (scaled by zoom). Scene y-axis runs plate(0)→mound(60):
    # positive-y eye looks from behind the MOUND; negative-y from behind the PLATE.
    _CAMS = {
        "Catcher (default)":   dict(eye=dict(x=0,          y=-1.55, z=0.4)),
        "Batter's box (RHH)":  dict(eye=dict(x=-0.6,       y=-1.0,  z=0.3)),
        "Batter's box (LHH)":  dict(eye=dict(x=0.6,        y=-1.0,  z=0.3)),
        "Behind the pitcher":  dict(eye=dict(x=0,          y=1.55,  z=0.4)),
        "First base side":     dict(eye=dict(x=1.5,        y=0.2,   z=0.35)),
        "Third base side":     dict(eye=dict(x=-1.5,       y=0.2,   z=0.35)),
        "Overhead":            dict(eye=dict(x=0,          y=0.15,  z=1.9)),
    }
    _cam_eye = _CAMS.get(viewpoint, _CAMS["Catcher (default)"])["eye"]

    # ── Build 3D figure ────────────────────────────────────────────────
    fig = go.Figure()
    _draw_strike_zone(fig)

    if mode == "Animated flight":
        # ── Animated pitch flight via a custom HTML/JS Plotly component. ──
        # We drive the ball with Plotly.restyle in a requestAnimationFrame loop,
        # which updates ONLY the trace data and never re-applies the scene camera.
        # This is what lets the user rotate to any angle and have it hold during
        # playback (Plotly's built-in frame animation resets the 3D camera).
        import json as _json
        avail_types = sorted(view["PitchType"].dropna().unique())
        if not avail_types:
            st.info("No pitches with trajectory data for this selection.")
            return
        sel_types = st.multiselect(
            "Pitches to animate (pick 1, 2, 3… or all)",
            options=avail_types, default=avail_types[:1], key="p3d_anim_types",
            help="Each selected pitch type animates its average path simultaneously, "
                 "on real relative timing — faster pitches arrive first.")
        if not sel_types:
            st.info("Pick at least one pitch type to animate.")
            return

        N_PTS = 60
        paths = {}
        max_t = 0.0
        for pt in sel_types:
            grp = view[view["PitchType"] == pt]
            avg_row = grp[["x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0"]].mean()
            avg_ext = grp["Extension"].mean() if "Extension" in grp.columns else 6.0
            result = _reconstruct(avg_row, n_points=N_PTS, extension=avg_ext)
            if result is None:
                continue
            xs, ys, zs, t_plate = result
            velo = grp["RelSpeed"].mean()
            paths[pt] = dict(
                xs=[round(float(v), 4) for v in xs],
                ys=[round(float(v), 4) for v in ys],
                zs=[round(float(v), 4) for v in zs],
                t=float(t_plate), velo=float(velo), n=int(len(grp)),
                color=PITCH_COLORS.get(pt, "#94a3b8"), name=pt)
            max_t = max(max_t, float(t_plate))
        if not paths:
            st.info("Couldn't reconstruct trajectories for the selected pitches.")
            return

        # Strike-zone box corners (feet) for drawing in JS
        zone = dict(x0=-0.83, x1=0.83, zb=1.755, zt=3.378, y=float(PLATE_Y))
        cam = dict(x=_cam_eye["x"]*zoom, y=_cam_eye["y"]*zoom, z=_cam_eye["z"]*zoom)
        payload = _json.dumps(dict(paths=list(paths.values()), max_t=max_t,
                                   zone=zone, cam=cam, n_pts=N_PTS))

        _html = """
<div id="wrap" style="background:#0a0e1a;border-radius:8px;padding:4px;">
  <div id="plot" style="width:100%;height:600px;"></div>
  <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;color:#e2e8f0;font-family:sans-serif;">
    <button id="play" style="background:#2563eb;color:#fff;border:none;border-radius:6px;
      padding:6px 16px;font-size:14px;font-weight:700;cursor:pointer;">▶ Play</button>
    <input id="scrub" type="range" min="0" max="1000" value="0" style="flex:1;cursor:pointer;">
    <span id="pct" style="min-width:42px;font-size:13px;">0%</span>
    <label style="font-size:12px;color:#94a3b8;">Speed
      <select id="spd" style="background:#111827;color:#e2e8f0;border:1px solid #334155;border-radius:4px;">
        <option value="0.15">0.15x</option><option value="0.25" selected>0.25x</option>
        <option value="0.5">0.5x</option><option value="1">1x (real)</option>
      </select>
    </label>
  </div>
</div>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const D = __PAYLOAD__;
const plot = document.getElementById('plot');
const traces = [];
// faint full path per pitch
D.paths.forEach(p => traces.push({type:'scatter3d',mode:'lines',x:p.xs,y:p.ys,z:p.zs,
  line:{color:p.color,width:2},opacity:0.25,name:p.name+' ('+p.velo.toFixed(0)+' mph, n='+p.n+')',hoverinfo:'skip'}));
// strike zone box
const z=D.zone;
traces.push({type:'scatter3d',mode:'lines',
  x:[z.x0,z.x1,z.x1,z.x0,z.x0], y:[z.y,z.y,z.y,z.y,z.y], z:[z.zb,z.zb,z.zt,z.zt,z.zb],
  line:{color:'rgba(255,255,255,0.55)',width:3},showlegend:false,hoverinfo:'skip'});
// animated trail + ball per pitch (indices tracked)
const trailIdx=[], ballIdx=[];
D.paths.forEach(p => {
  trailIdx.push(traces.length);
  traces.push({type:'scatter3d',mode:'lines',x:[p.xs[0]],y:[p.ys[0]],z:[p.zs[0]],
    line:{color:p.color,width:7},showlegend:false,hoverinfo:'skip'});
  ballIdx.push(traces.length);
  traces.push({type:'scatter3d',mode:'markers',x:[p.xs[0]],y:[p.ys[0]],z:[p.zs[0]],
    marker:{color:p.color,size:6},showlegend:false,hoverinfo:'skip'});
});
const layout={paper_bgcolor:'#0a0e1a',margin:{l:0,r:0,t:0,b:0},
  showlegend:true,legend:{bgcolor:'rgba(15,23,42,0.92)',bordercolor:'#3b4d6b',borderwidth:1,
    font:{color:'#fff',size:13},x:0.99,y:0.99,xanchor:'right',yanchor:'top'},
  scene:{bgcolor:'#0a0e1a',
    xaxis:{range:[-4,4],showticklabels:false,gridcolor:'#1e2d45',backgroundcolor:'#0a0e1a',title:''},
    yaxis:{range:[-2,63],gridcolor:'#1e2d45',backgroundcolor:'#0a0e1a',title:'Distance (ft)'},
    zaxis:{range:[0,7],gridcolor:'#1e2d45',backgroundcolor:'#0a0e1a',title:'Height (ft)'},
    aspectmode:'manual',aspectratio:{x:0.271,y:2.2,z:0.237},
    camera:{eye:{x:D.cam.x,y:D.cam.y,z:D.cam.z},center:{x:0,y:-0.05,z:-0.03},up:{x:0,y:0,z:1}},
    uirevision:'keep'}};
Plotly.newPlot(plot,traces,layout,{scrollZoom:true,displaylogo:false,responsive:true});

// Render the ball/trail positions at global time T (seconds). Camera untouched.
function renderAt(T){
  const xs=[],ys=[],zs=[],tx=[],ty=[],tz=[];
  D.paths.forEach(p=>{
    const frac=Math.min(T/p.t,1.0);
    const idx=Math.max(0,Math.floor(frac*(D.n_pts-1)));
    tx.push(p.xs.slice(0,idx+1)); ty.push(p.ys.slice(0,idx+1)); tz.push(p.zs.slice(0,idx+1));
    xs.push([p.xs[idx]]); ys.push([p.ys[idx]]); zs.push([p.zs[idx]]);
  });
  // restyle updates ONLY these traces' data — never the camera
  Plotly.restyle(plot,{x:tx,y:ty,z:tz},trailIdx);
  Plotly.restyle(plot,{x:xs,y:ys,z:zs},ballIdx);
}

let playing=false, T=0, last=null;
const btn=document.getElementById('play'), scrub=document.getElementById('scrub'),
      pct=document.getElementById('pct'), spd=document.getElementById('spd');
function setPct(){ const f=T/D.max_t; scrub.value=Math.round(f*1000); pct.textContent=Math.round(f*100)+'%'; }
function loop(ts){
  if(!playing) return;
  if(last===null) last=ts;
  const dt=(ts-last)/1000; last=ts;
  T += dt*parseFloat(spd.value);
  if(T>=D.max_t){ T=D.max_t; playing=false; btn.textContent='▶ Play'; }
  renderAt(T); setPct();
  if(playing) requestAnimationFrame(loop);
}
btn.onclick=()=>{
  if(playing){ playing=false; btn.textContent='▶ Play'; }
  else { if(T>=D.max_t) T=0; playing=true; btn.textContent='⏸ Pause'; last=null; requestAnimationFrame(loop); }
};
scrub.oninput=()=>{ playing=false; btn.textContent='▶ Play'; T=(scrub.value/1000)*D.max_t; renderAt(T); pct.textContent=Math.round((scrub.value/10))+'%'; };
renderAt(0);
</script>
""".replace("__PAYLOAD__", payload)

        components.html(_html, height=680, scrolling=False)
        st.caption("▶ plays the pitches on real relative timing (faster arrives first). Drag the "
                   "slider to scrub. **Rotate to any angle and it stays put during playback.** "
                   "Use the Speed menu — real pitch flight is ~0.4s, so slow it down to study.")
        return   # custom component handles everything; skip the shared Plotly layout below

    elif mode == "Average path per pitch type":
        for pt in sorted(view["PitchType"].dropna().unique()):
            grp = view[view["PitchType"] == pt]
            # Average the trajectory parameters
            avg_row = grp[["x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0"]].mean()
            avg_ext = grp["Extension"].mean() if "Extension" in grp.columns else 6.0
            result = _reconstruct(avg_row, extension=avg_ext)
            if result is None:
                continue
            xs, ys, zs, t_plate = result
            color = PITCH_COLORS.get(pt, "#94a3b8")
            velo = grp["RelSpeed"].mean()
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                name=f"{pt} ({velo:.0f}mph, n={len(grp)})",
                line=dict(color=color, width=6),
                hovertemplate=f"<b>{pt}</b><br>%{{z:.2f}} ft high<extra></extra>",
            ))
            # Mark the end point (at plate)
            fig.add_trace(go.Scatter3d(
                x=[xs[-1]], y=[ys[-1]], z=[zs[-1]], mode="markers",
                marker=dict(color=color, size=5),
                showlegend=False, hoverinfo="skip",
            ))
    else:
        # Individual pitches — cap at 60 for performance
        plot_pitches = view.head(60)
        for pt in sorted(plot_pitches["PitchType"].dropna().unique()):
            grp = plot_pitches[plot_pitches["PitchType"] == pt]
            color = PITCH_COLORS.get(pt, "#94a3b8")
            first = True
            for _, row in grp.iterrows():
                row_ext = row.get("Extension", 6.0)
                result = _reconstruct(row, n_points=30, extension=row_ext)
                if result is None:
                    continue
                xs, ys, zs, _ = result
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    name=pt if first else None,
                    legendgroup=pt,
                    showlegend=first,
                    line=dict(color=color, width=2),
                    opacity=0.5,
                    hoverinfo="skip",
                ))
                first = False

    # TRUE SCALE: aspectmode="data" renders all axes in identical feet —
    # no compression. Camera pulled back so mound, full paths, zone, and
    # plate are all in frame.
    # In animated mode use a CONSTANT uirevision matching the frames, so the
    # user's dragged camera is preserved across the redraws that move the ball.
    _uirev = "p3d_anim_lock" if mode == "Animated flight" else f"p3d_truescale_{zoom}_{viewpoint}"
    fig.update_layout(
        height=650,
        scene=dict(
            xaxis=dict(title="", range=[-4, 4], showgrid=True,
                       gridcolor="#1e2d45", backgroundcolor="#0a0e1a",
                       showticklabels=False),
            yaxis=dict(title="Distance to plate (ft)", range=[-2, 63],
                       gridcolor="#1e2d45", backgroundcolor="#0a0e1a"),
            zaxis=dict(title="Height (ft)", range=[0, 7],
                       gridcolor="#1e2d45", backgroundcolor="#0a0e1a"),
            aspectmode="manual",
            aspectratio=dict(x=0.271, y=2.2, z=0.237),
            camera=dict(eye=dict(x=_cam_eye["x"]*zoom, y=_cam_eye["y"]*zoom,
                                 z=_cam_eye["z"]*zoom),
                        center=dict(x=0, y=-0.05, z=-0.03),
                        up=dict(x=0, y=0, z=1)),
            uirevision=_uirev,
            bgcolor="#0a0e1a",
        ),
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e2e8f0"),
        uirevision=_uirev,
        legend=dict(bgcolor="rgba(15,23,42,0.92)", bordercolor="#3b4d6b",
                    borderwidth=1.5, font=dict(size=16, color="#ffffff"),
                    itemsizing="constant",
                    x=0.99, y=0.99, xanchor="right", yanchor="top"),
        margin=dict(l=0, r=0, t=10, b=0),
    )

    st.plotly_chart(fig, width='stretch', key="p3d_chart", config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": ["zoom3d", "pan3d", "resetCameraDefault3d"],
    })
    st.caption(
        f"{sel_pitcher} · Throws {throws} · {len(view)} pitches · "
        "Drag to rotate · Scroll to zoom · Catcher's view down the mound"
    )

    # ── Movement summary ───────────────────────────────────────────────
    if show_characteristics:
        st.divider()
        st.markdown("#### Pitch Characteristics")
        rows = []
        for pt in sorted(view["PitchType"].dropna().unique()):
            grp = view[view["PitchType"] == pt]
            rows.append({
                "Pitch":   pt,
                "Count":   len(grp),
                "Velo":    f"{grp['RelSpeed'].mean():.1f}",
                "IVB":     f"{grp['InducedVertBreak'].mean():+.1f}",
                "HB":      f"{grp['HorzBreak'].mean():+.1f}",
                "Rel Ht":  f"{grp['RelHeight'].mean():.2f}",
                "Rel Side":f"{grp['RelSide'].mean():+.2f}",
                "Ext":     f"{grp['Extension'].mean():.1f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
