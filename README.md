# Gamers

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Neon Racer — Car Dodger (Upgraded)</title>
  <style>
    :root{
      --bg:#05060a; --panel:#0b0f16; --neon1:#00f0ff; --neon2:#ff4dd2; --accent:#7cff6b; --muted:#9aa3b2; --glass: rgba(255,255,255,0.04);
      --road:#15181c; --lane:#1f2328; --white: #e6eef7;
      --ui-blur: 8px;
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial;color:var(--white);background:linear-gradient(180deg,#03040a 0,#07111a 100%);display:flex;align-items:center;justify-content:center;padding:18px}

    /* App container */
    .container{width:980px;max-width:98vw;display:grid;grid-template-columns:1fr 320px;gap:18px}

    /* Game Card */
    .game-card{position:relative;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.08));border-radius:14px;padding:14px;overflow:hidden;border:1px solid rgba(255,255,255,0.04);box-shadow:0 8px 30px rgba(0,0,0,0.6)}
    .canvas-wrap{position:relative;border-radius:10px;overflow:hidden;background:radial-gradient(1200px 300px at 50% 0%, rgba(0,240,255,0.02), transparent), var(--road);padding:10px}
    canvas{display:block;width:100%;height:auto;background:linear-gradient(#121417,#0c0e11)}

    /* Futuristic HUD panel */
    .hud{position:absolute;left:20px;top:20px;display:flex;gap:12px;align-items:center;z-index:30}
    .hud .panel{backdrop-filter: blur(var(--ui-blur));background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));border-radius:10px;padding:8px 12px;border:1px solid rgba(255,255,255,0.04);min-width:110px}
    .hud .title{font-size:12px;color:var(--muted);display:block}
    .hud .value{font-weight:700;font-size:16px;color:var(--white)}

    /* Right side UI */
    .sidebar{background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,0.03);height:100%}
    .sidebar h3{margin:0 0 8px 0;font-size:13px;color:var(--muted)}
    .stats{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    .stat-card{background:var(--glass);padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,0.02)}
    .stat-card .num{font-size:20px;font-weight:800}
    .controls{display:flex;gap:8px;margin-top:12px}
    .btn{flex:1;padding:10px;border-radius:10px;border:0;cursor:pointer;font-weight:700}
    .btn.start{background:linear-gradient(90deg,var(--neon1),var(--neon2));color:#061018}
    .btn.ghost{background:transparent;border:1px solid rgba(255,255,255,0.04);color:var(--white)}

    /* Overlay screens */
    .overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none}
    .overlay .card{pointer-events:auto;background:rgba(2,6,12,0.8);border-radius:12px;padding:18px 22px;text-align:center;border:1px solid rgba(255,255,255,0.04);backdrop-filter:blur(6px)}
    .overlay h2{margin:0 0 8px 0;font-size:22px}
    .overlay p{margin:0 0 12px 0;color:var(--muted)}

    /* Joystick */
    .joystick-wrap{position:absolute;left:16px;bottom:16px;z-index:40}
    .joystick{width:120px;height:120px;border-radius:999px;background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.1));border:1px solid rgba(255,255,255,0.03);display:flex;align-items:center;justify-content:center;touch-action:none}
    .stick{width:56px;height:56px;border-radius:999px;background:linear-gradient(180deg, rgba(255,255,255,0.04), rgba(0,0,0,0.12));box-shadow:0 6px 18px rgba(0,0,0,0.6)}

    /* small screens adjustments */
    @media (max-width:980px){.container{grid-template-columns:1fr;}.sidebar{order:2}}

    /* extra UI flourishes */
    .meter{height:8px;border-radius:999px;background:linear-gradient(90deg, rgba(255,255,255,0.03), rgba(0,0,0,0.05));overflow:hidden}
    .meter .fill{height:100%;width:20%;background:linear-gradient(90deg,var(--neon2),var(--neon1));}

    .footer-note{font-size:12px;color:var(--muted);margin-top:10px}
  </style>
</head>
<body>
  <div class="container">
    <div class="game-card" id="gameCard">
      <div class="canvas-wrap">
        <canvas id="gameCanvas" width="720" height="1200"></canvas>

        <!-- HUD -->
        <div class="hud">
          <div class="panel">
            <span class="title">SCORE</span>
            <div class="value" id="uiScore">0</div>
          </div>
          <div class="panel">
            <span class="title">HIGH</span>
            <div class="value" id="uiHigh">0</div>
          </div>
          <div class="panel">
            <span class="title">MULTI</span>
            <div class="value" id="uiMulti">x1</div>
          </div>
        </div>

        <!-- joystick for mobile -->
        <div class="joystick-wrap" id="joystickWrap">
          <div class="joystick" id="joystick"><div class="stick" id="stick"></div></div>
        </div>

        <!-- overlays -->
        <div class="overlay" id="startOverlay">
          <div class="card">
            <h2>NEON RACER</h2>
            <p>Drive the yellow prototype. Dodge traffic, build combos, and reach new highs.</p>
            <div style="display:flex;gap:8px;">
              <button class="btn start" id="startBtn">START</button>
              <button class="btn ghost" id="tutorialBtn">HOW TO PLAY</button>
            </div>
          </div>
        </div>

        <div class="overlay" id="gameOverOverlay" style="display:none">
          <div class="card">
            <h2 id="overTitle">GAME OVER</h2>
            <p id="overSub">You crashed.</p>
            <div style="display:flex;gap:8px;justify-content:center;margin-top:12px">
              <button class="btn start" id="retryBtn">RETRY</button>
              <button class="btn ghost" id="shareBtn">SAVE & SHARE</button>
            </div>
          </div>
        </div>

      </div>
    </div>

    <aside class="sidebar">
      <h3>RACER HUD</h3>
      <div class="stats">
        <div class="stat-card">
          <div class="num" id="statSpeed">0</div>
          <div class="muted">Speed</div>
        </div>
        <div class="stat-card">
          <div class="num" id="statTime">0s</div>
          <div class="muted">Time</div>
        </div>
        <div class="stat-card">
          <div class="num" id="statCombo">0</div>
          <div class="muted">Combo</div>
        </div>
        <div class="stat-card">
          <div class="num" id="statDodged">0</div>
          <div class="muted">Dodged</div>
        </div>
      </div>

      <div class="controls">
        <button class="btn start" id="uiStart">Start</button>
        <button class="btn ghost" id="uiPause">Pause</button>
      </div>

      <div style="margin-top:12px">
        <div class="footer-note">Controls: ← → keys / WASD / Mobile joystick. Avoid obstacles. Score increases per dodge.</div>
      </div>
    </aside>
  </div>

  <script>
    // Game variables
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    // UI elements
    const uiScore = document.getElementById('uiScore');
    const uiHigh = document.getElementById('uiHigh');
    const uiMulti = document.getElementById('uiMulti');
    const statSpeed = document.getElementById('statSpeed');
    const statTime = document.getElementById('statTime');
    const statCombo = document.getElementById('statCombo');
    const statDodged = document.getElementById('statDodged');

    const startOverlay = document.getElementById('startOverlay');
    const gameOverOverlay = document.getElementById('gameOverOverlay');
    const startBtn = document.getElementById('startBtn');
    const retryBtn = document.getElementById('retryBtn');
    const uiStart = document.getElementById('uiStart');
    const uiPause = document.getElementById('uiPause');

    // Joystick elements
    const joystick = document.getElementById('joystick');
    const stick = document.getElementById('stick');
    const joystickWrap = document.getElementById('joystickWrap');

    // Game state
    let running = false;
    let paused = false;
    let frameId = null;

    // Player
    const player = { w:70, h:120, x: W/2 - 35, y: H - 200, speed: 8, color: '#ffcc00' };

    // Road
    const road = { x:120, w:W-240, y:0, h:H, lanes:3 };

    // Obstacles
    let obstacles = [];
    let spawnTimer = 0;
    let spawnInterval = 80; // frames

    // Particles
    let particles = [];

    // Score and metrics
    let score = 0; let high = 0; let timeStart = 0; let elapsed = 0; let dodged = 0;
    let combo = 0; let bestCombo = 0; let multiplier = 1;

    // Input
    const keys = {};
    let joystickDir = {x:0,y:0};

    // Load highscore
    try{ high = parseInt(localStorage.getItem('neon_high')) || 0 }catch(e){ high = 0 }
    uiHigh.textContent = high;

    // Utility functions
    function rand(min,max){return Math.random()*(max-min)+min}
    function clamp(v,a,b){return Math.max(a,Math.min(b,v))}

    // Obstacle types
    const types = [
      {name:'car', w:60,h:100, color:'#e74c3c', score:10, speed:0},
      {name:'truck', w:100,h:140, color:'#ff8c42', score:18, speed:-0.4},
      {name:'bike', w:36,h:64, color:'#7bc8ff', score:6, speed:0.6}
    ];

    function spawnObstacle(){
      const laneW = road.w / road.lanes;
      const laneIndex = Math.floor(rand(0, road.lanes));
      const t = types[Math.floor(rand(0, types.length))];
      const w = t.w * (0.9 + rand(-0.1,0.15));
      const h = t.h * (0.9 + rand(-0.05,0.1));
      const x = road.x + laneIndex*laneW + (laneW - w)/2;
      const y = -h - rand(20,200);
      const baseSpeed = 3 + rand(0,2) + elapsed*0.01; // scale with time
      const lateral = rand(-0.5,0.5); // lane drift
      obstacles.push({x,y,w,h,baseSpeed,lateral,color:t.color,scoreVal:t.score});
    }

    function reset(){
      obstacles = []; particles = []; score=0; dodged=0; combo=0; multiplier=1; elapsed=0; spawnInterval=80; timeStart = performance.now();
      player.x = W/2 - player.w/2;
      uiScore.textContent = '0'; uiMulti.textContent='x1'; statDodged.textContent='0'; statCombo.textContent='0'; statTime.textContent='0s'; statSpeed.textContent='0';
    }

    function start(){
      reset(); running=true; paused=false; startOverlay.style.display='none'; gameOverOverlay.style.display='none'; timeStart = performance.now(); frameId = requestAnimationFrame(loop);
    }

    function endGame(){
      running=false; cancelAnimationFrame(frameId); gameOverOverlay.style.display='flex';
      document.getElementById('overTitle').textContent = 'CRASHED';
      document.getElementById('overSub').textContent = `Score: ${score} • Time: ${Math.floor(elapsed)}s • Best combo: ${bestCombo}`;
      // save high
      if(score>high){ high = score; try{ localStorage.setItem('neon_high', high) }catch(e){} uiHigh.textContent = high }
    }

    // collision box
    function rectsOverlap(a,b){
      return !(a.x + a.w < b.x || b.x + b.w < a.x || a.y + a.h < b.y || b.y + b.h < a.y);
    }

    // particles
    function spawnParticle(x,y,color){
      particles.push({x,y,vx:rand(-1.5,1.5),vy:rand(-3,-0.5),life:60,alpha:1,color});
    }

    function update(){
      if(!running || paused) return;
      const now = performance.now(); elapsed = (now - timeStart)/1000;

      // input movement
      let move = 0;
      if(keys['ArrowLeft']||keys['a']) move = -1;
      if(keys['ArrowRight']||keys['d']) move = 1;
      // joystick influence
      if(Math.abs(joystickDir.x) > 0.2) move = joystickDir.x > 0 ? 1 : -1;

      player.x += move * player.speed * (1 + elapsed*0.01);
      // clamp to road
      player.x = clamp(player.x, road.x + 8, road.x + road.w - player.w - 8);

      // spawn obstacles
      spawnTimer++;
      if(spawnTimer > spawnInterval){ spawnObstacle(); spawnTimer=0; spawnInterval = Math.max(36, Math.floor(80 - elapsed*0.6)); }

      // update obstacles
      for(let i=obstacles.length-1;i>=0;i--){
        let o = obstacles[i];
        o.y += o.baseSpeed * (1 + elapsed*0.01);
        o.x += o.lateral * Math.sin(o.y*0.01);
        // when passed bottom
        if(o.y > H + 120){
          obstacles.splice(i,1);
          dodged++;
          score += Math.floor(o.scoreVal * multiplier);
          combo++;
          bestCombo = Math.max(bestCombo, combo);
          // increment multiplier every 5 combos
          if(combo>0 && combo%5===0){ multiplier = +(multiplier * 1.5).toFixed(2); }
          uiScore.textContent = score;
          uiMulti.textContent = 'x'+multiplier;
          statDodged.textContent = dodged;
          statCombo.textContent = combo;
          // small reward particles
          for(let p=0;p<6;p++) spawnParticle(o.x + o.w/2, H-120, o.color);
        }
      }

      // collisions
      const playerBox = {x:player.x, y:player.y, w:player.w, h:player.h};
      for(let o of obstacles){
        if(rectsOverlap(playerBox, o)){
          // spawn crash particles
          for(let i=0;i<40;i++) spawnParticle(player.x + player.w/2, player.y + player.h/2, '#ffcc00');
          endGame();
          return;
        }
      }

      // particles update
      for(let i=particles.length-1;i>=0;i--){
        let p = particles[i]; p.x += p.vx; p.y += p.vy; p.vy += 0.08; p.life--; p.alpha = p.life/60;
        if(p.life<=0) particles.splice(i,1);
      }

      // combos drop after time without dodging
      if(combo>0 && Math.floor(elapsed)%7===0){ /* optional decay logic could be added */ }

      // score bonus per 30s
      if(Math.floor(elapsed)>0 && Math.floor(elapsed)%30===0){ /* kept simple to avoid repeated awarding on same second */ }

      // update UI metrics
      statTime.textContent = Math.floor(elapsed) + 's';
      statSpeed.textContent = Math.floor(3 + elapsed*0.6);
    }

    function draw(){
      // clear
      ctx.clearRect(0,0,W,H);

      // sky glow
      const g = ctx.createLinearGradient(0,0,0,H);
      g.addColorStop(0,'rgba(0,240,255,0.02)'); g.addColorStop(1,'rgba(0,0,0,0.1)');
      ctx.fillStyle = g; ctx.fillRect(0,0,W,H);

      // road backdrop
      roundRect(ctx, road.x-40, 0, road.w+80, H, 40, true, false, '#0b0e12');

      // lane stripes
      const laneW = road.w / road.lanes;
      ctx.lineWidth = 6; ctx.setLineDash([40,30]); ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      for(let i=1;i<road.lanes;i++){
        const x = road.x + laneW*i;
        ctx.beginPath(); ctx.moveTo(x, -40); ctx.lineTo(x, H+40); ctx.stroke();
      }
      ctx.setLineDash([]);

      // road edges glow
      ctx.fillStyle = 'rgba(0,255,255,0.02)'; ctx.fillRect(road.x-8,0,8,H); ctx.fillRect(road.x+road.w,0,8,H);

      // draw obstacles
      for(let o of obstacles){
        roundRect(ctx, o.x, o.y, o.w, o.h, 8, true, false, o.color);
        // subtle highlight
        ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(o.x+4, o.y+6, o.w-8, o.h*0.18);
      }

      // draw player car with headlights
      // shadow
      ctx.save(); ctx.shadowColor = 'rgba(0,0,0,0.6)'; ctx.shadowBlur = 20;
      roundRect(ctx, player.x, player.y, player.w, player.h, 12, true, false, player.color);
      ctx.restore();
      // windows
      ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(player.x + player.w*0.12, player.y + player.h*0.08, player.w*0.76, player.h*0.32);
      // headlights
      ctx.fillStyle = 'rgba(255,255,220,0.9)'; ctx.fillRect(player.x+8, player.y + player.h - 18, 14, 8); ctx.fillRect(player.x + player.w - 22, player.y + player.h - 18, 14, 8);

      // particles
      for(let p of particles){
        ctx.globalAlpha = p.alpha; ctx.fillStyle = p.color; ctx.fillRect(p.x, p.y, 4,4); ctx.globalAlpha = 1;
      }

      // HUD minimal (score rendered in DOM)
    }

    function loop(){ update(); draw(); if(running && !paused) frameId = requestAnimationFrame(loop); }

    // helpers: rounded rect with color override
    function roundRect(ctx,x,y,w,h,r,fill,stroke,color){
      ctx.beginPath(); ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
      if(color) ctx.fillStyle = color; if(fill) ctx.fill(); if(stroke) { ctx.strokeStyle = 'rgba(255,255,255,0.02)'; ctx.stroke(); }
    }

    // Input handlers
    window.addEventListener('keydown', e=>{ keys[e.key]=true; if(e.key===' ' && !running) start(); });
    window.addEventListener('keyup', e=>{ keys[e.key]=false; });

    // joystick pointer handling
    let active = false; let origin = null; const maxStick = 38;
    function toLocal(e){ const rect = joystick.getBoundingClientRect(); const client = e.touches? e.touches[0] : e; return {x: client.clientX - rect.left - rect.width/2, y: client.clientY - rect.top - rect.height/2}; }
    joystick.addEventListener('pointerdown', (e)=>{ joystick.setPointerCapture(e.pointerId); active=true; origin = toLocal(e); });
    joystick.addEventListener('pointermove', (e)=>{ if(!active) return; const p = toLocal(e); let dx = p.x; let dy = p.y; const d = Math.hypot(dx,dy); const s = Math.min(d, maxStick); if(d>0){ dx = dx/d * s; dy = dy/d * s } stick.style.transform = `translate(${dx}px, ${dy}px)`; joystickDir.x = dx/maxStick; joystickDir.y = dy/maxStick; });
    joystick.addEventListener('pointerup', (e)=>{ active=false; stick.style.transform='translate(0,0)'; joystickDir = {x:0,y:0}; });

    // Buttons
    startBtn.addEventListener('click', ()=>start());
    uiStart.addEventListener('click', ()=>start());
    retryBtn.addEventListener('click', ()=>start());
    uiPause.addEventListener('click', ()=>{ paused = !paused; uiPause.textContent = paused? 'Resume' : 'Pause'; if(!paused && running) frameId = requestAnimationFrame(loop); });

    // update DOM Score every second
    setInterval(()=>{ uiScore.textContent = score; uiMulti.textContent = 'x'+multiplier; statTime.textContent = Math.floor(elapsed)+'s'; statSpeed.textContent = Math.floor(3 + elapsed*0.6); statDodged.textContent = dodged; statCombo.textContent = combo; uiHigh.textContent = high; }, 200);

    // initial draw
    draw();
  </script>
</body>
</html>

