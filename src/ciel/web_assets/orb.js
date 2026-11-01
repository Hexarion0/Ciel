function createOrb() {
  const canvas = element('orb-canvas'), core = element('core');
  const context = canvas.getContext('2d');
  if (!context) {
    core.classList.toggle('no-canvas', true);
    return {update() {}};
  }
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const tau = Math.PI * 2;
  const profiles = {
    offline: {speed: 0, energy: .08, glow: .3},
    idle: {speed: .18, energy: .14, glow: .75},
    listening_wake: {speed: .22, energy: .22, glow: .9},
    recording: {speed: .48, energy: .8, glow: 1},
    processing: {speed: .9, energy: .45, glow: .95},
    speaking: {speed: .55, energy: 1, glow: 1}
  };
  let state = 'offline', muted = false, frame = null, lastFrame = null;
  let time = 0, turn = .55, rippleAge = 2;
  let width = 0;
  let current = {...profiles.offline};
  const pointer = {x: 0, y: 0, targetX: 0, targetY: 0, hover: 0, targetHover: 0};
  const active = () => ['recording', 'processing', 'speaking'].includes(state);
  const target = () => muted && !active() ? profiles.offline : profiles[state] || profiles.idle;
  const moving = () => state !== 'offline' && (!muted || active())
    && !document.hidden && !reducedMotion.matches;
  const frameRate = () => state === 'speaking' ? 16 : active() ? 20 : 12;
  const mesh = [];
  // Cache the silhouette's trigonometry instead of recalculating it every frame.
  const silhouette = Array.from({length: 33}, (_, step) => {
    const angle = step / 32 * tau;
    return {
      x: Math.cos(angle), y: Math.sin(angle),
      sin3: Math.sin(angle * 3), cos3: Math.cos(angle * 3),
      sin7: Math.sin(angle * 7), cos7: Math.cos(angle * 7),
      sin11: Math.sin(angle * 11), cos11: Math.cos(angle * 11)
    };
  });
  function spherePoint(latitude, longitude) {
    const ring = Math.cos(latitude);
    return [ring * Math.cos(longitude), Math.sin(latitude), ring * Math.sin(longitude)];
  }
  for (let latitude = -3; latitude <= 3; latitude++) {
    const points = [];
    for (let step = 0; step <= 24; step++) {
      points.push(spherePoint(latitude * Math.PI / 8, step * Math.PI / 12));
    }
    mesh.push(points);
  }
  for (let longitude = 0; longitude < 10; longitude++) {
    const points = [];
    for (let step = 0; step <= 24; step++) {
      points.push(spherePoint(-Math.PI / 2 + step * Math.PI / 24, longitude * Math.PI / 5));
    }
    mesh.push(points);
  }
  function draw() {
    if (!width) return;
    context.clearRect(0, 0, width, width);
    // Procedural voice rhythms follow the assistant's state, without requiring browser mic access.
    const voice = state === 'speaking'
      ? .5 + .3 * Math.sin(time * 6.8) + .2 * Math.sin(time * 11.3)
      : state === 'recording' ? .5 + .25 * Math.sin(time * 4.2) : .2;
    const breath = Math.sin(time * 1.7) * .012;
    const center = width / 2, radius = width * .34 * (1 + breath + voice * current.energy * .035);
    const brightness = current.glow, energy = current.energy;
    const glow = context.createRadialGradient(center, center, radius * .25,
      center, center, radius * 1.4);
    glow.addColorStop(0, `rgba(103, 232, 249, ${(.035 + voice * .035) * brightness})`);
    glow.addColorStop(.65, `rgba(139, 92, 246, ${.09 * brightness})`);
    glow.addColorStop(1, 'rgba(139, 92, 246, 0)');
    context.fillStyle = glow;
    context.beginPath(); context.arc(center, center, radius * 1.4, 0, tau); context.fill();
    const sinTurn = Math.sin(turn + pointer.x * .25), cosTurn = Math.cos(turn + pointer.x * .25);
    const tilt = -.35 + Math.sin(time * .35) * .12 + pointer.y * .2;
    const sinTilt = Math.sin(tilt), cosTilt = Math.cos(tilt);
    function project(point, scale = .8, deform = true) {
      const wave = deform ? 1 + energy * (.025 * Math.sin(point[1] * 6 + time * 2.3)
        + .018 * Math.sin(point[0] * 8 - point[2] * 5 + time * 1.7)) : 1;
      const x = point[0] * cosTurn - point[2] * sinTurn;
      const depth = point[0] * sinTurn + point[2] * cosTurn;
      const y = point[1] * cosTilt - depth * sinTilt;
      const z = point[1] * sinTilt + depth * cosTilt;
      const size = radius * scale * wave * (1 + z * .12);
      return [center + x * size, center + y * size, z];
    }
    const projected = mesh.map(line => line.map(point => project(point)));
    const particles = [];
    for (let index = 0; index < 12; index++) {
      const angle = index * 2.39996 + time * (.1 + current.speed * .16);
      const latitude = Math.sin(index * 1.73 + time * .16) * .85;
      const point = project(spherePoint(latitude, angle), 1.14, false);
      particles.push({point, index});
    }
    function drawParticles(front) {
      for (const {point, index} of particles) {
        if ((point[2] >= 0) !== front) continue;
        const twinkle = .65 + .35 * Math.sin(time * 1.6 + index);
        context.fillStyle = index % 3 === 0
          ? `rgba(164, 128, 250, ${(front ? .75 : .15) * brightness * twinkle})`
          : `rgba(103, 232, 249, ${(front ? .8 : .15) * brightness * twinkle})`;
        context.beginPath();
        context.arc(point[0], point[1], width * (front ? .003 : .002), 0, tau);
        context.fill();
      }
    }
    drawParticles(false);
    // A softly lit back surface gives the wire mesh depth.
    for (const front of [false, true]) {
      context.lineWidth = width / (front ? 650 : 850);
      for (let line = 0; line < projected.length; line++) {
        context.beginPath();
        let connected = false;
        for (const point of projected[line]) {
          if ((point[2] >= 0) !== front) { connected = false; continue; }
          if (connected) context.lineTo(point[0], point[1]);
          else context.moveTo(point[0], point[1]);
          connected = true;
        }
        const mix = line / projected.length;
        context.strokeStyle = `rgba(${Math.round(139 - mix * 36)}, ${Math.round(140 + mix * 92)}, 249, ${(front ? .38 : .07) * brightness})`;
        context.stroke();
      }
    }
    // Layered waves keep the silhouette alive as the core breathes and speaks.
    for (let strand = 0; strand < 6; strand++) {
      const seed = strand * 1.2;
      const wave3 = seed + time * .8, wave7 = -seed * 2 + time * 1.2;
      const wave11 = seed - time * .4;
      const sin3 = Math.sin(wave3), cos3 = Math.cos(wave3);
      const sin7 = Math.sin(wave7), cos7 = Math.cos(wave7);
      const sin11 = Math.sin(wave11), cos11 = Math.cos(wave11);
      const stretch = .97 + .02 * Math.sin(seed);
      context.beginPath();
      for (let index = 0; index < silhouette.length; index++) {
        const point = silhouette[index];
        const ripple = (point.sin3 * cos3 + point.cos3 * sin3) * .018
          + (point.sin7 * cos7 + point.cos7 * sin7) * .012
          + (point.cos11 * cos11 - point.sin11 * sin11) * .008;
        const distance = radius * (.94 + strand * .011 + ripple * (1 + energy * (1 + voice)));
        const x = center + point.x * distance;
        const y = center + point.y * distance * stretch;
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      }
      const mix = strand / 5;
      context.strokeStyle = `rgba(${Math.round(139 - mix * 36)}, ${Math.round(92 + mix * 140)}, 249, ${(.18 + mix * .3 + pointer.hover * .08) * brightness})`;
      context.lineWidth = width / 550;
      context.stroke();
    }
    // Light travels along two ribbons, fading on the sphere's far side.
    for (let ribbon = 0; ribbon < 2; ribbon++) {
      const segments = Array.from({length: 8}, () => []);
      let previous = null;
      for (let step = 0; step <= 32; step++) {
        const angle = step / 32 * tau;
        const latitude = Math.sin(angle * 2 + ribbon * Math.PI + time * .25) * .48;
        const point = project(spherePoint(latitude, angle + ribbon * 1.4), .9);
        if (previous) {
          const travel = .5 + .5 * Math.cos(angle - time * (1 + current.speed) + ribbon * Math.PI);
          const bucket = Math.min(3, Math.floor(travel * 4)) + (point[2] >= 0 ? 4 : 0);
          segments[bucket].push([previous, point]);
        }
        previous = point;
      }
      // Batch similarly lit segments to avoid a canvas stroke for every edge.
      context.lineWidth = width / 370;
      for (let bucket = 0; bucket < segments.length; bucket++) {
        if (!segments[bucket].length) continue;
        const alpha = (.15 + (bucket % 4) / 3 * .65) * (bucket >= 4 ? 1 : .18) * brightness;
        context.strokeStyle = ribbon === 0
          ? `rgba(164, 128, 250, ${alpha})` : `rgba(103, 232, 249, ${alpha})`;
        context.beginPath();
        for (const [from, to] of segments[bucket]) {
          context.moveTo(from[0], from[1]); context.lineTo(to[0], to[1]);
        }
        context.stroke();
      }
    }
    drawParticles(true);
    // Thinking has an orbiting scan; recording and speaking have expanding wave rings.
    if (state === 'processing') {
      for (let ring = 0; ring < 2; ring++) {
        context.strokeStyle = ring === 0
          ? `rgba(164, 128, 250, ${.55 * brightness})` : `rgba(103, 232, 249, ${.35 * brightness})`;
        context.lineWidth = width / 500;
        const start = time * (ring === 0 ? 1.2 : -.8) + ring * Math.PI;
        context.beginPath(); context.arc(center, center, radius * (1.17 + ring * .09), start, start + 1.7);
        context.stroke();
      }
    } else if (state === 'recording' || state === 'speaking') {
      for (let ring = 0; ring < 2; ring++) {
        const progress = (time * (state === 'speaking' ? .65 : .4) + ring * .5) % 1;
        context.strokeStyle = `rgba(103, 232, 249, ${Math.sin(progress * Math.PI) * .18 * brightness})`;
        context.lineWidth = width / 650;
        context.beginPath(); context.arc(center, center, radius * (1.06 + progress * .27), 0, tau);
        context.stroke();
      }
    }
    if (rippleAge < 1.4 && !reducedMotion.matches) {
      const progress = rippleAge / 1.4;
      context.strokeStyle = `rgba(164, 128, 250, ${(1 - progress) * .35 * brightness})`;
      context.lineWidth = width / 450;
      context.beginPath(); context.arc(center, center, radius * (.95 + progress * .42), 0, tau);
      context.stroke();
    }
    const nucleus = radius * (.22 + voice * energy * .06);
    const light = context.createRadialGradient(center - nucleus * .3, center - nucleus * .4,
      0, center, center, nucleus);
    light.addColorStop(0, `rgba(218, 237, 255, ${.7 * brightness})`);
    light.addColorStop(.3, `rgba(103, 232, 249, ${.3 * brightness})`);
    light.addColorStop(1, 'rgba(103, 232, 249, 0)');
    context.fillStyle = light;
    context.beginPath(); context.arc(center, center, nucleus, 0, tau); context.fill();
  }
  function animate(timestamp) {
    frame = null;
    if (!moving()) return;
    if (lastFrame === null || timestamp - lastFrame >= 1000 / frameRate()) {
      // Elapsed time keeps motion consistent across display refresh rates and pauses.
      const delta = lastFrame === null ? 1 / frameRate() : Math.min((timestamp - lastFrame) / 1000, .1);
      const blend = 1 - Math.exp(-delta * 5);
      const next = target();
      for (const key of ['speed', 'energy', 'glow']) current[key] += (next[key] - current[key]) * blend;
      pointer.x += (pointer.targetX - pointer.x) * blend;
      pointer.y += (pointer.targetY - pointer.y) * blend;
      pointer.hover += (pointer.targetHover - pointer.hover) * blend;
      time += delta; turn += delta * current.speed; rippleAge += delta;
      draw(); lastFrame = timestamp;
    }
    frame = requestAnimationFrame(animate);
  }
  function refresh() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null; lastFrame = null;
    // Use a smaller surface without high-DPI supersampling to keep the orb lightweight.
    width = Math.min(256, Math.round(canvas.clientWidth * Math.min(devicePixelRatio || 1, 1)));
    if (width && canvas.width !== width) { canvas.width = width; canvas.height = width; }
    if (!moving()) {
      current = {...target()};
      pointer.x = 0; pointer.y = 0; pointer.hover = 0;
      rippleAge = 2;
    }
    draw();
    if (moving()) frame = requestAnimationFrame(animate);
  }
  core.addEventListener('pointermove', event => {
    if (core.disabled || event.pointerType === 'touch') return;
    const bounds = core.getBoundingClientRect();
    pointer.targetX = Math.max(-1, Math.min(1, (event.clientX - bounds.left) / bounds.width * 2 - 1));
    pointer.targetY = Math.max(-1, Math.min(1, (event.clientY - bounds.top) / bounds.height * 2 - 1));
    pointer.targetHover = 1;
  });
  core.addEventListener('pointerleave', () => {
    pointer.targetX = 0; pointer.targetY = 0; pointer.targetHover = 0;
  });
  window.addEventListener('resize', refresh);
  document.addEventListener('visibilitychange', refresh);
  reducedMotion.addEventListener('change', refresh);
  refresh();
  return {update(nextState, nextMuted) {
    if (state === nextState && muted === nextMuted) return;
    state = nextState; muted = nextMuted;
    if (moving()) rippleAge = 0;
    refresh();
  }};
}
