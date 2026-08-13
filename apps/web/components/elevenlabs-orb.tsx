"use client";

// Adapted from ElevenLabs UI's MIT-licensed Orb component.
// https://github.com/elevenlabs/ui/blob/main/apps/www/registry/elevenlabs-ui/ui/orb.tsx
// Metis generates the noise texture locally so the visualizer has no runtime
// CDN dependency and remains usable when the rest of the app is offline.

import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

export type ElevenLabsOrbState = null | "thinking" | "listening" | "talking";

type ElevenLabsOrbProps = {
  colors?: [string, string];
  agentState?: ElevenLabsOrbState;
  getInputVolume?: () => number;
  getOutputVolume?: () => number;
  className?: string;
};

export function ElevenLabsOrb({
  colors = ["#73518b", "#ff7759"],
  agentState = null,
  getInputVolume,
  getOutputVolume,
  className,
}: ElevenLabsOrbProps) {
  return (
    <div className={className}>
      <Canvas
        resize={{ debounce: 80 }}
        gl={{ alpha: true, antialias: true, premultipliedAlpha: true }}
      >
        <OrbScene
          colors={colors}
          agentState={agentState}
          getInputVolume={getInputVolume}
          getOutputVolume={getOutputVolume}
        />
      </Canvas>
    </div>
  );
}

function OrbScene({
  colors,
  agentState,
  getInputVolume,
  getOutputVolume,
}: Required<Pick<ElevenLabsOrbProps, "colors" | "agentState">> &
  Pick<ElevenLabsOrbProps, "getInputVolume" | "getOutputVolume">) {
  const circleRef = useRef<THREE.Mesh<THREE.CircleGeometry, THREE.ShaderMaterial>>(null);
  const stateRef = useRef<ElevenLabsOrbState>(agentState);
  const inputRef = useRef(0);
  const outputRef = useRef(0);
  const targetColor1 = useRef(new THREE.Color(colors[0]));
  const targetColor2 = useRef(new THREE.Color(colors[1]));

  useEffect(() => {
    stateRef.current = agentState;
  }, [agentState]);

  useEffect(() => {
    targetColor1.current.set(colors[0]);
    targetColor2.current.set(colors[1]);
  }, [colors]);

  const noiseTexture = useMemo(() => makeNoiseTexture(), []);
  useEffect(() => () => noiseTexture.dispose(), [noiseTexture]);

  const uniforms = useMemo(
    () => ({
      uColor1: new THREE.Uniform(new THREE.Color(colors[0])),
      uColor2: new THREE.Uniform(new THREE.Color(colors[1])),
      uPerlinTexture: new THREE.Uniform(noiseTexture),
      uTime: new THREE.Uniform(0),
      uAnimation: new THREE.Uniform(0.1),
      uInputVolume: new THREE.Uniform(0),
      uOutputVolume: new THREE.Uniform(0),
      uOpacity: new THREE.Uniform(0),
    }),
    [colors, noiseTexture],
  );

  useFrame((_, delta) => {
    const material = circleRef.current?.material;
    if (!material) return;
    const state = stateRef.current;
    const measuredInput = safeVolume(getInputVolume);
    const measuredOutput = safeVolume(getOutputVolume);
    const fallbackInput = state === "listening" ? 0.22 : 0;
    const fallbackOutput = state === "talking" ? 0.58 : state === "thinking" ? 0.2 : 0.08;
    inputRef.current += ((measuredInput || fallbackInput) - inputRef.current) * 0.2;
    outputRef.current += ((measuredOutput || fallbackOutput) - outputRef.current) * 0.2;

    const u = material.uniforms;
    u.uTime.value += delta * 0.5;
    u.uAnimation.value += delta * (0.16 + outputRef.current * 0.7);
    u.uInputVolume.value = inputRef.current;
    u.uOutputVolume.value = outputRef.current;
    u.uOpacity.value = Math.min(1, u.uOpacity.value + delta * 2.5);
    u.uColor1.value.lerp(targetColor1.current, 0.08);
    u.uColor2.value.lerp(targetColor2.current, 0.08);
  });

  return (
    <mesh ref={circleRef}>
      <circleGeometry args={[3.5, 96]} />
      <shaderMaterial
        uniforms={uniforms}
        vertexShader={VERTEX_SHADER}
        fragmentShader={FRAGMENT_SHADER}
        transparent
      />
    </mesh>
  );
}

function safeVolume(reader?: () => number): number {
  try {
    const value = reader?.() ?? 0;
    return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
  } catch {
    return 0;
  }
}

function makeNoiseTexture(): THREE.DataTexture {
  const size = 96;
  const bytes = new Uint8Array(size * size * 4);
  let seed = 0x6d657469;
  for (let index = 0; index < size * size; index += 1) {
    seed = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    seed ^= seed + Math.imul(seed ^ (seed >>> 7), 61 | seed);
    const value = ((seed ^ (seed >>> 14)) >>> 0) & 255;
    const offset = index * 4;
    bytes[offset] = value;
    bytes[offset + 1] = value;
    bytes[offset + 2] = value;
    bytes[offset + 3] = 255;
  }
  const texture = new THREE.DataTexture(bytes, size, size, THREE.RGBAFormat);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

const VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = /* glsl */ `
uniform float uTime;
uniform float uAnimation;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform float uInputVolume;
uniform float uOutputVolume;
uniform float uOpacity;
uniform sampler2D uPerlinTexture;
varying vec2 vUv;

const float PI = 3.14159265358979323846;

vec2 hash2(vec2 p) {
  return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}

float noise2D(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float n = mix(
    mix(dot(hash2(i), f), dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
    mix(dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
        dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x),
    u.y
  );
  return 0.5 + 0.5 * n;
}

float flow(vec3 decomposed, float time) {
  return mix(
    texture2D(uPerlinTexture, vec2(time, decomposed.x / 2.0)).r,
    texture2D(uPerlinTexture, vec2(time, decomposed.y / 2.0)).r,
    decomposed.z
  );
}

vec3 colorRamp(float value, vec3 first, vec3 second) {
  if (value < 0.34) return mix(vec3(0.04), first, value / 0.34);
  if (value < 0.7) return mix(first, second, (value - 0.34) / 0.36);
  return mix(second, vec3(1.0), (value - 0.7) / 0.3);
}

void main() {
  vec2 uv = vUv * 2.0 - 1.0;
  float radius = length(uv);
  if (radius > 1.0) discard;

  float theta = atan(uv.y, uv.x);
  if (theta < 0.0) theta += 2.0 * PI;
  vec3 decomposed = vec3(
    theta / (2.0 * PI),
    mod(theta / (2.0 * PI) + 0.5, 1.0) + 1.0,
    abs(theta / PI - 1.0)
  );

  float drift = flow(decomposed, radius * 0.04 - uAnimation * 0.18) - 0.5;
  float bands = sin(theta * 2.7 + drift * 4.0 + uTime * 0.45);
  float folds = sin(radius * 11.0 - uAnimation * 2.2 + drift * 5.0);
  float grain = noise2D(vec2(theta * 1.8, radius * 5.0 - uAnimation));
  float value = clamp(0.5 + bands * 0.18 + folds * 0.13 + (grain - 0.5) * 0.22, 0.0, 1.0);
  value += uOutputVolume * 0.1;

  vec3 color = colorRamp(clamp(value, 0.0, 1.0), uColor1, uColor2);
  float highlight = smoothstep(0.42, 0.0, distance(uv, vec2(-0.32, 0.36)));
  color = mix(color, vec3(1.0), highlight * 0.48);

  float reactiveRing = smoothstep(0.055, 0.0, abs(radius - (0.76 + uInputVolume * 0.16)));
  color = mix(color, vec3(0.96, 1.0, 0.96), reactiveRing * (0.18 + uInputVolume * 0.52));
  float edge = 1.0 - smoothstep(0.975, 1.0, radius);
  gl_FragColor = vec4(color, edge * uOpacity);
}
`;
