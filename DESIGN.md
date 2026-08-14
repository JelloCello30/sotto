# Sotto — Design System & Voice

Sotto (from *sotto voce* — "under the voice") is a silent-speech input instrument.
You mouth words; it types them. No audio is ever captured. Everything runs on-device.

## Positioning (honest, always)

- It is a **research preview**. It does calibrated-vocabulary silent phrase input,
  not open-vocabulary dictation. Never imply otherwise.
- No fabricated numbers, testimonials, press logos, user counts, or fake download
  buttons. Confidence comes from the working demo, not from invented proof.
- Voice: plainspoken, precise, a little wry. Short sentences. No hype adjectives
  ("revolutionary", "game-changing", "seamless" are banned). No emoji. No exclamation
  marks in body copy. Explain limitations like an engineer who is proud of the thing.

## Brand

- Wordmark: lowercase `sotto.` set in Inter, weight 650, tracking -0.04em, with the
  final period in the accent color.
- Logo glyph: three horizontal bars of decreasing width, slightly rounded (a mouth
  closing / a fading voice). Provided in assets/logo.svg.

## Color

Light (default):
- `--paper`: #FAF7F2 (page background)
- `--ink`: #1C1917 (text)
- `--ink-soft`: #57534E (secondary text)
- `--line`: #E7E0D6 (hairlines, borders)
- `--card`: #FFFFFF (raised surfaces)
- `--accent`: #C2410C (burnt clay — CTAs, active states, the wordmark period)
- `--accent-ink`: #FFF7ED (text on accent)
- `--good`: #3F6212, `--warn`: #A16207, `--bad`: #9F1239

Dark (`@media (prefers-color-scheme: dark)`):
- `--paper`: #131110, `--ink`: #EDE7DE, `--ink-soft`: #A8A29E,
- `--line`: #2B2724, `--card`: #1C1917, `--accent`: #F97316, `--accent-ink`: #1C0A00
- `--good`: #A3E635, `--warn`: #FBBF24, `--bad`: #FB7185

Rules: text always uses `--ink`/`--ink-soft` on `--paper`/`--card`. Never hardcode
hex in components. Both themes must pass AA contrast — test both. (This is a hard
requirement, not a nice-to-have.)

## Type

- Family: `InterVariable` (vendored at vendor/inter/inter-var.woff2), fallback
  system-ui stack. `font-optical-sizing: auto`.
- Display: clamp(2.4rem, 6vw, 4.2rem), weight 640, line-height 1.04, tracking -0.035em.
- H2: 1.6–2rem, weight 620, tracking -0.02em.
- Body: 1.0625rem / 1.65, weight 400. Secondary text 0.9375rem.
- Mono (for keys/paths/status): ui-monospace stack, 0.875em.

## Layout & texture

- Max content width 1080px, gutters 24px (mobile) / 48px (desktop). 8px spacing grid.
- Radius: 12px cards, 8px controls, 999px pills.
- Borders are 1px `--line`; shadows are rare and soft (0 1px 2px rgb(0 0 0 / .06)).
- Asymmetry over symmetry: hero text sits left, artifact right. Avoid three-equal-
  cards rows where possible; prefer a 5/7 split or a stacked ledger list.
- Motion: 160ms ease-out on hover/press; one slow ambient animation max per screen
  (the lip-wave). Respect `prefers-reduced-motion`.

## Components (defined in css/styles.css)

- `.btn` (solid ink), `.btn-accent` (accent), `.btn-ghost` (border only)
- `.card` raised surface, `.pill` status pill, `.kbd` key hint
- `.nav` top bar: wordmark left, links right, hairline bottom border
- `.badge-preview` — small pill reading "Research preview", shown near the wordmark
  on every page. Never remove it.

## The lip-wave motif

A horizontal row of ~48 thin vertical bars whose heights animate like a mouth-
openness signal (irregular, organic, occasionally still). Used in the hero and as
the live visualizer in the app (where it renders REAL mouth-openness data).
The motif is the brand: the site version is decorative; the app version is data.
