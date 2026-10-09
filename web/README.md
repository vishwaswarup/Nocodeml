# NoCodeML website

The Next.js app: landing page, sign-in, project list, and the eight-section workspace (dataset, preprocessing, features,
split, models, regularization, training, results). It talks to the NoCodeML API ([`../engine`](../engine)) with the signed-in
user's Supabase token and signs in with Supabase directly.

**Stack:** Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS v4 · `@supabase/supabase-js` · `motion` · custom SVG charts.

> Next.js 16 differs from older versions. Before changing framework-level code, read the matching guide in
> `node_modules/next/dist/docs/` (see `AGENTS.md`).

## Run

```bash
npm install
npm run dev          # http://localhost:3000
```

Create `web/.env.local` (git-ignored; these values are public by design):

```
NEXT_PUBLIC_SUPABASE_URL=https://<project>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key>
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_MAX_UPLOAD_MB=100        # optional: lets the page refuse oversized files early (match the API's NOCODEML_MAX_UPLOAD_MB)
```

Never put the Supabase `service_role` key here.

## Checks

```bash
npx tsc --noEmit
npm run lint
npm run build
```

## Layout

```
src/
├── app/                routes: / (landing), /login, /auth/callback, /projects, /projects/[id], /privacy, /terms, /design
├── components/
│   ├── workspace/      the eight sections (+ Colab card, tuning panel)
│   ├── results/        comparison table, per-model detail, health checks, artifacts
│   ├── charts/         SVG charts, each with an accessible table twin
│   └── ui/             design-system pieces (buttons, fields, cards, badges)
└── lib/                API client (retries, wake-up notice), typed API shapes, pure helpers
public/
└── nocodeml-colab.ipynb   the Colab notebook; generated, see engine/nocodeml_engine/colab/notebook.py
```

Design notes: dark canvas, Geist + DM Mono, one signature ember → amber → sky gradient. A `/design` page shows the components.
Accessibility is part of "done": keyboard use, labelled controls, text next to colour, and a table alternative for every chart.
