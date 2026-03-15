# ⚡ VLSI RAG — Course Assistant

A browser-based Retrieval-Augmented Generation (RAG) app for VLSI design courses. Upload your lecture PDFs, circuit diagrams, Verilog files, and notes — then chat with them using Claude AI.

![VLSI RAG Screenshot](https://img.shields.io/badge/Stack-React%20+%20Vite%20+%20Claude%20API-00d4ff?style=for-the-badge)

## Features

- **PDF + Image Extraction** — Claude's vision API extracts text AND detailed descriptions of circuit schematics, band diagrams, timing diagrams, I-V plots, layouts, and more
- **BM25 Full-Text Search** — In-browser inverted index with TF-IDF scoring for fast, relevant chunk retrieval
- **Smart Chunking** — 300-word chunks with 50-word overlap to preserve context across chunk boundaries
- **LaTeX Math Rendering** — Equations render beautifully with KaTeX ($I_D = \frac{\mu_n C_{ox}}{2} \frac{W}{L}(V_{GS}-V_T)^2$)
- **Exam Prep Mode** — Generate practice exams with worked solutions from your course materials
- **Native EDA File Support** — `.v`, `.sv`, `.sdc`, `.spice`, `.lib`, `.lef`, `.def`, and more
- **Persistent Storage** — Documents and chat history survive browser refreshes via localStorage
- **Mobile Responsive** — Works on phone/tablet for on-the-go study sessions

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/vlsi-rag.git
cd vlsi-rag
npm install
```

### 2. Run Locally

```bash
npm run dev
```

Open [http://localhost:5173/vlsi-rag/](http://localhost:5173/vlsi-rag/) in your browser.

### 3. Set Your API Key

Go to **Settings** → paste your Anthropic API key from [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).

> Your key stays in your browser's localStorage and is only sent directly to Anthropic's API. It never touches any other server.

### 4. Upload & Chat

Go to **Documents** → drag in your lecture PDFs, screenshot images, and Verilog files → switch to **Chat** and start asking questions.

## Deploy to GitHub Pages

### Option A: GitHub Actions (recommended)

1. Push your repo to GitHub
2. Go to **Settings → Pages → Source** and select **GitHub Actions**
3. Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist
      - id: deployment
        uses: actions/deploy-pages@v4
```

4. Push → your app is live at `https://YOUR_USERNAME.github.io/vlsi-rag/`

### Option B: Manual Deploy

```bash
npm run deploy
```

This builds and pushes to the `gh-pages` branch. Make sure **Settings → Pages → Source** is set to the `gh-pages` branch.

> **Important**: Update `base` in `vite.config.js` to match your repo name if it's different from `vlsi-rag`.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Upload     │────▶│   Extract    │────▶│   Chunk &   │────▶│   BM25       │
│   PDF/IMG    │     │   via Claude │     │   Index     │     │   Search     │
│   Verilog    │     │   Vision API │     │   (browser) │     │   Index      │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                     │
                    ┌──────────────┐     ┌─────────────┐            │
                    │   Claude     │◀────│   Top-K     │◀───────────┘
                    │   Response   │     │   Chunks    │     User Query
                    └──────────────┘     └─────────────┘
```

1. **Upload** — PDFs and images are sent to Claude Sonnet's vision API with a VLSI-specialized prompt that extracts text AND describes every diagram in detail
2. **Index** — Extracted content is split into overlapping chunks. A BM25 inverted index (with stop-word removal and minimal stemming) is built entirely in the browser
3. **Retrieve** — When you ask a question, BM25 ranks all chunks by relevance and selects the top 6
4. **Generate** — The retrieved chunks are sent as context to Claude with a VLSI-specialized system prompt

## Supported File Types

| Category | Extensions |
|----------|-----------|
| Documents | `.pdf`, `.txt`, `.md`, `.csv` |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif` |
| HDL | `.v`, `.sv`, `.vh`, `.svh` |
| EDA | `.sdc`, `.lib`, `.lef`, `.def`, `.spice`, `.sp`, `.cir` |
| Scripts | `.py`, `.tcl` |
| Other | `.json`, `.yaml`, `.cfg`, `.log`, `.rpt` |

## Project Structure

```
vlsi-rag/
├── index.html          # Entry HTML with KaTeX CSS
├── vite.config.js      # Vite config with GH Pages base path
├── package.json
├── src/
│   ├── main.jsx        # React entry point
│   ├── App.jsx         # Main app (RAG engine + UI)
│   └── index.css       # All styles
└── public/
```

## Tech Stack

- **React 18** + **Vite** — fast dev & build
- **Claude Sonnet API** — chat, PDF extraction, image description
- **BM25** — in-browser full-text retrieval
- **react-markdown** + **KaTeX** — rich markdown + LaTeX math rendering
- **Lucide React** — icons
- **localStorage** — persistence

## Tips

- **Large PDFs**: Very large PDFs (50+ pages) may hit API token limits. Consider splitting them or uploading chapter-by-chapter.
- **Image Quality**: Higher resolution screenshots of diagrams give better extraction results.
- **Exam Prep**: The more relevant materials you upload, the better the generated exam questions match your course content.
- **Cost**: Each PDF page costs roughly $0.01–0.03 to extract via the API. Chat messages are typically $0.003–0.01 each.

## License

MIT
