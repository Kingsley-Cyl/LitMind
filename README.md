# LitMind

LitMind is a course project for intelligent literature management built as a two-part system:

- `client_windows/`: a `PyQt5 + PyQt-Fluent-Widgets` desktop client
- `server_ubuntu/`: a `FastAPI` service that handles PDF parsing, NLP analysis, embedding, search, and recommendation

## Features

- Batch PDF import from a server-side directory
- Metadata extraction for title, abstract, authors, year, and sections
- Keyword extraction with `TextRank + KeyBERT` fallback support
- Extractive summarization with the fixed scoring formula from the project plan
- Multilingual semantic retrieval with `sentence-transformers` when available
- Similar paper recommendations
- Fluent desktop GUI for import, library, search, and detail views

## Project Layout

```text
LitMind/
├── client_windows/
├── server_ubuntu/
├── shared/
└── tests/
```

## Quick Start

### 1. Start the server

```bash
cd server_ubuntu
pip install -r requirements.txt
python run_server.py
```

### 2. Start the Windows client

```bash
cd client_windows
pip install -r requirements.txt
python main.py
```

The default client base URL is `http://127.0.0.1:9000`.

## SSH Port Forwarding

If the NLP service runs on Ubuntu and the GUI runs on Windows, forward the service locally:

```bash
ssh -L 9000:127.0.0.1:9000 user@server_ip
```

Then point the client to `http://127.0.0.1:9000`.

## Notes

- The implementation prefers stable fallbacks when heavyweight NLP packages are unavailable.
- `sentence-transformers`, `faiss-cpu`, and `keybert` are optional at runtime but recommended.
- Imported data is stored in `server_ubuntu/server/data/`.

