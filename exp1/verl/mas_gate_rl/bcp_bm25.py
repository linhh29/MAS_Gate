"""Pure-Python BM25 fallback over the local BrowseComp-Plus parquet corpus.

This is used only when the Lucene/Pyserini dependency required by the released
BrowseComp-Plus searcher is unavailable.  It still performs real BM25 scoring
over the downloaded corpus under datasets/BrowseComp-Plus/browsecomp-plus-corpus.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class RetrievedDoc:
    docid: str
    text: str
    score: float | None = None


def _tok(text: str, max_chars: int | None = None) -> list[str]:
    if max_chars and len(text or "") > max_chars:
        text = (text or "")[:max_chars]
    return [x.group(0).lower() for x in _TOKEN_RE.finditer(text or "") if len(x.group(0)) > 1]


class ParquetBM25Retriever:
    def __init__(self, corpus_dir: str | Path, *, cache_path: str | Path | None = None, k1: float = 1.2, b: float = 0.75):
        self.corpus_dir = Path(corpus_dir)
        self.cache_path = Path(cache_path) if cache_path else None
        self.k1 = float(k1)
        self.b = float(b)
        self.max_doc_chars = int(__import__("os").environ.get("MAS_GATE_BCP_BM25_MAX_DOC_CHARS", "12000"))
        if self.cache_path and self.cache_path.exists():
            self._load_cache(self.cache_path)
        else:
            self._build()
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("wb") as f:
                    pickle.dump({
                        "docs": self.docs,
                        "doc_lens": self.doc_lens,
                        "avgdl": self.avgdl,
                        "inverted": self.inverted,
                        "idf": self.idf,
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_cache(self, path: Path) -> None:
        with path.open("rb") as f:
            obj = pickle.load(f)
        self.docs = obj["docs"]
        self.doc_lens = obj["doc_lens"]
        self.avgdl = obj["avgdl"]
        self.inverted = obj["inverted"]
        self.idf = obj["idf"]

    def _build(self) -> None:
        import pyarrow.parquet as pq

        files = sorted(p for p in (self.corpus_dir / "data").glob("*.parquet") if not p.name.startswith("._"))
        if not files:
            files = sorted(p for p in self.corpus_dir.glob("*.parquet") if not p.name.startswith("._"))
        if not files:
            raise FileNotFoundError(f"No parquet corpus files found under {self.corpus_dir}")
        self.docs: list[dict[str, str]] = []
        self.doc_lens: list[int] = []
        df: dict[str, int] = defaultdict(int)
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for fp in files:
            table = pq.read_table(fp, columns=["docid", "text"])
            for row in table.to_pylist():
                docid = str(row.get("docid", ""))
                text = str(row.get("text", ""))
                did = len(self.docs)
                self.docs.append({"docid": docid, "text": text})
                counts = Counter(_tok(text, self.max_doc_chars))
                dl = sum(counts.values()) or 1
                self.doc_lens.append(dl)
                for term, tf in counts.items():
                    df[term] += 1
                    self.inverted[term].append((did, int(tf)))
        n = max(1, len(self.docs))
        self.avgdl = float(sum(self.doc_lens) / n) if self.doc_lens else 1.0
        self.idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        scores: dict[int, float] = defaultdict(float)
        terms = Counter(_tok(query))
        for term, qtf in terms.items():
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for did, tf in postings:
                dl = self.doc_lens[did]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[did] += qtf * idf * (tf * (self.k1 + 1.0) / denom)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: int(k)]
        return [RetrievedDoc(docid=self.docs[did]["docid"], text=self.docs[did]["text"], score=float(score)) for did, score in top]
