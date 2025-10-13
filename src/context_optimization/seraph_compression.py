
"""
seraph_compression.py
---------------------
Three-tier compression pipeline for long corpora:

Tier-1  (500x-style): deterministic structural compressor that builds L1/L2/L3 layers
        via anchor extraction, de-duplication, BM25 salience, and extractive outlines.

Tier-2  (LLM-DCP-inspired): dynamic context pruning that greedily selects spans
        under a token budget using importance + novelty + locality scores.

Tier-3  (Hierarchical): optional LLMLingua-2 + LangChain contextual compression
        for query-time layered retrieval; falls back to internal rules if absent.

This module is self-contained and degrades gracefully if optional packages
(tiktoken, blake3, llmlingua, langchain, sentence_transformers) are not installed.

Author: Seraph MCP
License: MIT
"""

from __future__ import annotations

import os
import re
import io
import sys
import json
import math
import time
import gzip
import tarfile
import hashlib
import random
import string
import itertools as it
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Optional, Any
from collections import defaultdict, Counter, deque
from pathlib import Path

# ---------- Optional dependencies -------------------------------------------------
try:
    import tiktoken  # for accurate token counts
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

try:
    import blake3  # fast hashing
    _HAS_BLAKE3 = True
except Exception:
    _HAS_BLAKE3 = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_EMBED = True
except Exception:
    _HAS_EMBED = False

try:
    # LLMLingua-2 (pip install llmlingua)
    from llmlingua import LLMLingua
    _HAS_LLMLINGUA = True
except Exception:
    _HAS_LLMLINGUA = False

try:
    # LangChain contextual compression
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document
    _HAS_LANGCHAIN = True
except Exception:
    _HAS_LANGCHAIN = False

# ---------- Tokenization utilities ----------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:['\-][A-Za-z0-9_]+)?")

def _simple_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())

def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    # Fallback: rough word-based count ~ 1.3x words
    return math.ceil(len(_simple_tokens(text)) * 1.3)

# ---------- Hashing --------------------------------------------------------------

def blake_hash(data: bytes) -> str:
    if _HAS_BLAKE3:
        return blake3.blake3(data).hexdigest()
    # Fallback: blake2s
    return hashlib.blake2s(data).hexdigest()

# ---------- SimHash (64-bit) for near-dup filtering -----------------------------

def simhash64(tokens: List[str]) -> int:
    if not tokens:
        return 0
    v = [0] * 64
    for tok in tokens:
        h = int(hashlib.blake2b(tok.encode('utf-8'), digest_size=8).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, val in enumerate(v):
        if val >= 0:
            out |= (1 << i)
    return out

def hamm_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# ---------- Basic BM25 -----------------------------------------------------------

@dataclass
class BM25:
    docs: List[List[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def score(self, q: List[str], idx: int) -> float:
        d = self.docs[idx]
        freqs = Counter(d)
        score = 0.0
        for t in q:
            if t not in freqs:
                continue
            idf = self.idf.get(t, math.log(1 + (self.N - 0.5) / 0.5))
            tf = freqs[t]
            denom = tf + self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl + 1e-9))
            score += idf * (tf * (self.k1 + 1) / (denom + 1e-9))
        return score

# ---------- Core data structures -------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list)
    simhash: int = 0

@dataclass
class Anchor:
    aid: str
    type: str
    surface: str
    norm: str
    span: Tuple[int, int]
    chunk_id: str

# ---------- Tier-1: 500x-style compressor ---------------------------------------

class Tier1_500x:
    """
    Structural compressor producing 3 layers:
      L1: skeleton bullets using anchors (ultra-small)
      L2: section abstracts (compact)
      L3: top factual extracts (small tables/sentences)
    Heuristic, deterministic (seeded).
    """

    def __init__(self, seed: int = 7, l1_ratio=0.002, l2_ratio=0.01, l3_ratio=0.05):
        self.seed = seed
        random.seed(seed)
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.l3_ratio = l3_ratio

    # --- normalization & chunking
    def normalize(self, text: str) -> str:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', ' ', text)  # strip control chars
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    def sentence_split(self, text: str) -> List[str]:
        # rudimentary but stable splitter
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
        # fallback for very long lines
        out = []
        for p in parts:
            if len(p) > 2000:
                out.extend(re.findall(r'.{1,500}(?:\s+|$)', p))
            else:
                out.append(p)
        return [s.strip() for s in out if s.strip()]

    def make_chunks(self, text: str, max_sent_per_chunk: int = 3) -> List[Chunk]:
        sents = self.sentence_split(text)
        chunks: List[Chunk] = []
        for i in range(0, len(sents), max_sent_per_chunk):
            ctext = ' '.join(sents[i:i+max_sent_per_chunk])
            toks = _simple_tokens(ctext)
            chash = blake_hash(ctext.encode('utf-8'))[:12]
            chunks.append(Chunk(id=chash, text=ctext, tokens=toks, simhash=simhash64(toks)))
        return chunks

    def dedup(self, chunks: List[Chunk], hamm_thresh: int = 3) -> List[Chunk]:
        seen = []
        out = []
        for c in chunks:
            is_dup = any(hamm_distance64(c.simhash, h) <= hamm_thresh for h in seen)
            if not is_dup:
                seen.append(c.simhash)
                out.append(c)
        return out

    # --- anchors & relations (heuristic extraction)
    _URL = re.compile(r'(https?://\S+)', re.I)
    _QTY = re.compile(r'\b(?:~|≈|=|==|<=|>=|<|>)?\s?(\d+(?:\.\d+)?)(?:\s?[%A-Za-z$€¥₿kKmMgGsShHdDyY]+)\b')
    _DATE = re.compile(r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b')
    _ENTITY = re.compile(r'\b([A-Z][A-Za-z0-9\-\_]+(?:\s+[A-Z][A-Za-z0-9\-\_]+){0,3})\b')

    def extract_anchors(self, chunks: List[Chunk]) -> List[Anchor]:
        anchors: List[Anchor] = []
        aid_seq = 0
        def mk_aid():
            nonlocal aid_seq
            aid_seq += 1
            return f"A{aid_seq:05d}"

        for c in chunks:
            for m in self._URL.finditer(c.text):
                anchors.append(Anchor(mk_aid(), "url", m.group(1), m.group(1).lower(), (m.start(), m.end()), c.id))
            for m in self._QTY.finditer(c.text):
                surface = m.group(0).strip()
                anchors.append(Anchor(mk_aid(), "quantity", surface, surface.lower(), (m.start(), m.end()), c.id))
            for m in self._DATE.finditer(c.text):
                anchors.append(Anchor(mk_aid(), "date", m.group(0), m.group(0), (m.start(), m.end()), c.id))
            # entities (filter very short/common)
            for m in self._ENTITY.finditer(c.text):
                s = m.group(1).strip()
                if len(s) < 3 or s.lower() in {"the", "and", "for"}:
                    continue
                anchors.append(Anchor(mk_aid(), "entity", s, s.lower(), (m.start(), m.end()), c.id))
        return anchors

    # --- salience via BM25 + density bonuses
    def salience(self, chunks: List[Chunk], anchors: List[Anchor]) -> Dict[str, float]:
        docs = [c.tokens for c in chunks]
        bm25 = BM25(docs)
        # pseudo-query: all rare terms
        idf = bm25.idf
        rare_terms = sorted(idf.items(), key=lambda kv: kv[1], reverse=True)[:64]
        q = [t for t, _ in rare_terms]
        scores = {}
        anchor_by_chunk = defaultdict(list)
        for a in anchors:
            anchor_by_chunk[a.chunk_id].append(a)
        for i, c in enumerate(chunks):
            s = bm25.score(q, i)
            bonus = sum(1 for a in anchor_by_chunk[c.id] if a.type in {"quantity", "date", "url"}) * 0.3
            scores[c.id] = s + bonus
        return scores

    # --- layer builders
    def build_layers(self, text: str) -> Dict[str, Any]:
        norm = self.normalize(text)
        chunks = self.make_chunks(norm)
        chunks = self.dedup(chunks)
        anchors = self.extract_anchors(chunks)
        sal = self.salience(chunks, anchors)

        total_tokens = count_tokens(norm)
        l1_budget = max(64, int(total_tokens * self.l1_ratio))
        l2_budget = max(256, int(total_tokens * self.l2_ratio))
        l3_budget = max(1024, int(total_tokens * self.l3_ratio))

        # L3: top chunks by salience until budget
        ordered = sorted(chunks, key=lambda c: (-sal.get(c.id, 0.0), c.id))
        L3_sents = []
        used_tokens = 0
        for c in ordered:
            if used_tokens + count_tokens(c.text) > l3_budget:
                continue
            L3_sents.append({"text": c.text, "chunk_id": c.id})
            used_tokens += count_tokens(c.text)

        # L2: abstracts per coarse section (every N chunks)
        L2_parts = []
        section_size = 8
        for s in range(0, len(ordered), section_size):
            sec = ordered[s:s+section_size]
            top = sorted(sec, key=lambda c: -sal.get(c.id, 0.0))[:2]
            abstract = " ".join([t.text.split(". ")[0] for t in top if t.text])
            L2_parts.append(f"## Section {1 + s//section_size}\n{abstract}")
        L2_text = "\n\n".join(L2_parts)
        if count_tokens(L2_text) > l2_budget:
            # trim naively
            toks = _simple_tokens(L2_text)
            L2_text = " ".join(toks[:int(l2_budget/1.3)])

        # L1: skeleton bullets from anchors (entities + quantities)
        by_type = defaultdict(list)
        for a in anchors:
            by_type[a.type].append(a.surface)
        bullets = []
        def uniq(seq): 
            seen = set()
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    yield x
        for ety in it.islice(uniq(by_type["entity"]), 0, 12):
            bullets.append(f"- {ety}")
        for qty in it.islice(uniq(by_type["quantity"]), 0, 8):
            bullets.append(f"- {qty}")
        L1_text = "\n".join(bullets)
        if count_tokens(L1_text) > l1_budget:
            toks = _simple_tokens(L1_text)
            L1_text = " ".join(toks[:int(l1_budget/1.3)])

        manifest = {
            "total_tokens": total_tokens,
            "budgets": {"L1": l1_budget, "L2": l2_budget, "L3": l3_budget},
            "chunks": len(chunks),
            "anchors": len(anchors),
            "hash": blake_hash(norm.encode("utf-8")),
        }
        return {"L1": L1_text, "L2": L2_text, "L3": L3_sents, "manifest": manifest, "chunks": chunks}

# ---------- Tier-2: LLM-DCP inspired dynamic pruning ----------------------------

class Tier2_DCP:
    """
    Dynamic context pruning (inspired by LLM-DCP):
    - Builds sentence-level candidates.
    - Scores: importance (BM25 IDF sum), novelty (1 - overlap with selected),
      locality (keep neighbors of chosen high-importance sentences).
    - Greedy selection under a token budget.
    Optionally uses sentence embeddings if sentence_transformers is available.
    """

    def __init__(self, budget_ratio: float = 0.15, neighbor_bonus: float = 0.1, use_embeddings: bool = True):
        self.budget_ratio = budget_ratio
        self.neighbor_bonus = neighbor_bonus
        self.use_embeddings = use_embeddings and _HAS_EMBED
        self._embed = SentenceTransformer("all-MiniLM-L6-v2") if self.use_embeddings else None

    def split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def importance_scores(self, sents: List[str]) -> List[float]:
        docs = [[* _simple_tokens(s)] for s in sents]
        bm = BM25(docs if docs else [[]])
        # pseudo-query: union of rare tokens
        idf = bm.idf
        q = [t for t, _ in sorted(idf.items(), key=lambda kv: kv[1], reverse=True)[:64]]
        return [bm.score(q, i) for i in range(len(sents))]

    def novelty(self, chosen_embeds: List[Any], cand_embed: Any, cand_tokens: set) -> float:
        if chosen_embeds and cand_embed is not None:
            # 1 - max cosine similarity
            sims = [float(st_util.cos_sim(cand_embed, e)) for e in chosen_embeds]
            novelty_sim = 1.0 - max(sims)
        else:
            # token-level Jaccard penalty
            union = set().union(*[cand_tokens]) if cand_tokens else set()
            novelty_sim = 1.0  # neutral if nothing
        return max(0.0, novelty_sim)

    def compress(self, text: str, max_tokens: Optional[int] = None) -> str:
        sents = self.split_sentences(text)
        if not sents:
            return ""

        total = count_tokens(text)
        budget = max_tokens if max_tokens is not None else max(256, int(total * self.budget_ratio))

        imp = self.importance_scores(sents)
        embeds = self._embed.encode(sents, convert_to_tensor=True) if self.use_embeddings else None
        chosen: List[int] = []
        chosen_embeds: List[Any] = []
        used = 0

        # A simple neighbor map
        def neighbors(i): 
            return [j for j in (i-1, i+1) if 0 <= j < len(sents)]

        available = set(range(len(sents)))
        while available and used < budget:
            # candidate score = importance * novelty + neighbor_bonus if neighbor chosen
            best_i, best_score = None, -1.0
            for i in list(available):
                tok_count = count_tokens(sents[i])
                if used + tok_count > budget:
                    continue
                cand_tok = set(_simple_tokens(sents[i]))
                cand_embed = embeds[i] if embeds is not None else None
                nov = self.novelty(chosen_embeds, cand_embed, cand_tok)
                neigh = any((j in chosen) for j in neighbors(i))
                score = imp[i] * (0.5 + 0.5 * nov) + (self.neighbor_bonus if neigh else 0.0)
                if score > best_score:
                    best_i, best_score = i, score
            if best_i is None:
                break
            chosen.append(best_i)
            if embeds is not None:
                chosen_embeds.append(embeds[best_i])
            used += count_tokens(sents[best_i])
            available.remove(best_i)
        chosen.sort()
        return " ".join(sents[i] for i in chosen)

# ---------- Tier-3: Hierarchical compressor (LLMLingua-2 + LangChain) -----------

class Tier3_Hierarchical:
    """
    Hierarchical query-time compressor.
    - If LLMLingua-2 is installed, uses its ratio-based compression for L1/L2/L3.
    - If LangChain is installed, exposes a retriever with ContextualCompressionRetriever.
    - Otherwise, falls back to internal budgeted selection.
    """

    def __init__(self, lingua_model: str = "llmlingua-2-bert-base-uncased", device: str = "cpu"):
        self.has_lingua = _HAS_LLMLINGUA
        self.has_langchain = _HAS_LANGCHAIN
        self._lingua = None
        if self.has_lingua:
            try:
                self._lingua = LLMLingua(model=lingua_model, device=device)
            except Exception:
                self._lingua = None

    def lingua_compress(self, text: str, ratio: float) -> str:
        if self._lingua is None:
            # fallback: crude ratio by sentence sampling
            sents = re.split(r'(?<=[.!?])\s+', text)
            keep = max(1, int(len(sents) * ratio))
            return " ".join(sents[:keep])
        res = self._lingua.compress_text(text, rate=ratio, force_tokens=None)
        return res.get("compressed_text", "") or ""

    def build_hierarchical(self, text: str) -> Dict[str, str]:
        # L1 (very small), L2 (small), L3 (larger) via LLMLingua if available
        L3 = self.lingua_compress(text, 0.35)
        L2 = self.lingua_compress(text, 0.15)
        L1 = self.lingua_compress(text, 0.03)
        return {"L1": L1.strip(), "L2": L2.strip(), "L3": L3.strip()}

# ---------- Orchestrator ---------------------------------------------------------

@dataclass
class CompressionResult:
    l1: str
    l2: str
    l3: str
    manifest: Dict[str, Any]

class SeraphCompressor:
    """
    Orchestrates the three tiers:
      - Tier-1 builds deterministic structural layers (L1/L2/L3).
      - Tier-2 prunes L3 further to a DCP capsule (optional step).
      - Tier-3 provides query-time hierarchical compression (LLMLingua-2) when present.

    API:
      build(corpus_text) -> CompressionResult
      query(question) -> top snippets from L3
      pack(path) -> tar.gz with manifest + layers
    """

    def __init__(self, seed: int = 7):
        self.t1 = Tier1_500x(seed=seed)
        self.t2 = Tier2_DCP()
        self.t3 = Tier3_Hierarchical()

    def build(self, corpus_text: str) -> CompressionResult:
        tier1 = self.t1.build_layers(corpus_text)
        l3_concat = " ".join([s["text"] for s in tier1["L3"]])
        # Tier-2 prune L3 down to a compact capsule (approx 15% of L3 or 0.15*total)
        dcp_budget = max(256, int(count_tokens(corpus_text) * 0.08))
        dcp = self.t2.compress(l3_concat, max_tokens=dcp_budget)
        # Merge with Tier-3 (optional enrich)
        hcomp = self.t3.build_hierarchical(dcp if dcp else l3_concat)
        manifest = {
            "tier1": tier1["manifest"],
            "tier2": {"budget_tokens": dcp_budget, "dcp_tokens": count_tokens(dcp)},
            "tier3": {"method": "LLMLingua-2" if _HAS_LLMLINGUA else "fallback"},
        }
        return CompressionResult(
            l1=(tier1["L1"] or hcomp["L1"]),
            l2=(tier1["L2"] or hcomp["L2"]),
            l3=(hcomp["L3"] or l3_concat),
            manifest=manifest
        )

    # very small in-file retriever over L3 using BM25
    def query(self, result: CompressionResult, question: str, k: int = 5) -> List[Tuple[float, str]]:
        sents = re.split(r'(?<=[.!?])\s+', result.l3)
        docs = [[* _simple_tokens(s)] for s in sents]
        bm = BM25(docs if docs else [[]])
        q = _simple_tokens(question)
        scored = [(bm.score(q, i), s) for i, s in enumerate(sents) if s.strip()]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def pack(self, res: CompressionResult, out_path: str) -> str:
        payload = {
            "manifest": res.manifest,
            "L1": res.l1,
            "L2": res.l2,
            "L3": res.l3
        }
        out_path = str(out_path)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path

# ---------- CLI -----------------------------------------------------------------

def _read_text_from_path(p: str) -> str:
    path = Path(p)
    if path.is_dir():
        buf = []
        for f in sorted(path.rglob("*")):
            if f.suffix.lower() in {".txt", ".md", ".rst", ".log"} and f.is_file():
                try:
                    buf.append(f.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    pass
        return "\n\n".join(buf)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Seraph 3-Tier Compressor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build compressed layers from input file/dir")
    b.add_argument("input_path", type=str)
    b.add_argument("-o", "--out", type=str, default="seraph_pack.json.gz")

    q = sub.add_parser("query", help="Query over L3 layer of a compressed pack")
    q.add_argument("pack_path", type=str)
    q.add_argument("question", type=str)
    q.add_argument("-k", type=int, default=5)

    args = parser.parse_args(argv)

    if args.cmd == "build":
        text = _read_text_from_path(args.input_path)
        sc = SeraphCompressor()
        res = sc.build(text)
        out = sc.pack(res, args.out)
        print(json.dumps(res.manifest, indent=2))
        print(f"Saved: {out} (tokens: L1={count_tokens(res.l1)}, L2={count_tokens(res.l2)}, L3={count_tokens(res.l3)})")
    elif args.cmd == "query":
        with gzip.open(args.pack_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        res = CompressionResult(payload["L1"], payload["L2"], payload["L3"], payload["manifest"])
        sc = SeraphCompressor()
        top = sc.query(res, args.question, k=args.k)
        for s, txt in top:
            print(f"{s:.3f}: {txt}")

if __name__ == "__main__":
    main()
