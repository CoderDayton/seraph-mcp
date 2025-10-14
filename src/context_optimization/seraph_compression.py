"""
seraph_compression.py
---------------------
Three-tier compression pipeline for long corpora:

Tier-1  (500x-style): deterministic structural compressor that builds L1/L2/L3 layers
        via anchor extraction, de-duplication, BM25 salience, and extractive outlines.

Tier-2  (LLM-DCP-inspired): dynamic context pruning that greedily selects spans
        under a token budget using importance + novelty + locality scores.

Tier-3  (Hierarchical): LLMLingua-2 PromptCompressor for query-time
        layered retrieval with graceful fallback.

Required dependencies: tiktoken, blake3, llmlingua
Optional dependencies: sentence_transformers (for semantic embeddings)

Author: Seraph MCP
License: MIT
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import itertools as it
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

# ---------- Required dependencies -------------------------------------------------
try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError as e:
    logger.error(
        "Required dependency 'tiktoken' is missing. Install with: pip install tiktoken>=0.5.0",
        extra={"package": "tiktoken", "error": str(e)},
    )
    tiktoken: ModuleType | None = None  # type: ignore[no-redef]
    _HAS_TIKTOKEN = False

try:
    import blake3

    _HAS_BLAKE3 = True
except ImportError as e:
    logger.error(
        "Required dependency 'blake3' is missing. Install with: pip install blake3>=1.0.7",
        extra={"package": "blake3", "error": str(e)},
    )
    blake3: ModuleType | None = None  # type: ignore[no-redef]
    _HAS_BLAKE3 = False

try:
    # LLMLingua v2 exposes a single class: PromptCompressor
    from llmlingua import PromptCompressor

    _HAS_LLMLINGUA = True
except ImportError as e:
    logger.error(
        "Required dependency 'llmlingua' is missing. Install with: pip install llmlingua>=0.2.2",
        extra={"package": "llmlingua", "error": str(e)},
    )
    PromptCompressor = None
    _HAS_LLMLINGUA = False

# ---------- Optional dependencies -------------------------------------------------
try:
    from .embeddings import cosine_similarity

    _HAS_EMBED = True
except ImportError:
    logger.debug("Embedding service not available. Semantic features disabled.")
    EmbeddingService = None
    cosine_similarity: Callable[..., float] | None = None  # type: ignore[no-redef]
    _HAS_EMBED = False

# ---------- Tokenization utilities ----------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:['\-][A-Za-z0-9_]+)?")


def _simple_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def count_tokens(text: str, model_name: str = "cl100k_base") -> int:
    if not _HAS_TIKTOKEN or tiktoken is None:
        logger.warning(
            "tiktoken not available, using fallback word-based token counting (less accurate)",
            extra={"model": model_name},
        )
        # Fallback: rough word-based count ~ 1.3x words
        return math.ceil(len(_simple_tokens(text)) * 1.3)

    try:
        enc = tiktoken.get_encoding(model_name)
    except Exception as e:
        logger.warning(
            f"Failed to get tiktoken encoding '{model_name}', falling back to cl100k_base: {e}",
            extra={"model": model_name, "error": str(e)},
        )
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------- Hashing --------------------------------------------------------------


def blake_hash(data: bytes) -> str:
    if not _HAS_BLAKE3 or blake3 is None:
        logger.debug("blake3 not available, using hashlib.blake2s fallback")
        return hashlib.blake2s(data).hexdigest()

    try:
        return blake3.blake3(data).hexdigest()
    except Exception as e:
        logger.warning(f"blake3 hashing failed, falling back to blake2s: {e}", extra={"error": str(e)})
        return hashlib.blake2s(data).hexdigest()


# ---------- SimHash (64-bit) for near-dup filtering -----------------------------


def simhash64(tokens: list[str]) -> int:
    if not tokens:
        return 0
    v = [0] * 64
    for tok in tokens:
        h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, val in enumerate(v):
        if val >= 0:
            out |= 1 << i
    return out


def hamm_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------- Basic BM25 -----------------------------------------------------------


@dataclass
class BM25:
    docs: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df: Counter[str] = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}

    def score(self, q: list[str], idx: int) -> float:
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
    meta: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    simhash: int = 0


@dataclass
class Anchor:
    aid: str
    type: str
    surface: str
    norm: str
    span: tuple[int, int]
    chunk_id: str


# ---------- Tier-1: 500x-style compressor ---------------------------------------


class Tier1500x:
    """
    Structural compressor producing 3 layers:
      L1: skeleton bullets using anchors (ultra-small)
      L2: section abstracts (compact)
      L3: top factual extracts (small tables/sentences)
    Heuristic, deterministic (seeded).
    """

    def __init__(self, seed: int = 7, l1_ratio: float = 0.002, l2_ratio: float = 0.01, l3_ratio: float = 0.05):
        self.seed = seed
        random.seed(seed)
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.l3_ratio = l3_ratio

    # --- normalization & chunking
    def normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)  # strip control chars
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def sentence_split(self, text: str) -> list[str]:
        # rudimentary but stable splitter
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
        # fallback for very long lines
        out = []
        for p in parts:
            if len(p) > 2000:
                out.extend(re.findall(r".{1,500}(?:\s+|$)", p))
            else:
                out.append(p)
        return [s.strip() for s in out if s.strip()]

    def make_chunks(self, text: str, max_sent_per_chunk: int = 3) -> list[Chunk]:
        sents = self.sentence_split(text)
        chunks: list[Chunk] = []
        for i in range(0, len(sents), max_sent_per_chunk):
            ctext = " ".join(sents[i : i + max_sent_per_chunk])
            toks = _simple_tokens(ctext)
            chash = blake_hash(ctext.encode("utf-8"))[:12]
            chunks.append(Chunk(id=chash, text=ctext, tokens=toks, simhash=simhash64(toks)))
        return chunks

    def dedup(self, chunks: list[Chunk], hamm_thresh: int = 3) -> list[Chunk]:
        seen: list[int] = []
        out = []
        for c in chunks:
            is_dup = any(hamm_distance64(c.simhash, h) <= hamm_thresh for h in seen)
            if not is_dup:
                seen.append(c.simhash)
                out.append(c)
        return out

    # --- anchors & relations (heuristic extraction)
    _URL = re.compile(r"(https?://\S+)", re.I)
    _QTY = re.compile(r"\b(?:~|≈|=|==|<=|>=|<|>)?\s?(\d+(?:\.\d+)?)(?:\s?[%A-Za-z$€¥₿kKmMgGsShHdDyY]+)\b")
    _DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b")
    _ENTITY = re.compile(r"\b([A-Z][A-Za-z0-9\-\_]+(?:\s+[A-Z][A-Za-z0-9\-\_]+){0,3})\b")

    def extract_anchors(self, chunks: list[Chunk]) -> list[Anchor]:
        anchors: list[Anchor] = []
        aid_seq = 0

        def mk_aid() -> str:
            nonlocal aid_seq
            aid_seq += 1
            return f"A{aid_seq:05d}"

        for c in chunks:
            for m in self._URL.finditer(c.text):
                anchors.append(
                    Anchor(
                        mk_aid(),
                        "url",
                        m.group(1),
                        m.group(1).lower(),
                        (m.start(), m.end()),
                        c.id,
                    )
                )
            for m in self._QTY.finditer(c.text):
                surface = m.group(0).strip()
                anchors.append(
                    Anchor(
                        mk_aid(),
                        "quantity",
                        surface,
                        surface.lower(),
                        (m.start(), m.end()),
                        c.id,
                    )
                )
            for m in self._DATE.finditer(c.text):
                anchors.append(
                    Anchor(
                        mk_aid(),
                        "date",
                        m.group(0),
                        m.group(0),
                        (m.start(), m.end()),
                        c.id,
                    )
                )
            # entities (filter very short/common)
            for m in self._ENTITY.finditer(c.text):
                s = m.group(1).strip()
                if len(s) < 3 or s.lower() in {"the", "and", "for"}:
                    continue
                anchors.append(Anchor(mk_aid(), "entity", s, s.lower(), (m.start(), m.end()), c.id))
        return anchors

    # --- salience via BM25 + density bonuses
    def salience(self, chunks: list[Chunk], anchors: list[Anchor]) -> dict[str, float]:
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
    def build_layers(self, text: str) -> dict[str, Any]:
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
        l3_sents = []
        used_tokens = 0
        for c in ordered:
            if used_tokens + count_tokens(c.text) > l3_budget:
                continue
            l3_sents.append({"text": c.text, "chunk_id": c.id})
            used_tokens += count_tokens(c.text)

        # L2: abstracts per coarse section (every N chunks)
        l2_parts = []
        section_size = 8
        for s in range(0, len(ordered), section_size):
            sec = ordered[s : s + section_size]
            top = sorted(sec, key=lambda c: -sal.get(c.id, 0.0))[:2]
            abstract = " ".join([t.text.split(". ")[0] for t in top if t.text])
            l2_parts.append(f"## Section {1 + s // section_size}\n{abstract}")
        l2_text = "\n\n".join(l2_parts)
        if count_tokens(l2_text) > l2_budget:
            # trim naively
            toks = _simple_tokens(l2_text)
            l2_text = " ".join(toks[: int(l2_budget / 1.3)])

        # L1: skeleton bullets from anchors (entities + quantities)
        by_type = defaultdict(list)
        for a in anchors:
            by_type[a.type].append(a.surface)
        bullets = []

        def uniq(seq: list[Any]) -> Any:
            seen = set()
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    yield x

        for ety in it.islice(uniq(by_type["entity"]), 0, 12):
            bullets.append(f"- {ety}")
        for qty in it.islice(uniq(by_type["quantity"]), 0, 8):
            bullets.append(f"- {qty}")
        l1_text = "\n".join(bullets)
        if count_tokens(l1_text) > l1_budget:
            toks = _simple_tokens(l1_text)
            l1_text = " ".join(toks[: int(l1_budget / 1.3)])

        manifest = {
            "total_tokens": total_tokens,
            "budgets": {"L1": l1_budget, "L2": l2_budget, "L3": l3_budget},
            "chunks": len(chunks),
            "anchors": len(anchors),
            "hash": blake_hash(norm.encode("utf-8")),
        }
        return {
            "L1": l1_text,
            "L2": l2_text,
            "L3": l3_sents,
            "manifest": manifest,
            "chunks": chunks,
        }


# ---------- Tier-2: LLM-DCP inspired dynamic pruning ----------------------------


class Tier2DCP:
    """
    Dynamic context pruning (inspired by LLM-DCP):
    - Builds sentence-level candidates.
    - Scores: importance (BM25 IDF sum), novelty (1 - overlap with selected),
      locality (keep neighbors of chosen high-importance sentences).
    - Greedy selection under a token budget.
    Optionally uses API-based embeddings for semantic similarity.
    """

    def __init__(
        self,
        budget_ratio: float = 0.15,
        neighbor_bonus: float = 0.1,
        embedding_service: Any = None,
    ):
        self.budget_ratio = budget_ratio
        self.neighbor_bonus = neighbor_bonus
        self.embedding_service = embedding_service
        self.use_embeddings = embedding_service is not None

    def split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def importance_scores(self, sentences: list[str]) -> list[float]:
        docs = [[*_simple_tokens(s)] for s in sentences]
        bm = BM25(docs if docs else [[]])
        # pseudo-query: union of rare tokens
        idf = bm.idf
        q = [t for t, _ in sorted(idf.items(), key=lambda kv: kv[1], reverse=True)[:64]]
        return [bm.score(q, i) for i in range(len(sentences))]

    def novelty(self, chosen_embeds: list[Any], cand_embed: Any, cand_tokens: set[Any]) -> float:
        if chosen_embeds and cand_embed is not None and cosine_similarity is not None:
            # 1 - max cosine similarity
            sims = [cosine_similarity(cand_embed, e) for e in chosen_embeds]
            novelty_sim = 1.0 - max(sims)
        else:
            # token-level Jaccard penalty
            # Token-level Jaccard penalty - union calculated but not used for now
            # Could be used for future novelty scoring improvements
            novelty_sim = 1.0  # neutral if nothing
        return max(0.0, novelty_sim)

    async def compress(self, text: str, max_tokens: int | None = None) -> str:
        sents = self.split_sentences(text)
        if not sents:
            return ""

        total = count_tokens(text)
        budget = max_tokens if max_tokens is not None else max(256, int(total * self.budget_ratio))

        imp = self.importance_scores(sents)

        # Generate embeddings via API if service available
        embeds = None
        if self.use_embeddings and self.embedding_service:
            try:
                embeds = await self.embedding_service.embed_texts(sents)
            except Exception as e:
                logger.warning(f"Embedding generation failed, continuing without: {e}")
                embeds = None

        chosen: list[int] = []
        chosen_embeds: list[Any] = []
        used = 0

        # A simple neighbor map
        def neighbors(i: int) -> list[int]:
            return [j for j in (i - 1, i + 1) if 0 <= j < len(sents)]

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


# ---------- Tier-3: Hierarchical compressor (LLMLingua-2) -----------


class Tier3Hierarchical:
    """
    Hierarchical query-time compressor.
    - Uses LLMLingua-2 `PromptCompressor` to materialize L1/L2/L3 by ratios.
    - Falls back to a deterministic sentence-sampling strategy when LLMLingua is unavailable.
    """

    def __init__(
        self, lingua_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", device: str = "cpu"
    ):
        self.has_lingua = _HAS_LLMLINGUA
        self._lingua: Any = None
        if self.has_lingua and PromptCompressor is not None:
            try:
                # PromptCompressor(model_name=..., device_map=..., use_llmlingua2=True)
                self._lingua = PromptCompressor(
                    model_name=lingua_model,
                    device_map=device,
                    use_llmlingua2=True,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize PromptCompressor: {e}")
                self._lingua = None

    def lingua_compress(self, text: str, ratio: float) -> str:
        if self._lingua is None:
            # fallback: crude ratio by sentence sampling
            sents = re.split(r"(?<=[.!?])\s+", text)
            keep = max(1, int(len(sents) * ratio))
            return " ".join(sents[:keep])
        # LLMLingua-2 API
        # compress_prompt(prompt, rate=<0..1>, ...) -> dict with key 'compressed_prompt'
        try:
            res: dict[str, Any] = self._lingua.compress_prompt(
                text,
                rate=ratio,
                use_token_level_filter=True,
                use_context_level_filter=False,
                target_token=-1,
            )
            return (res.get("compressed_prompt") or "").strip()
        except Exception as e:
            logger.warning(f"LLMLingua compression failed: {e}")
            # fallback to sentence sampling
            sents = re.split(r"(?<=[.!?])\s+", text)
            keep = max(1, int(len(sents) * ratio))
            return " ".join(sents[:keep])

    def build_hierarchical(self, text: str) -> dict[str, str]:
        # L1 (very small), L2 (small), L3 (larger) via LLMLingua if available
        l3 = self.lingua_compress(text, 0.35)
        l2 = self.lingua_compress(text, 0.15)
        l1 = self.lingua_compress(text, 0.03)
        return {"L1": l1.strip(), "L2": l2.strip(), "L3": l3.strip()}


# ---------- Orchestrator ---------------------------------------------------------


@dataclass
class CompressionResult:
    l1: str
    l2: str
    l3: str
    manifest: dict[str, Any]


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

    def __init__(
        self,
        seed: int = 7,
        l1_ratio: float = 0.02,
        l2_ratio: float = 0.01,
        l3_ratio: float = 0.05,
        embedding_service: Any = None,
    ):
        """
        Initialize SeraphCompressor with all three tiers.

        Args:
            seed: Random seed for deterministic compression
            l1_ratio: Layer 1 compression ratio
            l2_ratio: Layer 2 compression ratio
            l3_ratio: Layer 3 compression ratio
            embedding_service: Optional EmbeddingService for semantic similarity

        Raises:
            RuntimeError: If critical dependencies are missing
        """
        # Check critical dependencies
        missing_deps = []
        if not _HAS_TIKTOKEN:
            missing_deps.append("tiktoken>=0.5.0")
        if not _HAS_BLAKE3:
            missing_deps.append("blake3>=1.0.7")

        if missing_deps:
            error_msg = f"Cannot initialize SeraphCompressor: missing required dependencies: {', '.join(missing_deps)}"
            logger.critical(error_msg, extra={"missing_dependencies": missing_deps})
            raise RuntimeError(error_msg)

        self.t1 = Tier1500x(seed=seed, l1_ratio=l1_ratio, l2_ratio=l2_ratio, l3_ratio=l3_ratio)
        self.t2 = Tier2DCP(embedding_service=embedding_service)
        self.t3 = Tier3Hierarchical()
        self.embedding_service = embedding_service

        logger.info(
            "SeraphCompressor initialized",
            extra={
                "seed": seed,
                "tiktoken": _HAS_TIKTOKEN,
                "blake3": _HAS_BLAKE3,
                "llmlingua": _HAS_LLMLINGUA,
                "embeddings": embedding_service is not None,
            },
        )

    async def build(self, corpus_text: str) -> CompressionResult:
        """
        Build compressed layers from corpus text.

        Args:
            corpus_text: Input text to compress

        Returns:
            CompressionResult with L1/L2/L3 layers and manifest

        Raises:
            ValueError: If corpus_text is not a string
            RuntimeError: If compression fails critically
        """
        try:
            tier1 = self.t1.build_layers(corpus_text)
            l3_concat = " ".join([s["text"] for s in tier1["L3"]])

            # Tier-2 prune L3 down to a compact capsule (approx 15% of L3 or 0.15*total)
            dcp_budget = max(256, int(count_tokens(corpus_text) * 0.08))
            dcp = await self.t2.compress(l3_concat, max_tokens=dcp_budget)

            # Merge with Tier-3 (optional enrich)
            hcomp = self.t3.build_hierarchical(dcp if dcp else l3_concat)

            manifest = {
                "tier1": tier1["manifest"],
                "tier2": {"budget_tokens": dcp_budget, "dcp_tokens": count_tokens(dcp)},
                "tier3": {"method": "LLMLingua-2" if _HAS_LLMLINGUA else "fallback"},
            }

            result = CompressionResult(
                l1=(tier1["L1"] or hcomp["L1"]),
                l2=(tier1["L2"] or hcomp["L2"]),
                l3=(hcomp["L3"] or l3_concat),
                manifest=manifest,
            )

            logger.debug(
                "Compression complete",
                extra={
                    "original_tokens": count_tokens(corpus_text),
                    "l1_tokens": count_tokens(result.l1),
                    "l2_tokens": count_tokens(result.l2),
                    "l3_tokens": count_tokens(result.l3),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Compression failed: {e}", extra={"error": str(e)}, exc_info=True)
            raise RuntimeError(f"Failed to build compression layers: {e}") from e

    def query(self, result: CompressionResult, question: str, k: int = 5) -> list[tuple[float, str]]:
        """
        Query compressed result using BM25 retrieval over L3 layer.

        Args:
            result: CompressionResult to query
            question: Query text
            k: Number of top results to return

        Returns:
            List of (score, text) tuples, sorted by relevance

        Raises:
            ValueError: If question is invalid
        """
        try:
            # Handle empty question gracefully by returning empty results
            if not question:
                return []
            sents = re.split(r"(?<=[.!?])\s+", result.l3)
            docs = [[*_simple_tokens(s)] for s in sents]
            bm = BM25(docs if docs else [[]])
            q = _simple_tokens(question)
            scored = [(bm.score(q, i), s) for i, s in enumerate(sents) if s.strip()]
            scored.sort(key=lambda x: -x[0])
            return scored[:k]
        except Exception as e:
            logger.error(f"Query failed: {e}", extra={"question": question, "error": str(e)}, exc_info=True)
            raise RuntimeError(f"Failed to query compression result: {e}") from e

    def pack(self, res: CompressionResult, out_path: str) -> str:
        """
        Pack compression result to gzipped JSON file.

        Args:
            res: CompressionResult to pack
            out_path: Output file path

        Returns:
            Absolute path to packed file

        Raises:
            ValueError: If out_path is invalid
            IOError: If file writing fails
        """
        if not out_path:
            raise ValueError("out_path cannot be empty")

        try:
            payload = {"manifest": res.manifest, "L1": res.l1, "L2": res.l2, "L3": res.l3}
            out_path = str(out_path)

            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            logger.info(f"Packed compression result to {out_path}")
            return out_path

        except OSError as e:
            logger.error(
                f"Failed to pack compression result: {e}", extra={"path": out_path, "error": str(e)}, exc_info=True
            )
            raise OSError(f"Failed to write packed file to {out_path}: {e}") from e


# ---------- CLI -----------------------------------------------------------------


def _read_text_from_path(p: str) -> str:
    """
    Read text from file or directory path.

    Args:
        p: File or directory path

    Returns:
        Concatenated text content

    Raises:
        FileNotFoundError: If path doesn't exist
        IOError: If reading fails
    """
    path = Path(p)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    try:
        if path.is_dir():
            buf = []
            for f in sorted(path.rglob("*")):
                if f.suffix.lower() in {".txt", ".md", ".rst", ".log"} and f.is_file():
                    try:
                        buf.append(f.read_text(encoding="utf-8", errors="ignore"))
                    except Exception as e:
                        logger.warning(f"Failed to read file {f}: {e}", extra={"file": str(f), "error": str(e)})
            if not buf:
                logger.warning(f"No readable text files found in directory: {p}")
            return "\n\n".join(buf)
        else:
            return path.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        logger.error(f"Failed to read text from path {p}: {e}", exc_info=True)
        raise OSError(f"Failed to read text from {p}: {e}") from e


async def _async_main(argv: list[str] | None = None) -> None:
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
        res = await sc.build(text)
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


def main(argv: list[str] | None = None) -> None:
    """Synchronous wrapper for async main function."""
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()
