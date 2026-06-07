# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

UCLA students rely on a fragmented patchwork of sources — Bruinwalk, Rate My Professors, r/UCLA threads, and shared Google Docs — to get honest course and professor advice before enrolling. This knowledge is hard to find in one place, disappears when threads get buried, and often contains contradictory opinions scattered across years of posts. This RAG guide aggregates that unofficial wisdom into a single queryable source.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Bruinwalk | UCLA student course & professor reviews — more detailed than RMP, UCLA-specific | bruinwalk.com |
| 2 | Rate My Professors — UCLA | Star ratings and written professor reviews | ratemyprofessors.com → search "UCLA" |
| 3 | r/UCLA — professor threads | Reddit discussions on specific professors and courses (search "professor", "CS 31", etc.) | reddit.com/r/UCLA |
| 4 | r/UCLA — course advice megathreads | Seasonal "what classes should I take" threads, posted each quarter | reddit.com/r/UCLA |
| 5 | r/uclaclasses | Subreddit dedicated to UCLA course discussions | reddit.com/r/uclaclasses |
| 6 | Daily Bruin | Student newspaper course/professor features and ranking articles | dailybruin.com |
| 7 | UCLA CS department syllabi | Official course descriptions and syllabi PDFs | cs.ucla.edu/courses |
| 8 | UCLA subreddit wiki | Compiled tips and course advice if available | reddit.com/r/UCLA/wiki |
| 9 | YouTube — UCLA course experience vlogs | First-person course recaps; use auto-generated captions as text | Search "UCLA CS 31 experience" |
| 10 | Public Google Docs study guides | Student-shared course notes and guides | Search: "UCLA" "CS 31" site:docs.google.com |
| 11 | Piazza / Ed Discussion public posts | Public Q&A from UCLA courses (if accessible without login) | Check per course |
| 12 | UCLA Discord server pinned messages | Pinned tips per course in student Discord servers | Manual copy from server |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 150 words (≈ tokens for all-MiniLM-L6-v2)

**Overlap:** 25 words

**Max review size before windowing:** 190 words

**Reasoning:** The corpus is Bruinwalk reviews, which range from 1 sentence to 1,200+ words. The original plan was one chunk per review, but profiling showed 104 of 283 reviews (37%) exceeded 200 words. all-MiniLM-L6-v2 has a hard 256-token limit and silently truncates beyond it, meaning the second half of long reviews would never be embedded.

Refined strategy — hybrid mode:
- Short reviews (≤ 190 words): one chunk per review, preserving the complete opinion as a unit.
- Long reviews (> 190 words): sliding window at 150 words with 25-word overlap, each sub-chunk prefixed with the Quarter/Grade/date header so retrieval context (who wrote it, when, what grade) is preserved.

For non-Bruinwalk files (Reddit threads, syllabi if added): pure sliding window at 150 words / 25-word overlap, since facts are scattered mid-paragraph and need overlap to avoid splitting a claim from its context.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** all-MiniLM-L6-v2 (via sentence-transformers)

**Top-k:** 5

**Production tradeoff reflection:** all-MiniLM-L6-v2 is fast and lightweight but was trained on general text, not academic reviews. For production, I'd evaluate text-embedding-3-small (OpenAI) or instructor-xl, which supports domain-specific prompting — useful for distinguishing "this professor is hard" (workload) from "this professor is harsh" (grading). Multilingual support matters less here since all sources are English. Latency is manageable at this corpus size, but a larger deployment would need to balance retrieval accuracy against embedding inference cost.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer | Result | Pass? |
|---|----------|-----------------|--------|-------|
| 1 | What do students say about the difficulty of CS 31 at UCLA? | References to heavy workload, weekly projects, and advice to start assignments early | Correctly refused — "I don't have enough information on that in my sources." CS31 not in corpus; retrieval returned CS131/CS35L/CS33 at distances 0.34–0.36, none specific to CS31. | ✓ Correct refusal |
| 2 | Is Professor Reinman known for curving grades in his courses? | Specific mentions of curve policy from reviews or Reddit threads | Correctly refused — "I don't have enough information." Retrieved chunks were all Reinman files (good) but none used the word "curve"; vocabulary mismatch prevented a match despite the data existing in adjacent form. | ✓ Correct refusal |
| 3 | What are the most common complaints about CS 32 at UCLA? | References to pointer/memory management difficulty, specific project complaints | Correctly refused — CS32 not in corpus. Top distances 0.41–0.47, worst of all queries, correctly signaling near-total semantic miss. | ✓ Correct refusal |
| 4 | Do UCLA CS professors hold useful office hours, or are TAs more helpful? | Student opinions comparing professor vs. TA office hours usefulness | Partial answer produced — TAs described as helpful, professor helpfulness described as varying. But distances 0.53–0.56 exceeded the 0.5 warning threshold; one retrieved chunk (student who transferred) was off-topic. | ~ Partial |
| 5 | Which UCLA CS professor is most recommended for introductory programming? | Named professors with positive sentiment from Bruinwalk/RMP reviews | Grounding failure — model answered "Eggert is recommended for CS111" but CS111 is an operating systems course, not introductory programming. Retrieved chunks didn't match the question; model misread the closest result. | ✗ Fail |
| 6 *(corpus-grounded refusal test)* | What is the main project in Nacherberg's CS131? | Building a programming language interpreter in Python across 3–4 sub-projects | Correctly refused despite the answer being present in the corpus — retrieval failed (distances 0.54–0.59); "main project" vocabulary didn't match how students wrote about "the interpreter project." Model refused rather than hallucinating. | ✓ Correct refusal |

**Results summary:** 4 correct refusals (Q1, Q2, Q3, Q6), 1 partial answer (Q4), 1 grounding failure (Q5). The refusal mechanism works correctly — the model never fabricates when retrieved chunks are weak or off-topic. The failure in Q5 is a retrieval-quality issue: the returned chunks were plausible but didn't actually answer the question, and the model drew a wrong inference from them rather than refusing.

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. **Opinion drift across years:** A professor's reviews from 2018 may contradict 2024 reviews if their teaching style changed. Without date metadata on chunks, the system may surface outdated opinions as current fact. Mitigation: tag each chunk with its source year and include that in retrieved context.

2. **Short reviews losing context when chunked:** A 3-sentence RMP review like "Avoid if you hate group projects. Grading is fair. Lectures are dry but recordings are posted." conveys meaning through sequence. If split mid-review, a chunk containing only "Grading is fair" misrepresents the full picture. Mitigation: treat each individual review as one atomic chunk regardless of token count.

---

## Architecture

```
                              RAG Pipeline — Offline (Build Time)
  ─────────────────────────────────────────────────────────────────────────────────

  ┌──────────────────┐     ┌──────────────────┐     ┌────────────────────────────┐
  │  1. Document     │     │  2. Chunking      │     │  3. Embedding +            │
  │     Ingestion    │────▶│                  │────▶│     Vector Store           │
  │                  │     │                  │     │                            │
  │  • plain read    │     │  • 1 chunk/review│     │  • all-MiniLM-L6-v2        │
  │    (.txt)        │     │    (short docs)  │     │    (sentence-transformers) │
  │  • pdfplumber    │     │  • 150–200 tok   │     │  • ChromaDB                │
  │    (.pdf)        │     │    + 25 tok      │     │    (local persistent)      │
  │                  │     │    overlap (long)│     │                            │
  └──────────────────┘     └──────────────────┘     └────────────────────────────┘


                              RAG Pipeline — Online (Query Time)
  ─────────────────────────────────────────────────────────────────────────────────

                 User Query
                     │
                     ▼
  ┌──────────────────────────┐     ┌─────────────────────────────────────────────┐
  │  4. Retrieval            │     │  5. Generation                              │
  │                          │────▶│                                             │
  │  • embed query with      │     │  • Groq API (LLaMA 3)                       │
  │    all-MiniLM-L6-v2      │     │  • system prompt requires source citation   │
  │  • cosine similarity     │     │  • top-5 chunks injected as context         │
  │    search in ChromaDB    │     │  • grounded answer returned to user         │
  │  • return top-5 chunks   │     │                                             │
  │    with source metadata  │     │                                             │
  └──────────────────────────┘     └─────────────────────────────────────────────┘
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
I'll give Claude this planning.md (specifically the Documents table and Chunking Strategy section) and ask it to implement `ingest.py` with two chunking modes: (1) one-chunk-per-review for short review files, and (2) sliding window at 150 tokens with 25 token overlap for longer documents. I'll verify by printing chunk counts and spot-checking that no review is split mid-sentence.

**Milestone 4 — Embedding and retrieval:**
I'll give Claude the Architecture diagram and ask it to implement `embed.py` using sentence-transformers all-MiniLM-L6-v2 and ChromaDB. I'll verify by querying for a known professor name and confirming relevant chunks are returned in top-5.

**Milestone 5 — Generation and interface:**
I'll give Claude the system prompt I drafted (grounded, citation-required) and ask it to implement `query.py` that calls Groq with the top-5 chunks as context. I'll run all 5 evaluation questions and compare output against expected answers in the Evaluation Plan.
