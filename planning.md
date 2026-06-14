# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

My domain is an unofficial student guide to eating at the University of Florida (which I attebded for undergrad). This knowledge is valuable because official UF dining pages explain locations, hours, and meal plan options, but they do not  answer practical student questions like whether meal plans are worth it, which locations students like, where students complain about the quality of the food, and what options seem convenient or healthy in daily life. Student-generated discussions are scattered across Reddit threads, so a RAG system makes that informal knowledge easier to search.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Florida Fresh Dining: Who We Are | Official dining portal with background on dining services and mission | https://new.dineoncampus.com/UF/who-we-are |
| 2 | Florida Fresh Dining: Hours of Operation | Official dining portal with location and hours | https://dineoncampus.com/uf/hours-of-operation |
| 3 | Navigating the Swamp: A Gator's Guide to UF Meal Plans | Unoffical guide to UF meal plans | Unoffical student guide to UF's meal plan options | https://prked.com/post/navigating-the-swamp-a-gators-guide-to-uf-meal-plans |
| 4 | UF Declining Balance | Official explanation/contact page for declining balance dining funds | https://businessservices.ufl.edu/services/dining/declining-balance/ |
| 5 | Florida Fresh Dining: FAQ | Frequently asked questions regarding dining and meal plans | https://new.dineoncampus.com/UF/faq |
| 6 | Reddit: Best meal plan for freshman? | Student discussion of whether freshman meal plans are worth it | https://www.reddit.com/r/ufl/comments/1b5y8qy/best_meal_plan_for_freshman/ |
| 7 | Reddit: UF on campus food tier list | Student opinions ranking campus food options | https://www.reddit.com/r/ufl/comments/xqubez/uf_on_campus_food_tier_list/ |
| 8 | Reddit: Meal plans Freshman Fall 2025 | Student discussion of price, quality, and convenience of meal plans | https://www.reddit.com/r/ufl/comments/1l77ghc/meal_plans_freshman_fall_2025/ |
| 9 | Reddit: Debating about Meal Plan or None | Student discussion comparing meal plans, dining halls, and cooking | https://www.reddit.com/r/ufl/comments/14uywgx/debating_about_meal_plan_or_none/ |
| 10 | Reddit: Lunch? | Student recommendations about campus lunch value and alternatives | https://www.reddit.com/r/ufl/comments/1mj9giu/lunch/ |

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**
About 500 characters per chunk

**Overlap:**
Approximately 100 characters of overlap between adjacent chunks

**Reasoning:**
The sources include a mix of official dining descriptions and short student opinions. Student comments are often compact and full of opinions, while the official pages contain more structured information (meal plan descriptions, hours, and locations). A 500-character chunk is small enough to keep one idea focused, such as meal plan recommendations or a complaint, but large enough to preserve context. The 100-character overlap helps prevent useful details from being split across chunk boundaries, especially when a student comment explains both a recommendation and the reason for it.

**Note for later milestones (noise in chunks):**
The Milestone 3 ingestion currently keeps the `SOURCE:` / `URL:` header lines at the top of each file and some Reddit UI artifacts in the comment threads (e.g. `Upvote`, `Downvote`, `Reply`, usernames, and timestamps like `4y ago`). This is acceptable for ingestion, but before embedding (Milestone 4) I should evaluate whether to strip these artifacts during text cleaning, since they add no dining-related meaning and may slightly dilute retrieval quality. I will inspect sample chunks first and decide whether the added cleaning is worth it.
---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**
sentence-transformers all-MiniLM-L6-v2

**Top-k:**
Retrieve the top 5 chunks for each query

**Production tradeoff reflection:**
For this project, all-MiniLM-L6-v2 is a good choice because it runs locally, is free, has low latency, and doesn't require an API key. In a production system, I would probably compare models based on retrieval accuracy, context length, cost, latency, multilingual support, and how well the model handles informal student language such as slang, abbreviations, and opinion-based reviews. I would also test whether a larger embedding model improves retrieval enough to justify the added compute cost.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about whether UF meal plans are worth it? | Students generally describe meal plans as convenient but not always the best value. Some students suggest smaller weekly swipe plans or cooking/groceries instead of unlimited plans |
| 2 | What are common student complaints about UF campus dining? | Common complaints include food quality, limited weekend options, prices, long lines or inconvenience, and frustration that there are many chain options but fewer satisfying full-meal options |
| 3 | What official dining options or services does UF provide? | UF provides residential dining halls, campus dining locations, national/local brands, meal plans, and declining balance or flex-style dining funds |
| 4 | What do students say is a good value for lunch on or near campus? | Some students say dining halls can be a good value for lunch because they provide filling meals, drinks, salads, and multiple options compared with buying lunch out |
| 5 | What are some student-mentioned alternatives to relying completely on a meal plan? | Students mention cooking, buying groceries, packing food, using smaller meal plans, using flex/declining balance, or eating at selected campus/off-campus locations |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Student comments may be noisy, sarcastic, outdated, or inconsistent. A retrieval system might return an opinion that sounds strong but only represents one student’s experience, so the generated answer needs to describe patterns cautiously instead of treating every comment as a fact.
2. Official dining pages and student discussions have different writing styles. Official pages may retrieve well for questions about meal plan structure or hours, while Reddit threads may retrieve better for opinion questions. If a query mixes official and subjective needs, the system may retrieve only one side of the answer.
3. Chunking could split a student recommendation from its reasoning. The overlap is intended to reduce this risk, but I will inspect sample chunks before embedding.
---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

Raw Documents
(.txt files in /documents)
     \/
Document Ingestion
(Python file reading + basic text cleaning)
     \/
Chunking
(custom chunk_text function, 500 chars, 100 char overlap)
     \/
Embedding + Vector Store
(sentence-transformers all-MiniLM-L6-v2 + ChromaDB)
     \/
Retrieval
(top-5 semantic search results with source metadata)
     \/
Generation
(Groq llama-3.3-70b-versatile, grounded prompt)
     \/
Query Interface
     \/
(Gradio web UI)

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
I will use Claude to help implement the document ingestion and chunking code. I will provide the Domain, Documents, and Chunking Strategy sections from this planning document and ask for Python code that loads .txt files from the documents/ folder, cleans whitespace and obvious artifacts, and splits text into approximately 500-character chunks with 100-character overlap. I will verify the output by printing at least 5 random chunks and checking that they are readable, substantive, and connected to the correct source file.

**Milestone 4 — Embedding and retrieval:**
I will use Claude to help implement the embedding and retrieval code. I will provide the Retrieval Approach and Architecture sections and ask for code that embeds chunks using sentence-transformers/all-MiniLM-L6-v2, stores them in ChromaDB with source metadata, and retrieves the top 5 chunks for a query. I will verify the output by testing at least 3 evaluation questions and checking whether the returned chunks are relevant and have reasonable distance scores.

**Milestone 5 — Generation and interface:**
I will use Claude to help wire retrieval into Groq response generation and a basic Gradio interface. I will provide the grounding requirement that the LLM must answer only from retrieved context and must cite sources. I will verify this by asking both in-scope questions and an out-of-scope question. The system should cite source documents for supported answers and refuse to guess when the retrieved context does not contain enough information.