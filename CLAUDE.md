# Project Rules & Design Contract

## Agent Guardrail — Read This First

Before writing **any** code, check if the request contradicts a rule below.
If it does: **stop immediately, explain the conflict in plain English, list what would break, and wait for explicit "yes" before touching a single file.**
After the user confirms, implement the change AND update this file to reflect the new approved design.

**After every code change — without being asked — do both:**
1. Update `CLAUDE.md` if any rule, threshold, flow, or file ownership described here is now outdated or no longer accurate.
2. Update `README.md` if any user-facing flow, tab description, architecture diagram, or feature description is affected by the change.

---

## Architecture Rules

### Rule 1 — Two Intents Only
`text_to_sql` and `analytical` are the **only** valid intents. No others.
- `text_to_sql` = specific fact / count / list / filter / live data lookup.
- `analytical` = pattern / trend / anomaly / comparison / insight over time.
- Router defaults to `text_to_sql` when unsure — it always produces an answer.
- Constants live in `src/app_constants.py`. The UI badge map in `streamlit_app.py` must match.

### Rule 2 — Domain Neutralization
Engine files must contain **zero** hardcoded business/domain strings.
All compliance-specific content lives exclusively in `src/business_schema/`:
- `domain.py` exports: `DOMAIN_NAME`, `DOMAIN_ANALYST_PERSONA`, `DOMAIN_DATA_ENG_PERSONA`, `EMBED_SEARCH_PERSONA`, `DB_SCRIPT_PATH`, `DOMAIN_SQL_RULES`, `EMBED_TEXT_STYLE`, `REGEN_COVERAGE`.
- `capsule_definitions.py` — `CAPSULE_DEFINITIONS` + `SCHEMA_DEFINITIONS`.
- `dbscript.sql` — database schema script.
To deploy on a new database: replace `src/business_schema/` only. Zero engine changes.

### Rule 3 — Three Qdrant Collections, Fixed Names
| Constant | Collection | Contents |
|---|---|---|
| `COLLECTION_ANALYTICAL` | `analytical` | Pre-computed capsule signals + result rows |
| `COLLECTION_SCHEMA` | `schema_context` | Schema metadata, join hints, SQL templates |
| `COLLECTION_LINKED` | `linked` | Risk-pattern linked capsules from graph |
Collection names come from `src/app_constants.py` only — never hardcode strings.

### Rule 4 — Confidence Gate = 0.65
Serve a capsule answer directly only when `overall_confidence >= 0.65` AND top hit is not a schema capsule. Otherwise fall through to SQL (`analytical_sql_fallback` route).

### Rule 5 — Anomaly Threshold = 0.7
`ANOMALY_DETECTED_THRESHOLD = 0.7` gates linked capsule generation and `anomaly_detected` tag. Never use `> 0.0`.
- Risk level: `anomaly_score >= 0.8` → `"critical"`, else `"high"`.
- `TAG_ESCALATING` added only when capsule type is `violation` AND trend is `"increasing"` AND anomaly > 0.7.
- Anomaly and trend detection require **minimum 4 rows** — returns `0.0` / `"flat"` otherwise.
- Anomaly uses `mean + 2 * stddev` (`ANOMALY_STDDEV_MULTIPLIER = 2.0`).
- Trend requires **≥ 10% change** between first and second half (`TREND_CHANGE_PCT = 10.0`).

### Rule 6 — Generate Before Clear (Never Clear First)
Any refresh operation must: **generate → clear → persist.**
Clearing before generation succeeds leaves an empty store on failure.
Exception: targeted refresh uses **upsert** (not clear) to preserve other capsules in the collection.

### Rule 7 — Always Invalidate Payload Cache After Persist
Call `invalidate_payload_cache()` (from `src/query_engine/context_packager.py`) after every rebuild, targeted refresh, or single-capsule upsert. Never skip.

### Rule 8 — Model Slots Are Deliberate
| Slot | Model | Used For |
|---|---|---|
| `groq_intent_model` | `llama-3.1-8b-instant` | Intent detection — fast, cheap |
| `groq_sql_model` | `llama-3.3-70b-versatile` | SQL gen + capsule regen — strongest |
| `groq_sql_fix_model` | `llama-3.3-70b-versatile` | SQL auto-fix on error |
| `groq_summary_model` | `llama-3.1-8b-instant` | Result row summarization |
| `groq_signal_model` | `llama-3.1-8b-instant` | Capsule signal generation |
Do not swap slots between tasks. Intent detection must stay fast/cheap.

### Rule 9 — No Duplicate Capsule IDs
`capsule_definitions.py` must have unique `capsule_id` values. Fix by renaming with a descriptive suffix. `_load_definitions()` logs an error when detected.

### Rule 10 — linked_capsule_ids and relationship_types Must Stay Equal Length
Enforced by Pydantic validator on `CapsuleDefinition`. Any code writing to these lists must append to **both** simultaneously.

### Rule 11 — User Capsules Are Stored in Two Places
When a user capsule is saved it goes to **both**:
1. `src/business_schema/capsule_definitions.py` — via `append_to_capsule_definitions_file()`
2. `data/user_capsules.json` — via `save_user_capsule_def()`
Never write to one without the other. Tag `user_defined` is auto-appended to all user capsule tags.

### Rule 12 — Groq Client is a Module-Level Singleton
`llm_service.py` holds `_groq_client`. Never instantiate `Groq()` per call.

### Rule 13 — F-string Prompts + .format() Placeholders
Prompts in `llm_instructions.py` are f-strings (domain vars injected at import time) that also use `.format()` at call time. Placeholders for `.format()` must be double-braced: `{{schema}}`, `{{fk_relationships}}`.

### Rule 14 — SQL Auto-fix is One Retry Only
`sql_autofix.py` attempts exactly one repair pass. Do not add loops or multiple retries.

### Rule 15 — Embedding Model and Dimension Are Fixed
- Model: `BAAI/bge-small-en-v1.5`
- Dimension: `EMBED_DIM = 384`
Changing either breaks **all existing vectors** in Qdrant — requires full wipe and rebuild.

### Rule 16 — SQL Generation Hard Rules
Generated SQL must follow these rules (enforced via LLM prompt in `llm_instructions.py`):
- SQLite syntax only (`LIMIT` not `TOP`, `COALESCE` not `ISNULL`, no `dbo.` prefix)
- No `SELECT *` — always name columns explicitly
- Always include `ORDER BY`
- Every selected column must have an explicit `AS` alias
- Explicit `JOIN ... ON` only — no comma joins, no implicit joins
- SQL must start with `select` or `with` (lowercase) — rejected otherwise
- `LIMIT 50` for regular queries, `LIMIT 100` for trend/time-series queries
- Only `SELECT` or `WITH` statements allowed — all others rejected at executor

### Rule 17 — Capsule Definition Rules
- **capsule_id**: unique, snake_case, no spaces, no hyphens
- **capsule_type**: one of `aggregation`, `trend`, `violation`, `risk`, `pattern`, `operational`, `distribution`, `sample`
- **priority**: `P1` (violation/critical), `P2` (monitoring/trend), `P3` (operational), `P4` (info)
- **signal_method**: `rule_based` for counts/aggregations, `llm_summary` for complex patterns, `sample` for random-row joins
- **ttl_hours**: P1 violation=6, operational=2, trend=12, other=24–48; user-defined: violation/risk=6, trend=12, aggregation/pattern/distribution=24, operational=2
- **tags**: 5–10 per capsule, all lowercase snake_case
- **Capsule regen must produce 40+ capsules** across 9 domain categories

### Rule 18 — Relationship Graph Rules
- `MIN_SHARED_TAGS_FOR_RELATION = 2` — two capsules need at least 2 shared tags to form a `corroborates` edge
- `MAX_HOP_DEPTH = 2` — graph traversal stops at 2 hops
- `MAX_LINKED_CAPSULES = 3` — max linked capsules included in a context package
- Relationship types: `corroborates`, `drills_down`, `aggregates_up`, `same_entity`
- Inferred edges must be **written back to in-memory capsule objects** during `build_graph()` before persisting to Qdrant

### Rule 19 — Search Score Thresholds
| Collection | Min Score | Top-K |
|---|---|---|
| `analytical` | 0.30 | 5 |
| `schema_context` | 0.20 | 5 |
| `linked` | 0.25 | 5 |
- Stale capsules (`is_stale=True`) are always filtered out from search results.
- Schema capsules are capped at **top 3** in context packaging.

### Rule 20 — Row Limits Throughout the Pipeline
| Stage | Limit |
|---|---|
| DB query execution | 500 rows (`DB_EXECUTE_MAX_ROWS`) |
| Capsule result rows stored in payload | 50 rows (`CAPSULE_RESULT_MAX_ROWS`) |
| SQL rows returned to UI | 50 rows |
| Sample signal rows sent to LLM | 15 rows |
| Result summarizer rows sent to LLM | 25 rows |
| Analytical answer combined context | 4000 chars |
| Activity comparison rows per capsule | 3 rows |

### Rule 21 — Capsule History Rules
- Snapshots stored in `MM-DD-YYYY` dated folders.
- Same-day rebuilds **overwrite** previous snapshot for that day.
- Snapshots older than **30 days** (`CAPSULE_HISTORY_RETENTION_DAYS`) are auto-deleted.
- Capsules are merged chronologically; later dates overwrite earlier for the same `capsule_id`.

---

## Key Flows

### User Asks a Question
```
Question
 → Intent Detection  (fast LLM / keyword fallback)
 ├─ analytical
 │    → Vector search Qdrant (analytical + schema collections)
 │    → confidence >= 0.65 AND not schema capsule?
 │         YES → return capsule answer            [route: capsule_direct / vector_retrieval]
 │         NO  → SQL path                         [route: analytical_sql_fallback]
 └─ text_to_sql
      → SQL path
           → Schema context from Qdrant (in-memory cached)
           → LLM generates SQL  (groq_sql_model, temp=0.0, max_tokens=700)
           → Validate: must start with select/with
           → Execute on SQLite (max 500 rows)
           → Error? → one auto-fix attempt (temp=0.0, max_tokens=700)
           → LLM summarizes first 25 rows  (groq_summary_model, temp=0.2, max_tokens=300)
           → Answer returned (UI shows max 50 rows)
```

### "Generate All Capsules" Button
```
Purge Qdrant storage entirely
 → Load CAPSULE_DEFINITIONS + user capsule defs  (check for duplicate IDs)
 → generate_all_capsules()        → GeneratedCapsule list
 → generate_schema_capsules()     → SchemaContextCapsule list
 → build_graph()                  → inferred edges written back to in-memory capsules
 → generate_linked_capsules()     → LinkedCapsule list  (anomaly_score >= 0.7 only)
 → Persist all three collections
 → invalidate_payload_cache()
 → Save refresh plan (.analytical_refresh_plan.json)
 → Save schema fingerprint (.schema_fingerprint.json)
```

### "Insert Capsule" (User-defined Capsule)
```
User fills form (what, tables, filters, tags)
 → enrich_user_capsule_metadata()   LLM fills: how / signal / embed_text / ttl  (max_tokens=512)
 → generate_capsule_sql()           LLM writes SQL  (max_tokens=512)
 → Preview shown → user confirms
 → build_single_capsule()           → GeneratedCapsule + embedding
 → append_to_capsule_definitions_file()   injected into capsule_definitions.py
 → save_user_capsule_def()                stored in user_capsules.json
 → Upsert into COLLECTION_ANALYTICAL
 → invalidate_payload_cache()
 → link_user_capsule_to_existing()   graph edges to related capsules (score >= 0.7)
```

### Schema Change Detection (app startup)
```
check_schema_and_refresh_if_needed()
 → compute current schema fingerprint
 → compare to .schema_fingerprint.json
 → changed → full rebuild (generate_all_capsule_collections)
 → unchanged → no-op  (first run saves baseline, no rebuild triggered)
```

---

## File Ownership

| Area | File(s) |
|---|---|
| Domain config | `src/business_schema/domain.py`, `capsule_definitions.py` |
| Intent routing | `src/query_engine/query_router.py`, `orchestrator.py` |
| Capsule build pipeline | `src/capsule_builder/store_manager.py`, `capsule_generator.py` |
| LLM prompts | `src/llm_instructions.py` — neutral; domain vars injected via f-string |
| Vector store ops | `src/vector_store.py` |
| UI | `streamlit_app.py` |
| Constants | `src/app_constants.py` — single source for intents, collections, all thresholds |
| Payload cache | `src/query_engine/context_packager.py` |
| Graph + linked capsules | `src/capsule_builder/relationship_builder.py` |
| Anomaly + trend ML | `src/capsule_builder/ml_enricher.py` |
| History snapshots | `src/capsule_history.py` |

---

## Changes That Require a Warning + Confirmation

| Change | What Breaks |
|---|---|
| Adding a new intent | Router, orchestrator, UI badge map — must all change together |
| Hardcoding domain strings in engine files | Domain neutralization — new DB deployment fails |
| Moving files out of `src/business_schema/` | Engine imports break |
| Changing confidence threshold (0.65) | How often SQL fallback fires vs capsule answer |
| Changing anomaly threshold (0.7) | Which capsules get linked — too low = noise |
| Changing TREND_CHANGE_PCT or ANOMALY_STDDEV_MULTIPLIER | Trend/anomaly scoring changes across all capsules |
| Changing embedding model or EMBED_DIM | All existing Qdrant vectors become incompatible — full wipe required |
| Changing min score thresholds | Search result quality and capsule retrieval changes |
| Clearing collection before generation | Empty store on any generation failure |
| Skipping `invalidate_payload_cache()` | Stale answers served after rebuild |
| Adding/removing `CapsuleDefinition` model fields | Breaks refresh plan deserialization on next startup |
| Changing `capsule_definitions.py` file format | `save_generated_definitions()` and `append_to_capsule_definitions_file()` use regex splice — silently breaks saves |
| Swapping model slots between tasks | Intent detection must stay fast/cheap; SQL must stay on strongest model |
| Writing user capsule to only one of the two stores | State divergence between `capsule_definitions.py` and `user_capsules.json` |
| Adding multiple SQL auto-fix retries | Intended as a single safety net, not a retry loop |
| Changing SQL generation rules (SELECT *, ORDER BY, aliases, JOIN syntax) | Generated SQL quality degrades; executor may reject output |
| Changing MAX_HOP_DEPTH, MAX_LINKED_CAPSULES, MIN_SHARED_TAGS | Graph traversal and context package composition changes |
| Changing row limits (CAPSULE_RESULT_MAX_ROWS, DB_EXECUTE_MAX_ROWS) | Memory, payload size, and Groq token limits affected |
| Changing capsule history retention (30 days) | Storage growth or loss of history data |
