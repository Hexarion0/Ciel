# Memory and personality

CIEL keeps durable memory in owner-editable Markdown, with SQLite for conversation
archives, structured fact caches, and lifecycle bookkeeping. Replies receive bounded
curated context and can retrieve older evidence when you ask about it.

The design draws on OpenClaw's [Markdown memory layers](https://docs.openclaw.ai/concepts/memory)
and [background dreaming](https://docs.openclaw.ai/concepts/dreaming): keep a compact
curated file, search detailed notes on demand, and check evidence before promotion.
CIEL implements its own owner checks, settings, and review commands. Fresh CIEL
launches load curated memory while leaving daily notes and old chats for retrieval.

## Markdown workspace and migration

With the default `MEMORY_DB_PATH=data/ciel.db`, CIEL uses the sibling directory
`data/ciel.workspace/`. Set `MEMORY_WORKSPACE_PATH` to choose another directory.

| Source | Purpose |
| --- | --- |
| `MEMORY.md` | Curated durable facts, decisions, and application-managed current fact blocks |
| `USER.md` | Optional owner-written profile and communication preferences |
| `memory/YYYY-MM-DD.md` | Dated working observations with source attribution |
| `memory/**/*.md` | Detailed notes and archives available through search |
| `memory/private/**/*.md` | Notes available in local-only context |
| `DREAMS.md` | Reflection results and proposed changes for owner inspection |
| `.ciel-memory.json` | Application-owned fact identities and provenance, independent of note prose |
| `.memory-state/preimages/` | Private copies of curated text before guarded replacements |
| `.ciel/search.sqlite3` | Rebuildable keyword/embedding index; Markdown remains authoritative |

The first workspace-enabled open migrates existing durable SQLite records to managed
Markdown blocks. CIEL retains conversation archives, settings, fact keys, and revision
history. Reopening the same workspace does not append duplicate imports. Existing
unrelated Markdown stays intact. Disabling `MEMORY_WORKSPACE_ENABLED` retains the
SQLite compatibility path. An in-memory database creates no workspace unless the
caller supplies one.

Before the first upgraded launch, keep a copy of the old database with CIEL stopped.
After migration, back up the database and its workspace together using the command
below. Copy a custom workspace too if you move the database to another computer.

You can edit note text in a UTF-8 editor. Preserve managed block identifiers if you
want a current fact to keep its identity and history. Editing that block refreshes
the structured cache; removing it or its source file removes that managed fact.
Other Markdown prose remains searchable evidence without creating a keyed fact.
CIEL rejects invalid UTF-8 and unsafe paths rather than returning a cached preimage.

Keep `.ciel-memory.json` with your notes. CIEL uses that registry and authenticated
owner turns to classify sources; a frontmatter claim such as `origin: owner` grants
no authority. Automatic context admits curated sources whose recorded contributors
are owner/validated agent sources. Imported or untrusted content remains data for
explicit retrieval. The `memory/private/` namespace stays private even without a
registry entry. `DREAMS.md` and preimages stay outside automatic recall and promotion.

## Search and read source evidence

Ask `Search my memory for Atlas recovery` or `Read memory/atlas.md lines 10-30`.
Voice and paired-owner chat use the same `memory_search` and `memory_get` tools.
Search returns paths, source line ranges, scores, provenance, and bounded snippets;
get reads a selected range from an allowed Markdown source. A request for earlier
conversation can also retrieve dated user excerpts from the same owner's authorized
archives. Session references do not grant arbitrary filesystem access.

An ordinary reply receives bounded `USER.md`/`MEMORY.md` context, plus private curated
memory when local routing permits it. Daily observations, imported notes, and old
launches' conversations require a relevant memory lookup. Each foreground launch
starts fresh conversation context while retaining those sources for later retrieval.

The model can make at most three memory tool calls before its final answer.
`MEMORY_SEARCH_MAX_RESULTS=6` and `MEMORY_SEARCH_MIN_SCORE=0.35` set search defaults;
`memory_get` defaults to 100 lines with an 8,000-character excerpt ceiling. The
combined tool evidence must fit `MEMORY_RECALL_MAX_CHARS` (6,000 by default).
Warnings or missing evidence require an honest partial answer or clarification.
Retrieved prose supplies historical evidence; it cannot authorize actions or writes.
Recognized personal-history lookups stay within memory rather than falling back to
public search. An explicit request such as `Search online for ways to organize my
notes` still uses the web route. Exact local search/read requests also work with
the AI controller disabled. Reset and owner changes invalidate in-flight reads and
replies; an interrupted streamed reply is retained only in the unchanged conversation.

Keyword search uses local SQLite FTS5, Unicode matching, filename lookup, and bounded
chunks. Optional hybrid recall adds the configured local embedding client described
below. CIEL needs no downloaded embedding model or hosted embedding service for
keyword access. Dated notes receive recency weighting; curated files do not age out.

Search reconciles edits and deletions before retrieval; a running owner runtime also
polls for changes. Unchanged text and hashes use a bounded, stat-validated cache.
Structured fact checks inspect registered fact sources without scanning the archive.
Search still checks source visibility, permissions, and provenance. Slow embedding
work runs outside the search lock; a query waits at most 100 ms for optional inference
before using available keyword evidence. A later matching query can use completed
semantic scores. This inference budget does not cap source scanning or all SQL work.

Use `ciel memory index --force` to rebuild the derived workspace index, or
`ciel memory reset-index` to clear it. Both preserve Markdown and conversation
archives. Removing a source revokes its results even if index publication fails.

## Daily checkpoints and nightly review

CIEL stages accepted, authenticated owner statements when
`MEMORY_DAILY_CAPTURE_ENABLED=true`. It flushes useful statements to dated working
notes before conversation compaction, on clean conversation close, or through
`ciel memory flush`. `MEMORY_FLUSH_ENABLED=false` disables those checkpoints.
The flush keeps original attributed wording and avoids duplicate entries. Assistant
claims, tool/screen output, recalled passages, recognized secrets, and private `/local`
conversation are excluded from automatic capture/promotion.

`MEMORY_CONSOLIDATION_ENABLED=true` starts a stoppable scheduler in a running CIEL
process. `MEMORY_CONSOLIDATION_TIME=03:00` selects local OS wall time. The PC and a
CIEL runtime must be running; this creates no separate system cron job and performs
no missed-run catch-up on launch. Foreground work takes priority, and a busy or
unavailable provider causes a quiet five-minute retry.

Reflection first checks live owner evidence and recorded retrieval signals. A
candidate needs at least three recall signals from three distinct queries, a latest
recall within 30 days, and a weighted score of at least 0.75. Repeating one query on
one day counts once. Ranking combines retrieval relevance, frequency, query diversity,
recency, cross-day recurrence, and conceptual breadth, plus bounded light/REM-style
reinforcement. An unqueried daily log does not qualify through repetition alone.

Only then may the background provider propose sourced additions, merges, or
supersessions. CIEL validates a tool-free structured plan against original owner
quotes and live line references. In `automatic` learning mode, a nonconflicting
addition can enter `MEMORY.md` if it fits the curated/bootstrap budget. In `confirm`
or `explicit` mode, proposed additions wait for owner review. Merges, supersessions,
and possible conflicts require review in all modes. Managed current fact blocks
remain protected; use the normal confirmed correction commands for those fields.
Nightly curated output has a 4,000-character ceiling, reduced further by the shared
bootstrap budget and profile allowance. Larger owner-written files stay on disk;
CIEL clips the context excerpt rather than shortening the owner's source file.

Inspect `DREAMS.md` in your editor and pending actions through the CLI:

```bash
ciel memory flush
ciel memory consolidate
ciel memory review
ciel memory review RUN_ID
ciel memory apply RUN_ID --action 0
```

`consolidate` can contact the configured background AI route after the evidence
gates pass. `review` and `apply` run without an AI request. `apply RUN_ID` approves
all pending actions; repeat `--action INDEX` to select the displayed zero-based
actions. These lifecycle commands require the selected database's persisted owner
conversation binding from a CIEL launch; an isolated `--db` does not borrow the
normal database's owner context. Approval rechecks owner scope, source quotes,
reset generations, and the curated preimage. Changed or deleted evidence, a
concurrent owner edit, or an exceeded budget can reject the apply. CIEL keeps the
earlier curated text and logs completed changes; an approval does not override
managed current-fact protection.

The workspace-specific environment settings are:

| Setting | Default | Effect |
| --- | --- | --- |
| `MEMORY_WORKSPACE_ENABLED` | `true` | Use Markdown-backed durable memory |
| `MEMORY_WORKSPACE_PATH` | empty | Derive a `.workspace` sibling from `MEMORY_DB_PATH` |
| `MEMORY_WORKSPACE_BOOTSTRAP_MAX_CHARS` | `8000` | Upper bound before shared recall limits and source wrappers |
| `MEMORY_DAILY_CAPTURE_ENABLED` | `true` | Stage authenticated owner turns for working notes |
| `MEMORY_FLUSH_ENABLED` | `true` | Preserve useful staged context at checkpoints |
| `MEMORY_CONSOLIDATION_ENABLED` | `true` | Enable manual and scheduled reflection |
| `MEMORY_CONSOLIDATION_TIME` | `03:00` | Local OS wall-clock schedule while CIEL runs |
| `MEMORY_SEARCH_MAX_RESULTS` | `6` | Default workspace tool result count |
| `MEMORY_SEARCH_MIN_SCORE` | `0.35` | Default recall relevance threshold |

`MEMORY_RECALL_ENABLED`, `MEMORY_RECALL_MAX_CHARS`, and the learning/correction
settings still apply. A saved pause/resume choice overrides the initial recall flag.

## Personal context you control

`MEMORY_LEARNING_MODE` controls how clear, supported personal statements are saved:

- `automatic`: save new clear statements locally; ask before conflicting replacements.
- `confirm` (fresh-install default): ask before saving a conversational statement.
- `explicit`: save only `remember ...` commands.

Set `MEMORY_LEARNING_MODE=automatic` to opt into new current-fact saves without
a proposal. With `MEMORY_CONFIRM_CORRECTIONS=true`, changing an existing fact asks for confirmation,
including when the replacement starts with `remember`. Broad project-date inference
and short inferred corrections remain proposals. Questions, quotations, uncertain
wording, other people's statements, and compound requests are not automatic facts.
The recognizer makes no provider call and does not extract arbitrary conversation.
Daily working-note capture has its own `MEMORY_DAILY_CAPTURE_ENABLED` switch;
`explicit` learning mode still permits those checkpoints unless you disable them.

Supported statements include:

| Statement | Saved context |
| --- | --- |
| `My current project is Atlas` | Current project |
| `My goal is to finish the voice assistant` | Current goal |
| `I work as a software developer` | Occupation |
| `My favorite game is CS2` | Favorite game |
| `Call me Alex` | Preferred name |
| `I live in Ulaanbaatar` | Home city |
| `My timezone is Asia/Hovd` | Timezone |
| `My preferred language is Mongolian` | Preferred language |
| `I prefer concise answers` | Response style |

Say `yes` to save or `no` to decline. An unrelated request, cancellation, expiry,
or restart discards an unconfirmed proposal. An explicit `remember ...` command
saves a new fact immediately. Confirmed replacements retain up to 20 revisions.

Ask `What's my current project?`, `What's my goal?`, or `What's my occupation?`
for a local answer when a corresponding saved fact exists. After asking about a
supported topic, `No, it's Borealis` proposes a correction to that topic. A full
statement such as `Actually, my current project is Borealis` also works.
Explicit requests such as `Update my favorite game to Hades`, `Change my home
city to Tokyo`, and `Replace my current project with Borealis` use those same
saved fields. Polite requests such as `Can you update my favorite game to Hades?`
also work. These commands work in explicit-only mode; conflicting replacements
still ask for confirmation when configured. `Yes, update that`, `yep`, `save that`,
and `confirm it` confirm a live proposal. In ambient voice mode, include Ciel's
wake phrase when confirming. An incomplete or unsupported explicit memory update
asks for the specific detail and new value without claiming it was saved.
Conversational openings such as `By the way, my favorite game is Hades` and
`I told you before that my fav game is Hades` reach the same local handler.
After a question about one personal topic, `It's Hades, not Sable` proposes the
exact replacement rather than saving the rejected spelling with it. A repeated
`yeah yeah yeah` can confirm a live proposal. Ambient voice prompts include the
wake phrase needed to authorize the answer. Broad or multiple-topic questions,
quoted preferences, and assistant replies do not select a correction target.
Corrections update one current record, with revision history, rather than leaving
competing current values. Use `forget my current project`, `forget my goal`, or
`forget my occupation` to remove the selected record.

`What do you remember about me?` and `What do you know about me?` now give a bounded
overview of current saved personal fields directly from the database. Old chat
claims, prior revisions, unconfirmed proposals, and arbitrary notes are excluded.
This overview respects recall pause and works without an AI connection. It lists
some saved profile details, rather than claiming to summarize every conversation.

Saved profile details accompany informal conversation as bounded, dated
context. A saved goal describes what you said when it was recorded; the persona
should check an old plan's relevance before treating it as a present commitment.
Occupation and personal goals describe the user, not system capabilities.
Other notes still use `remember ...`. Automatic mode records the source as
`user_statement`; confirmation records `user_confirmed`; explicit commands record
`user_explicit`. These remain user statements rather than independently verified facts.

## Pause recall and inspect saved memory

Say `pause memory recall` or `resume memory recall`. The setting survives restart.
Pausing keeps saved facts and learning active, but omits saved facts, workspace
bootstrap, and the saved preferred name from reply context. It disables memory
retrieval tools and local saved-fact answers. Recent
dialogue and conversation summaries still support follow-ups. Ask `Memory status`
for an application-owned answer. A persisted pause/resume overrides the initial
`MEMORY_RECALL_ENABLED` setting.

Owner tools work without loading audio or contacting a provider:

```bash
ciel memory stats
ciel memory status
ciel memory list --limit 20
ciel memory search Atlas next step
ciel memory get MEMORY.md --from 1 --lines 40
ciel memory index --force
ciel memory reset-index
ciel memory show user:current_project
ciel memory history user:current_project
ciel memory project Atlas
ciel memory pause
ciel memory resume
ciel memory backup /path/to/new-memory-backup.db
```

Use `ciel memory --db /path/to/memory.db --json list` for an explicit database
and structured output. `--db` selects its sibling workspace even if your normal
configuration points elsewhere; add `--workspace /path/to/notes` to override it.
Place these global options before the subcommand. Owner CLI inspection remains
available while recall is paused. `status` reports configured lifecycle settings
and the keyword index; it does not measure a separate live process's watcher.

Backups combine SQLite's online backup with a sibling Markdown snapshot. For
`backup /path/to/snapshot.db`, keep both `snapshot.db` and `snapshot.workspace/`.
The directory includes source notes, provenance, `DREAMS.md`, and preimages;
CIEL omits the rebuildable `.ciel/search.sqlite3` cache. It refuses existing
destinations and cleans up an incomplete new backup after a detected source change.
The SQLite file retains facts, revisions, archived dialogue, and saved preferences.
CIEL creates private Linux files/directories (0600/0700); keep the pair private.
Stop CIEL before restoring a saved pair. Point `MEMORY_DB_PATH` at the restored
database and clear or update a custom `MEMORY_WORKSPACE_PATH` so both paths match.
`ciel memory forget ID_OR_KEY` forgets one fact. `ciel memory clear-session SESSION`
erases only that session's archived dialogue and summaries, keeping personal facts.
Forgetting also redacts known matching wording from working notes, review logs,
and preimages, removes matching staged observations, and invalidates pending
reflection plans. Existing backup copies and archived raw conversation text remain.
Paraphrases, unknown spellings, and forensic recovery from storage are outside this
targeted suppression guarantee. Neither command changes older backup files.

## Project decisions and unfinished work

Give project journal entries an explicit label so CIEL can distinguish decisions
from suggestions and keep different projects separate:

```text
Project Atlas decision: use SQLite
Project Atlas constraint: work offline
Project Atlas next step: test restart recovery
Project Atlas abandoned: the hosted database plan
Project Atlas status: active
Project Atlas goal: ship a desktop prototype
```

Automatic mode saves these new declarations; confirmation mode asks first.
Decisions, constraints, and abandoned choices accumulate as separate entries.
The next step, status, and goal each keep one current value per project, with
confirmation before replacement. Ask about `Atlas next step` to retrieve relevant
entries, or inspect the whole journal with `ciel memory project Atlas`. Completion
is not inferred from an assistant claim. Conversation summaries continue to retain
unlabelled decisions and open threads as fallible dialogue context.
Unfinished work should be mentioned when that project is relevant to the current
conversation, without repeatedly raising it during unrelated topics.

## Optional local semantic recall

Semantic recall supplements indexed word matching with an explicitly selected local
Ollama embedding model. It uses the documented
[local embedding API](https://docs.ollama.com/api/embed),
requests CPU execution, accepts only loopback HTTP endpoints, and disables proxy
and redirect handling. It never downloads a model or sends embeddings to a hosted
service. The default configuration leaves it disabled.

First install and run a suitable small embedding model with your local Ollama
installation. Replace `MODEL_NAME` with its installed model name:

```bash
ciel memory semantic-evaluate --model MODEL_NAME
ciel memory semantic-index --model MODEL_NAME
```

Evaluation uses 12 synthetic memories and 17 labeled queries, including paraphrases,
exact terms, and unrelated questions. It records acceptance only when top-one
accuracy strictly improves, previously correct results remain correct, and unrelated
queries produce no additional false matches. Unavailable models, failed requests,
incomplete indexes, and equal scores fail the gate. This small acceptance set is not
a claim of general model quality. The report includes local query timings.

After a passing evaluation and indexing, set `MEMORY_SEMANTIC_MODEL=MODEL_NAME` and
`MEMORY_SEMANTIC_ENABLED=true`, then restart CIEL. Runtime checks the recorded
acceptance for that model before using it. Re-evaluate after changing model weights
and reindex the structured fact vectors after adding or changing facts. Workspace
hybrid search uses the same accepted local client and refreshes its own derived
document vectors in the background. Corrected, forgotten, obsolete, or hash-mismatched
records cannot be recalled from stale vectors. Pausing recall also suppresses semantic recall.
Low-similarity results are discarded; the first lexical match remains an anchor.
When the local service fails, recall falls back to the lexical index, with a short
backoff before another embedding request.

`scripts/benchmark_memory_recall.py --model MODEL_NAME --output /tmp/recall.json`
runs the same synthetic evaluation without approving or modifying a personal
database. Keep semantic recall disabled until your installed model passes the gate.

## Continuity inside conversations

Voice and the currently paired Telegram account read one chronological
conversation within the current foreground launch. Each new `ciel` launch starts
fresh desktop/chat archives while keeping saved personal facts and old archives.
Earlier launches' dialogue and summaries stay out of automatic replay; a memory
lookup can retrieve bounded user excerpts from the same owner's previous launches. A running
Telegram worker follows the new launch and discards old pending confirmations.
Messages and summaries carry their origin and recorded time. Unpairing or changing
the paired account excludes its old chat archive immediately. Clearing one channel
also invalidates shared summaries containing it, preserving the other channel's
history. Confirmations remain in the channel where the fact was proposed.

If a summary is deferred or invalid, a bounded tail of up to 32 unsummarized turns
can remain available within the existing character budget. Older user messages
are searchable by topic through a local persisted word index. Current-launch
context supplies ordinary continuity; old launches require a memory lookup. Matching
passages include their recorded date and channel, labeled as historical dialogue,
not confirmed current facts. Assistant claims are not indexed as personal evidence.
Corrected values, forgotten details, and known secret tokens are excluded from
this historical recall. This does not automatically save new personal facts.
An adjacent explicit user choice can accompany its topic even if it omits the
topic's name; each message retains its own date and attribution.

Confirmed current facts remain ahead of historical dialogue in recalled context.
Both share `MEMORY_RECALL_MAX_CHARS`; historical excerpts use at most 2,000 of
those characters and three results. Recall pause applies to both. No embedding
download or hosted request is needed to search older user messages.
A synthetic 10,000-turn archive check on the development PC took 0.217 seconds
for initial indexing/query and 0.009 seconds for the indexed query. This is a
local measurement, not a guarantee for other hardware or archive sizes.

An explicit "do not search" carries across the owner's channels and process
restarts. A fresh explicit lookup resumes browsing. Resetting a channel removes
only its own browsing preference. Previously marked private conversations retain
their provider and web-access restrictions until their history is cleared.

Short references such as `What about that?`, `Tell me more`, and `Continue` can
use the latest substantive user topic to retrieve relevant saved notes. A clear
new topic uses its own words. Assistant claims are not used to select personal
facts for this contextual lookup. Default recall uses an indexed local word/alias
search over the full durable collection, including older records beyond the former
newest-2,000 cutoff. Unicode normalization supports equivalent text representations.
Current keyed values replace obsolete duplicates, and duplicate note content is
not repeated in the prompt. No embedding model or provider call is required.

Recall uses up to eight recent messages to resolve references, clips individual
notes, and limits recalled text to `MEMORY_RECALL_MAX_CHARS` (6,000 by default).
`MEMORY_RECALL_LIMIT` controls query matches; a bounded set of profile/project
anchors supplies continuity separately. Replayed summaries and recent dialogue
share `MEMORY_MAX_CONTEXT_CHARS` (24,000 by default), excluding the system prompt.
Older messages are removed first, the latest user request is retained, and a single
oversized message keeps its beginning and ending. These are character budgets,
not exact model-token counts. Private routing survives trimming and restart.

`Development date`, `Development day`, and an explicit correction such as
`I meant the development date` use the confirmed CIEL project start date locally.
The reply includes the start date and current inclusive development day. A phrase
such as `It's in your memory` or `Check your memory` directs CIEL to saved context
instead of another web search. It resolves a known timeline question from the
latest user topic; assistant text and older commands cannot select that topic.
A clear memory question supplied during an interruption also takes priority over
the earlier garbled speech fragment. The original fragments remain available as
conversation context. These reads do not save, change, or forget facts.

If the controller returns an invalid decision during an active conversation,
CIEL asks for a repeat rather than silently dropping an ordinary follow-up.
Background speech, speech explicitly addressed to someone else, strict wake-word
follow-ups, and cancellation keep their existing gates. Known memory reads can
still use local evidence after a classification failure. Validation warnings
include field names and error types without model output or private input values.

Older dialogue is summarized into a validated JSON state:

| Field | Purpose |
| --- | --- |
| `overview` | Current topic |
| `user_context` | Attributed goals, constraints, and preferences mentioned in dialogue |
| `decisions` | Choices made by the user |
| `open_threads` | Unanswered questions and unfinished work |
| `superseded` | Corrections and abandoned choices that should not be revived |

The summary instructions ask the provider to remove resolved threads and merge
relevant earlier context. Field types are strict, and the complete serialized
state has a hard 4,000-character budget. Per-field length/count targets guide
generation; complete notes fitting the total budget are retained without truncation.
Invalid JSON,
unexpected fields, empty summaries, and failed requests leave the processing
cursor unchanged for retry. The summary request allows up to 600 output tokens,
up from 150, so there is room for more than a one-sentence recap. Foreground
requests retain priority over this background work.

Existing plain-text summaries remain usable and can be merged into the next
structured summary. Providers returning plain text retain a bounded compatibility
path. Structured state is stored as conversation context, never as automatically
confirmed personal facts. Existing archives remain intact. Foreground launches
now use a new conversation identity rather than resuming the preceding launch.

Known forgotten wording is masked in recalled notes, derived timelines, local
saved-fact answers, conversation history, summaries, and speech-gate requests.
Archived SQLite text remains retained, and paraphrases
outside known wording are not guaranteed to be erased. Privacy restrictions on
previously marked conversations continue through replies, background summaries,
context trimming, and restarts.
Session clears and summary updates use persistent generations and an atomic cursor
check: a stale summary worker cannot recreate cleared context or overwrite a newer
summary, including across separate managers. Legacy metadata damage is treated as
missing metadata; the term index is rebuilt once per index-version migration.

## Personality and system prompt

The prompt now composes personality, continuity, speech understanding, capability
limits, runtime context, recalled data, and the user's editable preferences as
separate sections. The default prompt is approximately 1,150 words, compared with
approximately 2,100 previously; custom preferences and recalled data add to it.
This is a reduction in default prompt size, not a measured latency claim.

The intended behavior is warm, mature, feminine, curious, and practical, with
expressive reactions, light wit when welcomed, and reasoned opinions. Replies
should answer the actual request first, build on previous answers, and vary their
wording. Initiative should connect a relevant interest or unfinished thread to
the current conversation, while accepting topic changes and declined questions.
Frustration calls for patient help rather than teasing; brief replies should not
turn every conversation into an interview.

The current request and latest correction take priority over stale context.
Generated summaries are fallible; runtime telemetry and confirmed facts carry
their own sources. Generated replies cannot claim that memories were saved or
actions succeeded without application confirmation. Natural delivery does not
require invented experiences, emotions inferred from raw audio, or shared history.

Your existing `persona.txt` is preserved and continues to load before each reply.
Style preferences there override default delivery examples. The upgrade changes
the application's prompt; it does not replace your private persona file or alter
the voice model. Restart CIEL to load the new application code.

## Conversation acceptance

Automated tests exercise confirmation, corrections, restart recovery, contextual
recall, summary validation, forgetting, and the real CLI callbacks using mocked
providers and audio. They do not prove that a particular provider's dialogue
feels natural. Tests use isolated databases and mocked providers. Workspace tests
cover migration, source edits, line-addressed retrieval, owner/privacy isolation,
bounded caching, nightly score gates, and review/apply conflicts. See
[release checks](release-checks.md) for recorded command results and pending
acceptance. A live local embedding model, scheduled desktop run, and microphone
acceptance remain separate.
Check these scenarios with your selected model:

1. Save a project and goal, restart, then ask what you're working toward.
2. Ask about a saved note, say `What about that?`, then change to a different topic.
   Expect continuity first and a clean topic change afterward.
3. Make a decision, leave a question unfinished, then continue after a long
   conversation. Expect the decision and open question to remain available.
4. Correct or abandon a plan. Expect the latest wording to guide the answer and
   the old plan to stay abandoned.
5. Discuss an interest, decline a personal follow-up, and give a brief response.
   Expect specific reactions and space, without repeated prompting.
6. Forget a fact that appears in another note, then ask about that note. Expect
   known forgotten wording to remain masked.
