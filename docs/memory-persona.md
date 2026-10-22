# Memory and personality

CIEL combines a stable companion personality with attributed personal facts,
recent dialogue, and a running account of older conversations. The upgrade keeps
those sources distinct so continuity does not become invented certainty.

## Personal context you control

`MEMORY_LEARNING_MODE` controls how clear, supported personal statements are saved:

- `automatic`: save new clear statements locally; ask before conflicting replacements.
- `confirm` (fresh-install default): ask before saving a conversational statement.
- `explicit`: save only `remember ...` commands.

The selected desktop configuration uses `automatic` and
`MEMORY_CONFIRM_CORRECTIONS=true`. Changing an existing fact asks for confirmation,
including when the replacement starts with `remember`. Broad project-date inference
and short inferred corrections remain proposals. Questions, quotations, uncertain
wording, other people's statements, and compound requests are not automatic facts.
The recognizer makes no provider call and does not extract arbitrary conversation.

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
Pausing keeps saved facts and learning active, but omits saved facts and the saved
preferred name from reply context and disables local saved-fact answers. Recent
dialogue and conversation summaries still support follow-ups. Ask `Memory status`
for an application-owned answer. A persisted pause/resume overrides the initial
`MEMORY_RECALL_ENABLED` setting.

Owner tools work without loading audio or contacting a provider:

```bash
ciel memory stats
ciel memory list --limit 20
ciel memory search Atlas next step
ciel memory show user:current_project
ciel memory history user:current_project
ciel memory project Atlas
ciel memory pause
ciel memory resume
ciel memory backup /path/to/new-memory-backup.db
```

Use `ciel memory --db /path/to/memory.db --json list` for an explicit database
and structured output. Inspection remains available while recall is paused.
Backups use SQLite's consistent online backup and refuse to overwrite files.
They contain facts, revision history, archived dialogue, and saved preferences;
keep them private. New backup files are readable only by their owner on Linux.
`ciel memory forget ID_OR_KEY` forgets one fact. `ciel memory clear-session SESSION`
erases only that session's archived dialogue and summaries, keeping personal facts.
Neither command changes older backup files.

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
Ollama embedding model. It uses the documented [local embedding API](https://docs.ollama.com/api/embed),
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
and reindex after adding or changing facts. Indexing stays outside the conversation
response path. Corrected, forgotten, obsolete, or hash-mismatched records cannot be
recalled from stale vectors. Pausing recall also suppresses semantic recall.
Low-similarity results are discarded; the first lexical match remains an anchor.
When the local service fails, recall falls back to the lexical index, with a short
backoff before another embedding request.

`scripts/benchmark_memory_recall.py --model MODEL_NAME --output /tmp/recall.json`
runs the same synthetic evaluation without approving or modifying a personal
database. The initial desktop check found no reachable local embedding service, so
semantic recall remains disabled until an actual model passes evaluation.

## Continuity inside conversations

Voice and the currently paired Telegram account now read one chronological
conversation. Existing archives stay intact; no migration or copying is needed.
Messages and summaries carry their origin and recorded time. Unpairing or changing
the paired account excludes its old chat archive immediately. Clearing one channel
also invalidates shared summaries containing it, preserving the other channel's
history. Confirmations remain in the channel where the fact was proposed.

If a summary is deferred or invalid, a bounded tail of up to 32 unsummarized turns
can remain available within the existing character budget. Older user messages
are also searchable by topic through a local persisted word index. Matching
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
relevant earlier context. Each field has size and type limits. Invalid JSON,
unexpected fields, empty summaries, and failed requests leave the processing
cursor unchanged for retry. The summary request allows up to 600 output tokens,
up from 150, so there is room for more than a one-sentence recap. Foreground
requests retain priority over this background work.

Existing plain-text summaries remain usable and can be merged into the next
structured summary. Providers returning plain text retain a bounded compatibility
path. Structured state is stored as conversation context, never as automatically
confirmed personal facts. The session identity is unchanged, so the upgrade
continues the existing conversation without deleting or resetting it.

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
feels natural. The complete suite passed 1,488 tests, with five integration cases
excluded, plus Ruff and dependency checks. Tests use isolated databases and mocked
providers; a live local embedding model and microphone acceptance remain separate.
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
