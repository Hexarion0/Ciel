# Apollo ideas for Ciel

Reviewed 2026-10-27 from `/mnt/storage/shared/APOLLO`. This is a source review;
Apollo was not executed. Credentials, personal configuration, databases, and chat
logs were not read. No Apollo code has been copied into Ciel.

| Priority | Idea | Benefit for Ciel | Apollo reference |
| --- | --- | --- | --- |
| First | Track unfinished user tasks separately from personal facts | Ciel can recall what still needs attention instead of reciting saved preferences | `apollo/scheduler/thought_engine.py:51`, `:127`, `:178` |
| Next | Publish real tool lifecycle events | Show checking, completed, failed, or cancelled from actual execution; avoid another model request for progress wording | `apollo/gateway.py:415`, `:475`, `:515` |
| Next | Add dedicated structured tools for predictable data | Weather can use geocoding and forecast fields instead of generic web snippets | `apollo/tools/internet.py:222` |
| Later | Share durable reminders across voice and Telegram | A reminder can survive restart and reach the owner's chosen channel | `apollo/tools/reminders.py:82`, `apollo/scheduler/cron.py:39` |
| Desktop phase | Use a bridge running inside the graphical session | Background Ciel can reach Wayland screenshots, media, and eventual desktop controls | `apollo/bridge/server.py:4`, `apollo/bridge/client.py:20` |

Apollo also validates the current main-model architecture: it supplies native tool
schemas to the response model, executes calls, and returns results in the same
conversation (`apollo/gateway.py:405`, `:423`, `:549`). Ciel's new default flow follows
that structure while retaining its own privacy, evidence, and original-request guards.

## Adaptation details

Unfinished tasks should record the user's actual requested outcome, latest progress,
next useful step, and completion/cancellation state. A fresh conversation may recall
a relevant unresolved task without replaying an old conversation. Suggested check-ins
should respect the user's notification preferences and stop when unanswered.
For Ciel, preserve the agreed minimum 15-minute gap and suppress initiative after
one ignored check-in until the owner reengages.

Apollo's check-in engine backs off by 2.5 and stashes after two attempts. Its pending
thought records are stored, but their timers use `persist=False`; startup restores
ordinary scheduled jobs, not these pending-thought timers. Ciel would need explicit
restart recovery. These are scheduled follow-ups, not independent reasoning.

Reminders also need absolute due times, owner/channel binding, and delivery receipts.
Apollo restores interval jobs from their interval, which alone does not guarantee
that a one-shot reminder keeps its original due time after restart.

The desktop bridge idea fits the previously deferred desktop phase. Adapt it with
authenticated local access and explicit capture scope. Apollo defaults to open access
when its token is empty (`bridge/server.py:40`), and can fall back from active-window
capture to the full screen (`:95`); those defaults should not be carried over.

## Features already stronger in Ciel

Apollo's search uses one DuckDuckGo HTML endpoint (`tools/internet.py:77`). Its memory
lookup uses SQL `LIKE` (`memory/store.py:113`). Ciel already has selected search
providers, fetched-page verification, owner-scoped workspace retrieval, and private
routing. Apollo's roadmap pooling and parallel-tool items are unchecked plans, not
implemented performance improvements.

Keep Ciel's action allowlists and memory-write authorization. Apollo's generated
shell/file arguments and its fallback phrase “Completed requested actions” after
synthesis failure (`gateway.py:579`) would weaken Ciel's truthful result handling.
