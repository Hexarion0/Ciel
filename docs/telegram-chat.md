# Telegram chat

```bash
ciel chat setup
ciel chat pair telegram
```

Setup also offers Telegram in the main `ciel setup` flow. Use
`ciel chat status` to inspect selection, pairing, and automatic startup. Configured
chat starts after setup, after pairing, whenever Ciel opens, and when you log in.
There is no manual-start choice or autostart toggle. Pairing and unpairing pause
and resume the managed worker automatically. Deselect the service in setup to
disable it while keeping the saved token.

The first remote channel is Telegram text chat, running on the user's PC. It can
start independently of voice mode. The PC must be on, logged in, and connected
to answer or deliver future reminders. Automatic startup requires Linux systemd
user services; failures are reported and retried rather than presented as success.

## First version

- Choose chat services during setup, like AI providers: choose the service before
  entering its credentials. Telegram is the first supported adapter; Discord and
  Signal remain future work. Keep existing choices or disable chat without
  deleting credentials.
- Offer a separate `ciel chat setup` path without microphone or speech-model
  prerequisites. Credentials stay in the local `.env` file.
- Pair a single owner through an expiring code displayed on the PC. Accept only
  that owner's private human messages. Group messages, forwards, and edits do not
  authorize work.
- Reuse Ciel's persona, provider routing, conversation storage, memory reads,
  guarded corrections, and public-web retrieval. Share saved facts with voice
  mode. Both channels read a chronological view of the desktop and currently
  paired account; original archives remain separate. Pending memory confirmations
  remain in the channel where they were proposed.
- Use a text-specific capability prompt. This version cannot operate desktop apps,
  process Telegram audio, or create reminders. It must not claim those actions
  succeeded.
- Automatically manage the Linux user service after setup saves. Keep credentials
  out of its definition. Old `CHAT_AUTOSTART=false` settings no longer suppress a
  selected service.

## Boundaries

The channel adapter owns Telegram polling, sender verification, pairing, message
delivery, and transport state. A headless conversation session owns text requests
and delegates to shared conversation components. Wake detection, playback,
captions, and microphone state stay in the desktop runtime.

Telegram receives and transports chat text. Existing AI-provider choices still
apply.

The chat service acknowledges incoming updates through a persisted offset and
claims each authorized request before processing it. Repeated updates must not
repeat memory writes. A crash can lose a reply; the service must not promise
exactly-once network delivery. Stale queued requests are skipped rather than
executed unexpectedly after a long offline period. Network errors and conflicting
pollers produce actionable messages without exposing the bot token.

Incoming requests older than five minutes are ignored. After a quiet week, the
poll cursor can recover from Telegram's randomized update numbering while keeping
recent duplicate receipts. These transport rules follow the
[Bot API update contract](https://core.telegram.org/bots/api#update).

In chat, `/memory` shows current saved personal details, `/cancel` discards an
unconfirmed fact, and `/reset` clears only this channel's conversation. Natural
memory corrections use the existing learning and confirmation policy. "Wait,"
"don't," and "do not search" are local conversation controls. Online lookups stay
paused until a fresh explicit search request. The most recent browsing control
carries between the owner's voice and chat
channels and survives process restart. Resetting a channel clears its own control.

## Conversation continuity

Voice and Telegram share recent conversation, topic-based recall of older user
messages, and a running summary for the current paired owner. Origins and recorded
times distinguish the channels. Confirmed facts remain authoritative; old dialogue
is historical context and never an automatic fact or permission to act.

Unpairing or changing owner excludes the previous chat archive from future context.
No archive migration or deletion is needed. `/reset` clears the Telegram archive
and invalidates summaries that included it, keeping voice history and saved facts.
Local-only routing stays enforced across the shared context and its summaries.

## Validation

Use mocked Telegram HTTP requests and temporary databases to check pairing,
owner restrictions, duplicate updates, restart behavior, token redaction, message
splitting, retries, cancellation, and shutdown. Check shared-memory corrections
and conversation continuity across separate archives. Verify setup cancellation
leaves configuration and autostart untouched. Verify chat help, setup, and text conversation do not import
or initialize the audio backend or speech models.

A live acceptance check requires the user's own bot token and owner pairing.
Automated checks do not establish live Telegram delivery or autostart on the
target desktop.

Initial validation on 2026-10-21: 2,456 automated tests passed, with five hardware/model
integration tests deselected. Ruff, wheel build, and systemd's read-only unit
verifier passed. Synthetic live NIM checks returned conversational opinions and
kept reflective follow-ups out of web search. No Telegram token was configured,
no real Telegram messages were sent, and no login service was installed by these checks.

The subsequent automatic-start/continuity upgrade passed 2,506 automated tests
with five integration tests deselected, Ruff, wheel build, and systemd unit
verification. Two live NIM checks using temporary synthetic voice/chat archives
correctly continued the voice topic and recalled its earlier decision after it
left the recent window. These checks did not contact Telegram or install a service.

## Next

Add persistent reminders after text chat is accepted. Store due times and delivery
state, support list/cancel and restart recovery, then deliver to the paired owner
through the same adapter. Recurrence and additional chat services come afterward.
