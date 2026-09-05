# The agent factory

A prospect brief in, a deployed ElevenLabs agent out. The `/agents` page
takes a company, a sentence or two about what the agent is for, up to six
pages to read, and any pasted material, and has your selected model draft the
whole agent: the system prompt, the opening line, knowledge documents, three
to five simulation tests, and a demo page. You review every field, press
Deploy, confirm once, and Metis creates the agent in your ElevenLabs account,
runs its tests, and serves a standalone demo page with the widget on it.

This is the one place Metis creates resources outside the machine. It is
built so that it cannot do so quietly.

## The shape

```
brief ─► draft (model, local) ─► review (you) ─► deploy (confirmed) ─► test
```

- **Draft** reads the pages through the same web client chat uses, joins them
  with the notes, and asks the model for a structured draft. The model never
  chooses a URL and never chooses a voice: the host adds the pages it was
  handed as URL documents and the configured stock voice, so an agent cannot
  be pointed at a page nobody supplied. Drafting reaches ElevenLabs for
  nothing.
- **Review** is the approval surface. What the page shows is exactly what is
  stored, and what is stored is exactly what deploys — the spec is one JSON
  column, and the deploy sends it and nothing else.
- **Deploy** is refused without `confirm: true`. The page asks once, naming
  what it is about to create: the agent (or which one it will update), how
  many documents, how many tests. Every id is recorded in the row as it is
  created, before the next call, so a failure halfway leaves a row that knows
  what to take back rather than orphans in the account. A redeploy replaces
  the documents and tests and patches the same agent, so the id — and any
  widget already pointing at it — stays stable.
- **Test** runs the agent's simulation tests through the platform's own
  testing API and polls until they settle. Results are derived data with a
  small stage machine (`'' → running → ready | failed`) and a Run tests
  button that reruns or joins.
- **Remove** deletes the tests, the documents, and the agent, in that order,
  and only then the row. A cleanup ElevenLabs refuses keeps the row so it can
  be tried again.

## The demo page

`GET /api/v1/agents/{id}/demo` renders one self-contained HTML file: the
headline, the opening line, things to try, and the platform's embed widget.
Save it, send it, open it anywhere. The same two-line snippet is shown on the
page for pasting into a prospect's own site.

The widget needs a public agent, which is the platform's own rule, so a
deploy sets `platform_settings.auth.enable_auth` to false explicitly rather
than assuming the default.

## Configuration

One setting, already used by dictation, meetings, voice and interviews:

```
WAQIL_ELEVENLABS_API_KEY=
```

The drafting model is whatever you selected in Settings. The voice is
`WAQIL_ELEVENLABS_VOICE_ID`; the TTS model is `WAQIL_ELEVENLABS_TTS_MODEL`.
The agent's LLM is the platform default unless the review screen picks one.

## API surface

| Route | What it does |
| --- | --- |
| `GET /api/v1/agents/availability` | Whether the factory can run; names the missing key |
| `GET /api/v1/agents` | Every agent, newest first |
| `POST /api/v1/agents/drafts` | Read the material, draft the agent, store it |
| `GET /api/v1/agents/{id}` | One agent with its spec and test results |
| `PUT /api/v1/agents/{id}/spec` | The reviewed spec, replacing the draft |
| `POST /api/v1/agents/{id}/deploy` | Create or update in ElevenLabs; 409 without `confirm` |
| `POST /api/v1/agents/{id}/tests` | Run or join the simulation tests |
| `GET /api/v1/agents/{id}/demo` | The standalone demo page |
| `DELETE /api/v1/agents/{id}` | Remove everything created, then the row |

Persistence is the `voice_agents` table (schema v33).

## Where things live

- API: `apps/api/src/waqil_api/agent_factory.py` (the service and the demo
  page), `elevenlabs_agents.py` (the thin API client), contracts in
  `contracts.py`, routes in `api.py`, table in `database.py`.
- Web: `apps/web/app/agents/page.tsx`, `components/agent-factory.tsx`, pure
  helpers in `lib/agents.ts`, calls in `lib/api.ts`.
- Tests: `apps/api/tests/test_agent_factory.py`,
  `apps/web/tests/agents-api.test.ts`.

## Manual test script

1. With `WAQIL_ELEVENLABS_API_KEY` unset, open `/agents`: the brief form
   names the missing variable and Draft is disabled.
2. Set it. Draft an agent from a real prospect page and a pasted FAQ. Check
   the material reached the model: the prompt and knowledge should reflect
   the page, not generic filler, and the URL document should be listed under
   Knowledge with the page's address.
3. Edit the first message and press Deploy. The confirmation must name one
   agent, the document count and the test count. Press Not now: nothing
   appears in the ElevenLabs dashboard. Press Deploy: the agent, documents
   and tests appear, and the page shows the agent id.
4. Wait for the tests. Each result carries the platform's rationale. Press
   Run tests again and confirm it joins rather than doubling the run.
5. Open the demo page in a new tab and talk to the widget. Copy the embed
   snippet into a blank HTML file and confirm it works there too.
6. Change the prompt and Redeploy: the agent id must not change, and the
   dashboard should show the old documents and tests replaced.
7. Remove the agent: the dashboard is clean and the row is gone.
