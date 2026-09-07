"use client";

// Building the room: the role, the round, and what to push on. The last
// time the user types context; after Start it is a stage to talk to.

import { Notice } from "@/components/ui/notice";
import { INTERVIEW_TYPES, parseFocusAreas, validateInterviewDraft, type InterviewDraft } from "@/lib/interviews";
import type { InterviewAvailability } from "@/lib/types";

const QUICK_FOCUS = ["Specific evidence", "Architecture trade-offs", "Leadership judgment", "Failure recovery", "First 90 days", "Stakeholder communication"];

export function InterviewSetup({ draft, onChange, availability, canBegin, onBegin }: {
  draft: InterviewDraft;
  onChange: (draft: InterviewDraft) => void;
  availability: InterviewAvailability | null;
  canBegin: boolean;
  onBegin: () => void;
}) {
  const problems = validateInterviewDraft(draft);
  const focus = parseFocusAreas(draft.focus_areas);
  const set = (field: keyof InterviewDraft, value: string) => onChange({ ...draft, [field]: value });
  const toggleFocus = (area: string) => {
    const has = focus.some((item) => item.toLowerCase() === area.toLowerCase());
    const next = has ? focus.filter((item) => item.toLowerCase() !== area.toLowerCase()) : focus.length < 6 ? [...focus, area] : focus;
    set("focus_areas", next.join(", "));
  };
  const field = (label: string, key: keyof InterviewDraft, control: React.ReactNode) => (
    <label className="ui-field">
      <span>{label}</span>
      {control}
      {problems[key] && draft[key] ? <small role="alert">{problems[key]}</small> : null}
    </label>
  );

  return (
    <form className="interview-setup" aria-label="Interview setup" onSubmit={(event) => { event.preventDefault(); if (canBegin) onBegin(); }}>
      {availability && !availability.available ? (
        <Notice kind="info" title="Interviews aren't configured yet"><ul>{(availability.missing.length ? availability.missing : [availability.reason]).map((item) => <li key={item}>{item}</li>)}</ul></Notice>
      ) : null}
      <div className="interview-setup-grid">
        <div className="interview-form">
          <section>
            <h2><span>01</span> The role</h2>
            <div className="interview-row">
              {field("Job title", "job_title", <input value={draft.job_title} maxLength={200} placeholder="Senior Data Engineer" onChange={(event) => set("job_title", event.target.value)} />)}
              {field("Company", "company_name", <input value={draft.company_name} maxLength={200} placeholder="Batelco" onChange={(event) => set("company_name", event.target.value)} />)}
            </div>
            {field("Job description", "job_description", <textarea rows={10} value={draft.job_description} placeholder="Paste the whole job description. Every question is drawn from it." onChange={(event) => set("job_description", event.target.value)} />)}
          </section>
          <section>
            <h2><span>02</span> The round</h2>
            <div className="interview-rounds" role="radiogroup" aria-label="Interview round">
              {INTERVIEW_TYPES.map((round) => (
                <label key={round.value} className={`interview-round${draft.interview_type === round.value ? " is-selected" : ""}`}>
                  <input type="radio" name="interview_type" value={round.value} checked={draft.interview_type === round.value} onChange={() => set("interview_type", round.value)} />
                  <strong>{round.label}</strong>
                  <span>{round.hint}</span>
                </label>
              ))}
            </div>
          </section>
          <section>
            <h2><span>03</span> Where to push <small>optional</small></h2>
            {field("What should this round probe?", "interview_objective", <textarea rows={3} value={draft.interview_objective} placeholder="e.g. Press on every architecture trade-off behind choosing ElevenLabs over building a voice stack." onChange={(event) => set("interview_objective", event.target.value)} />)}
            <div className="interview-quick">
              {QUICK_FOCUS.map((area) => {
                const on = focus.some((item) => item.toLowerCase() === area.toLowerCase());
                return <button key={area} type="button" className={`ui-chip${on ? " is-accent" : ""}`} aria-pressed={on} onClick={() => toggleFocus(area)}>{area}</button>;
              })}
            </div>
            {field("Focus areas", "focus_areas", <input value={draft.focus_areas} placeholder="up to six, comma separated" onChange={(event) => set("focus_areas", event.target.value)} />)}
          </section>
        </div>

        <aside className="interview-brief" aria-label="Session brief">
          <span className="ui-eyebrow">Session brief</span>
          <strong>{draft.job_title.trim() || "Your target role"}</strong>
          <span>{draft.company_name.trim() || "Company"} · {INTERVIEW_TYPES.find((round) => round.value === draft.interview_type)?.label ?? "Choose a round"}</span>
          <dl>
            <div><dt>Format</dt><dd>5 questions with live follow-ups</dd></div>
            <div><dt>Score</dt><dd>Full at 5 · provisional from 3</dd></div>
            <div><dt>Delivery</dt><dd>Pace, fillers, hedging, pauses</dd></div>
          </dl>
          {focus.length ? <div className="interview-brief-focus">{focus.map((area) => <span key={area} className="ui-chip">{area}</span>)}</div> : null}
          {draft.interview_objective.trim() ? <blockquote>{draft.interview_objective.trim()}</blockquote> : null}
          <button type="submit" className="ui-btn is-primary is-lg" disabled={!canBegin}>Begin the interview</button>
          <small>Chiron stays in character and scores only after the round.</small>
        </aside>
      </div>
    </form>
  );
}
