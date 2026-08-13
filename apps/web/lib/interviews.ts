// Pure interview logic: context validation, the local draft, question-number
// tracking, and the client-side gate on the evaluation payload. No React and
// no fetch, so every rule here is testable in the Node test runner exactly as
// the browser runs it.

import type {
  InterviewContext,
  InterviewEvaluation,
  InterviewImprovement,
  InterviewType,
} from "./types.ts";

export const INTERVIEW_QUESTION_LIMIT = 5;

export const INTERVIEW_TYPES: Array<{
  value: InterviewType;
  label: string;
  hint: string;
}> = [
  {
    value: "hiring_manager",
    label: "Hiring manager",
    hint: "Role fit, ownership, judgment under ambiguity. The hardest room.",
  },
  {
    value: "hr_recruiter",
    label: "HR / recruiter",
    hint: "Background, motivation, fit, and the screening concern you'd rather avoid.",
  },
  {
    value: "technical",
    label: "Technical",
    hint: "Depth in a named requirement, tradeoffs, failure. No trivia.",
  },
];

export interface InterviewDraft {
  job_title: string;
  company_name: string;
  job_description: string;
  interview_type: InterviewType | "";
}

export const EMPTY_DRAFT: InterviewDraft = {
  job_title: "",
  company_name: "",
  job_description: "",
  interview_type: "",
};

// Mirrors the backend contract bounds, so a paste that would be refused is
// refused next to the field rather than by a 422 after the Start click.
const MAX_TITLE = 200;
const MAX_COMPANY = 200;
const MAX_DESCRIPTION = 30_000;

export interface InterviewDraftProblems {
  job_title?: string;
  company_name?: string;
  job_description?: string;
  interview_type?: string;
}

/** Field-level problems with the draft; an empty object means it can start. */
export function validateInterviewDraft(draft: InterviewDraft): InterviewDraftProblems {
  const problems: InterviewDraftProblems = {};
  if (!draft.job_title.trim()) {
    problems.job_title = "Name the role you're interviewing for.";
  } else if (draft.job_title.length > MAX_TITLE) {
    problems.job_title = `Keep the job title under ${MAX_TITLE} characters.`;
  }
  if (!draft.company_name.trim()) {
    problems.company_name = "Name the company.";
  } else if (draft.company_name.length > MAX_COMPANY) {
    problems.company_name = `Keep the company name under ${MAX_COMPANY} characters.`;
  }
  if (!draft.job_description.trim()) {
    problems.job_description = "Paste the job description — the questions are drawn from it.";
  } else if (draft.job_description.length > MAX_DESCRIPTION) {
    problems.job_description = `That description is over ${MAX_DESCRIPTION.toLocaleString()} characters; trim it down.`;
  }
  if (!draft.interview_type) {
    problems.interview_type = "Pick which round to run.";
  } else if (!INTERVIEW_TYPES.some((round) => round.value === draft.interview_type)) {
    problems.interview_type = "Pick one of the three rounds.";
  }
  return problems;
}

/** The validated context, or null while anything is still missing. */
export function draftToContext(draft: InterviewDraft): InterviewContext | null {
  if (Object.keys(validateInterviewDraft(draft)).length > 0) return null;
  return {
    job_title: draft.job_title.trim(),
    company_name: draft.company_name.trim(),
    job_description: draft.job_description.trim(),
    interview_type: draft.interview_type as InterviewType,
  };
}

/** A pasted JD must survive accidental navigation, so the draft lives in
 * localStorage rather than in component state alone. */
const DRAFT_KEY = "metis.interviews.draft";

export function loadInterviewDraft(): InterviewDraft {
  if (typeof window === "undefined") return { ...EMPTY_DRAFT };
  try {
    const raw = window.localStorage.getItem(DRAFT_KEY);
    if (!raw) return { ...EMPTY_DRAFT };
    const parsed = JSON.parse(raw) as Partial<InterviewDraft>;
    return {
      job_title: typeof parsed.job_title === "string" ? parsed.job_title : "",
      company_name: typeof parsed.company_name === "string" ? parsed.company_name : "",
      job_description:
        typeof parsed.job_description === "string" ? parsed.job_description : "",
      interview_type: INTERVIEW_TYPES.some((round) => round.value === parsed.interview_type)
        ? (parsed.interview_type as InterviewType)
        : "",
    };
  } catch {
    return { ...EMPTY_DRAFT };
  }
}

export function saveInterviewDraft(draft: InterviewDraft): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch {
    // A full or blocked store costs the draft, never the page.
  }
}

export function clearInterviewDraft(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(DRAFT_KEY);
  } catch {
    // Same posture as saving.
  }
}

// "Question one." … "Question five." — every spoken question opens with its
// number, and this is what drives the page's progress counter.
const QUESTION_WORDS: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  "1": 1,
  "2": 2,
  "3": 3,
  "4": 4,
  "5": 5,
};

const QUESTION_PATTERN = /\bquestion\s+(one|two|three|four|five|[1-5])\b/i;

/** The question number an agent line announces, or null when it announces none. */
export function questionNumberFrom(text: string): number | null {
  const match = QUESTION_PATTERN.exec(text);
  if (!match) return null;
  return QUESTION_WORDS[match[1].toLowerCase()] ?? null;
}

/** Whether an agent line is the fixed hand-off into the evaluation. */
export function announcesEvaluation(text: string): boolean {
  return /that[’']?s the interview/i.test(text);
}

// The five criteria the tool must score, and the only ones it may.
const CRITERIA = [
  "specific_evidence",
  "role_depth",
  "relevance",
  "structure",
  "communication",
] as const;

function integerInRange(value: unknown, low: number, high: number): number | null {
  const parsed =
    typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  if (!Number.isInteger(parsed) || parsed < low || parsed > high) return null;
  return parsed;
}

function requiredText(value: unknown): string | null {
  if (typeof value !== "string" || !value.trim()) return null;
  return value.trim();
}

/** The frontend half of the evaluation gate.
 *
 * The backend revalidates everything; this exists so a malformed tool call is
 * refused in the same turn with a message the agent can act on, instead of a
 * round trip that 422s. Throws with the first problem named. Deliberately
 * refuses any payload carrying an overall score: the model does not do the
 * arithmetic.
 */
export function parseInterviewEvaluation(raw: unknown): InterviewEvaluation {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error("submit_interview_evaluation needs an object payload.");
  }
  const payload = raw as Record<string, unknown>;
  if ("overall_score" in payload || "recommendation" in payload) {
    throw new Error(
      "Do not submit an overall score or recommendation; Metis calculates both.",
    );
  }
  const scores = {} as Record<(typeof CRITERIA)[number], number>;
  for (const criterion of CRITERIA) {
    const score = integerInRange(payload[criterion], 1, 10);
    if (score === null) {
      throw new Error(`${criterion} must be an integer from 1 to 10.`);
    }
    scores[criterion] = score;
  }
  const verdict = requiredText(payload.verdict);
  if (!verdict) throw new Error("verdict must be one non-empty sentence.");
  const quote = requiredText(payload.strongest_answer_quote);
  if (!quote) {
    throw new Error("strongest_answer_quote must be the candidate's exact words.");
  }
  const reason = requiredText(payload.strongest_answer_reason);
  if (!reason) throw new Error("strongest_answer_reason must be non-empty.");
  if (!Array.isArray(payload.improvements) || payload.improvements.length !== 3) {
    throw new Error("improvements must contain exactly three objects.");
  }
  const improvements: InterviewImprovement[] = payload.improvements.map((item, index) => {
    const entry = (item ?? {}) as Record<string, unknown>;
    const what_happened = requiredText(entry.what_happened);
    const evidence = requiredText(entry.evidence);
    const why_it_hurt = requiredText(entry.why_it_hurt);
    const better_approach = requiredText(entry.better_approach);
    if (!what_happened || !evidence || !why_it_hurt || !better_approach) {
      throw new Error(
        `improvement ${index + 1} needs what_happened, evidence, why_it_hurt, and better_approach.`,
      );
    }
    return { what_happened, evidence, why_it_hurt, better_approach };
  });
  const drill = requiredText(payload.drill);
  if (!drill) throw new Error("drill must be one concrete ten-minute exercise.");
  const completed = integerInRange(payload.completed_question_count, 0, INTERVIEW_QUESTION_LIMIT);
  if (completed === null) {
    throw new Error("completed_question_count must be an integer from 0 to 5.");
  }
  const incomplete =
    typeof payload.incomplete === "boolean"
      ? payload.incomplete
      : payload.incomplete === "true"
        ? true
        : payload.incomplete === "false"
          ? false
          : null;
  if (incomplete === null) throw new Error("incomplete must be a boolean.");
  if (!incomplete && completed !== INTERVIEW_QUESTION_LIMIT) {
    throw new Error(
      "a complete interview answers all five questions; mark it incomplete or fix completed_question_count.",
    );
  }
  if (incomplete && completed >= INTERVIEW_QUESTION_LIMIT) {
    throw new Error("an interview with five answered questions is not incomplete.");
  }
  return {
    ...scores,
    verdict,
    strongest_answer_quote: quote,
    strongest_answer_reason: reason,
    improvements,
    drill,
    completed_question_count: completed,
    incomplete,
  };
}

/** mm:ss for the elapsed timer. */
export function elapsedLabel(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = String(Math.floor(whole / 60)).padStart(2, "0");
  return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

/** The recommendation, in the words the scorecard shows. */
export function recommendationLabel(
  recommendation: "advance" | "borderline" | "do_not_advance" | null,
): string {
  if (recommendation === "advance") return "Would advance";
  if (recommendation === "borderline") return "Borderline";
  if (recommendation === "do_not_advance") return "Would not advance";
  return "Not scored";
}
