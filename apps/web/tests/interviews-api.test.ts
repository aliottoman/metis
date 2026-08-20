import assert from "node:assert/strict";
import test from "node:test";

import {
  analyzeInterviewDelivery,
  appendInterviewTurns,
  endInterviewSession,
  getInterviewAvailability,
  startInterviewSession,
  submitInterviewEvaluation,
} from "../lib/api.ts";
import {
  EMPTY_DRAFT,
  announcesEvaluation,
  draftToContext,
  elapsedLabel,
  parseFocusAreas,
  parseInterviewEvaluation,
  questionNumberFrom,
  recommendationLabel,
  validateInterviewDraft,
} from "../lib/interviews.ts";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const DRAFT = {
  job_title: "Senior Data Engineer",
  company_name: "Batelco",
  job_description: "Own the lakehouse. Spark, Airflow, judgment.",
  interview_type: "technical" as const,
  interview_objective: "",
  focus_areas: "",
};

// What draftToContext(DRAFT) produces — the shape the API actually posts.
const CONTEXT = {
  job_title: DRAFT.job_title,
  company_name: DRAFT.company_name,
  job_description: DRAFT.job_description,
  interview_type: DRAFT.interview_type,
  interview_objective: "",
  focus_areas: [] as string[],
};

const EVALUATION = {
  specific_evidence: 8,
  role_depth: 7,
  relevance: 7,
  structure: 6,
  communication: 7,
  verdict: "Solid on evidence, thin on tradeoffs.",
  strongest_answer_quote: "We cut the nightly run from six hours to forty minutes.",
  strongest_answer_reason: "A number, a before, and an after.",
  improvements: [
    {
      what_happened: "Hid behind 'we'.",
      evidence: "\"We decided to move to Airflow.\"",
      why_it_hurt: "Nothing was attributable.",
      better_approach: "Name your part first.",
    },
    {
      what_happened: "Vague scenario answer.",
      evidence: "\"I would figure it out.\"",
      why_it_hurt: "Judgment went untested.",
      better_approach: "Commit to a first step.",
    },
    {
      what_happened: "Ran long.",
      evidence: "Three topics past the question.",
      why_it_hurt: "The point drowned.",
      better_approach: "Land it, stop.",
    },
  ],
  drill: "Rewrite the migration story in first person. Ten minutes.",
  completed_question_count: 5,
  incomplete: false,
};

// -- the setup form cannot start incomplete --------------------------------

test("every missing field blocks the start and is named next to its field", () => {
  const problems = validateInterviewDraft({ ...EMPTY_DRAFT });
  assert.ok(problems.job_title);
  assert.ok(problems.company_name);
  assert.ok(problems.job_description);
  assert.ok(problems.interview_type);
  assert.equal(draftToContext({ ...EMPTY_DRAFT }), null);
  // One missing field is still a blocked start.
  assert.equal(draftToContext({ ...DRAFT, job_description: "  " }), null);
  assert.notEqual(draftToContext({ ...DRAFT }), null);
});

test("the interview type is mandatory and restricted to the three rounds", () => {
  assert.equal(draftToContext({ ...DRAFT, interview_type: "" }), null);
  assert.ok(
    validateInterviewDraft({
      ...DRAFT,
      interview_type: "panel" as unknown as typeof DRAFT.interview_type,
    }).interview_type,
  );
  for (const round of ["hr_recruiter", "hiring_manager", "technical"] as const) {
    assert.equal(draftToContext({ ...DRAFT, interview_type: round })?.interview_type, round);
  }
});

test("the objective and focus areas are optional, parsed, and bounded", () => {
  // Optional: an empty pair still starts.
  const bare = draftToContext({ ...DRAFT });
  assert.deepEqual(bare?.focus_areas, []);
  assert.equal(bare?.interview_objective, "");
  // Commas, semicolons and newlines all separate; whitespace items drop.
  assert.deepEqual(
    parseFocusAreas("architecture trade-offs, why ElevenLabs;\n scaling , "),
    ["architecture trade-offs", "why ElevenLabs", "scaling"],
  );
  // A pasted repeat is noise, not emphasis — deduplicated in order, which
  // also keeps React chip keys unique.
  assert.deepEqual(parseFocusAreas("scaling, python, scaling"), [
    "scaling",
    "python",
  ]);
  const steered = draftToContext({
    ...DRAFT,
    interview_objective: "  Grill me on why ElevenLabs.  ",
    focus_areas: "trade-offs, scaling",
  });
  assert.equal(steered?.interview_objective, "Grill me on why ElevenLabs.");
  assert.deepEqual(steered?.focus_areas, ["trade-offs", "scaling"]);
  // Bounded: seven areas or a 121-char phrase block the start at the field.
  assert.ok(
    validateInterviewDraft({
      ...DRAFT,
      focus_areas: "a, b, c, d, e, f, g",
    }).focus_areas,
  );
  assert.ok(
    validateInterviewDraft({ ...DRAFT, focus_areas: "x".repeat(121) }).focus_areas,
  );
  assert.ok(
    validateInterviewDraft({
      ...DRAFT,
      interview_objective: "y".repeat(4_001),
    }).interview_objective,
  );
});

// -- api functions ----------------------------------------------------------

test("starting a session posts the context and hands back token and variables", async () => {
  const originalFetch = globalThis.fetch;
  let requested = "";
  let body: Record<string, unknown> = {};
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    requested = String(input);
    body = JSON.parse(String(init?.body));
    return jsonResponse({
      session: { id: "ivw_1", status: "active", question_limit: 5 },
      conversation_token: "conv-token",
      dynamic_variables: { question_limit: "5" },
    });
  };
  try {
    const opened = await startInterviewSession(CONTEXT);
    assert.ok(requested.endsWith("/api/v1/interviews/sessions"));
    assert.deepEqual(body, CONTEXT);
    assert.equal(opened.conversation_token, "conv-token");
    assert.equal(opened.dynamic_variables.question_limit, "5");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("availability, turns, evaluation and end hit their endpoints", async () => {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ url: string; method: string; body?: unknown }> = [];
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({
      url: String(input),
      method: init?.method ?? "GET",
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    });
    return jsonResponse({});
  };
  try {
    await getInterviewAvailability();
    await appendInterviewTurns("ivw_9", [{ ordinal: 0, role: "agent", text: "Question one." }]);
    await submitInterviewEvaluation("ivw_9", EVALUATION);
    await endInterviewSession("ivw_9", "ended_early");
    await analyzeInterviewDelivery("ivw_9");
    assert.ok(calls[0].url.endsWith("/api/v1/interviews/availability"));
    assert.ok(calls[1].url.endsWith("/api/v1/interviews/sessions/ivw_9/turns"));
    assert.deepEqual(calls[1].body, {
      turns: [{ ordinal: 0, role: "agent", text: "Question one." }],
    });
    assert.ok(calls[2].url.endsWith("/api/v1/interviews/sessions/ivw_9/evaluation"));
    assert.equal(calls[2].method, "POST");
    assert.ok(calls[3].url.endsWith("/api/v1/interviews/sessions/ivw_9/end"));
    assert.deepEqual(calls[3].body, { reason: "ended_early" });
    assert.ok(calls[4].url.endsWith("/api/v1/interviews/sessions/ivw_9/delivery"));
    assert.equal(calls[4].method, "POST");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

// -- question tracking ------------------------------------------------------

test("questions announce their number and the counter reads it", () => {
  assert.equal(questionNumberFrom("Question one. Walk me through your background."), 1);
  assert.equal(questionNumberFrom("Right. Question three: what broke?"), 3);
  assert.equal(questionNumberFrom("question 5, last one."), 5);
  assert.equal(questionNumberFrom("A question for you about scale."), null);
  assert.equal(questionNumberFrom("We'll run five questions."), null);
  assert.equal(questionNumberFrom("Question six."), null);
});

test("the evaluation hand-off phrase is recognized in both apostrophes", () => {
  assert.ok(announcesEvaluation("That's the interview. I'm evaluating it now."));
  assert.ok(announcesEvaluation("That’s the interview. I’m evaluating it now."));
  assert.ok(!announcesEvaluation("That was a strong answer to the interview question."));
});

// -- the evaluation gate ----------------------------------------------------

test("a well-formed evaluation passes and numeric strings are coerced", () => {
  const parsed = parseInterviewEvaluation({ ...EVALUATION, structure: "6" });
  assert.equal(parsed.structure, 6);
  assert.equal(parsed.completed_question_count, 5);
  assert.equal(parsed.incomplete, false);
});

test("malformed and out-of-range payloads are refused with a named problem", () => {
  const bad: Array<[Record<string, unknown>, RegExp]> = [
    [{ ...EVALUATION, specific_evidence: 0 }, /specific_evidence/],
    [{ ...EVALUATION, communication: 11 }, /communication/],
    [{ ...EVALUATION, relevance: 7.5 }, /relevance/],
    [{ ...EVALUATION, verdict: " " }, /verdict/],
    [{ ...EVALUATION, improvements: EVALUATION.improvements.slice(0, 2) }, /three/],
    [{ ...EVALUATION, improvements: [...EVALUATION.improvements, EVALUATION.improvements[0]] }, /three/],
    [{ ...EVALUATION, drill: "" }, /drill/],
    [{ ...EVALUATION, completed_question_count: 6 }, /completed_question_count/],
    [{ ...EVALUATION, incomplete: "maybe" }, /incomplete/],
    [{ ...EVALUATION, incomplete: false, completed_question_count: 3 }, /five questions/],
    [{ ...EVALUATION, incomplete: true, completed_question_count: 5 }, /not incomplete/],
  ];
  for (const [payload, pattern] of bad) {
    assert.throws(() => parseInterviewEvaluation(payload), pattern);
  }
});

test("the model may not submit its own overall score", () => {
  // The UI never invents a number either way: the only score it can show is
  // the one the backend calculated and returned.
  assert.throws(
    () => parseInterviewEvaluation({ ...EVALUATION, overall_score: 9.8 }),
    /Metis calculates/,
  );
  assert.throws(
    () => parseInterviewEvaluation({ ...EVALUATION, recommendation: "advance" }),
    /Metis calculates/,
  );
});

// -- labels -----------------------------------------------------------------

test("recommendations read as verdicts and a missing score is not one", () => {
  assert.equal(recommendationLabel("advance"), "Would advance");
  assert.equal(recommendationLabel("borderline"), "Borderline");
  assert.equal(recommendationLabel("do_not_advance"), "Would not advance");
  assert.equal(recommendationLabel(null), "Not scored");
});

test("the elapsed timer formats as mm:ss", () => {
  assert.equal(elapsedLabel(0), "00:00");
  assert.equal(elapsedLabel(61.9), "01:01");
  assert.equal(elapsedLabel(605), "10:05");
});
