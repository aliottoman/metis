"use client";

import { useState } from "react";

import type { ElicitationRequest } from "@/lib/types";

interface ElicitationCardProps {
  elicitation: ElicitationRequest;
  answered: boolean;
  answerBusy?: string | null;
  onAnswer: (
    elicitationId: string,
    reply: { option?: string; text?: string },
  ) => Promise<void>;
  /** "inline" renders in the reading column; the default suits the drawer. */
  variant?: "inline" | "drawer";
}

/**
 * One clarifying question the run paused to ask (ask_user): the question, a
 * button per offered choice, and an optional free-text reply. Modeled on
 * ApprovalCard so the inline dock reads as one family of pauses — the button
 * choice or the typed answer resumes the same turn.
 */
export function ElicitationCard({
  elicitation,
  answered,
  answerBusy,
  onAnswer,
  variant = "drawer",
}: ElicitationCardProps) {
  const [text, setText] = useState("");
  const busy = answerBusy === elicitation.id;

  // Reuses the approval card's classes so the two pauses read as one family;
  // only the options row and the free-text form are elicitation's own.
  return (
    <section
      className={`approvalCard elicitationCard ${variant === "inline" ? "isInline" : ""}`.trim()}
    >
      <div className="approvalTitle">
        <span>Question</span>
        <strong>{elicitation.question}</strong>
      </div>
      {answered ? (
        <div className="decisionRecorded">✓ Answer sent</div>
      ) : (
        <>
          {elicitation.options.length ? (
            <div className="elicitationOptions">
              {elicitation.options.map((option) => (
                <button
                  key={option}
                  type="button"
                  className="secondaryButton"
                  disabled={busy}
                  onClick={() => void onAnswer(elicitation.id, { option })}
                >
                  {option}
                </button>
              ))}
            </div>
          ) : null}
          {elicitation.allow_text ? (
            <form
              className="elicitationTextForm"
              onSubmit={(submit) => {
                submit.preventDefault();
                const value = text.trim();
                if (value) void onAnswer(elicitation.id, { text: value });
              }}
            >
              <input
                type="text"
                value={text}
                disabled={busy}
                placeholder={
                  elicitation.options.length
                    ? "Or type your own answer…"
                    : "Type your answer…"
                }
                onChange={(change) => setText(change.target.value)}
                aria-label="Your answer"
              />
              <button
                type="submit"
                className="primaryButton"
                disabled={busy || !text.trim()}
              >
                Send
              </button>
            </form>
          ) : null}
        </>
      )}
    </section>
  );
}
