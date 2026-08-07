import type { Metadata } from "next";

import { AnswerBank } from "@/components/answer-bank";

export const metadata: Metadata = { title: "Answers" };

export default function AnswersPage() {
  return <AnswerBank />;
}
