import type { AttachmentRef } from "@/lib/types";

export const CHAT_ATTACHMENT_ACCEPT = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".gif",
  "text/*",
  ".md",
  ".txt",
  ".rst",
  ".pdf",
  ".docx",
  ".pptx",
  ".xlsx",
  ".csv",
  ".tsv",
  ".rtf",
  ".ipynb",
  ".log",
  ".py",
  ".js",
  ".ts",
  ".tsx",
  ".jsx",
  ".json",
  ".yaml",
  ".yml",
  ".toml",
  ".ini",
  ".cfg",
  ".conf",
  ".xml",
  ".html",
  ".css",
  ".sql",
  ".sh",
  ".zsh",
  ".go",
  ".rs",
  ".java",
  ".kt",
  ".rb",
  ".php",
  ".c",
  ".h",
  ".cpp",
  ".hpp",
  ".cs",
  ".tf",
  ".tfvars",
  ".graphql",
  ".gql",
].join(",");

export function attachmentBadge(attachment: AttachmentRef): string {
  const extension = attachment.name.split(".").pop()?.toLowerCase();
  if (
    attachment.media_type?.toLowerCase().startsWith("image/")
    || ["png", "jpg", "jpeg", "webp", "gif"].includes(extension ?? "")
  ) return "IMG";
  if (extension === "pdf") return "PDF";
  if (extension === "docx") return "DOC";
  if (extension === "pptx") return "PPT";
  if (["xlsx", "csv", "tsv"].includes(extension ?? "")) return "XLS";
  if (["py", "js", "jsx", "ts", "tsx", "go", "rs", "java", "rb", "php", "c", "cpp", "h", "hpp", "cs", "sql", "sh", "zsh", "tf", "graphql", "gql"].includes(extension ?? "")) return "CODE";
  return "TXT";
}
