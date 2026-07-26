const FORBIDDEN_SVG_ELEMENTS = [
  "script",
  "foreignObject",
  "iframe",
  "object",
  "embed",
  "audio",
  "video",
  "style",
  "link",
  "meta",
  "base",
];

export function isAllowedSvgReference(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return (
    normalized === "" ||
    normalized.startsWith("#") ||
    /^data:image\/(?:png|jpeg|webp);base64,[a-z0-9+/=\s]+$/i.test(normalized)
  );
}

/**
 * Sanitize an SVG before placing it in a sandboxed srcdoc frame.
 *
 * The sandbox already disables scripts and same-origin access. This extra pass
 * also removes active elements and non-embedded references so merely opening a
 * generated preview cannot trigger a network or local-path request.
 */
export function sanitizeSvgForPreview(source: string): string {
  const documentNode = new DOMParser().parseFromString(source, "image/svg+xml");
  if (documentNode.querySelector("parsererror")) {
    throw new Error("The SVG is not well-formed XML.");
  }
  const root = documentNode.documentElement;
  if (root.localName.toLowerCase() !== "svg") {
    throw new Error("The artifact does not contain an SVG root element.");
  }

  // Graphviz writes fixed point dimensions. Preserve its viewBox but make the
  // preview responsive so large diagrams fit the modal instead of being
  // cropped at their intrinsic canvas size.
  if (root.hasAttribute("viewBox")) {
    root.setAttribute("width", "100%");
    root.setAttribute("height", "100%");
    root.setAttribute("preserveAspectRatio", "xMidYMid meet");
  }

  root.querySelectorAll(FORBIDDEN_SVG_ELEMENTS.join(",")).forEach((element) => {
    element.remove();
  });

  const elements = [root, ...Array.from(root.querySelectorAll("*"))];
  for (const element of elements) {
    for (const attribute of Array.from(element.attributes)) {
      const name = attribute.name.toLowerCase();
      const value = attribute.value.trim();
      const lowered = value.toLowerCase();
      if (name.startsWith("on") || lowered.includes("javascript:")) {
        element.removeAttribute(attribute.name);
        continue;
      }
      if ((name === "href" || name === "xlink:href") && !isAllowedSvgReference(value)) {
        element.removeAttribute(attribute.name);
        continue;
      }
      if (
        name === "style" &&
        (/url\s*\(/i.test(value) && !/url\s*\(\s*#[^)]+\)/i.test(value))
      ) {
        element.removeAttribute(attribute.name);
      }
    }
  }
  return new XMLSerializer().serializeToString(root);
}
