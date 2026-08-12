// Models the referenced timeout leaked by ClineCore 0.0.72 after an editor
// turn. The CLI shutdown path must exit without waiting for this handle.
setInterval(() => undefined, 60_000);
