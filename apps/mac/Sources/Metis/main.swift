// Metis.app — the native window.
//
// This is a host, not a port: the whole application still lives in apps/api
// and apps/web. Opening the app starts those servers through the same
// scripts/metis --serve the LaunchAgent used; quitting runs scripts/metis-stop,
// which is the one shutdown path that also releases whatever model Metis
// loaded into Ollama. In between, a WKWebView shows http://127.0.0.1:3000.
import AppKit

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
