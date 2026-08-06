import AppKit
import WebKit

/// The one window, and every browser behaviour the web app silently relies on.
///
/// A bare WKWebView is not a browser. Four things the app uses every day
/// simply do nothing without a delegate wired up, and all four fail silently:
/// `window.confirm` (deleting a conversation), `getUserMedia` (dictation),
/// file downloads (artifacts and exports), and links out to the wider web.
final class WebWindowController: NSObject, NSWindowDelegate, WKUIDelegate, WKNavigationDelegate, WKDownloadDelegate {
    let window: NSWindow
    let webView: WKWebView
    private let appURL = URL(string: "http://127.0.0.1:3000/")!
    /// Reached by the failure page's Try Again button.
    var onRetry: (() -> Void)?

    override init() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()   // keeps localStorage: sidebar width, scope, recents
        configuration.preferences.isElementFullscreenEnabled = true
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.allowsBackForwardNavigationGestures = false

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 840),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        window.title = "Metis"
        window.minSize = NSSize(width: 900, height: 600)
        window.contentView = webView
        window.center()
        window.setFrameAutosaveName("MetisMainWindow")
        window.tabbingMode = .disallowed

        super.init()
        webView.uiDelegate = self
        webView.navigationDelegate = self
        window.delegate = self
    }

    func show() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func loadApp() { webView.load(URLRequest(url: appURL)) }

    /// ⌘R: reload the page if we are on it, otherwise retry the whole startup.
    @objc func reload(_ sender: Any?) {
        if webView.url?.host == appURL.host { webView.reload() } else { onRetry?() }
    }

    // MARK: closing hides — quitting is ⌘Q's job

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        // The user chose this split deliberately: closing the window parks the
        // app (reopening is instant, a running build keeps running), and only
        // ⌘Q pays the full stop.
        sender.orderOut(nil)
        return false
    }

    // MARK: JavaScript panels

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = "Metis"
        alert.informativeText = message
        alert.runModal()
        completionHandler()
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        // Without this, window.confirm returns false unconditionally and
        // "Delete conversation" looks like a button that does nothing.
        let alert = NSAlert()
        alert.messageText = "Metis"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        completionHandler(alert.runModal() == .alertFirstButtonReturn)
    }

    func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
                 defaultText: String?, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (String?) -> Void) {
        let alert = NSAlert()
        alert.messageText = "Metis"
        alert.informativeText = prompt
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 260, height: 24))
        field.stringValue = defaultText ?? ""
        alert.accessoryView = field
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        completionHandler(alert.runModal() == .alertFirstButtonReturn ? field.stringValue : nil)
    }

    // MARK: microphone — dictation's second permission gate

    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        // Granted only to our own loopback origin, and only for audio — the
        // macOS-level microphone consent (NSMicrophoneUsageDescription) still
        // prompts the first time, so this is a narrowing, not a bypass.
        let local = origin.host == "127.0.0.1" || origin.host == "localhost"
        decisionHandler(local && type == .microphone ? .grant : .deny)
    }

    // MARK: navigation — stay home, send the web to the browser

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = navigationAction.request.url else { return decisionHandler(.allow) }
        if url.scheme == "metis-retry" {
            decisionHandler(.cancel)
            onRetry?()
            return
        }
        if let scheme = url.scheme, ["http", "https"].contains(scheme),
           let host = url.host, host != "127.0.0.1", host != "localhost" {
            // A citation, a docs link, a repo URL: that is a browsing task,
            // and this window deliberately is not a browser.
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
            return
        }
        if navigationAction.shouldPerformDownload {
            decisionHandler(.download)
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        decisionHandler(navigationResponse.canShowMIMEType ? .allow : .download)
    }

    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        // target=_blank. Local pages keep living in this window; anything
        // else goes to the default browser. Never a second WKWebView.
        if let url = navigationAction.request.url {
            if let host = url.host, host == "127.0.0.1" || host == "localhost" {
                webView.load(URLRequest(url: url))
            } else {
                NSWorkspace.shared.open(url)
            }
        }
        return nil
    }

    // MARK: downloads — artifacts and exports land in ~/Downloads

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse,
                  suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        var destination = downloads.appendingPathComponent(suggestedFilename)
        // Dedupe the way Finder does, because failing the download over a
        // name collision is the one wrong answer.
        let base = destination.deletingPathExtension().lastPathComponent
        let ext = destination.pathExtension
        var counter = 2
        while FileManager.default.fileExists(atPath: destination.path) {
            let name = ext.isEmpty ? "\(base) \(counter)" : "\(base) \(counter).\(ext)"
            destination = downloads.appendingPathComponent(name)
            counter += 1
        }
        lastDownload = destination
        completionHandler(destination)
    }

    private var lastDownload: URL?

    func downloadDidFinish(_ download: WKDownload) {
        if let file = lastDownload {
            NSWorkspace.shared.activateFileViewerSelecting([file])
        }
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        let alert = NSAlert()
        alert.messageText = "Download failed"
        alert.informativeText = error.localizedDescription
        alert.runModal()
    }
}
