import AppKit
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var web: WebWindowController!
    private var server: ServerController!
    private var sigtermSource: DispatchSourceSignal?
    private var stopping = false

    // MARK: lifecycle

    func applicationWillFinishLaunching(_ notification: Notification) {
        buildMenu()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // The repo path is stamped into Info.plist at build time, so the
        // binary carries no path of its own. Moving the repo = `make app`.
        let root = (Bundle.main.object(forInfoDictionaryKey: "MetisRepoRoot") as? String)
            ?? NSString(string: "~/Developer/metis").expandingTildeInPath
        server = ServerController(repoRoot: URL(fileURLWithPath: root, isDirectory: true))

        web = WebWindowController()
        web.onRetry = { [weak self] in self?.startServers() }
        web.webView.loadHTMLString(Pages.splash, baseURL: nil)
        web.show()

        // launchd, logout, and shutdown all speak SIGTERM. Route it through
        // the same quit path as ⌘Q so the servers always get their stop.
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        source.setEventHandler { NSApp.terminate(nil) }
        source.resume()
        sigtermSource = source

        startServers()
    }

    private func startServers() {
        web.webView.loadHTMLString(Pages.splash, baseURL: nil)
        server.ensureRunning(
            ready: { [weak self] in self?.web.loadApp() },
            failed: { [weak self] logTail in
                guard let self else { return }
                self.web.webView.loadHTMLString(Pages.failure(logPath: self.server.logPath, logTail: logTail), baseURL: nil)
            })
    }

    // MARK: quitting — the half of the promise the Safari stub could not keep

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if stopping { return .terminateNow }
        stopping = true
        web.window.makeKeyAndOrderFront(nil)
        web.webView.loadHTMLString(Pages.stopping, baseURL: nil)
        server.stop { NSApp.reply(toApplicationShouldTerminate: true) }
        return .terminateLater
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { web.show() }
        return true
    }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool { true }

    // MARK: menu

    /// A programmatic app gets no menu for free, and without an Edit menu
    /// ⌘C/⌘V/⌘A silently do nothing inside the web view — which reads as
    /// "the app is broken", not "the menu is missing".
    private func buildMenu() {
        let main = NSMenu()

        let appItem = NSMenuItem()
        main.addItem(appItem)
        let appMenu = NSMenu()
        appItem.submenu = appMenu
        appMenu.addItem(withTitle: "About Metis",
                        action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Hide Metis", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = appMenu.addItem(withTitle: "Hide Others",
                                         action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Metis", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        let editItem = NSMenuItem()
        main.addItem(editItem)
        let edit = NSMenu(title: "Edit")
        editItem.submenu = edit
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

        let viewItem = NSMenuItem()
        main.addItem(viewItem)
        let view = NSMenu(title: "View")
        viewItem.submenu = view
        view.addItem(withTitle: "Reload", action: #selector(WebWindowController.reload(_:)), keyEquivalent: "r")
        view.addItem(withTitle: "Open in Browser", action: #selector(AppDelegate.openInBrowser(_:)), keyEquivalent: "b")

        let windowItem = NSMenuItem()
        main.addItem(windowItem)
        let windowMenu = NSMenu(title: "Window")
        windowItem.submenu = windowMenu
        windowMenu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        windowMenu.addItem(withTitle: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        NSApp.windowsMenu = windowMenu

        NSApp.mainMenu = main
    }

    @objc func openInBrowser(_ sender: Any?) {
        NSWorkspace.shared.open(URL(string: "http://127.0.0.1:3000/")!)
    }
}
