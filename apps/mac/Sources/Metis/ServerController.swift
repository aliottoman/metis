import Foundation

/// Starts, watches, and stops the Metis servers.
///
/// The contract with the repo's own scripts is deliberately thin:
///   start = `scripts/metis --serve`   (frees stale ports, rebuilds the web
///           app only when sources changed, then execs the supervisor)
///   stop  = `scripts/metis-stop`      (supervisor first, then port holders,
///           then unloads whatever model Metis left in Ollama)
/// Everything those scripts already solved — launch locks, port squatters,
/// PATH for Finder-launched processes — stays solved in exactly one place.
final class ServerController {
    /// Where the repo lives. Baked into Info.plist by scripts/build-app so the
    /// binary itself stays path-free; moving the repo means rebuilding the app.
    let repoRoot: URL

    private let queue = DispatchQueue(label: "metis.server", qos: .userInitiated)
    private var spawnedPid: pid_t = -1
    private var exitWatcher: DispatchSourceProcess?
    private var childExited = false

    /// Written by the spawned tree; the failure page tails it, because "it
    /// didn't start" without the log is a support ticket with no evidence.
    var logPath: String { repoRoot.appendingPathComponent(".data/metis-app.log").path }

    init(repoRoot: URL) {
        self.repoRoot = repoRoot
    }

    // MARK: health

    private func responds(_ urlString: String) -> Bool {
        guard let url = URL(string: urlString) else { return false }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        var ok = false
        let done = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: request) { _, response, _ in
            if let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) {
                ok = true
            }
            done.signal()
        }.resume()
        done.wait()
        return ok
    }

    private func healthy() -> Bool {
        // Both halves, deliberately: the web server answers long before the
        // API on a cold start, and loading the UI then shows every panel in
        // its "API unavailable" state until a reload.
        responds("http://127.0.0.1:8000/api/v1/health") && responds("http://127.0.0.1:3000/")
    }

    // MARK: start

    /// Bring Metis up and call back on the main thread.
    ///
    /// If a healthy Metis is already serving (the user ran `make run` in a
    /// terminal), we attach to it rather than restarting it out from under
    /// them — but quit still stops it, because the promise on the box is
    /// "⌘Q stops Metis", not "⌘Q stops the Metis this window happened to start".
    func ensureRunning(ready: @escaping () -> Void, failed: @escaping (String) -> Void) {
        queue.async {
            if self.healthy() {
                DispatchQueue.main.async(execute: ready)
                return
            }
            if self.spawnedPid <= 0 || self.childExited {
                do {
                    try self.spawn()
                } catch {
                    DispatchQueue.main.async { failed("Could not launch scripts/metis: \(error.localizedDescription)") }
                    return
                }
            }
            // The window the launcher itself allows: a from-scratch web build
            // plus server start. Poll cheaply; give up loudly.
            let deadline = Date().addingTimeInterval(180)
            while Date() < deadline {
                if self.healthy() {
                    DispatchQueue.main.async(execute: ready)
                    return
                }
                // The launcher exiting without the ports ever answering is a
                // failed start, not a slow one — say so now, not in 3 minutes.
                if self.childExited && !self.healthy() {
                    Thread.sleep(forTimeInterval: 2)
                    if !self.healthy() { break }
                }
                Thread.sleep(forTimeInterval: 1)
            }
            DispatchQueue.main.async { failed(self.tailOfLog()) }
        }
    }

    private func spawn() throws {
        let script = repoRoot.appendingPathComponent("scripts/metis").path
        guard FileManager.default.isExecutableFile(atPath: script) else {
            throw NSError(domain: "Metis", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(script) is missing or not executable"])
        }
        try? FileManager.default.createDirectory(
            at: repoRoot.appendingPathComponent(".data"), withIntermediateDirectories: true)

        var fileActions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&fileActions)
        defer { posix_spawn_file_actions_destroy(&fileActions) }
        posix_spawn_file_actions_addopen(&fileActions, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_addopen(&fileActions, 1, logPath, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        posix_spawn_file_actions_adddup2(&fileActions, 1, 2)
        posix_spawn_file_actions_addchdir_np(&fileActions, repoRoot.path)

        // Its own process group, so the fallback kill can address the whole
        // server tree at once. metis-stop is the primary stop; this is the
        // seatbelt for the day that script is broken.
        var attrs: posix_spawnattr_t?
        posix_spawnattr_init(&attrs)
        defer { posix_spawnattr_destroy(&attrs) }
        posix_spawnattr_setpgroup(&attrs, 0)
        posix_spawnattr_setflags(&attrs, Int16(POSIX_SPAWN_SETPGROUP))

        let arguments = [script, "--serve"]
        var argv: [UnsafeMutablePointer<CChar>?] = arguments.map { strdup($0) }
        argv.append(nil)
        defer { argv.forEach { free($0) } }

        var pid: pid_t = 0
        let status = posix_spawn(&pid, script, &fileActions, &attrs, argv, environ)
        guard status == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(status),
                          userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(status))])
        }
        spawnedPid = pid
        childExited = false

        let watcher = DispatchSource.makeProcessSource(identifier: pid, eventMask: .exit, queue: queue)
        watcher.setEventHandler { [weak self] in self?.childExited = true }
        watcher.resume()
        exitWatcher = watcher
    }

    // MARK: stop

    /// The full stop, off the main thread; calls back on main when done.
    func stop(completion: @escaping () -> Void) {
        queue.async {
            let stopScript = self.repoRoot.appendingPathComponent("scripts/metis-stop")
            if FileManager.default.isExecutableFile(atPath: stopScript.path) {
                let process = Process()
                process.executableURL = stopScript
                process.currentDirectoryURL = self.repoRoot
                try? process.run()
                process.waitUntilExit()
            }
            // Seatbelt: if the stop script left our own spawn tree alive
            // (edited, broken, half-run), the process group still goes down.
            if self.spawnedPid > 0 && kill(-self.spawnedPid, 0) == 0 {
                kill(-self.spawnedPid, SIGTERM)
                for _ in 0..<20 {
                    if kill(-self.spawnedPid, 0) != 0 { break }
                    Thread.sleep(forTimeInterval: 0.25)
                }
                if kill(-self.spawnedPid, 0) == 0 { kill(-self.spawnedPid, SIGKILL) }
            }
            self.exitWatcher?.cancel()
            self.exitWatcher = nil
            self.spawnedPid = -1
            // NOT DispatchQueue.main.async. This completion carries
            // NSApp.reply(toApplicationShouldTerminate:), and when quitting
            // was triggered from a main-queue block (the SIGTERM source),
            // .terminateLater spins its modal wait INSIDE that block — the
            // serial main queue never drains again and the reply never
            // arrives. The run loop, however, keeps servicing blocks in
            // common and modal-panel modes throughout the wait.
            let modes = [CFRunLoopMode.commonModes.rawValue as Any,
                         "NSModalPanelRunLoopMode" as CFString as Any] as CFArray
            CFRunLoopPerformBlock(CFRunLoopGetMain(), modes, completion)
            CFRunLoopWakeUp(CFRunLoopGetMain())
        }
    }

    private func tailOfLog() -> String {
        guard let text = try? String(contentsOfFile: logPath, encoding: .utf8) else {
            return "Metis did not come up, and no log was written to \(logPath)."
        }
        return text.split(separator: "\n").suffix(25).joined(separator: "\n")
    }
}
