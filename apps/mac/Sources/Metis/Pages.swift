import Foundation

/// The three in-app pages that exist before, after, and instead of the web
/// app: waking, stopping, and could-not-start. Inline HTML rather than bundle
/// resources so the binary stays a single file, styled to the same purple
/// field the real front door uses — the splash should read as Metis arriving,
/// not as a loading screen in front of it.
enum Pages {
    private static let shell = """
    <meta charset="utf-8">
    <style>
      html, body { height: 100%; margin: 0; }
      body {
        display: grid; place-items: center;
        background: radial-gradient(120% 90% at 40% 30%, #cdb6df 0%, #c29ce0 55%, #a379ce 100%);
        font: 14px/1.5 -apple-system, BlinkMacSystemFont, sans-serif; color: #fff;
        -webkit-user-select: none; cursor: default;
      }
      main { text-align: center; max-width: 560px; padding: 0 32px; }
      .pearl {
        width: 64px; height: 64px; margin: 0 auto 22px; border-radius: 50%;
        background:
          radial-gradient(circle at 32% 28%, rgba(255,255,255,.95), rgba(255,255,255,0) 42%),
          conic-gradient(from 210deg, #ff7759, #7be3d8, #d18ee2, #ffe08a, #ff7759);
        filter: blur(.3px);
        animation: breathe 2.6s ease-in-out infinite;
      }
      @keyframes breathe { 0%,100% { transform: scale(1); } 50% { transform: scale(1.08); } }
      h1 { margin: 0 0 6px; font-size: 21px; font-weight: 500; letter-spacing: -.02em; }
      p  { margin: 0; color: rgba(255,255,255,.82); }
      code, pre { font: 11px ui-monospace, Menlo, monospace; }
      pre {
        margin: 18px auto 0; padding: 12px 14px; max-height: 200px; overflow: auto;
        text-align: left; background: rgba(38,22,48,.4); border-radius: 10px;
        color: rgba(255,255,255,.88); white-space: pre-wrap; word-break: break-all;
        -webkit-user-select: text;
      }
      a.button {
        display: inline-block; margin-top: 18px; padding: 9px 18px; border-radius: 999px;
        background: #1a1815; color: #fbf8f2; text-decoration: none; font-size: 13px;
      }
    </style>
    """

    static let splash = shell + """
    <main>
      <div class="pearl"></div>
      <h1>Metis is waking up</h1>
      <p>Starting the API and the interface. A moment — longer if the web app is rebuilding.</p>
    </main>
    """

    static let stopping = shell + """
    <main>
      <div class="pearl" style="animation-duration: 1.2s"></div>
      <h1>Stopping Metis</h1>
      <p>Shutting the servers down and giving the model memory back.</p>
    </main>
    """

    static func failure(logPath: String, logTail: String) -> String {
        let escaped = logTail
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
        return shell + """
        <main>
          <div class="pearl" style="animation: none; filter: saturate(.15) blur(.3px)"></div>
          <h1>Metis could not start</h1>
          <p>The servers never answered. The end of <code>\(logPath)</code>:</p>
          <pre>\(escaped)</pre>
          <a class="button" href="metis-retry://now">Try Again</a>
        </main>
        """
    }
}
