import Foundation

/// The three in-app pages that exist before, after, and instead of the web
/// app: waking, stopping, and could-not-start.
///
/// They are drawn as the front door itself — the same purple field, drifting
/// chrome forms, film grain, pearl, and wordmark the real welcome screen
/// carries — because the splash IS the app's first frame, and a generic
/// gradient there made opening Metis feel like passing through an airlock
/// that belonged to a different product. Inline HTML so the binary stays a
/// single file with no resource loading to fail while we are busy explaining
/// that something else failed.
enum Pages {
    private static let shell = """
    <meta charset="utf-8">
    <style>
      html, body { height: 100%; margin: 0; overflow: hidden; }
      body {
        display: grid; place-items: center; position: relative;
        background: radial-gradient(120% 90% at 40% 30%, #cdb6df 0%, #c29ce0 55%, #a379ce 100%);
        font: 14px/1.55 -apple-system, BlinkMacSystemFont, sans-serif; color: #fff;
        -webkit-user-select: none; cursor: default;
      }
      /* The chrome forms of the front door, adrift. Durations are mutually
         prime-ish so the field never falls into a visible cycle. */
      .form { position: absolute; border-radius: 50%; filter: blur(64px); }
      .f1 { width: 46vw; height: 40vh; left: -12vw; top: -14vh; background: #efe2f6; opacity: .55;
            animation: drift1 47s ease-in-out infinite; }
      .f2 { width: 38vw; height: 36vh; right: -10vw; top: -8vh; background: #8f66b8; opacity: .4;
            animation: drift2 61s ease-in-out infinite; }
      .f3 { width: 44vw; height: 38vh; right: -8vw; bottom: -16vh; background: #e8d0f2; opacity: .42;
            animation: drift3 53s ease-in-out infinite; }
      @keyframes drift1 { 50% { transform: translate(4vw, 3vh) scale(1.08); } }
      @keyframes drift2 { 50% { transform: translate(-3vw, 4vh) scale(.94); } }
      @keyframes drift3 { 50% { transform: translate(-4vw, -3vh) scale(1.06); } }
      /* The same grain the app runs, at the same scale. */
      .grain {
        position: absolute; inset: 0; pointer-events: none; opacity: .5; mix-blend-mode: overlay;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='g'><feTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/><feColorMatrix type='saturate' values='0'/></filter><rect width='220' height='220' filter='url(%23g)'/></svg>");
        background-size: 220px 220px;
      }
      main { position: relative; text-align: center; max-width: 640px; padding: 0 40px; }
      /* The pearl: lit body, turning film, specular — the companion, not a spinner. */
      .pearl { position: relative; width: 84px; height: 84px; margin: 0 auto 30px;
               animation: breathe 4.5s ease-in-out infinite; }
      .pearl::before {
        content: ""; position: absolute; inset: 0; border-radius: 50%;
        background: radial-gradient(circle at 33% 25%, #fffdfb 0%, #ffdccb 38%, #e3bbea 70%, #a67fc6 100%);
        box-shadow: inset -8px -10px 22px rgba(90,58,90,.3), inset 6px 8px 18px rgba(255,255,255,.95),
                    0 22px 40px -16px rgba(50,20,70,.6);
      }
      .film { position: absolute; inset: 6%; border-radius: 50%; overflow: hidden;
              -webkit-mask-image: radial-gradient(closest-side, #000 0 62%, transparent 88%); }
      .film i { position: absolute; inset: -30%; display: block; filter: blur(9px); opacity: .8;
                background:
                  radial-gradient(38% 34% at 68% 24%, #ff7f5c 0%, rgba(255,127,92,0) 66%),
                  radial-gradient(40% 36% at 78% 62%, #c77bee 0%, rgba(199,123,238,0) 66%),
                  radial-gradient(42% 38% at 34% 72%, #4fd3e4 0%, rgba(79,211,228,0) 66%),
                  radial-gradient(34% 32% at 26% 30%, #ffd25a 0%, rgba(255,210,90,0) 66%);
                animation: turn 11s linear infinite; }
      .spec { position: absolute; left: 24%; top: 17%; width: 34%; height: 24%; border-radius: 50%;
              background: radial-gradient(closest-side, rgba(255,255,255,.96), rgba(255,255,255,0)); }
      @keyframes breathe { 0%,100% { transform: scale(1); } 50% { transform: scale(1.06); } }
      @keyframes turn { to { transform: rotate(360deg); } }
      /* Every block centres on the same axis. The eyebrow overrode `margin`
         but kept the 46ch `max-width` below, which left it a narrow box
         pinned to the left edge — its text then centred inside that box,
         landing well left of the wordmark it is supposed to sit above. */
      p  {
        margin: 0 auto; max-width: 46ch; font-size: 15px;
        color: rgba(255,255,255,.88); text-wrap: balance;
      }
      .eyebrow {
        margin: 0 auto 14px; max-width: none;
        font: 12px ui-monospace, Menlo, monospace;
        letter-spacing: .16em; text-transform: uppercase;
        color: rgba(255,255,255,.82);
      }
      h1 {
        margin: 0 0 14px; font-size: 56px; line-height: 1.02; font-weight: 400;
        letter-spacing: -.034em; text-wrap: balance;
      }
      code, pre { font: 11px ui-monospace, Menlo, monospace; }
      pre {
        margin: 20px auto 0; padding: 12px 14px; max-height: 190px; overflow: auto;
        text-align: left; background: rgba(38,22,48,.42); border-radius: 12px;
        color: rgba(255,255,255,.88); white-space: pre-wrap; word-break: break-all;
        -webkit-user-select: text;
      }
      a.button {
        display: inline-block; margin-top: 20px; padding: 10px 20px; border-radius: 999px;
        background: #1a1815; color: #fbf8f2; text-decoration: none; font-size: 13px;
      }
      a.button:hover { background: #000; }
      @media (prefers-reduced-motion: reduce) { .form, .pearl, .film i { animation: none; } }
    </style>
    <span class="form f1"></span><span class="form f2"></span><span class="form f3"></span>
    <div class="grain"></div>
    """

    static let splash = shell + """
    <main>
      <div class="pearl"><span class="film"><i></i></span><span class="spec"></span></div>
      <p class="eyebrow">Your private thinking partner</p>
      <h1>Metis</h1>
      <p>Waking the API and the interface. This takes a moment longer whenever the web app is rebuilding.</p>
    </main>
    """

    static let stopping = shell + """
    <main>
      <div class="pearl" style="animation-duration: 2s"><span class="film"><i style="animation-duration: 4s"></i></span><span class="spec"></span></div>
      <p class="eyebrow">Closing up</p>
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
          <div class="pearl" style="animation: none"><span class="film"><i style="animation: none; filter: blur(9px) saturate(.15)"></i></span><span class="spec"></span></div>
          <p class="eyebrow">Something is wrong</p>
          <h1>Metis could not start</h1>
          <p>The servers never answered. The end of <code>\(logPath)</code>:</p>
          <pre>\(escaped)</pre>
          <a class="button" href="metis-retry://now">Try Again</a>
        </main>
        """
    }
}
