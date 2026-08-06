// swift-tools-version:5.9
// The native Metis window. Built with the Command Line Tools' SwiftPM —
// deliberately no Xcode project, so the repo stays buildable with what a
// Python/TypeScript machine already has. `make app` assembles the bundle.
import PackageDescription

let package = Package(
    name: "Metis",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "Metis", path: "Sources/Metis")
    ]
)
