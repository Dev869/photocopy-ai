import AppKit
import SwiftUI
import UniformTypeIdentifiers

// v1 is Devin-only: paths are constants, not preferences.
let workDir = URL(fileURLWithPath:
    "/Users/devinwilson/Documents/Projects/native-apps/photocopy-ai/work")

extension Color {
    private static func dyn(_ light: String, _ dark: String) -> Color {
        Color(nsColor: NSColor(name: nil) { $0.bestMatch(from: [.darkAqua]) == .darkAqua
            ? NSColor(hex: dark) : NSColor(hex: light) })
    }
    static let ink = dyn("1E1919", "F5F3F0")          // warm near-black
    static let inkSecondary = dyn("736C64", "A8A19A") // warm gray
    static let cream = dyn("F7F5F2", "26231F")        // rail / tint fills
    static let creamDeep = dyn("EDEAE5", "322E29")    // selected tile
    static let hairline = dyn("EBE7E2", "3A362F")
    static let paper = dyn("FFFFFF", "1C1A17")        // content background
    static let dbBlue = Color(nsColor: NSColor(hex: "0061FF"))
}

extension NSColor {
    convenience init(hex: String) {
        let v = UInt64(hex, radix: 16) ?? 0
        self.init(red: CGFloat((v >> 16) & 0xFF) / 255,
                  green: CGFloat((v >> 8) & 0xFF) / 255,
                  blue: CGFloat(v & 0xFF) / 255, alpha: 1)
    }
}

struct DaemonState: Codable {
    var status = "starting"
    var done = 0
    var total = 0
    var lastFile = ""
    var watchDir = ""
    var looks: [String] = []
    var updated = 0.0

    enum CodingKeys: String, CodingKey {
        case status, done, total, looks, updated
        case lastFile = "last_file"
        case watchDir = "watch_dir"
    }
}

struct DaemonConfig: Codable {
    var watchDir: String?
    var look: String?
    var edit = true
    var cull = false
    var cullTarget: Int?
    var sendToLightroom = true
    var paused = false

    enum CodingKeys: String, CodingKey {
        case look, edit, cull, paused
        case watchDir = "watch_dir"
        case cullTarget = "cull_target"
        case sendToLightroom = "send_to_lightroom"
    }
}

struct FeedEvent: Identifiable, Decodable {
    var id: Double { ts }
    let ts: Double
    let kind: String
    let text: String

    var icon: String {
        switch kind {
        case "start": "play.circle"
        case "photo": "photo"
        case "cull": "scissors"
        case "done": "checkmark.circle.fill"
        default: "circle"
        }
    }

    var time: String {
        Date(timeIntervalSince1970: ts).formatted(date: .omitted, time: .shortened)
    }
}

@MainActor @Observable
final class Agent {
    var state = DaemonState()
    var config = DaemonConfig()
    var events: [FeedEvent] = []
    var lastSeen = UserDefaults.standard.double(forKey: "lastSeenEventTs")

    var unseenCount: Int { events.filter { $0.ts > lastSeen }.count }

    func markSeen() {
        lastSeen = events.first?.ts ?? Date().timeIntervalSince1970
        UserDefaults.standard.set(lastSeen, forKey: "lastSeenEventTs")
    }
    var thumbs: [URL] = []
    private var daemon: Process?
    private var shouldRun = false   // explicit Start only; autonomy comes later
    private var staleTicks = 0

    var daemonAlive: Bool { Date().timeIntervalSince1970 - state.updated < 8 }

    init() {
        loadConfig()
        Task { [weak self] in
            while let self, !Task.isCancelled {
                self.poll()
                self.supervise()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private func supervise() {
        if daemonAlive {
            staleTicks = 0
        } else {
            staleTicks += 1
            if shouldRun, staleTicks >= 3, daemon?.isRunning != true {
                spawn()
            }
        }
    }

    private func spawn() {
        let p = Process()
        p.executableURL = workDir.appending(path: ".venv/bin/python")
        p.arguments = [workDir.appending(path: "agent.py").path]
        p.currentDirectoryURL = workDir
        try? p.run()
        daemon = p
        staleTicks = 0
    }

    func poll() {
        if let data = try? Data(contentsOf: workDir.appending(path: "state.json")),
           let s = try? JSONDecoder().decode(DaemonState.self, from: data) {
            state = s
        }
        if let text = try? String(contentsOf: workDir.appending(path: "events.jsonl"),
                                  encoding: .utf8) {
            let decoder = JSONDecoder()
            events = text.split(separator: "\n").suffix(80).reversed()
                .compactMap { try? decoder.decode(FeedEvent.self, from: Data($0.utf8)) }
        }
        let thumbsDir = workDir.appending(path: "thumbs")
        thumbs = ((try? FileManager.default.contentsOfDirectory(
            at: thumbsDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "jpg" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    func loadConfig() {
        if let data = try? Data(contentsOf: workDir.appending(path: "config.json")),
           let c = try? JSONDecoder().decode(DaemonConfig.self, from: data) {
            config = c
        }
    }

    func saveConfig() {
        if let data = try? JSONEncoder().encode(config) {
            try? data.write(to: workDir.appending(path: "config.json"))
        }
    }

    func startDaemon() {
        shouldRun = true
        if daemon?.isRunning != true, !daemonAlive {
            spawn()
        }
    }

    func stopDaemon() {
        shouldRun = false
        if let daemon, daemon.isRunning {
            daemon.terminate()
        } else {
            let kill = Process()
            kill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
            kill.arguments = ["-f", "agent.py"]
            try? kill.run()
        }
        daemon = nil
        state.updated = 0
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
    }
}

@main
struct PhotocopyApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var agent = Agent()

    var body: some Scene {
        MenuBarExtra {
            PanelView(agent: agent)
        } label: {
            Image(nsImage: menuBarImage(symbol: menuIcon,
                                        badged: agent.unseenCount > 0))
        }
        .menuBarExtraStyle(.window)
        // dev-only: PHOTOCOPY_PREVIEW=1 opens the panel as a window for screenshots
        WindowGroup(id: "preview") {
            if ProcessInfo.processInfo.environment["PHOTOCOPY_PREVIEW"] != nil {
                PanelView(agent: agent)
            }
        }
        .windowResizability(.contentSize)
        .defaultLaunchBehavior(ProcessInfo.processInfo.environment["PHOTOCOPY_PREVIEW"] != nil
                               ? .presented : .suppressed)
    }

    private var menuIcon: String {
        switch agent.state.status {
        case "processing": "camera.badge.clock"
        case "paused": "pause.circle"
        case "error": "exclamationmark.triangle"
        default: "camera"
        }
    }
}

/// Menu bar icon: template symbol normally; adds a red badge dot when there
/// are unseen activity events (non-template so the dot keeps its color).
@MainActor
func menuBarImage(symbol: String, badged: Bool) -> NSImage {
    let size = NSSize(width: 20, height: 18)
    let base = NSImage(systemSymbolName: symbol, accessibilityDescription: "Photocopy")!
        .withSymbolConfiguration(.init(pointSize: 14, weight: .regular))!
    let img = NSImage(size: size, flipped: false) { rect in
        let symbolRect = NSRect(x: 1, y: 1,
                                width: base.size.width, height: base.size.height)
        base.draw(in: symbolRect)
        if badged {
            NSColor.systemRed.setFill()
            NSBezierPath(ovalIn: NSRect(x: rect.maxX - 8, y: rect.maxY - 8,
                                        width: 7, height: 7)).fill()
        }
        return true
    }
    img.isTemplate = !badged   // template = adapts to menu bar; color only for the dot
    return img
}

enum Pane: String, CaseIterable {
    case home, activity

    var icon: String {
        switch self {
        case .home: "house"
        case .activity: "bell"
        }
    }

    var label: String { rawValue.capitalized }
}

struct PanelView: View {
    @Bindable var agent: Agent
    @State private var pane: Pane = .home
    @State private var expandedThumb: URL?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                rail
                Group {
                    switch pane {
                    case .home: HomePane(agent: agent, expandedThumb: $expandedThumb)
                    case .activity: ActivityPane(agent: agent)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
            statusStrip
        }
        .frame(width: 460, height: 520)
        .background(Color.paper)
        .tint(.dbBlue)
        .onAppear { agent.markSeen() }
        .onReceive(NotificationCenter.default.publisher(
            for: NSWindow.didBecomeKeyNotification)) { _ in agent.markSeen() }
        .overlay {
            if let url = expandedThumb {
                ExpandedThumb(url: url) { expandedThumb = nil }
            }
        }
    }

    private var rail: some View {
        VStack(spacing: 0) {
            Image(systemName: "camera.fill")
                .font(.system(size: 20))
                .foregroundStyle(Color.ink)
                .padding(.top, 18)
                .padding(.bottom, 26)
                .accessibilityLabel("Photocopy")
            VStack(spacing: 24) {
                ForEach(Pane.allCases, id: \.self) { p in
                    Button {
                        pane = p
                    } label: {
                        VStack(spacing: 5) {
                            Image(systemName: p.icon)
                                .font(.system(size: 17, weight: .medium))
                                .foregroundStyle(Color.ink)
                                .frame(width: 44, height: 36)
                                .background(pane == p ? Color.creamDeep : .clear,
                                            in: .rect(cornerRadius: 12))
                            Text(p.label)
                                .font(.system(size: 11))
                                .foregroundStyle(Color.inkSecondary)
                        }
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(p.label)
                }
            }
            Spacer()
            Button {
                agent.stopDaemon()
                NSApp.terminate(nil)
            } label: {
                Image(systemName: "power")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.inkSecondary)
                    .frame(width: 44, height: 32)
            }
            .buttonStyle(.plain)
            .padding(.bottom, 14)
            .accessibilityLabel("Quit")
        }
        .frame(width: 72)
        .frame(maxHeight: .infinity)
        .background(Color.cream)
    }

    private var statusStrip: some View {
        HStack(spacing: 8) {
            Image(systemName: stripIcon)
                .foregroundStyle(stripColor)
            Text(stripText)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(Color.ink)
                .lineLimit(1)
            Spacer()
            if agent.state.status == "processing", agent.state.total > 0 {
                ProgressView(value: Double(agent.state.done),
                             total: Double(agent.state.total))
                    .frame(width: 90)
            }
        }
        .padding(.horizontal, 18)
        .frame(height: 48)
        .background(Color.paper)
        .overlay(alignment: .top) { Color.hairline.frame(height: 1) }
    }

    private var stripIcon: String {
        if !agent.daemonAlive { return "stop.circle" }
        return switch agent.state.status {
        case "processing": "arrow.triangle.2.circlepath"
        case "watching": "checkmark.circle"
        case "paused": "pause.circle"
        case "error": "exclamationmark.triangle"
        default: "circle"
        }
    }

    private var stripColor: Color {
        if !agent.daemonAlive { return .inkSecondary }
        return switch agent.state.status {
        case "processing": .dbBlue
        case "watching": .ink
        case "paused": .inkSecondary
        case "error": .red
        default: .inkSecondary
        }
    }

    private var stripText: String {
        if !agent.daemonAlive { return "Stopped — press Start to edit" }
        return switch agent.state.status {
        case "processing": "Editing \(agent.state.done)/\(agent.state.total) — \(agent.state.lastFile)"
        case "watching": agent.state.total > 0
            ? "Found \(agent.state.total) photos — starting…"
            : "Up to date"
        case "paused": "Paused"
        case "no-folder": "Choose a folder of photos"
        case "error": "Error — see Activity"
        default: agent.state.status
        }
    }
}

struct HomePane: View {
    @Bindable var agent: Agent
    @Binding var expandedThumb: URL?
    @State private var dropTargeted = false

    private let columns = [GridItem(.adaptive(minimum: 78), spacing: 8)]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                startStop
                if agent.daemonAlive, agent.state.status == "processing",
                   agent.state.total > 0 {
                    progressCard
                }
                dropArea
                actionRow
                if agent.config.edit {
                    profileRow
                }
                Toggle("Open in Lightroom when done", isOn: $agent.config.sendToLightroom)
                    .onChange(of: agent.config.sendToLightroom) { agent.saveConfig() }
                if !agent.thumbs.isEmpty {
                    thumbGrid
                }
            }
            .padding(16)
        }
    }

    private var progressCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: Double(agent.state.done), total: Double(agent.state.total))
                .tint(.dbBlue)
            HStack {
                Text(agent.state.done > 0
                     ? "Editing \(agent.state.done) of \(agent.state.total)"
                     : "Getting ready…")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Color.ink)
                Spacer()
                Text(agent.state.lastFile)
                    .font(.system(size: 12))
                    .foregroundStyle(Color.inkSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(12)
        .background(Color.cream, in: .rect(cornerRadius: 12))
    }

    private var startStop: some View {
        Button {
            agent.daemonAlive ? agent.stopDaemon() : agent.startDaemon()
        } label: {
            Label(agent.daemonAlive ? "Stop" : "Start editing",
                  systemImage: agent.daemonAlive ? "stop.fill" : "play.fill")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
                .frame(height: 46)
                .background(agent.daemonAlive ? Color.ink : Color.dbBlue,
                            in: .capsule)
        }
        .buttonStyle(.plain)
        .keyboardShortcut(.defaultAction)
        .accessibilityLabel(agent.daemonAlive ? "Stop" : "Start editing")
    }

    private var dropArea: some View {
        VStack(spacing: 6) {
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 24))
                .foregroundStyle(dropTargeted ? Color.dbBlue : Color.inkSecondary)
            Text(agent.config.watchDir.map {
                    ($0 as NSString).abbreviatingWithTildeInPath
                } ?? "Drop a folder of photos — or click to choose")
                .font(.system(size: 14))
                .foregroundStyle(agent.config.watchDir == nil ? Color.inkSecondary : Color.ink)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 22)
        .background(Color.cream, in: .rect(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(dropTargeted ? Color.dbBlue : Color.hairline,
                              style: StrokeStyle(lineWidth: 1.5, dash: [6, 5]))
        )
        .contentShape(.rect)
        .onTapGesture(perform: chooseFolder)
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
            _ = providers.first?.loadObject(ofClass: URL.self) { url, _ in
                if let url, (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                    Task { @MainActor in
                        agent.config.watchDir = url.path
                        agent.saveConfig()
                    }
                }
            }
            return true
        }
        .accessibilityLabel("Import folder")
    }

    private var actionRow: some View {
        HStack(spacing: 18) {
            Toggle("Edit", isOn: $agent.config.edit)
                .onChange(of: agent.config.edit) { agent.saveConfig() }
            Toggle("Cull", isOn: $agent.config.cull)
                .onChange(of: agent.config.cull) { agent.saveConfig() }
            if agent.config.cull {
                TextField("keep", value: cullTargetBinding, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 60)
                Text("keepers")
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .toggleStyle(.checkbox)
        .font(.system(size: 14))
        .foregroundStyle(Color.ink)
    }

    private var profileRow: some View {
        Picker("Profile", selection: $agent.config.look) {
            Text("All history").tag(String?.none)
            ForEach(agent.state.looks, id: \.self) { look in
                Text(look).tag(String?.some(look))
            }
        }
        .onChange(of: agent.config.look) { agent.saveConfig() }
    }

    private var thumbGrid: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Preview — predicted edits")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Color.inkSecondary)
            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(agent.thumbs, id: \.self) { url in
                    ThumbCell(url: url)
                        .onTapGesture { expandedThumb = url }
                }
            }
        }
    }

    private var cullTargetBinding: Binding<Int?> {
        Binding(get: { agent.config.cullTarget },
                set: { agent.config.cullTarget = ($0 ?? 0) > 0 ? $0 : nil
                       agent.saveConfig() })
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url {
            agent.config.watchDir = url.path
            agent.saveConfig()
        }
    }
}

struct ThumbCell: View {
    let url: URL

    var body: some View {
        AsyncImage(url: url) { image in
            image.resizable().scaledToFill()
        } placeholder: {
            Color(nsColor: .underPageBackgroundColor)
        }
        .frame(width: 78, height: 78)
        .clipShape(.rect(cornerRadius: 6))
        .accessibilityLabel(url.deletingPathExtension().lastPathComponent)
    }
}

struct ExpandedThumb: View {
    let url: URL
    let dismiss: () -> Void

    var body: some View {
        ZStack {
            Color.black.opacity(0.75)
            VStack(spacing: 8) {
                if let img = NSImage(contentsOf: url) {
                    Image(nsImage: img)
                        .resizable()
                        .scaledToFit()
                        .clipShape(.rect(cornerRadius: 8))
                }
                Text(url.deletingPathExtension().lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.white)
            }
            .padding(20)
        }
        .contentShape(.rect)
        .onTapGesture(perform: dismiss)
        .accessibilityAddTraits(.isButton)
        .accessibilityLabel("Close preview")
    }
}

struct ActivityPane: View {
    @Bindable var agent: Agent

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(agent.events) { event in
                    HStack(spacing: 12) {
                        Image(systemName: event.icon)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(event.kind == "done" ? Color.dbBlue : Color.ink)
                            .frame(width: 30, height: 30)
                            .background(Color.cream, in: .circle)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(event.text)
                                .font(.system(size: 13))
                                .foregroundStyle(Color.ink)
                                .lineLimit(2)
                            Text(event.time)
                                .font(.system(size: 11))
                                .foregroundStyle(Color.inkSecondary)
                        }
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 16)
                    Color.hairline.frame(height: 1).padding(.leading, 58)
                }
                if agent.events.isEmpty {
                    Text("No activity yet — edits will show up here")
                        .font(.system(size: 13))
                        .foregroundStyle(Color.inkSecondary)
                        .padding(20)
                }
            }
        }
        .overlay(alignment: .bottomTrailing) {
            Button("Open full log") {
                NSWorkspace.shared.open(workDir.appending(path: "agent.log"))
            }
            .controlSize(.small)
            .padding(10)
        }
    }
}
