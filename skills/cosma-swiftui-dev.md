# CosmaSense SwiftUI Frontend Development Guide for AI Agents

Use this skill when modifying the CosmaSense macOS SwiftUI app at `fileSearchForntend/`.

## Project Location

```
/Users/ethanpan/Documents/code/SAIL/fileSearchForntend/
```

## Architecture

- **State management**: Single `@Observable` `AppModel` class annotated `@MainActor`
- **API client**: `APIClient.swift` using `URLSession` async/await
- **Real-time updates**: `UpdatesStream.swift` (SSE via `URLSessionDataDelegate`)
- **Navigation**: `NavigationSplitView` with sidebar enum `SidebarItem`

## Key Files

| File | Purpose |
|------|---------|
| `AppModel.swift` | Central state, orchestrates API calls and SSE handling |
| `APIClient.swift` | HTTP client (GET/POST/PUT/DELETE helpers) |
| `APIModels.swift` | Codable request/response models |
| `UpdatesStream.swift` | SSE client with auto-reconnect |
| `WatchedFolder.swift` | Represents indexed folder with progress tracking |
| `HomeView.swift` | Search interface with @token parsing |
| `JobsView.swift` | Watched folders list |
| `SettingsView.swift` | Backend URL configuration |
| `ContentView.swift` | Root NavigationSplitView layout |

## Backend API Base URL

Default: `http://localhost:60534` (configurable in Settings, stored in UserDefaults).

## CodingKeys Pattern

Backend uses Python snake_case. Swift uses camelCase. Every Codable struct needs explicit `CodingKeys`:

```swift
struct FileResponse: Codable {
    let filePath: String
    let fileExtension: String  // avoid Swift keyword "extension"

    enum CodingKeys: String, CodingKey {
        case filePath = "file_path"
        case fileExtension = "extension"
    }
}
```

## Adding a New API Call

### Step 1: Add models to `APIModels.swift`

```swift
struct MyRequest: Codable {
    let someField: String

    enum CodingKeys: String, CodingKey {
        case someField = "some_field"
    }
}

struct MyResponse: Codable {
    let success: Bool
    let message: String
}
```

### Step 2: Add method to `APIClient.swift`

```swift
func doSomething(someField: String) async throws -> MyResponse {
    let url = baseURL.appendingPathComponent("/api/myfeature/do-something")
    let request = MyRequest(someField: someField)
    return try await post(url: url, body: request)
}
```

### Step 3: Call from `AppModel.swift`

```swift
func doSomething() async {
    do {
        let response = try await apiClient.doSomething(someField: "test")
        // Update @Observable state here
    } catch {
        self.statusMessage = "Error: \(error.localizedDescription)"
    }
}
```

## Handling New SSE Events

### Step 1: Add opcode to `EventOpcode` enum in `APIModels.swift`

```swift
enum EventOpcode: String, Codable {
    // ... existing opcodes ...
    case myEvent = "my_event"
}
```

### Step 2: Handle in `AppModel.handleBackend(event:)`

```swift
func handleBackend(event: BackendEvent) {
    switch event.opcode {
    // ... existing cases ...
    case .myEvent:
        if let data = event.data {
            // Process the event data
        }
    }
}
```

## Trailing Slash Convention

- POST endpoints that take a body: use trailing slash (`/api/search/`, `/api/watch/`)
- GET list endpoints: no trailing slash (`/api/watch/jobs`, `/api/queue/status`)
- DELETE with path param: no trailing slash (`/api/watch/jobs/{id}`)
- Settings: trailing slash (`/api/settings/`)

## Build & Run

```bash
# Build
xcodebuild -project fileSearchForntend.xcodeproj -scheme fileSearchForntend -configuration Debug build

# Clean build
xcodebuild clean -project fileSearchForntend.xcodeproj -scheme fileSearchForntend
rm -rf ~/Library/Developer/Xcode/DerivedData/fileSearchForntend-*
```

## Entitlements

Network client entitlement is required for localhost connections:
```xml
<key>com.apple.security.network.client</key>
<true/>
```

Verify: `plutil -lint fileSearchForntend/fileSearchForntend.entitlements`

## Token-Based Search Scoping

Users type `@FolderName` in the search bar to scope searches:
1. `@Doc` triggers autocomplete from `watchedFolders`
2. Tab/Enter creates a blue chip token
3. Backend receives `directory` parameter extracted from the token
