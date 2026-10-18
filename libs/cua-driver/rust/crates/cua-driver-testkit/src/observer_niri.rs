//! Independent niri observations over niri's IPC socket. This module never
//! calls the driver's niri adapter, capture, geometry, or input code.
//!
//! niri's IPC reports no stacking order and no position for tiled windows, so
//! occlusion is classified only where niri's own rendering rule decides it:
//! floating windows are not drawn while a tiled window in a fullscreen column
//! holds focus on the same workspace (`Workspace::is_floating_visible`). That
//! rule needs a static view, focus that does not follow the mouse, and a
//! fullscreen (not maximized) column, which IPC cannot tell apart by size. The
//! canonical niri lane (`CUA_E2E_WAYLAND_SESSION=niri`) guarantees all three,
//! so the rule is trusted only there. Every other overlap is reported as
//! ambiguous instead of guessed.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::Deserialize;

use super::{DesktopSnapshot, ObserverError, TargetWindow, TargetZ};

const QUERY_TIMEOUT: Duration = Duration::from_secs(1);
const REPLY_LIMIT: u64 = 1024 * 1024;
/// Logical pixels of slack when comparing a fullscreen tile with its output.
const FULLSCREEN_TOLERANCE: f64 = 1.0;

/// Connects without blocking past `timeout`: a wedged compositor stops
/// accepting, and a full backlog would block a plain connect forever.
fn connect(path: &Path, timeout: Duration) -> std::io::Result<UnixStream> {
    use std::os::unix::ffi::OsStrExt as _;

    let bytes = path.as_os_str().as_bytes();
    // SAFETY: an all-zero sockaddr_un is a valid empty address.
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    if bytes.len() >= address.sun_path.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "socket path is too long",
        ));
    }
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (slot, byte) in address.sun_path.iter_mut().zip(bytes) {
        *slot = *byte as libc::c_char;
    }
    let deadline = Instant::now() + timeout;
    loop {
        // SAFETY: plain socket creation; ownership moves into UnixStream below.
        let fd = unsafe {
            libc::socket(
                libc::AF_UNIX,
                libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
                0,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: fd is a fresh socket owned only here.
        let stream = unsafe { UnixStream::from_raw_fd(fd) };
        // SAFETY: address is a valid sockaddr_un of the given length.
        let connected = unsafe {
            libc::connect(
                fd,
                (&address as *const libc::sockaddr_un).cast(),
                std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
            )
        };
        if connected == 0 {
            stream.set_nonblocking(false)?;
            return Ok(stream);
        }
        let error = std::io::Error::last_os_error();
        // A Unix stream connect either completes or reports a full backlog.
        if error.raw_os_error() != Some(libc::EAGAIN) || Instant::now() >= deadline {
            return Err(error);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn peer_pid(stream: &UnixStream) -> std::io::Result<i32> {
    // SAFETY: an all-zero ucred is valid output storage.
    let mut credentials: libc::ucred = unsafe { std::mem::zeroed() };
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: the pointer and length describe `credentials`.
    let result = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut credentials as *mut libc::ucred).cast(),
            &mut length,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(credentials.pid)
}

fn socket_peer(path: &Path) -> std::io::Result<i32> {
    peer_pid(&connect(path, QUERY_TIMEOUT)?)
}

fn niri_socket() -> Option<PathBuf> {
    std::env::var_os("NIRI_SOCKET")
        .filter(|socket| !socket.is_empty())
        .map(PathBuf::from)
}

fn wayland_socket() -> Option<PathBuf> {
    let display = PathBuf::from(std::env::var_os("WAYLAND_DISPLAY")?);
    if display.is_absolute() {
        return Some(display);
    }
    Some(PathBuf::from(std::env::var_os("XDG_RUNTIME_DIR")?).join(display))
}

/// The niri pid this process attested first. A later niri, for example a
/// restarted compositor, is a different session and is refused.
static ATTESTED: OnceLock<i32> = OnceLock::new();

/// Whether `NIRI_SOCKET` belongs to the compositor serving `WAYLAND_DISPLAY`.
/// `Ok(false)` means an unrelated or stale socket, inherited for example by a
/// nested session started from a niri desktop, so other observers may claim
/// the session. A pid that differs from the attested one is an error.
pub(super) fn attest() -> Result<bool, ObserverError> {
    let (Some(niri), Some(wayland)) = (niri_socket(), wayland_socket()) else {
        return Ok(false);
    };
    let (Ok(niri), Ok(wayland)) = (socket_peer(&niri), socket_peer(&wayland)) else {
        return Ok(false);
    };
    if niri != wayland {
        return Ok(false);
    }
    let attested = *ATTESTED.get_or_init(|| niri);
    if attested != niri {
        return Err(ObserverError::new(format!(
            "niri pid changed from {attested} to {niri}; refusing a new compositor"
        )));
    }
    Ok(true)
}

/// Sends one request on a fresh connection and returns the `Ok` payload.
fn request(request: &serde_json::Value) -> Result<serde_json::Value, ObserverError> {
    let socket =
        niri_socket().ok_or_else(|| ObserverError::new("niri observer requires NIRI_SOCKET"))?;
    let error = |operation: &str, error: std::io::Error| {
        ObserverError::new(format!("niri IPC {operation} {request}: {error}"))
    };
    if ATTESTED.get().is_none() && !attest()? {
        return Err(ObserverError::new(
            "NIRI_SOCKET is not served by the WAYLAND_DISPLAY compositor",
        ));
    }
    let mut stream = connect(&socket, QUERY_TIMEOUT).map_err(|e| error("connect", e))?;
    let peer = peer_pid(&stream).map_err(|e| error("read peer of", e))?;
    if Some(&peer) != ATTESTED.get() {
        return Err(ObserverError::new(format!(
            "niri IPC peer {peer} is not the attested niri {:?}",
            ATTESTED.get()
        )));
    }
    stream
        .set_read_timeout(Some(QUERY_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(QUERY_TIMEOUT)))
        .map_err(|e| error("configure", e))?;
    writeln!(stream, "{request}").map_err(|e| error("write", e))?;
    let mut line = String::new();
    BufReader::new(stream.take(REPLY_LIMIT))
        .read_line(&mut line)
        .map_err(|e| error("read", e))?;
    parse_reply(&line)
}

fn parse_reply(line: &str) -> Result<serde_json::Value, ObserverError> {
    let reply: serde_json::Value = serde_json::from_str(line)
        .map_err(|error| ObserverError::new(format!("invalid niri IPC reply: {error}")))?;
    match reply {
        serde_json::Value::Object(mut reply) if reply.len() == 1 => {
            if let Some(payload) = reply.remove("Ok") {
                Ok(payload)
            } else if let Some(message) = reply.remove("Err") {
                Err(ObserverError::new(format!("niri IPC refused: {message}")))
            } else {
                Err(ObserverError::new("niri IPC reply is neither Ok nor Err"))
            }
        }
        _ => Err(ObserverError::new("niri IPC reply is neither Ok nor Err")),
    }
}

/// Reads one response kind, e.g. `{"Ok":{"Windows":[...]}}` for `"Windows"`.
fn query<T: DeserializeOwned>(name: &str) -> Result<T, ObserverError> {
    let mut payload = request(&serde_json::Value::String(name.to_owned()))?;
    let value = payload
        .get_mut(name)
        .map(serde_json::Value::take)
        .ok_or_else(|| ObserverError::new(format!("niri answered {name} with another response")))?;
    serde_json::from_value(value)
        .map_err(|error| ObserverError::new(format!("invalid niri {name} JSON: {error}")))
}

#[derive(Clone, Debug, Deserialize)]
struct Layout {
    tile_size: (f64, f64),
    tile_pos_in_workspace_view: Option<(f64, f64)>,
}

#[derive(Clone, Debug, Deserialize)]
struct Window {
    id: u64,
    pid: Option<i64>,
    workspace_id: Option<u64>,
    is_focused: bool,
    is_floating: bool,
    layout: Layout,
}

#[derive(Clone, Debug, Deserialize)]
struct Workspace {
    id: u64,
    output: Option<String>,
    is_active: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct Logical {
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct Output {
    logical: Option<Logical>,
}

#[derive(Clone, Debug, Deserialize)]
struct Overview {
    is_open: bool,
}

struct Desktop {
    windows: Vec<Window>,
    workspaces: Vec<Workspace>,
    outputs: HashMap<String, Output>,
    overview_open: bool,
    /// The canonical niri lane's config: animations and focus-follows-mouse
    /// off, and only the sentinel's fullscreen column covers the output.
    lane_config: bool,
}

fn desktop() -> Result<Desktop, ObserverError> {
    Ok(Desktop {
        windows: query("Windows")?,
        workspaces: query("Workspaces")?,
        outputs: query("Outputs")?,
        overview_open: query::<Overview>("OverviewState")?.is_open,
        lane_config: std::env::var("CUA_E2E_WAYLAND_SESSION").as_deref() == Ok("niri"),
    })
}

fn focused(windows: &[Window]) -> Result<Option<&Window>, ObserverError> {
    let mut focused = windows.iter().filter(|window| window.is_focused);
    let first = focused.next();
    if focused.next().is_some() {
        return Err(ObserverError::new(
            "niri reports more than one focused window",
        ));
    }
    Ok(first)
}

/// The target by exact niri id, or by its pid's only window when no id is known.
fn select(windows: &[Window], target: TargetWindow) -> Result<Option<&Window>, ObserverError> {
    let pid = Some(i64::from(target.pid));
    if target.native_id != 0 {
        let window = windows.iter().find(|window| window.id == target.native_id);
        if window.is_some_and(|window| window.pid != pid) {
            return Err(ObserverError::new(
                "niri target window id belongs to a different pid",
            ));
        }
        return Ok(window);
    }
    let mut matches = windows.iter().filter(|window| window.pid == pid);
    let first = matches.next();
    if matches.next().is_some() {
        return Err(ObserverError::new(
            "niri target pid has multiple windows; native_id required",
        ));
    }
    Ok(first)
}

fn covers_output(window: &Window, output: &Logical) -> bool {
    let (width, height) = window.layout.tile_size;
    width + FULLSCREEN_TOLERANCE >= f64::from(output.width)
        && height + FULLSCREEN_TOLERANCE >= f64::from(output.height)
}

fn floating_overlap(a: &Window, b: &Window) -> Option<bool> {
    let (ax, ay) = a.layout.tile_pos_in_workspace_view?;
    let (bx, by) = b.layout.tile_pos_in_workspace_view?;
    let (aw, ah) = a.layout.tile_size;
    let (bw, bh) = b.layout.tile_size;
    Some(ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah)
}

fn classify(desktop: &Desktop, target: TargetWindow) -> Result<TargetZ, ObserverError> {
    let Some(target) = select(&desktop.windows, target)? else {
        return Ok(TargetZ::NotFound);
    };
    if target.is_focused {
        return Ok(TargetZ::Foreground);
    }
    let workspace = target.workspace_id.and_then(|id| {
        desktop
            .workspaces
            .iter()
            .find(|workspace| workspace.id == id)
    });
    let Some(workspace) = workspace.filter(|workspace| workspace.is_active) else {
        // niri has no minimize; a window off the active workspace is not shown.
        return Ok(TargetZ::Minimized);
    };
    if desktop.overview_open {
        return Err(ObserverError::new(
            "niri overview is open; z-order is ambiguous",
        ));
    }
    let output = workspace
        .output
        .as_ref()
        .and_then(|name| desktop.outputs.get(name))
        .and_then(|output| output.logical.as_ref())
        .ok_or_else(|| ObserverError::new("niri target workspace has no logical output"))?;
    if !target.is_floating {
        return Err(ObserverError::new(
            "niri IPC reports no position for tiled windows; z-order is ambiguous",
        ));
    }
    let focused = focused(&desktop.windows)?;
    if focused.is_some_and(|focused| {
        !focused.is_floating
            && focused.workspace_id == Some(workspace.id)
            && covers_output(focused, output)
    }) {
        if !desktop.lane_config {
            return Err(ObserverError::new(
                "niri fullscreen occlusion is only certified under the niri lane config",
            ));
        }
        return Ok(TargetZ::BackgroundOccluded);
    }
    for other in &desktop.windows {
        if other.id == target.id || !other.is_floating || other.workspace_id != target.workspace_id
        {
            continue;
        }
        // IPC has no floating stacking order, so any overlap is ambiguous.
        if floating_overlap(target, other) != Some(false) {
            return Err(ObserverError::new(
                "niri overlapping floating windows have ambiguous z-order",
            ));
        }
    }
    Ok(TargetZ::BackgroundVisible)
}

pub(super) fn probe() -> Result<(), ObserverError> {
    let version: serde_json::Value = query("Version")?;
    if !version.is_string() {
        return Err(ObserverError::new(
            "niri answered Version without a version",
        ));
    }
    desktop().map(|_| ())
}

pub(super) fn snapshot(target: TargetWindow) -> Result<DesktopSnapshot, ObserverError> {
    let desktop = desktop()?;
    let focus = focused(&desktop.windows)?.map(|window| window.id);
    Ok(DesktopSnapshot {
        foreground: focus,
        input_focus: focus,
        target_z: classify(&desktop, target)?,
        cursor_pos: None,
    })
}

pub(super) fn focus_identity() -> Result<Option<u64>, ObserverError> {
    Ok(focused(&query::<Vec<Window>>("Windows")?)?.map(|window| window.id))
}

/// Whether focus left `from` for `to`, both resolved by exact identity. This
/// does not use z-order: the tiled sentinel has no IPC position, so its own
/// z-order is ambiguous once a floating window holds focus.
fn focus_transferred(
    windows: &[Window],
    from: TargetWindow,
    to: TargetWindow,
) -> Result<bool, ObserverError> {
    let resolve = |target| {
        select(windows, target)?
            .map(|window| window.id)
            .ok_or_else(|| ObserverError::new("niri focus-transfer window not found"))
    };
    let (from, to) = (resolve(from)?, resolve(to)?);
    if from == to {
        return Err(ObserverError::new(
            "niri focus transfer requires two distinct windows",
        ));
    }
    Ok(focused(windows)?.map(|window| window.id) == Some(to))
}

pub(super) fn wait_for_focus_transfer(
    from: TargetWindow,
    to: TargetWindow,
    timeout: Duration,
) -> Result<(), ObserverError> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let windows: Vec<Window> = query("Windows")?;
        if focus_transferred(&windows, from, to)? {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(ObserverError::new(format!(
                "niri focus did not move from window {} to window {} within {timeout:?}; focused: {:?}",
                from.native_id,
                to.native_id,
                focused(&windows)?.map(|window| window.id)
            )));
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Focuses the target through niri's own action, for test setup only.
pub(super) fn focus(target: TargetWindow) -> Result<(), ObserverError> {
    let id = select(&query::<Vec<Window>>("Windows")?, target)?
        .ok_or_else(|| ObserverError::new("niri target window not found"))?
        .id;
    let reply = request(&serde_json::json!({ "Action": { "FocusWindow": { "id": id } } }))?;
    if reply == serde_json::json!("Handled") {
        Ok(())
    } else {
        Err(ObserverError::new(format!(
            "niri answered FocusWindow with {reply}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn window(id: u64, pid: i64, floating: bool, focused: bool) -> serde_json::Value {
        json!({
            "id": id, "title": format!("window {id}"), "app_id": "fixture", "pid": pid,
            "workspace_id": 1, "is_focused": focused, "is_floating": floating,
            "is_urgent": false, "focus_timestamp": null,
            "layout": {
                "pos_in_scrolling_layout": if floating { json!(null) } else { json!([1, 1]) },
                "tile_size": [940.0, 780.0], "window_size": [940, 780],
                "tile_pos_in_workspace_view": if floating { json!([0.0, 0.0]) } else { json!(null) },
                "window_offset_in_tile": [0.0, 0.0],
            },
        })
    }

    fn fullscreen(mut window: serde_json::Value) -> serde_json::Value {
        window["layout"]["tile_size"] = json!([1920.0, 1080.0]);
        window["layout"]["window_size"] = json!([1920, 1080]);
        window
    }

    fn desktop(windows: Vec<serde_json::Value>) -> Desktop {
        Desktop {
            windows: serde_json::from_value(json!(windows)).unwrap(),
            workspaces: serde_json::from_value(json!([
                {"id": 1, "idx": 1, "name": null, "output": "winit", "is_urgent": false,
                 "is_active": true, "is_focused": true, "active_window_id": 10},
                {"id": 2, "idx": 2, "name": null, "output": "winit", "is_urgent": false,
                 "is_active": false, "is_focused": false, "active_window_id": null},
            ]))
            .unwrap(),
            outputs: serde_json::from_value(json!({
                "winit": {"name": "winit", "make": "Smithay", "model": "Winit",
                    "serial": null, "physical_size": null, "modes": [], "current_mode": null,
                    "is_custom_mode": true, "vrr_supported": false, "vrr_enabled": false,
                    "logical": {"x": 0, "y": 0, "width": 1920, "height": 1080, "scale": 1.0,
                        "transform": "Flipped180"}}
            }))
            .unwrap(),
            overview_open: false,
            lane_config: true,
        }
    }

    fn target(native_id: u64, pid: u32) -> TargetWindow {
        TargetWindow { pid, native_id }
    }

    #[test]
    fn floating_target_behind_focused_fullscreen_sentinel_is_occluded() {
        let desktop = desktop(vec![
            fullscreen(window(10, 100, false, true)),
            window(20, 200, true, false),
        ]);
        assert_eq!(
            classify(&desktop, target(20, 200)).unwrap(),
            TargetZ::BackgroundOccluded
        );
        assert_eq!(
            classify(&desktop, target(10, 100)).unwrap(),
            TargetZ::Foreground
        );
    }

    #[test]
    fn fullscreen_sized_tile_outside_the_lane_config_is_ambiguous() {
        // A maximized column, focus-follows-mouse or a moving view leaves
        // floating windows visible over a tile of the same size.
        let mut desktop = desktop(vec![
            fullscreen(window(10, 100, false, true)),
            window(20, 200, true, false),
        ]);
        desktop.lane_config = false;
        assert!(classify(&desktop, target(20, 200))
            .unwrap_err()
            .to_string()
            .contains("only certified under the niri lane config"));
    }

    #[test]
    fn floating_target_above_a_non_fullscreen_tile_is_visible() {
        let desktop = desktop(vec![
            window(10, 100, false, true),
            window(20, 200, true, false),
        ]);
        assert_eq!(
            classify(&desktop, target(20, 200)).unwrap(),
            TargetZ::BackgroundVisible
        );
    }

    #[test]
    fn floating_overlap_tiled_target_and_open_overview_are_ambiguous() {
        let overlapping = desktop(vec![
            window(10, 100, true, true),
            window(20, 200, true, false),
        ]);
        assert!(classify(&overlapping, target(20, 200))
            .unwrap_err()
            .to_string()
            .contains("overlapping floating windows"));

        let tiled = desktop(vec![
            fullscreen(window(10, 100, false, true)),
            window(20, 200, false, false),
        ]);
        assert!(classify(&tiled, target(20, 200))
            .unwrap_err()
            .to_string()
            .contains("no position for tiled windows"));

        let mut overview = desktop(vec![
            fullscreen(window(10, 100, false, true)),
            window(20, 200, true, false),
        ]);
        overview.overview_open = true;
        assert!(classify(&overview, target(20, 200))
            .unwrap_err()
            .to_string()
            .contains("overview is open"));
    }

    #[test]
    fn window_on_an_inactive_workspace_is_not_shown() {
        let mut hidden = window(20, 200, true, false);
        hidden["workspace_id"] = json!(2);
        let desktop = desktop(vec![fullscreen(window(10, 100, false, true)), hidden]);
        assert_eq!(
            classify(&desktop, target(20, 200)).unwrap(),
            TargetZ::Minimized
        );
    }

    #[test]
    fn identity_is_the_exact_niri_id_and_never_a_pid_sibling() {
        let desktop = desktop(vec![
            fullscreen(window(10, 100, false, true)),
            window(20, 200, true, false),
            window(21, 200, true, false),
        ]);
        assert_eq!(
            classify(&desktop, target(30, 200)).unwrap(),
            TargetZ::NotFound
        );
        assert!(classify(&desktop, target(20, 300))
            .unwrap_err()
            .to_string()
            .contains("different pid"));
        assert!(classify(&desktop, target(0, 200))
            .unwrap_err()
            .to_string()
            .contains("native_id required"));
    }

    #[test]
    fn focus_canary_sees_the_floating_target_take_focus_from_the_sentinel() {
        let sentinel = target(10, 100);
        let harness = target(20, 200);
        let before = desktop(vec![
            fullscreen(window(10, 100, false, true)),
            window(20, 200, true, false),
        ]);
        assert!(!focus_transferred(&before.windows, sentinel, harness).unwrap());

        let after = desktop(vec![
            fullscreen(window(10, 100, false, false)),
            window(20, 200, true, true),
        ]);
        assert!(focus_transferred(&after.windows, sentinel, harness).unwrap());
        // The unfocused tiled sentinel has no IPC position, so z-order cannot
        // decide this canary.
        assert!(classify(&after, sentinel).is_err());
        assert!(focus_transferred(&after.windows, sentinel, sentinel).is_err());
    }

    #[test]
    fn socket_peer_is_the_listening_process() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("niri.sock");
        let _listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        assert_eq!(socket_peer(&path).unwrap() as u32, std::process::id());
        assert!(socket_peer(&directory.path().join("missing.sock")).is_err());
    }

    #[test]
    fn replies_unwrap_ok_and_preserve_err() {
        assert_eq!(
            parse_reply(r#"{"Ok":{"Version":"26.04"}}"#).unwrap(),
            json!({"Version": "26.04"})
        );
        assert!(parse_reply(r#"{"Err":"no such window"}"#)
            .unwrap_err()
            .to_string()
            .contains("no such window"));
        assert!(parse_reply("").is_err());
    }
}
