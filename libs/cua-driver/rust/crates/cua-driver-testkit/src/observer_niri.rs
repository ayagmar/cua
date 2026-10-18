//! Independent niri observations over niri's IPC socket. This module never
//! calls the driver's niri adapter, capture, geometry, or input code.
//!
//! niri's IPC reports no stacking order and no position for tiled windows, so
//! occlusion is classified only where niri's own rendering rule decides it:
//! floating windows are not drawn while a tiled window in a fullscreen column
//! holds focus on the same workspace (`Workspace::is_floating_visible`). That
//! rule needs a static view, so it holds only with animations off and focus
//! not following the mouse, as the canonical niri lane configures. Every other
//! overlap is reported as ambiguous instead of guessed.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::Deserialize;

use super::{DesktopSnapshot, ObserverError, TargetWindow, TargetZ};

const QUERY_TIMEOUT: Duration = Duration::from_secs(1);
const REPLY_LIMIT: u64 = 1024 * 1024;
/// Logical pixels of slack when comparing a fullscreen tile with its output.
const FULLSCREEN_TOLERANCE: f64 = 1.0;

/// Sends one request on a fresh connection and returns the `Ok` payload.
fn request(request: &serde_json::Value) -> Result<serde_json::Value, ObserverError> {
    let socket = std::env::var_os("NIRI_SOCKET")
        .filter(|socket| !socket.is_empty())
        .ok_or_else(|| ObserverError::new("niri observer requires NIRI_SOCKET"))?;
    let error = |operation: &str, error: std::io::Error| {
        ObserverError::new(format!("niri IPC {operation} {request}: {error}"))
    };
    let mut stream = UnixStream::connect(&socket).map_err(|e| error("connect", e))?;
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
}

fn desktop() -> Result<Desktop, ObserverError> {
    Ok(Desktop {
        windows: query("Windows")?,
        workspaces: query("Workspaces")?,
        outputs: query("Outputs")?,
        overview_open: query::<Overview>("OverviewState")?.is_open,
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
