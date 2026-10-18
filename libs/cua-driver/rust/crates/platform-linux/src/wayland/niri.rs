//! niri identity, geometry and focus over niri's IPC socket.
//!
//! Window ids are niri's own window ids. Every IPC connection is attested:
//! the `NIRI_SOCKET` peer and the `WAYLAND_DISPLAY` peer must both be the
//! compositor process first recognised for this driver process. Once niri is
//! recognised, an IPC failure refuses instead of falling back to the generic
//! wlr-foreign-toplevel title join, which would lose the niri id namespace.
//!
//! niri 26.04 reports a position only for floating windows
//! (`tile_pos_in_workspace_view`): a tiled window's place depends on the
//! view's scroll position, which IPC does not report. A tiled window's
//! screenshot is therefore its whole output, and its window coordinates are
//! pixels of that screenshot while niri reports no layout change, for at
//! most [`FRAME_MAX_AGE`] (see [`OutputFrame`]). IPC positions also omit niri's animation offset, so
//! pointer and crop routes wait until the layout has been quiet for a while
//! (see [`History`]).

use anyhow::{anyhow, bail, Context, Result};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::compositor_peer::{peer_pid, wayland_connection_with_timeout};

const QUERY_TIMEOUT: Duration = Duration::from_secs(1);
const QUERY_TOTAL_TIMEOUT: Duration = Duration::from_secs(3);
const QUERY_RETRY_BACKOFF: Duration = Duration::from_millis(50);
const QUERY_MAX_ATTEMPTS: usize = 2;
const MAX_REPLY_BYTES: usize = 4 * 1024 * 1024;
const MAX_LOGICAL_PIXELS: u64 = 64 * 1024 * 1024;
/// How long niri may take to apply and report a focus change.
const FOCUS_CONFIRM_TIMEOUT: Duration = Duration::from_millis(500);
/// How long the layout must stay unchanged before IPC geometry is taken to
/// be what niri renders. This is an assumption, not proof: niri's IPC
/// positions exclude animation offsets and niri reports no animation state,
/// so an animation longer than this (a slow `animations` config) is not
/// detected, and input or a window capture can then use the destination
/// while the window is still drawn elsewhere. niri's default animations
/// finish well inside it.
const SETTLE_QUIET: Duration = Duration::from_secs(1);
/// How long a geometry-bound route waits for the layout to go quiet.
const SETTLE_LIMIT: Duration = Duration::from_secs(4);
/// Recent window-local to layout translations kept for the pointer gate.
const MAX_AIMS: usize = 64;
/// How long a tiled window's output screenshot stays usable for pointer
/// input. niri scrolls a view without reporting it (see [`Shown`]), so an
/// unchanged layout proves less the older the screenshot is.
pub const FRAME_MAX_AGE: Duration = Duration::from_secs(60);

// ── Session ──────────────────────────────────────────────────────────────────

/// The compositor process recognised as niri for this driver process.
static RECOGNIZED: OnceLock<libc::pid_t> = OnceLock::new();

#[derive(Debug, PartialEq, Eq)]
enum Attestation {
    /// `NIRI_SOCKET` belongs to another session: not niri for this process.
    Unrelated(String),
    Attested(libc::pid_t),
    /// niri was recognised, but this connection cannot be proven to reach it.
    Refused(String),
}

/// Session identity is tri-state. Before recognition, IPC and Wayland peers
/// that differ (an inherited socket of another desktop) mean an unrelated
/// session. After recognition, any loss, mismatch or new compositor process
/// refuses rather than falling through to generic discovery.
fn attest(
    pinned: Option<libc::pid_t>,
    ipc_peer: Result<libc::pid_t>,
    wayland_peer: Result<libc::pid_t>,
) -> Attestation {
    let (ipc_peer, wayland_peer) = match (ipc_peer, wayland_peer) {
        (Ok(ipc), Ok(wayland)) => (ipc, wayland),
        (ipc, wayland) => {
            let reason = format!(
                "niri IPC or WAYLAND_DISPLAY peer is unavailable (IPC: {}, Wayland: {})",
                ipc.map_or_else(|e| format!("{e:#}"), |pid| format!("pid {pid}")),
                wayland.map_or_else(|e| format!("{e:#}"), |pid| format!("pid {pid}")),
            );
            return match pinned {
                None => Attestation::Unrelated(reason),
                Some(_) => Attestation::Refused(reason),
            };
        }
    };
    if ipc_peer != wayland_peer {
        let reason = format!(
            "NIRI_SOCKET is served by pid {ipc_peer} but WAYLAND_DISPLAY by pid {wayland_peer}"
        );
        return match pinned {
            None => Attestation::Unrelated(reason),
            Some(_) => Attestation::Refused(reason),
        };
    }
    match pinned {
        Some(pinned) if pinned != ipc_peer => Attestation::Refused(format!(
            "niri was recognised as pid {pinned}, but the compositor is now pid {ipc_peer}"
        )),
        _ => Attestation::Attested(ipc_peer),
    }
}

fn socket_path() -> Result<PathBuf> {
    let path = PathBuf::from(std::env::var_os("NIRI_SOCKET").context("missing NIRI_SOCKET")?);
    if !path.is_absolute() {
        bail!("NIRI_SOCKET must be an absolute path");
    }
    Ok(path)
}

fn connect(timeout: Duration) -> Result<UnixStream> {
    let socket = socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None)?;
    socket.connect_timeout(&socket2::SockAddr::unix(socket_path()?)?, timeout)?;
    Ok(socket.into())
}

/// The attestation, the IPC connection, and whether the answer is lasting:
/// a socket nobody serves, or two different live peers. A timeout is not.
fn probe(
    pinned: Option<libc::pid_t>,
    timeout: Duration,
) -> (Attestation, Option<UnixStream>, bool) {
    let ipc = connect(timeout);
    let ipc_peer = ipc
        .as_ref()
        .map_err(|e| anyhow!("{e:#}"))
        .and_then(|ipc| peer_pid(ipc.as_raw_fd()));
    // Without an IPC peer there is nothing to compare the Wayland peer with.
    let wayland_peer = match &ipc_peer {
        Ok(_) => wayland_connection_with_timeout(timeout)
            .and_then(|wayland| peer_pid(wayland.backend().poll_fd().as_raw_fd())),
        Err(_) => Err(anyhow!("not checked")),
    };
    let lasting = match &ipc {
        Err(error) => error.downcast_ref::<io::Error>().is_some_and(|error| {
            matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            )
        }),
        Ok(_) => ipc_peer.is_ok() && wayland_peer.is_ok(),
    };
    (attest(pinned, ipc_peer, wayland_peer), ipc.ok(), lasting)
}

/// True when this process drives a niri session: `NIRI_SOCKET` and
/// `WAYLAND_DISPLAY` were attested to be the same niri process, now or
/// earlier in this process. Recognition is remembered, so a later socket
/// loss refuses instead of turning this off.
pub fn is_session() -> bool {
    match session_check(
        !super::is_inject_mode() && super::is_wayland(),
        RECOGNIZED.get().is_some(),
        std::env::var_os("NIRI_SOCKET").is_some_and(|socket| !socket.is_empty()),
    ) {
        SessionCheck::NotNiri => return false,
        SessionCheck::Recognized => return true,
        SessionCheck::Probe => {}
    }
    // An unrelated socket stays unrelated for the same environment: a stale
    // NIRI_SOCKET on another desktop must not cost a probe on every call.
    static UNRELATED: Mutex<Option<(OsString, OsString)>> = Mutex::new(None);
    let environment = (
        std::env::var_os("NIRI_SOCKET").unwrap_or_default(),
        std::env::var_os("WAYLAND_DISPLAY").unwrap_or_default(),
    );
    if UNRELATED
        .lock()
        .is_ok_and(|unrelated| unrelated.as_ref() == Some(&environment))
    {
        return false;
    }
    let (attestation, _, lasting) = probe(None, QUERY_TIMEOUT);
    match attestation {
        Attestation::Attested(pid) => {
            let pinned = *RECOGNIZED.get_or_init(|| pid);
            tracing::info!(pid = pinned, "niri session recognised");
            true
        }
        Attestation::Unrelated(reason) | Attestation::Refused(reason) => {
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("NIRI_SOCKET is not this Wayland session's niri: {reason}");
            }
            if let (true, Ok(mut unrelated)) = (lasting, UNRELATED.lock()) {
                *unrelated = Some(environment);
            }
            false
        }
    }
}

#[derive(Debug, PartialEq)]
enum SessionCheck {
    NotNiri,
    /// Recognised earlier: stay niri even if the socket is gone, so routes
    /// refuse instead of falling through to generic Wayland discovery.
    Recognized,
    Probe,
}

fn session_check(adapter_on: bool, recognized: bool, socket_set: bool) -> SessionCheck {
    if !adapter_on {
        SessionCheck::NotNiri
    } else if recognized {
        SessionCheck::Recognized
    } else if socket_set {
        SessionCheck::Probe
    } else {
        SessionCheck::NotNiri
    }
}

/// A connection to the recognised niri process, re-attested each time.
fn attested_connection(deadline: Instant) -> Result<UnixStream> {
    let pinned = *RECOGNIZED
        .get()
        .context("niri session has not been recognised")?;
    match probe(Some(pinned), time_remaining(deadline)?) {
        (Attestation::Attested(_), Some(ipc), _) => Ok(ipc),
        (Attestation::Unrelated(reason) | Attestation::Refused(reason), ..) => bail!("{reason}"),
        (Attestation::Attested(_), None, _) => bail!("niri IPC connection unavailable"),
    }
}

/// Bind a Wayland connection used for capture or input to the recognised
/// niri process.
pub(super) fn verify_wayland_peer(connection: &wayland_client::Connection) -> Result<()> {
    let pinned = *RECOGNIZED
        .get()
        .context("niri session has not been recognised")?;
    let peer = peer_pid(connection.backend().poll_fd().as_raw_fd())?;
    if peer != pinned {
        bail!("WAYLAND_DISPLAY is served by pid {peer}, not the recognised niri pid {pinned}");
    }
    Ok(())
}

/// A Wayland client socket proven to reach the recognised niri process, for
/// a keyboard helper that must not rediscover the display on its own.
pub(super) fn attested_wayland_socket() -> Result<UnixStream> {
    let pinned = *RECOGNIZED
        .get()
        .context("niri session has not been recognised")?;
    let socket = super::compositor_peer::wayland_socket_with_timeout(QUERY_TIMEOUT)?;
    let peer = peer_pid(socket.as_raw_fd())?;
    if peer != pinned {
        bail!("WAYLAND_DISPLAY is served by pid {peer}, not the recognised niri pid {pinned}");
    }
    Ok(socket)
}

/// IPC to the recognised niri process. One value serves one action: its
/// requests share one attested connection.
type AttestedIpc = Ipc<fn(Instant) -> Result<UnixStream>>;

fn ipc() -> AttestedIpc {
    Ipc {
        connect: attested_connection,
        per_attempt: QUERY_TIMEOUT,
        total: QUERY_TOTAL_TIMEOUT,
        backoff: QUERY_RETRY_BACKOFF,
        lines: None,
    }
}

// ── IPC transport ────────────────────────────────────────────────────────────

fn time_remaining(deadline: Instant) -> Result<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(io::Error::new(io::ErrorKind::TimedOut, "niri IPC timed out").into());
    }
    Ok(remaining)
}

fn is_timeout(error: &anyhow::Error) -> bool {
    error.downcast_ref::<io::Error>().is_some_and(|error| {
        matches!(
            error.kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
        )
    })
}

/// niri replies one JSON line per request; the event stream sends one line
/// per event.
struct Lines {
    stream: UnixStream,
    buffer: Vec<u8>,
    /// Bytes of `buffer` already known to hold no line end.
    scanned: usize,
}

impl Lines {
    fn new(stream: UnixStream) -> Self {
        Self {
            stream,
            buffer: Vec::new(),
            scanned: 0,
        }
    }

    fn send(&mut self, line: &str, deadline: Instant) -> Result<()> {
        self.stream
            .set_write_timeout(Some(time_remaining(deadline)?))?;
        self.stream.write_all(line.as_bytes())?;
        self.stream.write_all(b"\n")?;
        Ok(())
    }

    /// The next complete line, or `None` once `deadline` passes first.
    fn next(&mut self, deadline: Instant) -> Result<Option<Vec<u8>>> {
        let mut chunk = [0u8; 8192];
        loop {
            if let Some(end) = self.buffer[self.scanned..]
                .iter()
                .position(|byte| *byte == b'\n')
            {
                let mut line: Vec<u8> = self.buffer.drain(..=self.scanned + end).collect();
                line.pop();
                self.scanned = 0;
                return Ok(Some(line));
            }
            self.scanned = self.buffer.len();
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            self.stream.set_read_timeout(Some(remaining))?;
            match self.stream.read(&mut chunk) {
                Ok(0) => bail!("niri closed the IPC connection"),
                Ok(count) => {
                    if self.buffer.len() + count > MAX_REPLY_BYTES {
                        bail!("niri IPC reply exceeds size limit");
                    }
                    self.buffer.extend_from_slice(&chunk[..count]);
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    return Ok(None)
                }
                Err(error) => return Err(error).context("niri IPC reply unavailable"),
            }
        }
    }

    /// One request and its reply line.
    fn exchange(&mut self, request: &str, deadline: Instant) -> Result<serde_json::Value> {
        self.send(request, deadline)?;
        let line = self.next(deadline)?.ok_or_else(|| {
            anyhow::Error::from(io::Error::new(
                io::ErrorKind::TimedOut,
                "niri IPC reply timed out",
            ))
        })?;
        serde_json::from_slice(&line).context("invalid niri IPC JSON")
    }
}

/// The successful payload of a reply, or niri's own error text.
fn reply_ok(request: &str, reply: serde_json::Value) -> Result<serde_json::Value> {
    let serde_json::Value::Object(mut reply) = reply else {
        bail!("unexpected niri reply to {request}");
    };
    if let Some(error) = reply.remove("Err") {
        bail!(
            "niri refused {request}: {}",
            error
                .as_str()
                .map_or_else(|| error.to_string(), str::to_owned)
        );
    }
    reply
        .remove("Ok")
        .with_context(|| format!("unexpected niri reply to {request}"))
}

/// The niri IPC requests this adapter reads. Only these are retried.
#[derive(Clone, Copy, Debug)]
enum Request {
    Version,
    Windows,
    Workspaces,
    Outputs,
    FocusedWindow,
    OverviewState,
}

impl Request {
    fn name(self) -> &'static str {
        match self {
            Self::Version => "Version",
            Self::Windows => "Windows",
            Self::Workspaces => "Workspaces",
            Self::Outputs => "Outputs",
            Self::FocusedWindow => "FocusedWindow",
            Self::OverviewState => "OverviewState",
        }
    }
}

struct Ipc<C> {
    connect: C,
    per_attempt: Duration,
    total: Duration,
    backoff: Duration,
    /// The connection earlier requests used. A connected socket keeps its
    /// peer, so it stays attested; niri answers one connection's requests
    /// in order. A failed exchange drops it, so a late reply is never read
    /// as the answer to a later request.
    lines: Option<Lines>,
}

impl<C: FnMut(Instant) -> Result<UnixStream>> Ipc<C> {
    fn connection(&mut self, deadline: Instant) -> Result<Lines> {
        match self.lines.take() {
            Some(lines) => Ok(lines),
            None => Ok(Lines::new((self.connect)(deadline)?)),
        }
    }

    /// A read-only request. Only a timed-out request/reply is retried, on a
    /// newly attested connection; connection and attestation errors are final.
    fn query<T: DeserializeOwned>(&mut self, request: Request) -> Result<T> {
        let deadline = Instant::now() + self.total;
        let name = request.name();
        for attempt in 1..=QUERY_MAX_ATTEMPTS {
            time_remaining(deadline)?;
            let attempt_deadline = deadline.min(Instant::now() + self.per_attempt);
            let mut lines = self.connection(attempt_deadline)?;
            match lines.exchange(&format!("\"{name}\""), attempt_deadline) {
                Ok(reply) => {
                    self.lines = Some(lines);
                    let mut payload = reply_ok(name, reply)?;
                    let value = payload
                        .get_mut(name)
                        .map(serde_json::Value::take)
                        .with_context(|| format!("niri reply to {name} has no {name}"))?;
                    return serde_json::from_value(value)
                        .with_context(|| format!("invalid niri {name} reply"));
                }
                Err(error) => {
                    if !is_timeout(&error)
                        || attempt == QUERY_MAX_ATTEMPTS
                        || deadline.saturating_duration_since(Instant::now()) <= self.backoff
                    {
                        return Err(error).with_context(|| {
                            format!("niri IPC {name} failed after {attempt} attempt(s)")
                        });
                    }
                    drop(lines);
                    tracing::warn!(name, attempt, "niri IPC request timed out; retrying");
                    std::thread::sleep(self.backoff);
                }
            }
        }
        unreachable!("the final query attempt always returns")
    }

    /// One compositor action. Never retried: after a timeout niri may have
    /// applied it.
    fn action(&mut self, action: serde_json::Value) -> Result<()> {
        let deadline = Instant::now() + self.per_attempt;
        let request = serde_json::json!({ "Action": action }).to_string();
        let mut lines = self.connection(deadline)?;
        let reply = lines
            .exchange(&request, deadline)
            .context("niri action outcome is unknown")?;
        self.lines = Some(lines);
        match reply_ok(&request, reply)? {
            serde_json::Value::String(handled) if handled == "Handled" => Ok(()),
            other => bail!("unexpected niri reply to {request}: {other}"),
        }
    }

    fn focused_window(&mut self) -> Result<Option<IpcWindow>> {
        self.query(Request::FocusedWindow)
    }

    fn scene(&mut self) -> Result<Scene> {
        let windows = self.query(Request::Windows)?;
        let workspaces = self.query(Request::Workspaces)?;
        let outputs = self.query(Request::Outputs)?;
        let overview: Overview = self.query(Request::OverviewState)?;
        Ok(Scene {
            windows,
            workspaces,
            outputs,
            overview_open: overview.is_open,
        })
    }
}

// ── Replies ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct IpcWindow {
    id: u64,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    app_id: Option<String>,
    pid: Option<i32>,
    workspace_id: Option<u64>,
    is_focused: bool,
    /// When the window last had focus; absent on older releases.
    #[serde(default)]
    focus_timestamp: Option<IpcTimestamp>,
    /// Added in niri 25.08; absent on older releases.
    #[serde(default)]
    layout: Option<IpcLayout>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct IpcTimestamp {
    secs: u64,
    nanos: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct IpcLayout {
    pos_in_scrolling_layout: Option<(u64, u64)>,
    tile_size: (f64, f64),
    window_size: (i32, i32),
    tile_pos_in_workspace_view: Option<(f64, f64)>,
    window_offset_in_tile: (f64, f64),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct IpcWorkspace {
    id: u64,
    output: Option<String>,
    is_active: bool,
    is_focused: bool,
    active_window_id: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct IpcOutput {
    name: String,
    modes: Vec<IpcMode>,
    current_mode: Option<usize>,
    logical: Option<IpcLogical>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
struct IpcMode {
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct IpcLogical {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    scale: f64,
    transform: String,
}

#[derive(Deserialize)]
struct Overview {
    is_open: bool,
}

/// wl_output transform number of a niri transform name.
fn transform_code(name: &str) -> Result<u32> {
    Ok(match name {
        "Normal" => 0,
        "90" => 1,
        "180" => 2,
        "270" => 3,
        "Flipped" => 4,
        "Flipped90" => 5,
        "Flipped180" => 6,
        "Flipped270" => 7,
        other => bail!("unknown niri output transform {other:?}"),
    })
}

fn valid_dimensions(width: u32, height: u32) -> bool {
    width > 0 && height > 0 && u64::from(width) * u64::from(height) <= MAX_LOGICAL_PIXELS
}

impl IpcLogical {
    fn checked(&self) -> Result<&Self> {
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || !valid_dimensions(self.width, self.height)
        {
            bail!("invalid niri output geometry");
        }
        Ok(self)
    }
}

impl IpcOutput {
    /// The output's size in niri's pointer space. Absolute virtual-pointer
    /// motion uses Smithay's output geometry, which rounds `mode / scale` up,
    /// while IPC truncates it (1280 px at 1.5 is 853 in IPC, 854 here).
    fn pointer_rect(&self) -> Result<Option<Rect>> {
        let Some(logical) = &self.logical else {
            return Ok(None);
        };
        let logical = logical.checked()?;
        let mode = self
            .current_mode
            .and_then(|index| self.modes.get(index))
            .with_context(|| format!("niri output {} has no current mode", self.name))?;
        let (width, height) = super::logical_output_size(
            (mode.width, mode.height),
            transform_code(&logical.transform)?,
        );
        let width = (f64::from(width) / logical.scale).ceil();
        let height = (f64::from(height) / logical.scale).ceil();
        if !(1.0..=f64::from(u32::MAX)).contains(&width)
            || !(1.0..=f64::from(u32::MAX)).contains(&height)
        {
            bail!("invalid niri output mode");
        }
        Ok(Some(Rect {
            x: logical.x,
            y: logical.y,
            width: width as u32,
            height: height as u32,
        }))
    }
}

// ── Geometry ─────────────────────────────────────────────────────────────────

/// A rectangle in niri's global logical layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    fn right(&self) -> i64 {
        i64::from(self.x) + i64::from(self.width)
    }

    fn bottom(&self) -> i64 {
        i64::from(self.y) + i64::from(self.height)
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        i64::from(x) >= i64::from(self.x)
            && i64::from(y) >= i64::from(self.y)
            && i64::from(x) < self.right()
            && i64::from(y) < self.bottom()
    }

    fn inside(&self, outer: &Rect) -> bool {
        self.x >= outer.x
            && self.y >= outer.y
            && self.right() <= outer.right()
            && self.bottom() <= outer.bottom()
    }

    fn from_f64(x: f64, y: f64, width: f64, height: f64) -> Option<Self> {
        let (x, y, width, height) = (x.round(), y.round(), width.round(), height.round());
        let in_i32 = |v: f64| (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&v);
        (in_i32(x) && in_i32(y) && width >= 1.0 && height >= 1.0 && in_i32(width) && in_i32(height))
            .then_some(Self {
                x: x as i32,
                y: y as i32,
                width: width as u32,
                height: height as u32,
            })
            .filter(|rect| valid_dimensions(rect.width, rect.height))
    }
}

/// Why niri reports no on-screen place for a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unplaced {
    /// In the scrolling layout: niri does not report where the view is.
    Tiled,
    /// On an inactive workspace, on no output, or the overview is open.
    NotShowing,
    /// A tiled window as large as the output, likely fullscreen, hides the
    /// workspace's floating windows.
    Covered,
    /// This niri release reports no window layout.
    NoLayout,
}

impl Unplaced {
    pub fn reason(self) -> &'static str {
        match self {
            Self::Tiled => "niri does not report where a tiled window is on screen",
            Self::NotShowing => {
                "the window is not showing: its workspace is not active on an output, or the niri overview is open"
            }
            Self::Covered => {
                "a tiled window as large as the output, likely fullscreen, covers the workspace's floating windows"
            }
            Self::NoLayout => "this niri release does not report window layout (niri 25.08 or later is required)",
        }
    }
}

/// A floating window's place on screen, from one compositor snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    /// The window's geometry in the global logical layout, rounded to whole
    /// logical pixels.
    pub rect: Rect,
    /// The exact logical origin niri reports: niri aligns windows to
    /// physical pixels, so at a fractional scale it is fractional.
    pub origin: (f64, f64),
    pub output: String,
    /// The output's logical rectangle: niri shows no part of the window
    /// outside it.
    pub area: Rect,
    /// The exact tiles of the workspace's other floating windows. IPC
    /// reports no stacking order, so any of them may be above this window.
    pub others: Vec<Exact>,
}

/// A rectangle in exact, possibly fractional, logical coordinates: at a
/// fractional scale niri aligns windows to physical pixels, and rounding to
/// whole logical pixels can hide a one-pixel overlap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exact {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Exact {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }

    fn intersects(&self, other: &Exact) -> bool {
        self.x < other.x + other.width
            && other.x < self.x + self.width
            && self.y < other.y + other.height
            && other.y < self.y + self.height
    }
}

impl Placed {
    /// The window's exact geometry.
    pub fn exact(&self) -> Exact {
        Exact {
            x: self.origin.0,
            y: self.origin.1,
            width: f64::from(self.rect.width),
            height: f64::from(self.rect.height),
        }
    }

    pub fn overlapped(&self) -> bool {
        let exact = self.exact();
        self.others.iter().any(|other| other.intersects(&exact))
    }

    /// Whether a layout point lies inside the exact window geometry rather
    /// than only inside its rounded rectangle.
    pub fn covers(&self, x: i32, y: i32) -> bool {
        self.exact().contains(f64::from(x), f64::from(y))
    }

    /// Whether a layout point surely lands on this window.
    pub fn hits(&self, x: i32, y: i32) -> bool {
        self.covers(x, y)
            && !self
                .others
                .iter()
                .any(|other| other.contains(f64::from(x), f64::from(y)))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub id: u64,
    pub pid: Option<u32>,
    pub title: String,
    pub app_id: String,
    pub workspace: Option<u64>,
    pub focused: bool,
    /// Focus recency among all windows, higher is more recent and the
    /// focused window highest. niri reports no stacking order; this is the
    /// best available "which of a process's windows is in use".
    pub focus_rank: usize,
    /// niri shows it: a placed floating window, or the active tiled window of
    /// an active workspace while the overview is closed.
    pub on_screen: bool,
    /// The window's logical size, which niri reports for tiled windows too.
    /// Identity only: a tiled window has no position, so this is never input
    /// or capture geometry.
    pub size: Option<(u32, u32)>,
    pub placement: Result<Placed, Unplaced>,
}

struct Scene {
    windows: Vec<IpcWindow>,
    workspaces: Vec<IpcWorkspace>,
    outputs: BTreeMap<String, IpcOutput>,
    overview_open: bool,
}

impl Scene {
    fn logical(&self, workspace: &IpcWorkspace) -> Option<&IpcLogical> {
        let output = self.outputs.get(workspace.output.as_ref()?)?;
        output.logical.as_ref()
    }

    fn workspace(&self, window: &IpcWindow) -> Option<&IpcWorkspace> {
        let id = window.workspace_id?;
        self.workspaces.iter().find(|workspace| workspace.id == id)
    }

    fn placement(&self, window: &IpcWindow) -> Result<Placed, Unplaced> {
        let layout = window.layout.as_ref().ok_or(Unplaced::NoLayout)?;
        if layout.pos_in_scrolling_layout.is_some() {
            return Err(Unplaced::Tiled);
        }
        if self.overview_open {
            return Err(Unplaced::NotShowing);
        }
        let workspace = self
            .workspace(window)
            .filter(|workspace| workspace.is_active)
            .ok_or(Unplaced::NotShowing)?;
        let logical = self.logical(workspace).ok_or(Unplaced::NotShowing)?;
        if self.covered(workspace, logical) {
            return Err(Unplaced::Covered);
        }
        let (tile_x, tile_y) = layout
            .tile_pos_in_workspace_view
            .ok_or(Unplaced::NotShowing)?;
        let (offset_x, offset_y) = layout.window_offset_in_tile;
        let (width, height) = layout.window_size;
        let origin = (
            f64::from(logical.x) + tile_x + offset_x,
            f64::from(logical.y) + tile_y + offset_y,
        );
        let rect = Rect::from_f64(origin.0, origin.1, f64::from(width), f64::from(height))
            .ok_or(Unplaced::NotShowing)?;
        let others = self
            .windows
            .iter()
            .filter(|other| other.id != window.id && other.workspace_id == window.workspace_id)
            .filter_map(|other| floating_tile(logical, other.layout.as_ref()?))
            .collect();
        Ok(Placed {
            rect,
            origin,
            output: workspace.output.clone().unwrap_or_default(),
            area: Rect {
                x: logical.x,
                y: logical.y,
                width: logical.width,
                height: logical.height,
            },
            others,
        })
    }

    /// niri hides a workspace's floating windows while its active column is
    /// fullscreen. IPC has no fullscreen flag, so an active tiled tile as
    /// large as the output (give or take a pixel) counts as covering them.
    /// This is refusal evidence only, never proof that a window is visible.
    fn covered(&self, workspace: &IpcWorkspace, logical: &IpcLogical) -> bool {
        self.windows
            .iter()
            .filter(|window| Some(window.id) == workspace.active_window_id)
            .filter_map(|window| window.layout.as_ref())
            .any(|layout| {
                layout.pos_in_scrolling_layout.is_some()
                    && layout.tile_size.0 >= f64::from(logical.width) - 1.0
                    && layout.tile_size.1 >= f64::from(logical.height) - 1.0
            })
    }

    fn on_screen(&self, window: &IpcWindow, placement: &Result<Placed, Unplaced>) -> bool {
        match placement {
            Ok(_) => true,
            Err(Unplaced::Tiled) => {
                !self.overview_open
                    && self.workspace(window).is_some_and(|workspace| {
                        workspace.is_active && workspace.active_window_id == Some(window.id)
                    })
            }
            Err(_) => false,
        }
    }

    fn windows(&self) -> Result<Vec<Window>> {
        let mut seen = HashSet::new();
        let mut windows = Vec::with_capacity(self.windows.len());
        for window in &self.windows {
            if !seen.insert(window.id) {
                bail!("duplicate niri window id {}", window.id);
            }
            let placement = self.placement(window);
            windows.push(Window {
                id: window.id,
                pid: window
                    .pid
                    .and_then(|pid| u32::try_from(pid).ok())
                    .filter(|pid| *pid > 0),
                title: window.title.clone().unwrap_or_default(),
                app_id: window.app_id.clone().unwrap_or_default(),
                workspace: window.workspace_id,
                focused: window.is_focused,
                focus_rank: self
                    .windows
                    .iter()
                    .filter(|other| {
                        (other.is_focused, other.focus_timestamp)
                            < (window.is_focused, window.focus_timestamp)
                    })
                    .count(),
                on_screen: self.on_screen(window, &placement),
                size: window.layout.as_ref().and_then(|layout| {
                    let (width, height) = layout.window_size;
                    Some((u32::try_from(width).ok()?, u32::try_from(height).ok()?))
                }),
                placement,
            });
        }
        Ok(windows)
    }

    fn window(&self, id: u64) -> Result<Window> {
        self.windows()?
            .into_iter()
            .find(|window| window.id == id)
            .with_context(|| format!("niri window {id} no longer exists; refresh list_windows"))
    }

    /// Outputs niri lays out (enabled ones have a logical rectangle).
    fn laid_out(&self) -> Vec<(&IpcOutput, &IpcLogical)> {
        self.outputs
            .values()
            .filter_map(|output| Some((output, output.logical.as_ref()?)))
            .collect()
    }

    /// Absolute virtual-pointer motion from an unbound pointer maps over the
    /// bounding box of every output's pointer rectangle.
    fn pointer_space(&self) -> Result<(i32, i32, u32, u32)> {
        pointer_space_of(&self.outputs)
    }

    /// The one laid-out output. Display capture copies a single output, so
    /// output-pixel routes need a layout where that output is the only one.
    fn single_output(&self) -> Result<(&IpcOutput, &IpcLogical)> {
        match self.laid_out().as_slice() {
            [(output, logical)] => Ok((*output, logical.checked()?)),
            [] => bail!("niri has no enabled output"),
            outputs => bail!(
                "niri window capture and desktop input support a single output; this layout has {}",
                outputs.len()
            ),
        }
    }
}

fn pointer_space_of(outputs: &BTreeMap<String, IpcOutput>) -> Result<(i32, i32, u32, u32)> {
    let mut rects = Vec::new();
    for output in outputs.values() {
        rects.extend(output.pointer_rect()?);
    }
    bounding_box(&rects).context("niri has no enabled output")
}

fn floating_tile(logical: &IpcLogical, layout: &IpcLayout) -> Option<Exact> {
    if layout.pos_in_scrolling_layout.is_some() {
        return None;
    }
    let (x, y) = layout.tile_pos_in_workspace_view?;
    let tile = Exact {
        x: f64::from(logical.x) + x,
        y: f64::from(logical.y) + y,
        width: layout.tile_size.0,
        height: layout.tile_size.1,
    };
    [tile.x, tile.y, tile.width, tile.height]
        .iter()
        .all(|v| v.is_finite())
        .then_some(tile)
}

fn bounding_box(rects: &[Rect]) -> Option<(i32, i32, u32, u32)> {
    let min_x = rects.iter().map(|r| r.x).min()?;
    let min_y = rects.iter().map(|r| r.y).min()?;
    let max_x = rects.iter().map(Rect::right).max()?;
    let max_y = rects.iter().map(Rect::bottom).max()?;
    let width = u32::try_from(max_x - i64::from(min_x)).ok()?;
    let height = u32::try_from(max_y - i64::from(min_y)).ok()?;
    Some((min_x, min_y, width, height))
}

// ── Observation ──────────────────────────────────────────────────────────────

pub fn list_windows() -> Result<Vec<Window>> {
    ipc().scene()?.windows()
}

pub fn window_for_id(id: u64) -> Result<Window> {
    ipc().scene()?.window(id)
}

/// The process's only niri window. A process with several windows needs an
/// exact window id; none is guessed.
pub fn window_for_pid(pid: u32) -> Option<Window> {
    let mut owned = list_windows()
        .ok()?
        .into_iter()
        .filter(|window| window.pid == Some(pid));
    let first = owned.next()?;
    owned.next().is_none().then_some(first)
}

/// The window a pid-only keyboard action means: the pid's on-screen window
/// focused most recently (the focused one first). This is focus recency,
/// not stacking, so it is used only to pick a keyboard target.
pub fn window_in_use(pid: u32) -> Option<u64> {
    pick_window_in_use(&list_windows().ok()?, pid)
}

fn pick_window_in_use(windows: &[Window], pid: u32) -> Option<u64> {
    windows
        .iter()
        .filter(|window| window.pid == Some(pid) && window.on_screen)
        .max_by_key(|window| window.focus_rank)
        .map(|window| window.id)
}

pub fn version() -> Result<String> {
    ipc().query(Request::Version)
}

/// Pointer space for `motion_absolute`: `(origin_x, origin_y, width, height)`.
/// Only outputs are queried: the gate rechecks this inside a double click.
pub fn pointer_space() -> Result<(i32, i32, u32, u32)> {
    pointer_space_of(&ipc().query(Request::Outputs)?)
}

/// The desktop frame (the single output's logical rectangle) and the pointer
/// layout, from one snapshot.
pub fn desktop_snapshot() -> Result<super::hyprland::DesktopSnapshot> {
    desktop_snapshot_in(&ipc().scene()?)
}

fn desktop_snapshot_in(scene: &Scene) -> Result<super::hyprland::DesktopSnapshot> {
    let (output, logical) = scene.single_output()?;
    Ok(super::hyprland::DesktopSnapshot {
        frame: super::hyprland::DesktopFrame {
            x: logical.x,
            y: logical.y,
            width: logical.width,
            height: logical.height,
            scale: logical.scale,
            outputs: vec![super::hyprland::FrameOutput {
                name: output.name.clone(),
                x: logical.x,
                y: logical.y,
                width: logical.width,
                height: logical.height,
            }],
        },
        pointer: Some(scene.pointer_space()?),
    })
}

/// Every enabled output's logical rectangle, and their bounding box with the
/// largest scale, as Hyprland reports its frame. Desktop-scope input still
/// needs a single output; reporting sizes does not.
fn screen_frame_in(scene: &Scene) -> Result<super::hyprland::DesktopFrame> {
    let mut outputs = Vec::new();
    let mut scale: f64 = 0.0;
    for (output, logical) in scene.laid_out() {
        let logical = logical.checked()?;
        scale = scale.max(logical.scale);
        outputs.push(super::hyprland::FrameOutput {
            name: output.name.clone(),
            x: logical.x,
            y: logical.y,
            width: logical.width,
            height: logical.height,
        });
    }
    let rects: Vec<Rect> = outputs
        .iter()
        .map(|output| Rect {
            x: output.x,
            y: output.y,
            width: output.width,
            height: output.height,
        })
        .collect();
    let (x, y, width, height) = bounding_box(&rects).context("niri has no enabled output")?;
    Ok(super::hyprland::DesktopFrame {
        x,
        y,
        width,
        height,
        scale,
        outputs,
    })
}

/// The output frame plus the monitor list `get_screen_size` reports, in the
/// shape the Hyprland adapter publishes.
pub fn screen_report() -> Result<(super::hyprland::DesktopFrame, serde_json::Value)> {
    let scene = ipc().scene()?;
    let frame = screen_frame_in(&scene)?;
    let monitors = frame
        .outputs
        .iter()
        .map(|output| {
            serde_json::json!({
                "name": output.name,
                "width": output.width,
                "height": output.height,
                "scale": scene.outputs[&output.name]
                    .logical
                    .as_ref()
                    .map_or(frame.scale, |logical| logical.scale),
                "powered": true,
                "frame_x": output.x - frame.x,
                "frame_y": output.y - frame.y,
            })
        })
        .collect();
    Ok((frame, serde_json::Value::Array(monitors)))
}

/// wl_output transform of every enabled output, by name, for orienting a
/// capture of one of them.
pub fn output_transforms() -> Result<HashMap<String, u32>> {
    ipc()
        .scene()?
        .laid_out()
        .into_iter()
        .map(|(output, logical)| Ok((output.name.clone(), transform_code(&logical.transform)?)))
        .collect()
}

/// Doctor note: `None` without `NIRI_SOCKET`, otherwise whether this process
/// drives niri, its version and the single-output limit.
pub fn health_note() -> Option<String> {
    if std::env::var_os("NIRI_SOCKET").is_none_or(|socket| socket.is_empty()) {
        return None;
    }
    if !is_session() {
        let session = match probe(RECOGNIZED.get().copied(), QUERY_TIMEOUT).0 {
            Attestation::Unrelated(reason) | Attestation::Refused(reason) => {
                HealthSession::Foreign(reason)
            }
            Attestation::Attested(_) => HealthSession::Inactive,
        };
        return Some(describe_health(session, Ok(1)));
    }
    let scene = ipc().scene();
    Some(describe_health(
        HealthSession::Attested(version()),
        scene.map(|scene| scene.laid_out().len()),
    ))
}

enum HealthSession {
    /// NIRI_SOCKET belongs to another compositor, with the reason.
    Foreign(String),
    /// This session's niri, but this process does not use the adapter.
    Inactive,
    /// This session's niri, with its version.
    Attested(Result<String>),
}

fn describe_health(session: HealthSession, outputs: Result<usize>) -> String {
    let version = match session {
        HealthSession::Foreign(reason) => {
            return format!(
                "niri: NIRI_SOCKET is not this session's compositor ({reason}); niri window \
                 ids, focus and geometry are off."
            )
        }
        HealthSession::Inactive => {
            return "niri: NIRI_SOCKET is this session's niri, but this process does not use \
                    the niri adapter (inject mode or the native Wayland backend is off)."
                .to_owned()
        }
        HealthSession::Attested(Ok(version)) => version,
        HealthSession::Attested(Err(error)) => format!("version unavailable: {error:#}"),
    };
    match outputs {
        Ok(1) => format!("niri {version}: IPC attested to WAYLAND_DISPLAY."),
        Ok(count) => format!(
            "niri {version}: IPC attested to WAYLAND_DISPLAY; window capture and desktop \
             input refuse because {count} outputs are enabled (one is supported)."
        ),
        Err(error) => format!(
            "niri {version}: IPC attested to WAYLAND_DISPLAY; the output layout is \
             unavailable: {error:#}."
        ),
    }
}

// ── Focus ────────────────────────────────────────────────────────────────────

fn focus_window_action(id: u64) -> serde_json::Value {
    serde_json::json!({ "FocusWindow": { "id": id } })
}

fn foreground_unavailable(detail: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("foreground_unavailable: {detail}")
}

/// An IPC failure on an input route, with its whole cause chain. A refusal
/// that already carries the code is kept as it is.
pub(crate) fn input_unavailable(error: anyhow::Error) -> anyhow::Error {
    if error.to_string().starts_with("foreground_unavailable") {
        return error;
    }
    foreground_unavailable(format!("{error:#}"))
}

/// Check that `id` can take focus for `pid`. Returns the window when it is
/// not focused yet, `None` when it already is.
fn focus_target<C: FnMut(Instant) -> Result<UnixStream>>(
    ipc: &mut Ipc<C>,
    id: u64,
    pid: Option<u32>,
) -> Result<Option<IpcWindow>> {
    let overview: Overview = ipc.query(Request::OverviewState)?;
    if overview.is_open {
        return Err(foreground_unavailable(
            "the niri overview is open, so input would go to the overview instead of the window",
        ));
    }
    let windows: Vec<IpcWindow> = ipc.query(Request::Windows)?;
    let target = windows
        .into_iter()
        .find(|window| window.id == id)
        .ok_or_else(|| foreground_unavailable(format!("niri window {id} no longer exists")))?;
    let target_pid = target.pid.and_then(|pid| u32::try_from(pid).ok());
    if let Some(pid) = pid {
        if target_pid != Some(pid) {
            return Err(foreground_unavailable(format!(
                "niri window {id} belongs to pid {}, not pid {pid}",
                target_pid.map_or_else(|| "unknown".to_owned(), |pid| pid.to_string())
            )));
        }
    }
    Ok((!target.is_focused).then_some(target))
}

/// Focus exactly `id` and read the focus back. niri answers `Handled` even
/// for an unknown id, and reports no focused window while the overview, a
/// layer-shell surface or the lock screen holds focus, so the read-back is
/// the guarantee.
fn activate_in<C: FnMut(Instant) -> Result<UnixStream>>(
    ipc: &mut Ipc<C>,
    id: u64,
    pid: Option<u32>,
    confirm: Duration,
) -> Result<()> {
    match focus_target(ipc, id, pid)? {
        None => Ok(()),
        Some(target) => focus_window(ipc, id, target.pid, confirm),
    }
}

fn focus_window<C: FnMut(Instant) -> Result<UnixStream>>(
    ipc: &mut Ipc<C>,
    id: u64,
    pid: Option<i32>,
    confirm: Duration,
) -> Result<()> {
    ipc.action(focus_window_action(id))?;
    confirm_focus(ipc, id, pid, confirm)
}

fn confirm_focus<C: FnMut(Instant) -> Result<UnixStream>>(
    ipc: &mut Ipc<C>,
    id: u64,
    pid: Option<i32>,
    confirm: Duration,
) -> Result<()> {
    let deadline = Instant::now() + confirm;
    loop {
        let focused = ipc.focused_window()?;
        if focused
            .as_ref()
            .is_some_and(|focused| focused.id == id && focused.pid == pid)
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(foreground_unavailable(format!(
                "niri did not confirm focus on window {id} (focused: {})",
                focused.map_or_else(|| "none".to_owned(), |focused| focused.id.to_string())
            )));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Focus an exact niri window for input, confirmed by read-back.
pub fn activate(id: u64, pid: Option<u32>) -> Result<()> {
    activate_in(&mut ipc(), id, pid, FOCUS_CONFIRM_TIMEOUT).map_err(input_unavailable)
}

/// Briefly focus one niri window for `pid`, run `body`, then restore the
/// previously focused window, or the previously focused workspace when no
/// window had focus, and each output's previously active workspace.
/// Focusing a window on another workspace switches to it; the restore
/// switches back, also after a focus niri did not confirm.
pub fn with_focused_window<T>(pid: u32, id: u64, body: impl FnOnce() -> Result<T>) -> Result<T> {
    with_focused_window_in(&mut ipc(), pid, id, FOCUS_CONFIRM_TIMEOUT, body)
}

fn with_focused_window_in<C: FnMut(Instant) -> Result<UnixStream>, T>(
    ipc: &mut Ipc<C>,
    pid: u32,
    id: u64,
    confirm: Duration,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let prior = ipc.focused_window().map_err(input_unavailable)?;
    let workspaces: Vec<IpcWorkspace> =
        ipc.query(Request::Workspaces).map_err(input_unavailable)?;
    let Some(target) = focus_target(ipc, id, Some(pid)).map_err(input_unavailable)? else {
        return body();
    };
    // Once FocusWindow is sent, focus may have moved even when niri does not
    // confirm it, so the restore runs either way.
    let result = focus_window(ipc, id, target.pid, confirm)
        .map_err(input_unavailable)
        .and_then(|()| body());
    let restore = restore_focus(ipc, prior.as_ref(), &workspaces, &target, confirm);
    match (result, restore) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(restore)) => {
            let note =
                format!("{error}; the prior niri focus also could not be restored: {restore:#}");
            Err(error.context(note))
        }
    }
}

/// Undo what focusing `target` changed: the active window of its workspace,
/// the active workspace of every output, then the focused window or
/// workspace.
fn restore_focus<C: FnMut(Instant) -> Result<UnixStream>>(
    ipc: &mut Ipc<C>,
    prior: Option<&IpcWindow>,
    before: &[IpcWorkspace],
    target: &IpcWindow,
    confirm: Duration,
) -> Result<()> {
    let prior_id = prior.map(|prior| prior.id);
    // Someone focused another window meanwhile: theirs is the newer choice.
    if let Some(chosen) = ipc
        .focused_window()?
        .map(|focused| focused.id)
        .filter(|focused| *focused != target.id && Some(*focused) != prior_id)
    {
        tracing::info!(
            window = chosen,
            "niri focus moved to another window during input; leaving it there"
        );
        return Ok(());
    }
    // The same for a workspace, which may hold no window to focus.
    let prior_workspace = before
        .iter()
        .find(|workspace| workspace.is_focused)
        .map(|workspace| workspace.id);
    if let Some(chosen) = ipc
        .query::<Vec<IpcWorkspace>>(Request::Workspaces)?
        .into_iter()
        .find(|workspace| workspace.is_focused)
        .map(|workspace| workspace.id)
        .filter(|focused| {
            Some(*focused) != target.workspace_id && Some(*focused) != prior_workspace
        })
    {
        tracing::info!(
            workspace = chosen,
            "niri focus moved to another workspace during input; leaving it there"
        );
        return Ok(());
    }
    if let Some(active) = before
        .iter()
        .find(|workspace| Some(workspace.id) == target.workspace_id)
        .and_then(|workspace| workspace.active_window_id)
        .filter(|active| *active != target.id && Some(*active) != prior_id)
    {
        ipc.action(focus_window_action(active))
            .with_context(|| format!("niri could not restore active window {active}"))?;
    }
    let now: Vec<IpcWorkspace> = ipc.query(Request::Workspaces)?;
    for was in before
        .iter()
        .filter(|was| was.is_active && !was.is_focused)
        .filter(|was| now.iter().any(|w| w.id == was.id && !w.is_active))
    {
        ipc.action(focus_workspace_action(was.id))
            .with_context(|| format!("niri could not restore workspace {}", was.id))?;
    }
    match (prior, before.iter().find(|workspace| workspace.is_focused)) {
        (Some(prior), _) => focus_window(ipc, prior.id, prior.pid, confirm)
            .with_context(|| format!("niri could not restore prior window {}", prior.id))?,
        (None, Some(workspace)) => restore_workspace(ipc, workspace.id, confirm)
            .with_context(|| format!("niri could not restore prior workspace {}", workspace.id))?,
        (None, None) => {}
    }
    // niri applies workspace switches asynchronously: poll like the
    // single-workspace restore does.
    let deadline = Instant::now() + confirm;
    loop {
        let now: Vec<IpcWorkspace> = ipc.query(Request::Workspaces)?;
        let Some(was) = before
            .iter()
            .filter(|was| was.is_active)
            .find(|was| now.iter().any(|w| w.id == was.id && !w.is_active))
        else {
            return Ok(());
        };
        if Instant::now() >= deadline {
            bail!(
                "niri did not restore workspace {} on output {}",
                was.id,
                was.output.as_deref().unwrap_or("unknown")
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn focus_workspace_action(id: u64) -> serde_json::Value {
    serde_json::json!({ "FocusWorkspace": { "reference": { "Id": id } } })
}

fn restore_workspace<C: FnMut(Instant) -> Result<UnixStream>>(
    ipc: &mut Ipc<C>,
    workspace: u64,
    confirm: Duration,
) -> Result<()> {
    ipc.action(focus_workspace_action(workspace))?;
    let deadline = Instant::now() + confirm;
    loop {
        let workspaces: Vec<IpcWorkspace> = ipc.query(Request::Workspaces)?;
        if workspaces
            .iter()
            .any(|candidate| candidate.id == workspace && candidate.is_focused)
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("niri did not confirm focus on workspace {workspace}");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

// ── Settled geometry ─────────────────────────────────────────────────────────

/// What an event-stream event did to the shown layout.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Change {
    /// Anything that can move, uncover or animate a window, and any event
    /// this adapter does not know, so a newer niri fails closed.
    Moved,
    /// Keyboard focus went to this window, or to none.
    Focus(Option<u64>),
    /// This workspace became active on its output.
    Activated(Option<u64>),
    /// This window became its workspace's active window.
    Active(Option<u64>),
}

/// The focus change a pointer action makes itself. Focus moves no floating
/// window, so the events naming the window it focuses are expected.
#[derive(Clone, Copy, Debug, Default)]
struct Expected {
    window: Option<u64>,
    /// The window's workspace, already active on its output: when focus
    /// moves there from another output, niri reports it activated.
    workspace: Option<u64>,
}

impl Change {
    fn counts(self, expected: Expected) -> bool {
        match self {
            Self::Moved => true,
            Self::Focus(id) | Self::Active(id) => {
                expected.window.is_none() || id != expected.window
            }
            Self::Activated(id) => expected.workspace.is_none() || id != expected.workspace,
        }
    }
}

/// Classifies event-stream events. A window event that only changes a title
/// or similar metadata changes nothing.
#[derive(Default)]
struct LayoutWatch {
    layouts: HashMap<u64, serde_json::Value>,
}

impl LayoutWatch {
    /// The parts of a window that place it on screen.
    fn placing(window: &serde_json::Value) -> serde_json::Value {
        serde_json::json!([
            window.get("workspace_id"),
            window.get("is_floating"),
            window.get("layout")
        ])
    }

    fn change(&mut self, event: &serde_json::Value) -> Option<Change> {
        let Some((name, body)) = event.as_object().and_then(|event| event.iter().next()) else {
            return Some(Change::Moved);
        };
        let id = |key: &str| body.get(key).and_then(serde_json::Value::as_u64);
        match name.as_str() {
            "WindowFocusTimestampChanged"
            | "WindowUrgencyChanged"
            | "WorkspaceUrgencyChanged"
            | "KeyboardLayoutsChanged"
            | "KeyboardLayoutSwitched"
            | "ScreenshotCaptured"
            | "CastsChanged"
            | "CastStartedOrChanged"
            | "CastStopped" => None,
            // A reload can move tiled columns (struts, gaps, centering) and
            // change animation lengths without any other event.
            "ConfigLoaded" => Some(Change::Moved),
            "WindowFocusChanged" => Some(Change::Focus(id("id"))),
            "WorkspaceActivated" => Some(Change::Activated(id("id"))),
            "WorkspaceActiveWindowChanged" => Some(Change::Active(id("active_window_id"))),
            "WindowsChanged" => {
                self.layouts = body
                    .get("windows")
                    .and_then(serde_json::Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|window| Some((window.get("id")?.as_u64()?, Self::placing(window))))
                    .collect();
                Some(Change::Moved)
            }
            "WindowOpenedOrChanged" => {
                let Some(window) = body.get("window") else {
                    return Some(Change::Moved);
                };
                let Some(id) = window.get("id").and_then(serde_json::Value::as_u64) else {
                    return Some(Change::Moved);
                };
                let placing = Self::placing(window);
                (self.layouts.insert(id, placing.clone()) != Some(placing)).then_some(Change::Moved)
            }
            _ => Some(Change::Moved),
        }
    }
}

/// How many recent changes a history keeps for checks still in flight; a
/// check older than that counts as changed.
const MAX_CHANGES: usize = 1024;

/// niri's layout changes as one long-lived, attested event stream reports
/// them. niri reports a new place at once and animates to it, and the stream
/// does not report the animation's end, so the layout counts as settled once
/// it has not changed for [`SETTLE_QUIET`]; that covers only animations no
/// longer than it (see [`SETTLE_QUIET`]).
struct Replica {
    lines: Lines,
    generation: u64,
    watch: LayoutWatch,
    /// When the layout last changed. Connecting counts: an animation may
    /// have started before the stream did.
    changed_at: Instant,
    /// Recent changes, the first numbered `first`.
    changes: VecDeque<Change>,
    first: u64,
}

impl Replica {
    fn start(stream: UnixStream, generation: u64) -> Result<Self> {
        let deadline = Instant::now() + QUERY_TIMEOUT;
        let mut lines = Lines::new(stream);
        lines.send("\"EventStream\"", deadline)?;
        let first = lines
            .next(deadline)?
            .context("niri event stream did not start")?;
        reply_ok(
            "EventStream",
            serde_json::from_slice(&first).context("invalid niri IPC JSON")?,
        )?;
        // From here on, reads take only what niri has already sent.
        lines.stream.set_nonblocking(true)?;
        Ok(Self {
            lines,
            generation,
            watch: LayoutWatch::default(),
            changed_at: Instant::now(),
            changes: VecDeque::new(),
            first: 0,
        })
    }

    fn mark(&self) -> Mark {
        Mark {
            generation: self.generation,
            at: self.first + self.changes.len() as u64,
        }
    }

    /// Apply every event niri has sent so far.
    fn catch_up(&mut self) -> Result<()> {
        while let Some(line) = self.lines.next(Instant::now() + QUERY_TIMEOUT)? {
            let event: serde_json::Value =
                serde_json::from_slice(&line).context("invalid niri event")?;
            if let Some(change) = self.watch.change(&event) {
                self.changed_at = Instant::now();
                self.changes.push_back(change);
                if self.changes.len() > MAX_CHANGES {
                    self.changes.pop_front();
                    self.first += 1;
                }
            }
        }
        Ok(())
    }
}

/// Where a layout history stood when a check passed: later changes are new.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mark {
    generation: u64,
    at: u64,
}

/// The process's niri layout history: one event stream, kept open between
/// actions so an action on a layout that has been quiet for
/// [`SETTLE_QUIET`] need not wait. A background thread reads it as events
/// arrive, which dates each change and keeps niri from dropping a stream
/// that reads too slowly; every check also reads what has arrived itself.
/// A stream that fails is dropped and the next settle connects again; a
/// check made against an earlier connection refuses.
struct History {
    replica: Mutex<Option<Replica>>,
    generations: AtomicU64,
}

impl History {
    const fn new() -> Self {
        Self {
            replica: Mutex::new(None),
            generations: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Option<Replica>>> {
        self.replica
            .lock()
            .map_err(|_| anyhow!("niri layout history is unavailable"))
    }

    /// Wait until the layout has not changed for `quiet`, at most `limit`,
    /// connecting through `connect` when no stream is open.
    fn settle(
        &'static self,
        connect: impl FnOnce() -> Result<UnixStream>,
        quiet: Duration,
        limit: Duration,
    ) -> Result<Mark> {
        let started = Instant::now();
        let mut connect = Some(connect);
        loop {
            let settled = {
                let mut replica = self.lock()?;
                if replica.is_none() {
                    let connect = connect
                        .take()
                        .context("niri event stream closed while settling")?;
                    *replica = Some(self.start(connect()?)?);
                }
                let current = replica.as_mut().expect("connected above");
                if let Err(error) = current.catch_up() {
                    *replica = None;
                    return Err(error.context("niri event stream failed"));
                }
                let settled = current.changed_at + quiet;
                if settled <= Instant::now() {
                    return Ok(current.mark());
                }
                settled
            };
            if settled > started + limit {
                return Err(foreground_unavailable(format!(
                    "the niri layout did not stay unchanged for {} ms within {} ms, so window geometry may not match the screen",
                    quiet.as_millis(),
                    limit.as_millis()
                )));
            }
            std::thread::sleep(settled.saturating_duration_since(Instant::now()));
        }
    }

    fn start(&'static self, stream: UnixStream) -> Result<Replica> {
        let generation = self.generations.fetch_add(1, Ordering::Relaxed) + 1;
        let replica = Replica::start(stream, generation)?;
        let watched = replica.lines.stream.try_clone()?;
        std::thread::Builder::new()
            .name("cua-niri-events".into())
            .spawn(move || self.watch(generation, &watched))?;
        Ok(replica)
    }

    /// Read events as they arrive until this generation's stream is gone.
    fn watch(&self, generation: u64, stream: &UnixStream) {
        loop {
            let mut poll = libc::pollfd {
                fd: stream.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one valid pollfd that outlives the call.
            if unsafe { libc::poll(&mut poll, 1, 1000) } < 0
                && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted
            {
                return;
            }
            let Ok(mut replica) = self.replica.lock() else {
                return;
            };
            match replica.as_mut() {
                Some(current) if current.generation == generation => {
                    if let Err(error) = current.catch_up() {
                        tracing::warn!("niri event stream failed: {error:#}");
                        *replica = None;
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    /// Whether the layout changed after `mark`, other than as `expected`.
    /// A stream that failed or reconnected since cannot tell, so it refuses.
    fn changed_since(&self, mark: Mark, expected: Expected) -> Result<bool> {
        let mut replica = self.lock()?;
        let Some(current) = replica
            .as_mut()
            .filter(|current| current.generation == mark.generation)
        else {
            bail!("niri's event stream reconnected after the layout was checked");
        };
        if let Err(error) = current.catch_up() {
            *replica = None;
            return Err(error.context("niri event stream failed"));
        }
        let Some(skip) = mark.at.checked_sub(current.first) else {
            return Ok(true);
        };
        Ok(current
            .changes
            .iter()
            .skip(usize::try_from(skip)?)
            .any(|change| change.counts(expected)))
    }
}

static HISTORY: History = History::new();

/// Wait until niri's layout has been quiet for [`SETTLE_QUIET`]: at once
/// when it already has been.
fn settled() -> Result<Mark> {
    HISTORY.settle(
        || attested_connection(Instant::now() + QUERY_TIMEOUT),
        SETTLE_QUIET,
        SETTLE_LIMIT,
    )
}

struct Aim {
    window: u64,
    point: (i32, i32),
    rect: Rect,
}

fn aims() -> &'static Mutex<Vec<Aim>> {
    static AIMS: OnceLock<Mutex<Vec<Aim>>> = OnceLock::new();
    AIMS.get_or_init(|| Mutex::new(Vec::new()))
}

fn placed_or_refuse(window: &Window) -> Result<&Placed> {
    window.placement.as_ref().map_err(|unplaced| {
        foreground_unavailable(format!("niri window {}: {}", window.id, unplaced.reason()))
    })
}

/// Translate window coordinates into the global layout. A floating window's
/// coordinates are window-local; the geometry used is remembered so the
/// pointer gate can prove it still holds. A tiled window's are pixels of its
/// output frame (see [`OutputFrame`]); without one it refuses instead of
/// passing `(x, y)` through as output coordinates.
pub fn aim(window_id: u64, x: i32, y: i32) -> Result<(i32, i32)> {
    let scene = ipc().scene().map_err(input_unavailable)?;
    let window = scene.window(window_id).map_err(input_unavailable)?;
    if matches!(window.placement, Err(Unplaced::Tiled)) {
        return adopted_frame(window_id)?.aim_window(&scene, &window, &HISTORY, x, y);
    }
    let (point, rect) = aim_at(&window, x, y)?;
    if let Ok(mut aims) = aims().lock() {
        aims.retain(|aim| (aim.window, aim.point) != (window_id, point));
        if aims.len() >= MAX_AIMS {
            aims.remove(0);
        }
        aims.push(Aim {
            window: window_id,
            point,
            rect,
        });
    }
    Ok(point)
}

fn aim_at(window: &Window, x: i32, y: i32) -> Result<((i32, i32), Rect)> {
    let rect = placed_or_refuse(window)?.rect;
    Ok(((rect.x.saturating_add(x), rect.y.saturating_add(y)), rect))
}

fn aimed_rect(window: u64, point: (i32, i32)) -> Option<Rect> {
    aims()
        .lock()
        .ok()?
        .iter()
        .rev()
        .find(|aim| (aim.window, aim.point) == (window, point))
        .map(|aim| aim.rect)
}

/// A window-addressed pointer action admitted on a settled layout. Each
/// confirmation reads the layout history again, so a change after admission
/// refuses the next event.
pub struct PointerGate {
    history: &'static History,
    mark: Mark,
    expected: Expected,
    ipc: AttestedIpc,
}

impl PointerGate {
    /// Refuse when the layout changed after admission. Call right before the
    /// first motion and again right before every press or wheel event.
    pub fn confirm(&mut self) -> Result<()> {
        if self
            .history
            .changed_since(self.mark, self.expected)
            .map_err(input_unavailable)?
        {
            return Err(foreground_unavailable(
                "the niri layout changed after the pointer target was checked; take a new screenshot",
            ));
        }
        Ok(())
    }

    /// The pointer space now, read on the action's connection: niri sends
    /// no event when outputs change.
    pub fn pointer_space(&mut self) -> Result<(i32, i32, u32, u32)> {
        pointer_space_of(&self.ipc.query(Request::Outputs)?)
    }
}

/// Gate window-addressed pointer input. A floating window: wait until the
/// layout is quiet, then require the target to be placed where its
/// coordinates were computed and every point to land on it rather than on
/// another floating window. A tiled window: require its output frame to
/// still hold (see [`OutputFrame`]). With `activate`, focus the window only
/// once all of that holds, so a refused action leaves focus alone.
pub fn gate_pointer(window_id: u64, points: &[(i32, i32)], activate: bool) -> Result<PointerGate> {
    let mut ipc = ipc();
    let scene = ipc.scene().map_err(input_unavailable)?;
    let window = scene.window(window_id).map_err(input_unavailable)?;
    if matches!(window.placement, Err(Unplaced::Tiled)) {
        let frame = adopted_frame(window_id)?;
        frame.admit(&scene, &window, points, activate, &HISTORY)?;
        return Ok(PointerGate {
            history: &HISTORY,
            mark: frame.mark,
            expected: Expected::default(),
            ipc,
        });
    }
    let before = placed_or_refuse(&window)?.rect;
    let mark = settled().map_err(input_unavailable)?;
    let window = ipc
        .scene()
        .and_then(|scene| scene.window(window_id))
        .map_err(input_unavailable)?;
    pointer_admission(&window, before, points, |point| {
        aimed_rect(window_id, point)
    })?;
    // Changes queued while the admission scene was read refuse here, before
    // focus moves.
    let mut gate = PointerGate {
        history: &HISTORY,
        mark,
        expected: Expected::default(),
        ipc,
    };
    gate.confirm()?;
    if activate {
        gate.expected = Expected {
            window: Some(window_id),
            workspace: window.workspace,
        };
        activate_in(&mut gate.ipc, window_id, None, FOCUS_CONFIRM_TIMEOUT)
            .map_err(input_unavailable)?;
    }
    Ok(gate)
}

fn pointer_admission(
    window: &Window,
    before: Rect,
    points: &[(i32, i32)],
    aimed: impl Fn((i32, i32)) -> Option<Rect>,
) -> Result<()> {
    let placed = placed_or_refuse(window)?;
    for &(x, y) in points {
        let expected = aimed((x, y)).unwrap_or(before);
        if placed.rect != expected {
            return Err(foreground_unavailable(format!(
                "niri window {} moved from {expected:?} to {:?} after its coordinates were computed; take a new screenshot",
                window.id, placed.rect
            )));
        }
        if !placed.covers(x, y) {
            return Err(foreground_unavailable(format!(
                "point ({x}, {y}) is outside niri window {} at {:?}",
                window.id, placed.rect
            )));
        }
        if !placed.area.contains(x, y) {
            return Err(foreground_unavailable(format!(
                "point ({x}, {y}) is past the edge of niri output {}, where window {} is not shown",
                placed.output, window.id
            )));
        }
        if !placed.hits(x, y) {
            return Err(foreground_unavailable(format!(
                "point ({x}, {y}) lies under another floating window, and niri reports no stacking order"
            )));
        }
    }
    Ok(())
}

// ── Tiled windows ────────────────────────────────────────────────────────────

/// What one output shows, as far as niri reports it: the output, its active
/// workspace with that workspace's active window and every window's layout,
/// the focused window and the overview. niri can also scroll the view and
/// so move every tiled window without changing any of this or sending any
/// event: the `center-column`, `center-visible-columns` and `center-window`
/// actions, and a touchpad or mouse view gesture that ends on the same
/// column. Nothing in IPC detects those; [`FRAME_MAX_AGE`] bounds them.
#[derive(Clone, Debug, PartialEq)]
pub struct Shown {
    output: IpcOutput,
    workspace: u64,
    active: Option<u64>,
    focused: Option<u64>,
    overview_open: bool,
    windows: Vec<(u64, Option<IpcLayout>)>,
}

impl Scene {
    fn shown(&self, output: &str) -> Option<Shown> {
        let workspace = self
            .workspaces
            .iter()
            .find(|workspace| workspace.is_active && workspace.output.as_deref() == Some(output))?;
        let mut windows: Vec<_> = self
            .windows
            .iter()
            .filter(|window| window.workspace_id == Some(workspace.id))
            .map(|window| (window.id, window.layout.clone()))
            .collect();
        windows.sort_by_key(|(id, _)| *id);
        Some(Shown {
            output: self.outputs.get(output)?.clone(),
            workspace: workspace.id,
            active: workspace.active_window_id,
            focused: self
                .windows
                .iter()
                .find(|window| window.is_focused)
                .map(|window| window.id),
            overview_open: self.overview_open,
            windows,
        })
    }
}

/// A tiled window's pixel frame. niri does not report where a tiled window
/// is, so its `get_window_state` screenshot is its whole output and its
/// window coordinates are that output's logical pixels. They are admitted
/// while niri's event stream reports no layout change after `mark`, taken
/// once the layout had settled before the capture, the output still shows
/// what `shown` recorded, and the screenshot is at most [`FRAME_MAX_AGE`]
/// old. That proves only that niri reported no change: a view scroll niri
/// does not report (see [`Shown`]) or an animation longer than
/// [`SETTLE_QUIET`] still moves the pixels, and a point can be on another
/// tiled window or a layer-shell surface, which IPC does not place. A point
/// inside a floating window, drawn above every tiled one, refuses.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputFrame {
    window: u64,
    area: Rect,
    shown: Shown,
    mark: Mark,
    /// When the capture started.
    taken: Instant,
}

impl OutputFrame {
    pub fn output(&self) -> &str {
        &self.shown.output.name
    }

    fn changed(&self, why: &str) -> anyhow::Error {
        foreground_unavailable(format!(
            "{why} after the screenshot of tiled niri window {}, so its pixels may be elsewhere now; take a new screenshot",
            self.window
        ))
    }

    fn holds(&self, scene: &Scene, history: &History) -> Result<()> {
        if self.taken.elapsed() > FRAME_MAX_AGE {
            return Err(foreground_unavailable(format!(
                "the screenshot of tiled niri window {} is over {} s old, and niri does not report every scroll of its view; take a new screenshot",
                self.window,
                FRAME_MAX_AGE.as_secs()
            )));
        }
        if scene.shown(self.output()).as_ref() != Some(&self.shown) {
            return Err(self.changed(&format!("niri output {} changed", self.output())));
        }
        if history
            .changed_since(self.mark, Expected::default())
            .map_err(input_unavailable)?
        {
            return Err(self.changed("the niri layout changed"));
        }
        Ok(())
    }

    /// Focusing a tiled window can scroll niri's view, which niri does not
    /// report, so the screenshot could no longer show the point.
    fn not_focused(&self) -> anyhow::Error {
        foreground_unavailable(format!(
            "tiled niri window {} is not focused, and niri may scroll its view to focus it; focus it, then take a new screenshot",
            self.window
        ))
    }

    /// The layout point of pixel `(x, y)` for window-addressed input on
    /// `window`, the frame's window as `scene` reports it now. It must
    /// already be focused: this runs before any route focuses the window,
    /// so an unfocused one refuses before focus or the view move.
    fn aim_window(
        &self,
        scene: &Scene,
        window: &Window,
        history: &History,
        x: i32,
        y: i32,
    ) -> Result<(i32, i32)> {
        if !window.focused {
            return Err(self.not_focused());
        }
        self.holds(scene, history)?;
        self.aim(x, y)
    }

    /// The layout point of pixel `(x, y)` of the screenshot.
    fn aim(&self, x: i32, y: i32) -> Result<(i32, i32)> {
        let point = (self.area.x.saturating_add(x), self.area.y.saturating_add(y));
        if x < 0 || y < 0 || !self.area.contains(point.0, point.1) {
            return Err(foreground_unavailable(format!(
                "point ({x}, {y}) is outside the screenshot of tiled niri window {}, which is its whole output, {}x{}",
                self.window, self.area.width, self.area.height
            )));
        }
        Ok(point)
    }

    /// Admit pointer input at layout `points` for `window`, the frame's
    /// window as `scene` reports it now.
    fn admit(
        &self,
        scene: &Scene,
        window: &Window,
        points: &[(i32, i32)],
        activate: bool,
        history: &History,
    ) -> Result<()> {
        self.holds(scene, history)?;
        let logical = self.shown.output.logical.as_ref();
        for &(x, y) in points {
            if !self.area.contains(x, y) {
                return Err(foreground_unavailable(format!(
                    "point ({x}, {y}) is past the edge of niri output {}, which tiled window {} was captured on",
                    self.output(),
                    self.window
                )));
            }
            let floating = self
                .shown
                .windows
                .iter()
                .filter_map(|(id, layout)| Some((id, floating_tile(logical?, layout.as_ref()?)?)))
                .find(|(_, tile)| tile.contains(f64::from(x), f64::from(y)));
            if let Some((id, _)) = floating {
                return Err(foreground_unavailable(format!(
                    "point ({x}, {y}) lies on floating niri window {id}, which niri draws above tiled window {}",
                    self.window
                )));
            }
        }
        if activate && !window.focused {
            return Err(self.not_focused());
        }
        Ok(())
    }
}

/// A tiled window's frame, from its latest `get_window_state` screenshot,
/// and the capture id that screenshot was published as.
type Adopted = (OutputFrame, Option<String>);

fn frames() -> &'static Mutex<Vec<Adopted>> {
    static FRAMES: OnceLock<Mutex<Vec<Adopted>>> = OnceLock::new();
    FRAMES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Record a window's latest `get_window_state` screenshot: a tiled window's
/// output `frame`, published as `capture_id`, or `None` for a window whose
/// screenshot shows only itself, which drops any earlier frame.
pub fn adopt_frame(window: u64, frame: Option<OutputFrame>, capture_id: Option<&str>) {
    let Ok(mut frames) = frames().lock() else {
        return;
    };
    frames.retain(|(frame, _)| frame.window != window);
    if let Some(frame) = frame {
        if frames.len() >= MAX_AIMS {
            frames.remove(0);
        }
        frames.push((frame, capture_id.map(str::to_owned)));
    }
}

fn adopted(window: u64) -> Option<Adopted> {
    frames()
        .lock()
        .ok()?
        .iter()
        .find(|(frame, _)| frame.window == window)
        .cloned()
}

fn adopted_frame(window: u64) -> Result<OutputFrame> {
    adopted(window).map(|(frame, _)| frame).ok_or_else(|| {
        foreground_unavailable(format!(
            "niri does not report where tiled window {window} is, so its coordinates are pixels of its get_window_state screenshot, which shows its whole output; take one first"
        ))
    })
}

/// A click bound to `capture_id` on a tiled window needs that capture to be
/// the window's latest screenshot: only that one's frame is checked against
/// the layout, and an earlier one may show another.
pub fn frame_admits_capture(window: u64, capture_id: &str) -> Result<()> {
    match adopted(window) {
        Some((_, adopted)) if adopted.as_deref() != Some(capture_id) => {
            Err(foreground_unavailable(format!(
                "capture {capture_id} is not the latest screenshot of tiled niri window {window}, and only that one's pixels are checked against niri's layout; use the latest get_window_state screenshot"
            )))
        }
        _ => Ok(()),
    }
}

// ── Window capture ───────────────────────────────────────────────────────────

/// Where a window is in a single-output display capture, and the output
/// geometry that mapping depends on.
#[derive(Clone, Debug, PartialEq)]
pub struct CapturePlan {
    /// Capture pixels, after the capture is oriented to the logical layout.
    pub pixels: Rect,
    /// The window's logical size, the size of the returned window image.
    pub width: u32,
    pub height: u32,
    /// The output the capture must come from, its logical rectangle and
    /// scale, and the oriented capture's size in pixels.
    pub output: String,
    pub area: Rect,
    pub scale: f64,
    pub transform: u32,
    pub image: (u32, u32),
    /// For a tiled window, captured as its whole output, what that output
    /// shows.
    pub shown: Option<Shown>,
}

/// An oriented display capture, the output it copied and the transform the
/// buffer was oriented with.
pub struct DisplayCapture {
    pub png: Vec<u8>,
    pub output: String,
    pub transform: u32,
    pub size: (u32, u32),
}

/// A window capture: the oriented display capture, where the window is in
/// it, and for a tiled window the frame its coordinates use once a
/// `get_window_state` screenshot adopts it.
#[derive(Debug)]
pub struct WindowCapture {
    pub png: Vec<u8>,
    pub plan: CapturePlan,
    pub frame: Option<OutputFrame>,
}

fn capture_plan_in(scene: &Scene, id: u64, pid: Option<u32>) -> Result<CapturePlan, &'static str> {
    let window = scene
        .window(id)
        .map_err(|_| "niri reports no such window; refresh list_windows")?;
    if pid.is_some_and(|pid| window.pid != Some(pid)) {
        return Err("niri reports the window belongs to a different process");
    }
    let (output, logical) = scene
        .single_output()
        .map_err(|_| "niri window capture supports a single output, and this layout has several")?;
    let mode = output
        .current_mode
        .and_then(|index| output.modes.get(index))
        .ok_or("niri reports no current mode for the output")?;
    let transform =
        transform_code(&logical.transform).map_err(|_| "unknown niri output transform")?;
    let image = super::logical_output_size((mode.width, mode.height), transform);
    let scale = logical.scale;
    let placed = match &window.placement {
        Ok(placed) => placed,
        Err(Unplaced::Tiled) => {
            return tiled_plan(scene, &window, &output.name, logical, transform, image)
        }
        Err(unplaced) => return Err(unplaced.reason()),
    };
    if placed.output != output.name {
        return Err("the window is not on the captured output");
    }
    if !placed.rect.inside(&placed.area) {
        return Err("the window extends past its output's edge");
    }
    if placed.overlapped() {
        return Err("another floating window overlaps it, and niri reports no stacking order");
    }
    // Round only in pixel space: the fractional origin is what niri drew.
    let pixels = Rect::from_f64(
        (placed.origin.0 - f64::from(logical.x)) * scale,
        (placed.origin.1 - f64::from(logical.y)) * scale,
        f64::from(placed.rect.width) * scale,
        f64::from(placed.rect.height) * scale,
    )
    .ok_or("invalid niri window geometry")?;
    Ok(CapturePlan {
        pixels,
        width: placed.rect.width,
        height: placed.rect.height,
        output: output.name.clone(),
        area: placed.area,
        scale,
        transform,
        image,
        shown: None,
    })
}

/// A tiled window is captured as its whole output, and only when niri
/// surely shows it: as the active window of the workspace shown on that
/// output, with the overview closed. niri keeps the active column in view;
/// any other tiled window may be scrolled out of it.
fn tiled_plan(
    scene: &Scene,
    window: &Window,
    output: &str,
    logical: &IpcLogical,
    transform: u32,
    image: (u32, u32),
) -> Result<CapturePlan, &'static str> {
    if scene.overview_open {
        return Err(Unplaced::NotShowing.reason());
    }
    let workspace = scene
        .workspaces
        .iter()
        .find(|workspace| Some(workspace.id) == window.workspace)
        .filter(|workspace| workspace.is_active && workspace.output.as_deref() == Some(output))
        .ok_or(Unplaced::NotShowing.reason())?;
    if workspace.active_window_id != Some(window.id) {
        return Err("niri does not report where a tiled window is, and shows for certain only its workspace's active window; focus the window first");
    }
    Ok(CapturePlan {
        pixels: Rect {
            x: 0,
            y: 0,
            width: image.0,
            height: image.1,
        },
        width: logical.width,
        height: logical.height,
        output: output.to_owned(),
        area: Rect {
            x: logical.x,
            y: logical.y,
            width: logical.width,
            height: logical.height,
        },
        scale: logical.scale,
        transform,
        image,
        shown: scene.shown(output),
    })
}

/// Capture one window from a settled display capture: a floating window
/// cropped to its place, a tiled one as its whole output. The layout must
/// stay unchanged through the capture, and the window and its output too.
/// Errors are refusal reasons for `surface_identity_unproven`.
pub fn capture_window(
    id: u64,
    pid: Option<u32>,
    capture_display: impl FnOnce() -> Result<DisplayCapture>,
) -> Result<WindowCapture, String> {
    let mut ipc = ipc();
    capture_window_in(
        || ipc.scene(),
        settled,
        |mark| HISTORY.changed_since(mark, Expected::default()),
        id,
        pid,
        capture_display,
    )
}

fn capture_window_in(
    mut scene: impl FnMut() -> Result<Scene>,
    settle: impl FnOnce() -> Result<Mark>,
    changed_since: impl FnOnce(Mark) -> Result<bool>,
    id: u64,
    pid: Option<u32>,
    capture_display: impl FnOnce() -> Result<DisplayCapture>,
) -> Result<WindowCapture, String> {
    let taken = Instant::now();
    let mut plan_now = || -> Result<CapturePlan, String> {
        Ok(capture_plan_in(
            &scene().map_err(|e| format!("{e:#}"))?,
            id,
            pid,
        )?)
    };
    let before = plan_now()?;
    let mark = settle().map_err(|e| format!("{e:#}"))?;
    let plan = plan_now()?;
    if plan != before {
        return Err("the window moved while niri was settling".into());
    }
    let capture = capture_display().map_err(|e| format!("{e:#}"))?;
    if changed_since(mark).map_err(|e| format!("{e:#}"))? {
        return Err("the niri layout changed during capture".into());
    }
    // niri has no output-change event: compare the output itself.
    if plan_now()? != plan {
        return Err("the niri window or its output changed during capture".into());
    }
    if capture.output != plan.output {
        return Err(format!(
            "the display capture copied output {:?}, not the window's output {:?}",
            capture.output, plan.output
        ));
    }
    if capture.transform != plan.transform {
        return Err(format!(
            "the capture was oriented for output transform {}, but the window's output has {}",
            capture.transform, plan.transform
        ));
    }
    if capture.size != plan.image {
        return Err(format!(
            "the capture is {}x{} pixels, but niri reports {}x{} for output {:?}",
            capture.size.0, capture.size.1, plan.image.0, plan.image.1, plan.output
        ));
    }
    let frame = plan.shown.clone().map(|shown| OutputFrame {
        window: id,
        area: plan.area,
        shown,
        mark,
        taken,
    });
    Ok(WindowCapture {
        png: capture.png,
        plan,
        frame,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex};

    // Shapes recorded from niri 26.04 (8ed0da4) in a nested session.
    fn floating(id: u64, workspace: u64, at: (f64, f64), size: (i32, i32)) -> Value {
        json!({"id": id, "title": format!("Float {id}"), "app_id": "float.a", "pid": 100 + id,
            "workspace_id": workspace, "is_focused": false, "is_floating": true, "is_urgent": false,
            "layout": {"pos_in_scrolling_layout": null, "tile_size": [f64::from(size.0), f64::from(size.1)],
                "window_size": [size.0, size.1], "tile_pos_in_workspace_view": [at.0, at.1],
                "window_offset_in_tile": [0.0, 0.0]},
            "focus_timestamp": {"secs": 804, "nanos": 995946831}})
    }

    fn tiled(id: u64, workspace: u64, tile: (f64, f64)) -> Value {
        json!({"id": id, "title": "Tiled", "app_id": "tiled.b", "pid": 100 + id,
            "workspace_id": workspace, "is_focused": false, "is_floating": false, "is_urgent": false,
            "layout": {"pos_in_scrolling_layout": [1, 1], "tile_size": [tile.0, tile.1],
                "window_size": [tile.0 as i32, tile.1 as i32], "tile_pos_in_workspace_view": null,
                "window_offset_in_tile": [0.0, 0.0]},
            "focus_timestamp": {"secs": 805, "nanos": 828275861}})
    }

    fn workspace(id: u64, output: &str, active: bool, active_window: Option<u64>) -> Value {
        json!({"id": id, "idx": id, "name": null, "output": output, "is_urgent": false,
            "is_active": active, "is_focused": active, "active_window_id": active_window})
    }

    fn output(name: &str, x: i32, y: i32, mode: (u32, u32), scale: f64, transform: &str) -> Value {
        let width = (f64::from(mode.0) / scale) as u32;
        let height = (f64::from(mode.1) / scale) as u32;
        json!({"name": name, "make": "Smithay", "model": "Winit", "serial": null, "physical_size": null,
            "modes": [{"width": mode.0, "height": mode.1, "refresh_rate": 60000, "is_preferred": true}],
            "current_mode": 0, "is_custom_mode": true, "vrr_supported": false, "vrr_enabled": false,
            "logical": {"x": x, "y": y, "width": width, "height": height, "scale": scale, "transform": transform}})
    }

    fn scene(
        windows: Vec<Value>,
        workspaces: Vec<Value>,
        outputs: Vec<Value>,
        overview: bool,
    ) -> Scene {
        Scene {
            windows: serde_json::from_value(Value::Array(windows)).unwrap(),
            workspaces: serde_json::from_value(Value::Array(workspaces)).unwrap(),
            outputs: outputs
                .into_iter()
                .map(|o| {
                    (
                        o["name"].as_str().unwrap().to_owned(),
                        serde_json::from_value(o).unwrap(),
                    )
                })
                .collect(),
            overview_open: overview,
        }
    }

    fn one_output(windows: Vec<Value>) -> Scene {
        scene(
            windows,
            vec![
                workspace(1, "winit", true, None),
                workspace(2, "winit", false, None),
            ],
            vec![output("winit", 0, 0, (1280, 720), 1.5, "Flipped180")],
            false,
        )
    }

    fn placement(scene: &Scene, id: u64) -> Result<Placed, Unplaced> {
        scene.window(id).unwrap().placement
    }

    // ── Fake niri ───────────────────────────────────────────────────────────

    #[derive(Default)]
    struct Fake {
        windows: Vec<Value>,
        workspaces: Vec<Value>,
        overview: bool,
        /// FocusWindow is answered `Handled` but changes nothing.
        ignore_focus: bool,
        /// FocusWindow on these ids changes nothing (a window that closed).
        ignore_focus_on: Vec<u64>,
        /// FocusWindow on these ids switches their workspace, but keyboard
        /// focus stays elsewhere (a layer-shell surface took it).
        focus_taken_on: Vec<u64>,
        /// After a FocusWorkspace, this many Workspaces replies still show
        /// the layout from before it (niri applies it asynchronously).
        workspace_lag: usize,
        stale: Option<(usize, Vec<Value>)>,
        requests: Vec<String>,
        connections: usize,
    }

    impl Fake {
        fn reply(&mut self, request: &str) -> Value {
            self.requests.push(request.to_owned());
            let request: Value = serde_json::from_str(request).unwrap();
            match request.as_str() {
                Some("Windows") => json!({"Ok": {"Windows": self.windows}}),
                Some("Workspaces") => match &mut self.stale {
                    Some((reads, stale)) if *reads > 0 => {
                        *reads -= 1;
                        json!({"Ok": {"Workspaces": stale}})
                    }
                    _ => json!({"Ok": {"Workspaces": self.workspaces}}),
                },
                Some("OverviewState") => {
                    json!({"Ok": {"OverviewState": {"is_open": self.overview}}})
                }
                Some("FocusedWindow") => json!({"Ok": {"FocusedWindow":
                    self.windows.iter().find(|w| w["is_focused"] == true)}}),
                _ => {
                    let action = &request["Action"];
                    if let Some(id) = action["FocusWindow"]["id"].as_u64() {
                        if !self.ignore_focus && !self.ignore_focus_on.contains(&id) {
                            let workspace = self
                                .windows
                                .iter()
                                .find(|w| w["id"] == id)
                                .map(|w| w["workspace_id"].clone());
                            if let Some(workspace) = workspace.and_then(|w| w.as_u64()) {
                                self.show_workspace(workspace, Some(id));
                            }
                            if self.focus_taken_on.contains(&id) {
                                for window in &mut self.windows {
                                    window["is_focused"] = json!(false);
                                }
                            }
                        }
                    } else if let Some(id) = action["FocusWorkspace"]["reference"]["Id"].as_u64() {
                        let active = self
                            .workspaces
                            .iter()
                            .find(|w| w["id"] == id)
                            .and_then(|w| w["active_window_id"].as_u64());
                        if self.workspace_lag > 0 {
                            self.stale = Some((self.workspace_lag, self.workspaces.clone()));
                        }
                        self.show_workspace(id, active);
                    } else {
                        return json!({"Err": "error parsing request"});
                    }
                    json!({"Ok": "Handled"})
                }
            }
        }

        /// niri's focus change: the workspace becomes active on its output
        /// and focused, and `window` its active, focused window.
        fn show_workspace(&mut self, id: u64, window: Option<u64>) {
            let output = self
                .workspaces
                .iter()
                .find(|w| w["id"] == id)
                .map(|w| w["output"].clone());
            for workspace in &mut self.workspaces {
                let this = workspace["id"] == id;
                if Some(&workspace["output"]) == output.as_ref() {
                    workspace["is_active"] = json!(this);
                }
                workspace["is_focused"] = json!(this);
                if this && window.is_some() {
                    workspace["active_window_id"] = json!(window);
                }
            }
            for w in &mut self.windows {
                w["is_focused"] = json!(window.is_some() && w["id"] == json!(window));
            }
        }

        fn active(&self, id: u64) -> bool {
            self.workspaces
                .iter()
                .any(|w| w["id"] == id && w["is_active"] == true)
        }

        fn focused(&self) -> Option<u64> {
            self.windows
                .iter()
                .find(|w| w["is_focused"] == true)
                .and_then(|w| w["id"].as_u64())
        }
    }

    fn fake_ipc(fake: &Arc<Mutex<Fake>>) -> Ipc<impl FnMut(Instant) -> Result<UnixStream>> {
        let fake = fake.clone();
        Ipc {
            connect: move |_| {
                let (client, server) = UnixStream::pair()?;
                let fake = fake.clone();
                fake.lock().unwrap().connections += 1;
                std::thread::spawn(move || {
                    let mut lines = Lines::new(server);
                    while let Ok(Some(line)) = lines.next(Instant::now() + Duration::from_secs(2)) {
                        let reply = fake
                            .lock()
                            .unwrap()
                            .reply(std::str::from_utf8(&line).unwrap());
                        if lines
                            .stream
                            .write_all(format!("{reply}\n").as_bytes())
                            .is_err()
                        {
                            return;
                        }
                    }
                });
                Ok(client)
            },
            per_attempt: Duration::from_secs(1),
            total: Duration::from_secs(2),
            backoff: Duration::from_millis(1),
            lines: None,
        }
    }

    /// A connection whose peer sends `reply` (raw bytes) and then holds it open.
    fn replying(reply: Vec<u8>) -> UnixStream {
        let (client, mut server) = UnixStream::pair().unwrap();
        std::thread::spawn(move || {
            let _ = server.write_all(&reply);
            std::thread::sleep(Duration::from_secs(2));
        });
        client
    }

    fn ipc_with(
        connect: impl FnMut(Instant) -> Result<UnixStream>,
        per_attempt: Duration,
        total: Duration,
    ) -> Ipc<impl FnMut(Instant) -> Result<UnixStream>> {
        Ipc {
            connect,
            per_attempt,
            total,
            backoff: Duration::from_millis(1),
            lines: None,
        }
    }

    // ── Session attestation ─────────────────────────────────────────────────

    #[test]
    fn foreign_socket_before_recognition_is_an_unrelated_session() {
        // A host desktop's inherited NIRI_SOCKET next to a nested compositor.
        assert_eq!(
            attest(None, Ok(10), Ok(20)),
            Attestation::Unrelated(
                "NIRI_SOCKET is served by pid 10 but WAYLAND_DISPLAY by pid 20".into()
            )
        );
        assert!(matches!(
            attest(None, Err(anyhow!("connection refused")), Ok(20)),
            Attestation::Unrelated(_)
        ));
        assert_eq!(attest(None, Ok(20), Ok(20)), Attestation::Attested(20));
    }

    #[test]
    fn a_recognised_niri_stays_niri_after_its_socket_disappears() {
        assert_eq!(session_check(true, true, false), SessionCheck::Recognized);
        assert_eq!(session_check(true, false, false), SessionCheck::NotNiri);
        assert_eq!(session_check(true, false, true), SessionCheck::Probe);
        assert_eq!(session_check(false, true, true), SessionCheck::NotNiri);
    }

    #[test]
    fn recognised_niri_refuses_loss_mismatch_and_a_new_compositor() {
        assert_eq!(attest(Some(20), Ok(20), Ok(20)), Attestation::Attested(20));
        match attest(Some(20), Err(anyhow!("No such file or directory")), Ok(20)) {
            Attestation::Refused(reason) => assert!(reason.contains("No such file or directory")),
            other => panic!("socket loss must refuse, got {other:?}"),
        }
        assert_eq!(
            attest(Some(20), Ok(20), Ok(30)),
            Attestation::Refused(
                "NIRI_SOCKET is served by pid 20 but WAYLAND_DISPLAY by pid 30".into()
            )
        );
        assert_eq!(
            attest(Some(20), Ok(40), Ok(40)),
            Attestation::Refused(
                "niri was recognised as pid 20, but the compositor is now pid 40".into()
            )
        );
    }

    // ── Transport ───────────────────────────────────────────────────────────

    #[test]
    fn stalled_niri_hits_the_deadline_and_retries_a_read_once() {
        let mut attempts = 0;
        let mut held = Vec::new();
        let started = Instant::now();
        let error = ipc_with(
            |_| {
                attempts += 1;
                let (client, server) = UnixStream::pair()?;
                held.push(server);
                Ok(client)
            },
            Duration::from_millis(40),
            Duration::from_millis(500),
        )
        .query::<Value>(Request::Windows)
        .unwrap_err();
        assert!(is_timeout(&error), "{error:#}");
        assert_eq!(attempts, 2);
        // Bounded by the deadlines, not by the peer; loose for loaded CI.
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn one_action_attests_one_connection_for_all_its_requests() {
        let fake = focus_fixture();
        let mut ipc = fake_ipc(&fake);
        ipc.focused_window().unwrap();
        activate_in(&mut ipc, 7, Some(107), Duration::from_millis(200)).unwrap();
        let fake = fake.lock().unwrap();
        assert_eq!(fake.requests.len(), 5, "{:?}", fake.requests);
        assert_eq!(fake.connections, 1);
    }

    #[test]
    fn an_action_is_never_retried_after_an_ambiguous_timeout() {
        let mut attempts = 0;
        let mut held = Vec::new();
        let error = ipc_with(
            |_| {
                attempts += 1;
                let (client, server) = UnixStream::pair()?;
                held.push(server);
                Ok(client)
            },
            Duration::from_millis(40),
            Duration::from_millis(500),
        )
        .action(focus_window_action(7))
        .unwrap_err();
        assert_eq!(attempts, 1);
        assert_eq!(error.to_string(), "niri action outcome is unknown");
    }

    #[test]
    fn connection_and_attestation_errors_are_not_retried() {
        let mut attempts = 0;
        let error = ipc_with(
            |_| {
                attempts += 1;
                bail!("NIRI_SOCKET is served by pid 10 but WAYLAND_DISPLAY by pid 20")
            },
            QUERY_TIMEOUT,
            QUERY_TOTAL_TIMEOUT,
        )
        .query::<Value>(Request::Windows)
        .unwrap_err();
        assert_eq!(attempts, 1);
        assert_eq!(
            error.to_string(),
            "NIRI_SOCKET is served by pid 10 but WAYLAND_DISPLAY by pid 20"
        );
    }

    #[test]
    fn niri_error_text_is_kept_verbatim() {
        let error = ipc_with(
            |_| Ok(replying(b"{\"Err\":\"error parsing request\"}\n".to_vec())),
            QUERY_TIMEOUT,
            QUERY_TOTAL_TIMEOUT,
        )
        .action(focus_window_action(7))
        .unwrap_err();
        assert!(error.to_string().ends_with(
            "niri refused {\"Action\":{\"FocusWindow\":{\"id\":7}}}: error parsing request"
        ));
    }

    #[test]
    fn oversize_and_malformed_replies_refuse() {
        let reply = |bytes: Vec<u8>| {
            ipc_with(
                move |_| Ok(replying(bytes.clone())),
                QUERY_TIMEOUT,
                QUERY_TOTAL_TIMEOUT,
            )
            .query::<Vec<IpcWindow>>(Request::Windows)
            .unwrap_err()
        };
        let oversize = reply(vec![b' '; MAX_REPLY_BYTES + 1]);
        assert!(format!("{oversize:#}").contains("niri IPC reply exceeds size limit"));
        let id_as_string =
            br#"{"Ok":{"Windows":[{"id":"7","pid":1,"workspace_id":1,"is_focused":false}]}}
"#;
        assert!(
            format!("{:#}", reply(id_as_string.to_vec())).contains("invalid niri Windows reply")
        );
        let other_variant = b"{\"Ok\":{\"Workspaces\":[]}}\n";
        assert!(format!("{:#}", reply(other_variant.to_vec()))
            .contains("niri reply to Windows has no Windows"));
    }

    #[test]
    fn unknown_fields_are_ignored_and_null_pid_has_no_owner() {
        let mut window = floating(7, 1, (100.0, 60.0), (300, 200));
        window["pid"] = Value::Null;
        window["is_new_in_a_later_release"] = json!({"nested": [1, 2]});
        window["layout"]["also_new"] = json!(true);
        let scene = one_output(vec![window]);
        let window = scene.window(7).unwrap();
        assert_eq!(window.pid, None);
        assert!(window.placement.is_ok());
    }

    #[test]
    fn unknown_output_transform_refuses_pointer_space() {
        let scene = scene(
            vec![],
            vec![workspace(1, "winit", true, None)],
            vec![output("winit", 0, 0, (1280, 720), 1.5, "Diagonal")],
            false,
        );
        assert_eq!(
            scene.pointer_space().unwrap_err().to_string(),
            "unknown niri output transform \"Diagonal\""
        );
    }

    #[test]
    fn duplicate_window_ids_refuse_the_snapshot() {
        let scene = one_output(vec![
            floating(7, 1, (0.0, 0.0), (10, 10)),
            floating(7, 1, (50.0, 0.0), (10, 10)),
        ]);
        assert_eq!(
            scene.windows().unwrap_err().to_string(),
            "duplicate niri window id 7"
        );
    }

    // ── Placement ───────────────────────────────────────────────────────────

    #[test]
    fn the_focused_then_most_recently_focused_window_ranks_highest() {
        // A main window focused at 805 s and its floating dialog at 804 s.
        let main = tiled(5, 1, (402.0, 448.0));
        let mut dialog = floating(7, 1, (100.0, 60.0), (300, 200));
        let rank = |windows: Vec<Value>, id| one_output(windows).window(id).unwrap().focus_rank;
        assert!(
            rank(vec![main.clone(), dialog.clone()], 5)
                > rank(vec![main.clone(), dialog.clone()], 7)
        );
        dialog["is_focused"] = json!(true);
        assert!(rank(vec![main.clone(), dialog.clone()], 7) > rank(vec![main, dialog], 5));
    }

    #[test]
    fn a_pid_only_keyboard_action_goes_to_the_window_in_use() {
        // A main window focused at 805 s and its focused floating dialog.
        let main = tiled(5, 1, (402.0, 448.0));
        let mut dialog = floating(7, 1, (100.0, 60.0), (300, 200));
        dialog["pid"] = json!(105);
        dialog["is_focused"] = json!(true);
        let mut scene = one_output(vec![main, dialog]);
        scene.workspaces[0].active_window_id = Some(5);
        let windows = scene.windows().unwrap();
        assert_eq!(pick_window_in_use(&windows, 105), Some(7));
        assert_eq!(pick_window_in_use(&windows, 999), None);
    }

    #[test]
    fn floating_window_geometry_adds_output_origin_tile_and_offset() {
        let mut window = floating(7, 1, (100.25, 60.0), (300, 200));
        window["layout"]["window_offset_in_tile"] = json!([4.0, 4.0]);
        let scene = scene(
            vec![window],
            vec![workspace(1, "DP-2", true, None)],
            vec![output("DP-2", 1920, 0, (2560, 1440), 1.25, "Normal")],
            false,
        );
        assert_eq!(
            placement(&scene, 7).unwrap().rect,
            Rect {
                x: 2024,
                y: 64,
                width: 300,
                height: 200
            }
        );
    }

    #[test]
    fn tiled_windows_get_no_geometry() {
        let scene = one_output(vec![tiled(5, 1, (402.0, 448.0))]);
        assert_eq!(placement(&scene, 5), Err(Unplaced::Tiled));
        assert!(
            !scene.window(5).unwrap().on_screen,
            "an inactive tiled window is not proven on screen"
        );
    }

    #[test]
    fn active_tiled_window_is_on_screen_without_a_place() {
        let scene = scene(
            vec![tiled(5, 1, (402.0, 448.0))],
            vec![workspace(1, "winit", true, Some(5))],
            vec![output("winit", 0, 0, (1280, 720), 1.5, "Normal")],
            false,
        );
        let window = scene.window(5).unwrap();
        assert!(window.on_screen);
        assert_eq!(window.placement, Err(Unplaced::Tiled));
        // The size still identifies it, for AT-SPI root correlation.
        assert_eq!(window.size, Some((402, 448)));
    }

    #[test]
    fn floating_window_on_an_inactive_workspace_is_not_showing() {
        let scene = one_output(vec![floating(7, 2, (100.0, 60.0), (300, 200))]);
        assert_eq!(placement(&scene, 7), Err(Unplaced::NotShowing));
        assert!(!scene.window(7).unwrap().on_screen);
    }

    #[test]
    fn open_overview_hides_every_place() {
        let scene = scene(
            vec![floating(7, 1, (100.0, 60.0), (300, 200))],
            vec![workspace(1, "winit", true, None)],
            vec![output("winit", 0, 0, (1280, 720), 1.5, "Normal")],
            true,
        );
        assert_eq!(placement(&scene, 7), Err(Unplaced::NotShowing));
    }

    #[test]
    fn output_sized_active_tile_covers_floating_windows() {
        // Logical output 853x480; a fullscreen tile reports about that size.
        for (tile, covered) in [
            ((853.0, 480.0), true),
            ((852.4, 479.2), true),
            ((402.0, 448.0), false),
        ] {
            let scene = scene(
                vec![floating(7, 1, (100.0, 60.0), (300, 200)), tiled(5, 1, tile)],
                vec![workspace(1, "winit", true, Some(5))],
                vec![output("winit", 0, 0, (1280, 720), 1.5, "Normal")],
                false,
            );
            assert_eq!(
                placement(&scene, 7) == Err(Unplaced::Covered),
                covered,
                "{tile:?}"
            );
        }
    }

    #[test]
    fn layoutless_niri_refuses_geometry() {
        let mut window = floating(7, 1, (100.0, 60.0), (300, 200));
        window.as_object_mut().unwrap().remove("layout");
        assert_eq!(
            placement(&one_output(vec![window]), 7),
            Err(Unplaced::NoLayout)
        );
    }

    #[test]
    fn any_floating_overlap_is_ambiguous_whatever_the_focus_order() {
        let stamps = [
            (
                json!({"secs": 1, "nanos": 0}),
                json!({"secs": 2, "nanos": 0}),
            ),
            (
                json!({"secs": 2, "nanos": 0}),
                json!({"secs": 1, "nanos": 0}),
            ),
            (
                json!({"secs": 1, "nanos": 0}),
                json!({"secs": 1, "nanos": 0}),
            ),
            (Value::Null, Value::Null),
        ];
        for (target_stamp, other_stamp) in stamps {
            for target_focused in [false, true] {
                let mut target = floating(7, 1, (100.0, 60.0), (300, 200));
                target["focus_timestamp"] = target_stamp.clone();
                target["is_focused"] = json!(target_focused);
                // A dialog (child) over its parent looks the same in IPC.
                let mut other = floating(8, 1, (350.0, 200.0), (200, 100));
                other["focus_timestamp"] = other_stamp.clone();
                let placed = placement(&one_output(vec![target, other]), 7).unwrap();
                assert!(placed.overlapped());
                assert!(
                    !placed.hits(360, 210),
                    "a point under the other tile is ambiguous"
                );
                assert!(placed.hits(110, 70));
            }
        }
    }

    #[test]
    fn floating_windows_on_other_workspaces_do_not_overlap() {
        let placed = placement(
            &one_output(vec![
                floating(7, 1, (100.0, 60.0), (300, 200)),
                floating(8, 2, (100.0, 60.0), (300, 200)),
                tiled(5, 1, (402.0, 448.0)),
            ]),
            7,
        )
        .unwrap();
        assert!(!placed.overlapped());
    }

    #[test]
    fn pointer_space_rounds_each_output_up_and_spans_all_outputs() {
        let single = |mode, scale| {
            scene(
                vec![],
                vec![],
                vec![output("winit", 0, 0, mode, scale, "Normal")],
                false,
            )
            .pointer_space()
            .unwrap()
        };
        assert_eq!(single((1280, 720), 1.0), (0, 0, 1280, 720));
        assert_eq!(single((1280, 720), 1.25), (0, 0, 1024, 576));
        assert_eq!(single((1280, 720), 1.5), (0, 0, 854, 480));
        assert_eq!(single((1280, 720), 2.0), (0, 0, 640, 360));
        let pair = scene(
            vec![],
            vec![],
            vec![
                output("eDP-1", 0, 0, (1280, 720), 1.5, "Normal"),
                output("DP-2", 853, -100, (1920, 1080), 1.0, "Normal"),
            ],
            false,
        );
        assert_eq!(pair.pointer_space().unwrap(), (0, -100, 2773, 1080));
        let rotated = scene(
            vec![],
            vec![],
            vec![output("DP-2", 0, 0, (1920, 1080), 1.0, "90")],
            false,
        );
        assert_eq!(rotated.pointer_space().unwrap(), (0, 0, 1080, 1920));
    }

    #[test]
    fn disabled_outputs_are_left_out_of_pointer_space() {
        let mut off = output("HDMI-A-1", 0, 0, (1920, 1080), 1.0, "Normal");
        off["logical"] = Value::Null;
        off["current_mode"] = Value::Null;
        let scene = scene(
            vec![],
            vec![],
            vec![output("winit", 0, 0, (1280, 720), 1.5, "Normal"), off],
            false,
        );
        assert_eq!(scene.pointer_space().unwrap(), (0, 0, 854, 480));
    }

    #[test]
    fn desktop_snapshot_uses_the_single_output_and_refuses_several() {
        let one = scene(
            vec![],
            vec![],
            vec![output("DP-2", 100, 50, (1280, 720), 1.5, "Normal")],
            false,
        );
        let snapshot = desktop_snapshot_in(&one).unwrap();
        assert_eq!(
            (
                snapshot.frame.x,
                snapshot.frame.y,
                snapshot.frame.width,
                snapshot.frame.height
            ),
            (100, 50, 853, 480)
        );
        assert_eq!(snapshot.frame.scale, 1.5);
        assert_eq!(snapshot.pointer, Some((100, 50, 854, 480)));
        let two = scene(
            vec![],
            vec![],
            vec![
                output("DP-1", 0, 0, (1920, 1080), 1.0, "Normal"),
                output("DP-2", 1920, 0, (1920, 1080), 1.0, "Normal"),
            ],
            false,
        );
        assert_eq!(
            desktop_snapshot_in(&two).unwrap_err().to_string(),
            "niri window capture and desktop input support a single output; this layout has 2"
        );
    }

    // ── Capture plan ────────────────────────────────────────────────────────

    #[test]
    fn crop_maps_global_logical_geometry_to_output_pixels() {
        let scene = scene(
            vec![floating(7, 1, (100.0, 60.0), (300, 200))],
            vec![workspace(1, "DP-2", true, None)],
            vec![output("DP-2", 1000, 40, (1920, 1080), 1.5, "Normal")],
            false,
        );
        assert_eq!(
            capture_plan_in(&scene, 7, Some(107)),
            Ok(CapturePlan {
                pixels: Rect {
                    x: 150,
                    y: 90,
                    width: 450,
                    height: 300
                },
                width: 300,
                height: 200,
                output: "DP-2".into(),
                area: Rect {
                    x: 1000,
                    y: 40,
                    width: 1280,
                    height: 720
                },
                scale: 1.5,
                transform: 0,
                image: (1920, 1080),
                shown: None,
            })
        );
    }

    #[test]
    fn a_one_physical_pixel_overlap_at_a_fractional_scale_refuses() {
        // At 1.5 the target spans physical x 152..602 and the other tile
        // starts at physical 601. Rounded to logical pixels (101 and 401)
        // they would only touch.
        let scene = one_output(vec![
            floating(7, 1, (152.0 / 1.5, 60.0), (300, 200)),
            floating(8, 1, (601.0 / 1.5, 60.0), (100, 100)),
        ]);
        assert_eq!(
            capture_plan_in(&scene, 7, None),
            Err("another floating window overlaps it, and niri reports no stacking order")
        );
        let placed = placement(&scene, 7).unwrap();
        assert!(placed.hits(400, 100));
        assert!(
            !placed.hits(401, 100),
            "x=401 is also on the tile from 400.667"
        );
    }

    #[test]
    fn crop_keeps_a_pixel_aligned_fractional_origin() {
        // At scale 1.5 niri places a window on physical x=151, which IPC
        // reports as logical 100.667; rounding it first would crop from 152.
        let scene = one_output(vec![floating(7, 1, (151.0 / 1.5, 60.0), (300, 200))]);
        let plan = capture_plan_in(&scene, 7, None).unwrap();
        assert_eq!((plan.pixels.x, plan.pixels.width), (151, 450));
        let placed = placement(&scene, 7).unwrap();
        assert!(!placed.covers(100, 100), "x=100 is left of 100.667");
        assert!(placed.covers(101, 100));
    }

    fn settled_now() -> Result<Mark> {
        Ok(Mark {
            generation: 1,
            at: 0,
        })
    }

    fn unchanged(_: Mark) -> Result<bool> {
        Ok(false)
    }

    fn display(output: &str, size: (u32, u32)) -> Result<DisplayCapture> {
        oriented(output, 6, size)
    }

    fn oriented(output: &str, transform: u32, size: (u32, u32)) -> Result<DisplayCapture> {
        Ok(DisplayCapture {
            png: vec![1],
            output: output.into(),
            transform,
            size,
        })
    }

    #[test]
    fn capture_refuses_an_output_change_niri_sends_no_event_for() {
        let window = || floating(7, 1, (100.0, 60.0), (300, 200));
        let mut calls = 0;
        let error = capture_window_in(
            || {
                calls += 1;
                // The scale changes during the capture; the layout does not.
                let scale = if calls <= 2 { 1.5 } else { 1.0 };
                Ok(scene(
                    vec![window()],
                    vec![workspace(1, "winit", true, None)],
                    vec![output("winit", 0, 0, (1280, 720), scale, "Normal")],
                    false,
                ))
            },
            settled_now,
            unchanged,
            7,
            None,
            || display("winit", (1280, 720)),
        )
        .unwrap_err();
        assert_eq!(
            error,
            "the niri window or its output changed during capture"
        );
    }

    #[test]
    fn capture_refuses_a_transform_change_that_keeps_every_size() {
        // one_output's winit output is Flipped180 (6); Normal (0) keeps the
        // name, area, scale and image size.
        let one = || Ok(one_output(vec![floating(7, 1, (100.0, 60.0), (300, 200))]));
        assert_eq!(
            capture_window_in(one, settled_now, unchanged, 7, None, || oriented(
                "winit",
                0,
                (1280, 720)
            ))
            .unwrap_err(),
            "the capture was oriented for output transform 0, but the window's output has 6"
        );
        let mut calls = 0;
        let error = capture_window_in(
            || {
                calls += 1;
                let transform = if calls <= 2 { "Flipped180" } else { "Normal" };
                Ok(scene(
                    vec![floating(7, 1, (100.0, 60.0), (300, 200))],
                    vec![workspace(1, "winit", true, None)],
                    vec![output("winit", 0, 0, (1280, 720), 1.5, transform)],
                    false,
                ))
            },
            settled_now,
            unchanged,
            7,
            None,
            || display("winit", (1280, 720)),
        )
        .unwrap_err();
        assert_eq!(
            error,
            "the niri window or its output changed during capture"
        );
    }

    #[test]
    fn capture_refuses_another_output_or_size() {
        let one = || Ok(one_output(vec![floating(7, 1, (100.0, 60.0), (300, 200))]));
        assert_eq!(
            capture_window_in(one, settled_now, unchanged, 7, None, || display(
                "DP-1",
                (1280, 720)
            ))
            .unwrap_err(),
            "the display capture copied output \"DP-1\", not the window's output \"winit\""
        );
        assert_eq!(
            capture_window_in(one, settled_now, unchanged, 7, None, || display(
                "winit",
                (1920, 1080)
            ))
            .unwrap_err(),
            "the capture is 1920x1080 pixels, but niri reports 1280x720 for output \"winit\""
        );
        assert!(
            capture_window_in(one, settled_now, unchanged, 7, None, || display(
                "winit",
                (1280, 720)
            ))
            .is_ok()
        );
    }

    #[test]
    fn screen_size_spans_every_output_with_the_largest_scale() {
        let two = scene(
            vec![],
            vec![],
            vec![
                output("DP-1", 0, 0, (1920, 1080), 1.0, "Normal"),
                output("DP-2", 1920, 0, (2560, 1440), 2.0, "Normal"),
            ],
            false,
        );
        let frame = screen_frame_in(&two).unwrap();
        assert_eq!(
            (frame.x, frame.y, frame.width, frame.height, frame.scale),
            (0, 0, 3200, 1080, 2.0)
        );
        assert_eq!(frame.outputs.len(), 2);
    }

    #[test]
    fn crop_refuses_other_outputs_overlap_and_other_pids() {
        let two_outputs = scene(
            vec![floating(7, 2, (100.0, 60.0), (300, 200))],
            vec![
                workspace(1, "DP-1", true, None),
                workspace(2, "DP-2", true, None),
            ],
            vec![
                output("DP-1", 0, 0, (1920, 1080), 1.0, "Normal"),
                output("DP-2", 1920, 0, (1920, 1080), 1.0, "Normal"),
            ],
            false,
        );
        assert_eq!(
            capture_plan_in(&two_outputs, 7, None),
            Err("niri window capture supports a single output, and this layout has several")
        );
        let overlapped = one_output(vec![
            floating(7, 1, (100.0, 60.0), (300, 200)),
            floating(8, 1, (390.0, 250.0), (100, 100)),
        ]);
        assert_eq!(
            capture_plan_in(&overlapped, 7, None),
            Err("another floating window overlaps it, and niri reports no stacking order")
        );
        let window = one_output(vec![floating(7, 1, (100.0, 60.0), (300, 200))]);
        assert_eq!(
            capture_plan_in(&window, 7, Some(1)),
            Err("niri reports the window belongs to a different process")
        );
        let past_edge = one_output(vec![floating(7, 1, (700.0, 60.0), (300, 200))]);
        assert_eq!(
            capture_plan_in(&past_edge, 7, None),
            Err("the window extends past its output's edge")
        );
    }

    // ── Focus transactions ──────────────────────────────────────────────────

    fn focus_fixture() -> Arc<Mutex<Fake>> {
        let mut prior = tiled(5, 1, (402.0, 448.0));
        prior["is_focused"] = json!(true);
        Arc::new(Mutex::new(Fake {
            windows: vec![prior, floating(7, 2, (100.0, 60.0), (300, 200))],
            workspaces: vec![
                workspace(1, "winit", true, Some(5)),
                workspace(2, "winit", false, Some(7)),
            ],
            ..Fake::default()
        }))
    }

    #[test]
    fn focus_transaction_confirms_runs_and_restores() {
        let fake = focus_fixture();
        let seen = with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(100),
            || Ok(fake.lock().unwrap().focused()),
        )
        .unwrap();
        assert_eq!(seen, Some(7));
        assert_eq!(fake.lock().unwrap().focused(), Some(5));
    }

    #[test]
    fn unconfirmed_focus_refuses_before_the_body() {
        let fake = focus_fixture();
        fake.lock().unwrap().ignore_focus = true;
        let mut ran = false;
        let error = with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(50),
            || {
                ran = true;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(!ran);
        assert_eq!(
            error.to_string(),
            "foreground_unavailable: niri did not confirm focus on window 7 (focused: 5)"
        );
    }

    #[test]
    fn a_lost_niri_refuses_focus_bound_input_with_the_typed_code() {
        let mut ran = false;
        let error = with_focused_window_in(
            &mut ipc_with(
                |_| bail!("niri was recognised as pid 20, but the compositor is now pid 40"),
                Duration::from_millis(50),
                Duration::from_millis(100),
            ),
            107,
            7,
            Duration::from_millis(50),
            || {
                ran = true;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(!ran);
        assert_eq!(
            error.to_string(),
            "foreground_unavailable: niri was recognised as pid 20, but the compositor is now pid 40"
        );
    }

    #[test]
    fn another_pid_refuses_without_focusing() {
        let fake = focus_fixture();
        let error = with_focused_window_in(
            &mut fake_ipc(&fake),
            999,
            7,
            Duration::from_millis(50),
            || Ok(()),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "foreground_unavailable: niri window 7 belongs to pid 107, not pid 999"
        );
        assert!(!fake
            .lock()
            .unwrap()
            .requests
            .iter()
            .any(|r| r.contains("FocusWindow")));
    }

    #[test]
    fn null_pid_window_refuses_a_pid_bound_transaction() {
        let fake = focus_fixture();
        fake.lock().unwrap().windows[1]["pid"] = Value::Null;
        let error = with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(50),
            || Ok(()),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "foreground_unavailable: niri window 7 belongs to pid unknown, not pid 107"
        );
    }

    #[test]
    fn open_overview_refuses_focus_bound_input() {
        let fake = focus_fixture();
        fake.lock().unwrap().overview = true;
        let error = with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(50),
            || Ok(()),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "foreground_unavailable: the niri overview is open, so input would go to the overview instead of the window"
        );
    }

    #[test]
    fn failed_restore_is_reported_after_the_body_ran() {
        let fake = focus_fixture();
        fake.lock().unwrap().ignore_focus_on = vec![5];
        let mut ran = false;
        let error = with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(50),
            || {
                ran = true;
                Ok(())
            },
        )
        .unwrap_err();
        assert!(ran);
        assert_eq!(error.to_string(), "niri could not restore prior window 5");
        assert_eq!(
            format!("{:#}", error.root_cause()),
            "foreground_unavailable: niri did not confirm focus on window 5 (focused: 7)"
        );
    }

    #[test]
    fn body_error_keeps_the_restore_error_as_context() {
        let fake = focus_fixture();
        fake.lock().unwrap().ignore_focus_on = vec![5];
        let error = with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(50),
            || Err::<(), _>(anyhow!("wtype failed")),
        )
        .unwrap_err();
        assert!(error.to_string().starts_with(
            "wtype failed; the prior niri focus also could not be restored: niri could not restore prior window 5"
        ));
        assert_eq!(error.root_cause().to_string(), "wtype failed");
    }

    #[test]
    fn focus_niri_did_not_confirm_is_still_undone() {
        let fake = focus_fixture();
        fake.lock().unwrap().focus_taken_on = vec![7];
        let error = with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(50),
            || Ok(()),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "foreground_unavailable: niri did not confirm focus on window 7 (focused: none)"
        );
        let fake = fake.lock().unwrap();
        assert!(fake.active(1) && !fake.active(2));
        assert_eq!(fake.focused(), Some(5));
    }

    #[test]
    fn a_window_focused_by_someone_else_during_input_keeps_focus() {
        let fake = focus_fixture();
        fake.lock()
            .unwrap()
            .windows
            .push(floating(9, 1, (500.0, 60.0), (100, 100)));
        with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(100),
            || {
                fake.lock()
                    .unwrap()
                    .reply(r#"{"Action":{"FocusWindow":{"id":9}}}"#);
                Ok(())
            },
        )
        .unwrap();
        let fake = fake.lock().unwrap();
        assert_eq!(fake.focused(), Some(9));
        assert!(!fake
            .requests
            .iter()
            .any(|r| r.contains(r#""FocusWindow":{"id":5}"#)));
    }

    #[test]
    fn an_empty_workspace_chosen_during_input_keeps_focus() {
        let fake = focus_fixture();
        fake.lock()
            .unwrap()
            .workspaces
            .push(workspace(3, "winit", false, None));
        with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(100),
            || {
                fake.lock()
                    .unwrap()
                    .reply(r#"{"Action":{"FocusWorkspace":{"reference":{"Id":3}}}}"#);
                Ok(())
            },
        )
        .unwrap();
        let fake = fake.lock().unwrap();
        assert!(fake.active(3) && !fake.active(1));
        assert_eq!(fake.focused(), None);
    }

    #[test]
    fn every_output_gets_its_active_workspace_back() {
        let mut prior = tiled(5, 1, (402.0, 448.0));
        prior["is_focused"] = json!(true);
        let fake = Arc::new(Mutex::new(Fake {
            windows: vec![prior, floating(7, 4, (100.0, 60.0), (300, 200))],
            workspaces: vec![
                workspace(1, "left", true, Some(5)),
                workspace(3, "right", true, None),
                workspace(4, "right", false, Some(7)),
            ],
            ..Fake::default()
        }));
        {
            let mut fake = fake.lock().unwrap();
            fake.workspaces[1]["is_focused"] = json!(false);
            fake.workspace_lag = 2;
        }
        let seen = with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(100),
            || {
                let fake = fake.lock().unwrap();
                Ok((fake.focused(), fake.active(4)))
            },
        )
        .unwrap();
        assert_eq!(seen, (Some(7), true));
        let fake = fake.lock().unwrap();
        assert!(fake.active(1) && fake.active(3) && !fake.active(4));
        assert_eq!(fake.focused(), Some(5));
    }

    #[test]
    fn no_prior_window_restores_the_prior_workspace() {
        let fake = focus_fixture();
        {
            let mut fake = fake.lock().unwrap();
            fake.windows[0]["is_focused"] = json!(false);
            fake.workspaces.push(workspace(3, "winit", false, None));
            for ws in &mut fake.workspaces {
                ws["is_focused"] = json!(ws["id"] == 3);
                ws["is_active"] = json!(ws["id"] == 3);
            }
        }
        with_focused_window_in(
            &mut fake_ipc(&fake),
            107,
            7,
            Duration::from_millis(100),
            || Ok(()),
        )
        .unwrap();
        let fake = fake.lock().unwrap();
        assert!(fake
            .requests
            .iter()
            .any(|r| r.contains(r#""FocusWorkspace":{"reference":{"Id":3}}"#)));
        assert!(fake
            .workspaces
            .iter()
            .any(|ws| ws["id"] == 3 && ws["is_focused"] == true));
    }

    #[test]
    fn an_unknown_window_refuses_instead_of_trusting_handled() {
        // niri answers FocusWindow on an unknown id with Handled.
        let fake = focus_fixture();
        let error =
            activate_in(&mut fake_ipc(&fake), 999, None, Duration::from_millis(50)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "foreground_unavailable: niri window 999 no longer exists"
        );
    }

    // ── Settling ────────────────────────────────────────────────────────────

    #[test]
    fn a_config_reload_counts_as_a_layout_change() {
        // A reload can move tiled columns (struts, gaps, centering) or
        // lengthen animations past the settle period, with no other event.
        let mut watch = LayoutWatch::default();
        for event in [
            json!({"ConfigLoaded": {"failed": false}}),
            json!({"ConfigLoaded": {"failed": true}}),
        ] {
            assert_eq!(watch.change(&event), Some(Change::Moved), "{event}");
        }
    }

    #[test]
    fn title_changes_do_not_count_as_layout_changes() {
        let mut watch = LayoutWatch::default();
        let window = floating(7, 1, (100.0, 60.0), (300, 200));
        let mut change = |event: Value| watch.change(&event);
        assert_eq!(
            change(json!({"WindowsChanged": {"windows": [window.clone()]}})),
            Some(Change::Moved)
        );
        let mut retitled = window.clone();
        retitled["title"] = json!("Float 7 clicked 1");
        assert_eq!(
            change(json!({"WindowOpenedOrChanged": {"window": retitled}})),
            None
        );
        let mut moved = window;
        moved["layout"]["tile_pos_in_workspace_view"] = json!([200.0, 100.0]);
        assert_eq!(
            change(json!({"WindowOpenedOrChanged": {"window": moved}})),
            Some(Change::Moved)
        );
        assert_eq!(
            change(
                json!({"WindowOpenedOrChanged": {"window": floating(9, 1, (0.0, 0.0), (10, 10))}})
            ),
            Some(Change::Moved)
        );
        assert_eq!(
            change(json!({"WindowFocusTimestampChanged": {"id": 7}})),
            None
        );
        assert_eq!(
            change(json!({"SomethingFromALaterNiri": {}})),
            Some(Change::Moved)
        );
    }

    #[test]
    fn only_the_drivers_own_focus_change_is_expected() {
        let mut watch = LayoutWatch::default();
        let mut counts =
            |event: Value, expected: Expected| watch.change(&event).unwrap().counts(expected);
        let none = Expected::default();
        assert!(counts(json!({"WindowFocusChanged": {"id": 7}}), none));
        let focusing = Expected {
            window: Some(7),
            workspace: Some(4),
        };
        assert!(!counts(json!({"WindowFocusChanged": {"id": 7}}), focusing));
        assert!(!counts(
            json!({"WorkspaceActiveWindowChanged": {"workspace_id": 1, "active_window_id": 7}}),
            focusing
        ));
        assert!(counts(json!({"WindowFocusChanged": {"id": 8}}), focusing));
        assert!(counts(
            json!({"WindowFocusChanged": {"id": null}}),
            focusing
        ));
        assert!(counts(
            json!({"WorkspaceActiveWindowChanged": {"workspace_id": 1, "active_window_id": 8}}),
            focusing
        ));
        // Focus moving to the target's (already active) workspace on another
        // output is expected; another workspace is not.
        assert!(!counts(
            json!({"WorkspaceActivated": {"id": 4, "focused": true}}),
            focusing
        ));
        assert!(counts(
            json!({"WorkspaceActivated": {"id": 3, "focused": true}}),
            focusing
        ));
        assert!(counts(
            json!({"WorkspaceActivated": {"id": 4, "focused": true}}),
            none
        ));
    }

    /// A fresh history; each test needs its own event stream.
    fn history() -> &'static History {
        Box::leak(Box::new(History::new()))
    }

    /// A niri event stream that sends `events`, each after its delay, then
    /// stays open for `open`.
    fn event_stream_open(events: Vec<(Duration, Value)>, open: Duration) -> UnixStream {
        let (client, mut server) = UnixStream::pair().unwrap();
        std::thread::spawn(move || {
            let mut request = [0u8; 64];
            let _ = server.read(&mut request);
            let _ = server.write_all(b"{\"Ok\":\"Handled\"}\n");
            for (delay, event) in events {
                std::thread::sleep(delay);
                if server.write_all(format!("{event}\n").as_bytes()).is_err() {
                    return;
                }
            }
            std::thread::sleep(open);
        });
        client
    }

    fn event_stream(events: Vec<(Duration, Value)>) -> UnixStream {
        event_stream_open(events, Duration::from_secs(5))
    }

    fn workspaces_changed() -> Value {
        json!({"WorkspacesChanged": {"workspaces": []}})
    }

    #[test]
    fn a_layout_that_stayed_quiet_settles_at_once_for_the_next_action() {
        let history = history();
        let quiet = Duration::from_millis(200);
        let started = Instant::now();
        let first = history
            .settle(
                || Ok(event_stream(vec![(Duration::ZERO, workspaces_changed())])),
                quiet,
                Duration::from_secs(2),
            )
            .unwrap();
        // Connecting counts as a change: an animation may be running.
        assert!(started.elapsed() >= quiet);
        let again = Instant::now();
        let second = history
            .settle(
                || bail!("the stream stays open between actions"),
                quiet,
                Duration::from_secs(2),
            )
            .unwrap();
        assert!(
            again.elapsed() < Duration::from_millis(100),
            "{:?}",
            again.elapsed()
        );
        assert_eq!(first, second);
    }

    #[test]
    fn a_change_is_dated_when_niri_sends_it_not_when_an_action_looks() {
        let history = history();
        let quiet = Duration::from_millis(200);
        let started = Instant::now();
        // The change comes 50 ms in, while the first wait sleeps: read then,
        // the layout is quiet 250 ms in; read when the wait wakes, 400 ms.
        history
            .settle(
                || {
                    Ok(event_stream(vec![(
                        Duration::from_millis(50),
                        json!({"WindowLayoutsChanged": {"changes": []}}),
                    )]))
                },
                quiet,
                Duration::from_secs(2),
            )
            .unwrap();
        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_millis(250), "{elapsed:?}");
        assert!(elapsed < Duration::from_millis(380), "{elapsed:?}");
    }

    #[test]
    fn a_recent_change_still_waits_out_the_quiet_period() {
        let history = history();
        let quiet = Duration::from_millis(200);
        history
            .settle(
                || {
                    Ok(event_stream(vec![(
                        Duration::from_millis(400),
                        json!({"WindowLayoutsChanged": {"changes": []}}),
                    )]))
                },
                quiet,
                Duration::from_secs(2),
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(500).saturating_sub(quiet));
        let again = Instant::now();
        history
            .settle(|| bail!("still connected"), quiet, Duration::from_secs(2))
            .unwrap();
        assert!(
            again.elapsed() >= Duration::from_millis(80),
            "{:?}",
            again.elapsed()
        );
    }

    #[test]
    fn a_layout_that_keeps_moving_refuses() {
        let events = (0..40)
            .map(|_| {
                (
                    Duration::from_millis(20),
                    json!({"WindowLayoutsChanged": {"changes": []}}),
                )
            })
            .collect();
        let error = history()
            .settle(
                || Ok(event_stream(events)),
                Duration::from_millis(100),
                Duration::from_millis(300),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "foreground_unavailable: the niri layout did not stay unchanged for 100 ms within 300 ms, so window geometry may not match the screen"
        );
    }

    #[test]
    fn a_change_after_settling_is_detected() {
        let history = history();
        let mark = history
            .settle(
                || {
                    Ok(event_stream(vec![(
                        Duration::from_millis(150),
                        json!({"WorkspaceActivated": {"id": 2, "focused": true}}),
                    )]))
                },
                Duration::from_millis(100),
                Duration::from_secs(1),
            )
            .unwrap();
        assert!(!history.changed_since(mark, Expected::default()).unwrap());
        std::thread::sleep(Duration::from_millis(150));
        assert!(history.changed_since(mark, Expected::default()).unwrap());
    }

    #[test]
    fn a_lost_or_reconnected_stream_cannot_vouch_for_an_earlier_check() {
        let history = history();
        let quiet = Duration::from_millis(50);
        let mark = history
            .settle(
                || Ok(event_stream_open(vec![], Duration::from_millis(100))),
                quiet,
                Duration::from_secs(1),
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(200));
        // niri closed the stream: whatever happened since is unknown.
        assert!(history.changed_since(mark, Expected::default()).is_err());
        let again = history
            .settle(|| Ok(event_stream(vec![])), quiet, Duration::from_secs(1))
            .unwrap();
        assert_ne!(again, mark);
        assert_eq!(
            history
                .changed_since(mark, Expected::default())
                .unwrap_err()
                .to_string(),
            "niri's event stream reconnected after the layout was checked"
        );
        assert!(!history.changed_since(again, Expected::default()).unwrap());
    }

    fn offline_ipc() -> AttestedIpc {
        Ipc {
            connect: |_| bail!("no niri in this test"),
            per_attempt: QUERY_TIMEOUT,
            total: QUERY_TOTAL_TIMEOUT,
            backoff: QUERY_RETRY_BACKOFF,
            lines: None,
        }
    }

    #[test]
    fn a_layout_change_after_positioning_refuses_the_press() {
        let history = history();
        let mark = history
            .settle(
                || {
                    Ok(event_stream(vec![(
                        Duration::from_millis(250),
                        json!({"WindowLayoutsChanged": {"changes": []}}),
                    )]))
                },
                Duration::from_millis(100),
                Duration::from_secs(1),
            )
            .unwrap();
        let mut gate = PointerGate {
            history,
            mark,
            expected: Expected::default(),
            ipc: offline_ipc(),
        };
        // Still quiet before the first motion; a window moves over the point
        // while the pointer is positioned, so the press must refuse.
        gate.confirm().unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            gate.confirm().unwrap_err().to_string(),
            "foreground_unavailable: the niri layout changed after the pointer target was checked; take a new screenshot"
        );
    }

    // ── Pointer admission ───────────────────────────────────────────────────

    #[test]
    fn floating_window_coordinates_add_the_window_origin() {
        let scene = one_output(vec![floating(7, 1, (100.0, 60.0), (300, 200))]);
        let (point, rect) = aim_at(&scene.window(7).unwrap(), 10, 20).unwrap();
        assert_eq!(point, (110, 80));
        assert_eq!(
            rect,
            Rect {
                x: 100,
                y: 60,
                width: 300,
                height: 200
            }
        );
    }

    fn placed_window(scene: &Scene, id: u64) -> Window {
        scene.window(id).unwrap()
    }

    #[test]
    fn pointer_admission_refuses_moved_windows_and_ambiguous_points() {
        let scene = one_output(vec![
            floating(7, 1, (100.0, 60.0), (300, 200)),
            floating(8, 1, (350.0, 200.0), (200, 100)),
        ]);
        let window = placed_window(&scene, 7);
        let rect = window.placement.as_ref().unwrap().rect;
        assert!(pointer_admission(&window, rect, &[(150, 100)], |_| None).is_ok());
        assert!(pointer_admission(&window, rect, &[(150, 100)], |_| Some(rect)).is_ok());
        let earlier = Rect { x: 0, ..rect };
        assert_eq!(
            pointer_admission(&window, rect, &[(150, 100)], |_| Some(earlier)).unwrap_err().to_string(),
            "foreground_unavailable: niri window 7 moved from Rect { x: 0, y: 60, width: 300, height: 200 } to Rect { x: 100, y: 60, width: 300, height: 200 } after its coordinates were computed; take a new screenshot"
        );
        assert_eq!(
            pointer_admission(&window, rect, &[(360, 210)], |_| None).unwrap_err().to_string(),
            "foreground_unavailable: point (360, 210) lies under another floating window, and niri reports no stacking order"
        );
        assert_eq!(
            pointer_admission(&window, rect, &[(150, 100), (5, 5)], |_| None).unwrap_err().to_string(),
            "foreground_unavailable: point (5, 5) is outside niri window 7 at Rect { x: 100, y: 60, width: 300, height: 200 }"
        );
        // Taller than the 853x480 output: its bottom is not shown.
        let scene = one_output(vec![floating(9, 1, (100.0, 60.0), (300, 500))]);
        let tall = placed_window(&scene, 9);
        let tall_rect = tall.placement.as_ref().unwrap().rect;
        assert_eq!(
            pointer_admission(&tall, tall_rect, &[(150, 500)], |_| None).unwrap_err().to_string(),
            "foreground_unavailable: point (150, 500) is past the edge of niri output winit, where window 9 is not shown"
        );
        let tiled = one_output(vec![tiled(5, 1, (402.0, 448.0))])
            .window(5)
            .unwrap();
        assert_eq!(
            pointer_admission(&tiled, rect, &[(150, 100)], |_| None)
                .unwrap_err()
                .to_string(),
            format!(
                "foreground_unavailable: niri window 5: {}",
                Unplaced::Tiled.reason()
            )
        );
    }

    // ── Tiled windows ───────────────────────────────────────────────────────

    /// Tiled window 5 (active) and 6 beside it, and floating window 7, on
    /// workspace 1 of output winit placed at x=1920; workspace 2 hidden.
    fn tiled_scene(focused: Option<u64>) -> Scene {
        let mut windows = vec![
            tiled(5, 1, (402.0, 448.0)),
            tiled(6, 1, (402.0, 448.0)),
            floating(7, 1, (100.0, 60.0), (300, 200)),
            tiled(8, 2, (402.0, 448.0)),
        ];
        for window in &mut windows {
            window["is_focused"] = json!(Some(window["id"].as_u64().unwrap()) == focused);
        }
        scene(
            windows,
            vec![
                workspace(1, "winit", true, Some(5)),
                workspace(2, "winit", false, Some(8)),
            ],
            vec![output("winit", 1920, 0, (1280, 720), 1.5, "Flipped180")],
            false,
        )
    }

    #[test]
    fn a_tiled_window_is_captured_as_its_output_while_niri_surely_shows_it() {
        let scene = tiled_scene(Some(5));
        let plan = capture_plan_in(&scene, 5, Some(105)).unwrap();
        assert_eq!(
            plan.pixels,
            Rect {
                x: 0,
                y: 0,
                width: 1280,
                height: 720
            }
        );
        assert_eq!((plan.width, plan.height), (853, 480));
        assert_eq!(
            plan.area,
            Rect {
                x: 1920,
                y: 0,
                width: 853,
                height: 480
            }
        );
        assert_eq!(plan.shown, scene.shown("winit"));
        assert_eq!(
            capture_plan_in(&scene, 6, None),
            Err("niri does not report where a tiled window is, and shows for certain only its workspace's active window; focus the window first")
        );
        assert_eq!(
            capture_plan_in(&scene, 8, None),
            Err(Unplaced::NotShowing.reason())
        );
        let mut overview = tiled_scene(Some(5));
        overview.overview_open = true;
        assert_eq!(
            capture_plan_in(&overview, 5, None),
            Err(Unplaced::NotShowing.reason())
        );
    }

    #[test]
    fn a_tiled_capture_yields_the_frame_of_its_settled_layout() {
        let mark = Mark {
            generation: 3,
            at: 9,
        };
        let capture = capture_window_in(
            || Ok(tiled_scene(Some(5))),
            || Ok(mark),
            unchanged,
            5,
            None,
            || display("winit", (1280, 720)),
        )
        .unwrap();
        let frame = capture.frame.unwrap();
        assert_eq!((frame.window, frame.mark), (5, mark));
        assert_eq!(frame.area, capture.plan.area);
        assert!(capture_window_in(
            || Ok(one_output(vec![floating(7, 1, (100.0, 60.0), (300, 200))])),
            settled_now,
            unchanged,
            7,
            None,
            || display("winit", (1280, 720)),
        )
        .unwrap()
        .frame
        .is_none());
    }

    /// The frame window 5's capture gives, checked against `history`.
    fn tiled_frame(history: &'static History) -> OutputFrame {
        let mark = history
            .settle(
                || Ok(event_stream(vec![])),
                Duration::from_millis(10),
                Duration::from_secs(1),
            )
            .unwrap();
        capture_window_in(
            || Ok(tiled_scene(Some(5))),
            || Ok(mark),
            |mark| history.changed_since(mark, Expected::default()),
            5,
            None,
            || display("winit", (1280, 720)),
        )
        .unwrap()
        .frame
        .unwrap()
    }

    #[test]
    fn tiled_window_pixels_aim_at_the_output_while_its_layout_holds() {
        let history = history();
        let frame = tiled_frame(history);
        let scene = tiled_scene(Some(5));
        let window = scene.window(5).unwrap();
        frame.holds(&scene, history).unwrap();
        assert_eq!(frame.aim(10, 20).unwrap(), (1930, 20));
        assert_eq!(frame.aim(852, 479).unwrap(), (2772, 479));
        assert_eq!(
            frame.aim(853, 20).unwrap_err().to_string(),
            "foreground_unavailable: point (853, 20) is outside the screenshot of tiled niri window 5, which is its whole output, 853x480"
        );
        assert!(frame.aim(-1, 20).is_err());
        frame
            .admit(&scene, &window, &[(2500, 400)], true, history)
            .unwrap();
        // Floating window 7 covers layout x 2020..2320, y 60..260.
        assert_eq!(
            frame
                .admit(&scene, &window, &[(2100, 100)], true, history)
                .unwrap_err()
                .to_string(),
            "foreground_unavailable: point (2100, 100) lies on floating niri window 7, which niri draws above tiled window 5"
        );
        assert!(frame
            .admit(&scene, &window, &[(10, 100)], true, history)
            .is_err());
    }

    #[test]
    fn tiled_window_pixels_refuse_once_the_layout_changed() {
        let history = history();
        let frame = tiled_frame(history);
        let changed = "foreground_unavailable: niri output winit changed after the screenshot of tiled niri window 5, so its pixels may be elsewhere now; take a new screenshot";
        // Focus moved to the tiled window beside it: niri may have scrolled.
        let mut focus = tiled_scene(Some(6));
        focus.workspaces[0].active_window_id = Some(6);
        assert_eq!(
            frame.holds(&focus, history).unwrap_err().to_string(),
            changed
        );
        // A column resized or opened shifts the columns after it.
        let mut resized = tiled_scene(Some(5));
        resized.windows[1].layout.as_mut().unwrap().tile_size = (500.0, 448.0);
        assert_eq!(
            frame.holds(&resized, history).unwrap_err().to_string(),
            changed
        );
        let mut overview = tiled_scene(Some(5));
        overview.overview_open = true;
        assert!(frame.holds(&overview, history).is_err());
        // A window on the hidden workspace changes nothing shown.
        let mut hidden = tiled_scene(Some(5));
        hidden.windows[3].layout.as_mut().unwrap().tile_size = (500.0, 448.0);
        frame.holds(&hidden, history).unwrap();
    }

    #[test]
    fn a_change_niri_reported_after_the_capture_refuses_tiled_pixels() {
        let captured = tiled_frame(history());
        let history = history();
        let mark = history
            .settle(
                || {
                    Ok(event_stream(vec![(
                        Duration::from_millis(100),
                        json!({"WindowFocusChanged": {"id": 6}}),
                    )]))
                },
                Duration::from_millis(50),
                Duration::from_secs(1),
            )
            .unwrap();
        let frame = OutputFrame { mark, ..captured };
        let scene = tiled_scene(Some(5));
        frame.holds(&scene, history).unwrap();
        // Focus went elsewhere and back: the scene matches again, but niri
        // may have scrolled its view in between.
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            frame.holds(&scene, history).unwrap_err().to_string(),
            "foreground_unavailable: the niri layout changed after the screenshot of tiled niri window 5, so its pixels may be elsewhere now; take a new screenshot"
        );
    }

    #[test]
    fn an_unfocused_tiled_window_is_not_focused_for_a_pointer_action() {
        let history = history();
        // Focus is on a layer-shell surface: no window has it.
        let unfocused = tiled_scene(None);
        let frame = OutputFrame {
            shown: unfocused.shown("winit").unwrap(),
            ..tiled_frame(history)
        };
        let window = unfocused.window(5).unwrap();
        assert_eq!(
            frame
                .admit(&unfocused, &window, &[(2500, 400)], true, history)
                .unwrap_err()
                .to_string(),
            "foreground_unavailable: tiled niri window 5 is not focused, and niri may scroll its view to focus it; focus it, then take a new screenshot"
        );
        // Desktop-scope input focuses nothing.
        frame
            .admit(&unfocused, &window, &[(2500, 400)], false, history)
            .unwrap();
    }

    #[test]
    fn a_tiled_frame_expires_because_niri_does_not_report_every_view_scroll() {
        let history = history();
        let scene = tiled_scene(Some(5));
        let frame = tiled_frame(history);
        let aged = |age: Duration| OutputFrame {
            taken: Instant::now().checked_sub(age).unwrap(),
            ..frame.clone()
        };
        aged(Duration::from_secs(59))
            .holds(&scene, history)
            .unwrap();
        assert_eq!(
            aged(Duration::from_secs(61))
                .holds(&scene, history)
                .unwrap_err()
                .to_string(),
            "foreground_unavailable: the screenshot of tiled niri window 5 is over 60 s old, and niri does not report every scroll of its view; take a new screenshot"
        );
    }

    #[test]
    fn an_unfocused_tiled_window_refuses_before_any_route_focuses_it() {
        let history = history();
        // Captured and still shown while a layer-shell surface has focus.
        let unfocused = tiled_scene(None);
        let frame = OutputFrame {
            shown: unfocused.shown("winit").unwrap(),
            ..tiled_frame(history)
        };
        let window = unfocused.window(5).unwrap();
        assert_eq!(
            frame
                .aim_window(&unfocused, &window, history, 10, 20)
                .unwrap_err()
                .to_string(),
            "foreground_unavailable: tiled niri window 5 is not focused, and niri may scroll its view to focus it; focus it, then take a new screenshot"
        );
        let focused = tiled_scene(Some(5));
        let frame = tiled_frame(history);
        assert_eq!(
            frame
                .aim_window(&focused, &focused.window(5).unwrap(), history, 10, 20)
                .unwrap(),
            (1930, 20)
        );
    }

    #[test]
    fn only_the_latest_screenshot_of_a_tiled_window_aims() {
        // Window ids no other test adopts: the frames are process-wide.
        let frame = OutputFrame {
            window: 9001,
            ..tiled_frame(history())
        };
        assert!(adopted_frame(9001)
            .unwrap_err()
            .to_string()
            .ends_with("which shows its whole output; take one first"));
        adopt_frame(9001, Some(frame.clone()), Some("c1"));
        adopt_frame(9001, Some(frame.clone()), Some("c2"));
        assert_eq!(adopted_frame(9001).unwrap(), frame);
        frame_admits_capture(9001, "c2").unwrap();
        assert!(frame_admits_capture(9001, "c1")
            .unwrap_err()
            .to_string()
            .starts_with("foreground_unavailable: capture c1 is not the latest screenshot"));
        // Its latest screenshot shows only the window (it was floated).
        adopt_frame(9001, None, Some("c3"));
        assert!(adopted_frame(9001).is_err());
        frame_admits_capture(9001, "c1").unwrap();
    }

    #[test]
    fn health_note_states_attestation_and_the_single_output_limit() {
        assert_eq!(
            describe_health(HealthSession::Attested(Ok("26.04 (8ed0da4)".into())), Ok(1)),
            "niri 26.04 (8ed0da4): IPC attested to WAYLAND_DISPLAY."
        );
        assert!(
            describe_health(HealthSession::Attested(Ok("26.04".into())), Ok(2))
                .contains("refuse because 2 outputs are enabled")
        );
        assert_eq!(
            describe_health(
                HealthSession::Foreign(
                    "NIRI_SOCKET is served by pid 1 but WAYLAND_DISPLAY by pid 2".into()
                ),
                Ok(1)
            ),
            "niri: NIRI_SOCKET is not this session's compositor (NIRI_SOCKET is served by pid 1 \
             but WAYLAND_DISPLAY by pid 2); niri window ids, focus and geometry are off."
        );
        assert!(describe_health(HealthSession::Inactive, Ok(1)).starts_with(
            "niri: NIRI_SOCKET is this session's niri, but this process does not use"
        ));
    }
}
