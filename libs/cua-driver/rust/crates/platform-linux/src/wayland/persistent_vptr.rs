//! Persistent virtual-pointer for stateful `mouse_button_down` / `mouse_drag` /
//! `mouse_button_up`.
//!
//! The non-persistent path in [`crate::wayland::click`] opens its own
//! `ZwlrVirtualPointerV1`, presses, releases, drops the connection — useful
//! for one-shot clicks but useless for held-button drags: each tool call
//! emits a fresh device whose press/release pair is matched by the
//! compositor, so apps that distinguish a real drag (press, motion+, release)
//! from a series of clicks (press, release, press, release, …) miss the
//! drag entirely.
//!
//! This module keeps the virtual-pointer alive across tool calls. A single
//! owner thread per process owns one Wayland `Connection`, one `EventQueue`,
//! and a map of `cursor_id -> ActivePointer`. Commands are sent over a
//! `crossbeam-channel`; replies come back on a per-call reply channel so the
//! caller blocks until the compositor has roundtripped.
//!
//! Lifecycle:
//! - First `press` for a cursor_id binds a fresh `ZwlrVirtualPointerV1`,
//!   activates the foreign-toplevel target window once, presses the button,
//!   adds to the held-button set, roundtrips.
//! - Subsequent `move_to` calls emit `motion_absolute` on the same vptr (no
//!   activate — would steal focus mid-drag) and roundtrip.
//! - `release` emits a button release, removes from the held set; if the
//!   set is empty the vptr is destroyed and the map entry dropped.
//! - On `Connection` roundtrip failure (compositor restart / disconnect)
//!   the owner thread tears down its connection and accepts the next
//!   command on a fresh one, emitting a typed error for the in-flight call.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::thread;

use crossbeam_channel::{bounded, Receiver, Sender};
use wayland_client::{protocol::wl_pointer::ButtonState, Connection};
use wayland_protocols_wlr::virtual_pointer::v1::client::zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1;

use super::{evdev_pointer_button, open_vptr_session};

/// One in-flight command from the public API to the owner thread.
enum Cmd {
    Press {
        cursor_id: String,
        window_id: u64,
        x: i32,
        y: i32,
        button: u8,
        reply: Sender<anyhow::Result<()>>,
    },
    MoveTo {
        cursor_id: String,
        x: i32,
        y: i32,
        reply: Sender<anyhow::Result<()>>,
    },
    Release {
        cursor_id: String,
        button: u8,
        reply: Sender<anyhow::Result<()>>,
    },
    /// Drop the entry for a cursor_id without sending wire events — used to
    /// recover when the compositor disconnected mid-life.
    Forget {
        cursor_id: String,
        reply: Sender<anyhow::Result<()>>,
    },
}

/// The device a held cursor drives: the virtual pointer, or a recording
/// fake in tests. Each call sends one event and its frame.
trait HeldDevice {
    fn motion(&self, x: u32, y: u32, extent: (u32, u32));
    fn button(&self, button: u32, state: ButtonState);
    fn destroy(&self);
}

impl HeldDevice for ZwlrVirtualPointerV1 {
    fn motion(&self, x: u32, y: u32, (w, h): (u32, u32)) {
        self.motion_absolute(0, x, y, w, h);
        self.frame();
    }

    fn button(&self, button: u32, state: ButtonState) {
        ZwlrVirtualPointerV1::button(self, 0, button, state);
        self.frame();
    }

    fn destroy(&self) {
        ZwlrVirtualPointerV1::destroy(self);
    }
}

/// A held cursor's motion refused because the pointer space it scales by
/// changed, or could not be read. The owner sent the release of its buttons
/// where the pointer was, so the app may have seen a drop, and discarded the
/// cursor: it holds nothing any more, whether or not niri confirmed the
/// release.
#[derive(Debug)]
pub struct HeldPointerEnded {
    detail: String,
    /// niri answered the round trip after the release.
    pub release_confirmed: bool,
}

impl std::fmt::Display for HeldPointerEnded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for HeldPointerEnded {}

/// State held inside the owner thread for one cursor_id.
struct ActivePointer<D = ZwlrVirtualPointerV1> {
    vptr: D,
    /// evdev codes of buttons currently held down. When this set becomes
    /// empty the vptr is destroyed and the entry dropped from the map.
    held: HashSet<u32>,
    /// Pointer-space extent and layout origin at session open time — needed
    /// for motion_absolute.
    out_w: u32,
    out_h: u32,
    origin_x: i32,
    origin_y: i32,
}

/// Process-global command channel into the owner thread. Lazily started on
/// first use.
static TX: OnceLock<Sender<Cmd>> = OnceLock::new();

fn tx() -> &'static Sender<Cmd> {
    TX.get_or_init(|| {
        let (tx, rx) = bounded::<Cmd>(32);
        thread::Builder::new()
            .name("cua-persistent-vptr".into())
            .spawn(move || owner_thread(rx))
            .expect("spawn cua-persistent-vptr thread");
        tx
    })
}

fn owner_thread(rx: Receiver<Cmd>) {
    let mut active: HashMap<String, ActivePointer> = HashMap::new();
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Press {
                cursor_id,
                window_id,
                x,
                y,
                button,
                reply,
            } => {
                let r = handle_press(&mut active, &cursor_id, window_id, x, y, button);
                let _ = reply.send(r);
            }
            Cmd::MoveTo {
                cursor_id,
                x,
                y,
                reply,
            } => {
                let r = handle_move(&mut active, &cursor_id, x, y);
                let _ = reply.send(r);
            }
            Cmd::Release {
                cursor_id,
                button,
                reply,
            } => {
                let r = handle_release(&mut active, &cursor_id, button);
                let _ = reply.send(r);
            }
            Cmd::Forget { cursor_id, reply } => {
                if let Some(p) = active.remove(&cursor_id) {
                    p.vptr.destroy();
                }
                let _ = reply.send(Ok(()));
            }
        }
    }
}

fn handle_press(
    active: &mut HashMap<String, ActivePointer>,
    cursor_id: &str,
    window_id: u64,
    x: i32,
    y: i32,
    button: u8,
) -> anyhow::Result<()> {
    // Open a fresh session for this press — this binds the seat, the foreign-
    // toplevel manager, activates the target window, and creates a new vptr.
    // Keep the (out_w, out_h) but drop the queue + state at end of scope; the
    // vptr itself remains alive (Wayland objects survive their original queue
    // as long as the Connection is alive).
    let mut sess = open_vptr_session(Some(window_id))?;
    let (w, h) = (sess.output_w, sess.output_h);
    let (px, py) = sess.abs(x, y);
    let mut gate = super::niri_pointer_gate(Some(window_id), None, &sess, &[(x, y)])?;
    let btn = evdev_pointer_button(button);
    super::niri_gate_holds(&mut gate, &sess)?;
    sess.motion(px, py)?;
    // A press whose round trip fails is released when `sess` drops.
    sess.press(btn)?;

    // The vptr proxy stays valid while its connection lives: keep both for
    // the cursor's later commands.
    let (origin_x, origin_y) = (sess.origin_x, sess.origin_y);
    let niri = sess.deadline.is_some();
    let (conn, vptr) = sess.keep();
    persist_conn(cursor_id, conn, niri);

    let mut held = HashSet::new();
    held.insert(btn);
    active.insert(
        cursor_id.to_string(),
        ActivePointer {
            vptr,
            held,
            out_w: w,
            out_h: h,
            origin_x,
            origin_y,
        },
    );
    Ok(())
}

fn handle_move(
    active: &mut HashMap<String, ActivePointer>,
    cursor_id: &str,
    x: i32,
    y: i32,
) -> anyhow::Result<()> {
    move_held(
        active,
        cursor_id,
        (x, y),
        super::niri_pointer_space_now,
        roundtrip_on_persistent,
    )
}

/// Move a held cursor. On niri the pointer space is read again first: the
/// motion is scaled by the extent saved at the press, and niri sends no
/// event when an output is added or rescaled. If it changed, no motion goes
/// out; the held buttons are released and the cursor is dropped.
fn move_held<D: HeldDevice>(
    active: &mut HashMap<String, ActivePointer<D>>,
    cursor_id: &str,
    (x, y): (i32, i32),
    space_now: impl FnOnce() -> anyhow::Result<Option<(i32, i32, u32, u32)>>,
    mut roundtrip: impl FnMut(&str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let entry = active.get_mut(cursor_id).ok_or_else(|| {
        anyhow::anyhow!(
            "no held mouse button for cursor '{cursor_id}'; call mouse_button_down first"
        )
    })?;
    let saved = (entry.origin_x, entry.origin_y, entry.out_w, entry.out_h);
    if let Some(refusal) = space_now()
        .and_then(|now| now.map_or(Ok(()), |now| super::pointer_space_holds(saved, now)))
        .err()
    {
        let entry = active.remove(cursor_id).expect("found above");
        for button in &entry.held {
            entry.vptr.button(*button, ButtonState::Released);
        }
        entry.vptr.destroy();
        let released = roundtrip(cursor_id);
        forget_conn(cursor_id);
        let refusal = refusal
            .to_string()
            .trim_start_matches("foreground_unavailable: ")
            .to_owned();
        return Err(match released {
            Ok(()) => HeldPointerEnded {
                detail: format!(
                    "{refusal}; no motion was sent, and the held button was released where the pointer was, which the app may take as a drop"
                ),
                release_confirmed: true,
            },
            Err(error) => HeldPointerEnded {
                detail: format!(
                    "{refusal}; no motion was sent, and the held button's release was sent but not confirmed ({error:#}); the cursor no longer holds it"
                ),
                release_confirmed: false,
            },
        }
        .into());
    }
    let (px, py) = super::pointer_abs(
        entry.origin_x,
        entry.origin_y,
        entry.out_w,
        entry.out_h,
        x,
        y,
    );
    entry.vptr.motion(px, py, (entry.out_w, entry.out_h));
    roundtrip(cursor_id)
}

fn handle_release(
    active: &mut HashMap<String, ActivePointer>,
    cursor_id: &str,
    button: u8,
) -> anyhow::Result<()> {
    let btn = evdev_pointer_button(button);
    let drop_entry = {
        let entry = active
            .get_mut(cursor_id)
            .ok_or_else(|| anyhow::anyhow!("no held mouse button for cursor '{cursor_id}'"))?;
        entry.vptr.button(0, btn, ButtonState::Released);
        entry.vptr.frame();
        roundtrip_on_persistent(cursor_id)?;
        entry.held.remove(&btn);
        entry.held.is_empty()
    };
    if drop_entry {
        if let Some(p) = active.remove(cursor_id) {
            p.vptr.destroy();
            roundtrip_on_persistent(cursor_id).ok();
        }
        forget_conn(cursor_id);
    }
    Ok(())
}

// Process-static slots for Connection + EventQueue keyed by cursor_id. The
// EventQueue is !Send but we only touch these on the owner thread, so wrap
// in a thread-local-by-construction pattern: store inside the same map so
// the owner thread is the sole accessor.
//
// We use a per-thread static rather than a Mutex<HashMap> because the owner
// thread is the only accessor (no contention possible).
/// A held cursor's connection, its queue, and whether it is niri's, whose
/// round trips are bounded.
type Persisted = (Connection, wayland_client::EventQueue<super::State>, bool);

thread_local! {
    static CONNS: std::cell::RefCell<HashMap<String, Persisted>>
        = std::cell::RefCell::new(HashMap::new());
}

fn persist_conn(cursor_id: &str, conn: Connection, niri: bool) {
    let queue = conn.new_event_queue::<super::State>();
    CONNS.with(|c| {
        c.borrow_mut()
            .insert(cursor_id.to_string(), (conn, queue, niri));
    });
}

fn forget_conn(cursor_id: &str) {
    CONNS.with(|c| {
        c.borrow_mut().remove(cursor_id);
    });
}

fn roundtrip_on_persistent(cursor_id: &str) -> anyhow::Result<()> {
    CONNS.with(|c| {
        let mut b = c.borrow_mut();
        let (conn, queue, niri) = b
            .get_mut(cursor_id)
            .ok_or_else(|| anyhow::anyhow!("no persistent connection for cursor '{cursor_id}'"))?;
        let mut tmp = super::State::default();
        let deadline = niri.then(|| std::time::Instant::now() + super::NIRI_ROUNDTRIP_TIMEOUT);
        super::sync_by(conn, queue, &mut tmp, deadline)
            .map_err(|e| anyhow::anyhow!("compositor roundtrip failed: {e:#}"))
    })
}

// ── public API ────────────────────────────────────────────────────────────

/// Press and HOLD `button` (evdev code) at output coordinates `(x, y)` on the
/// toplevel identified by `window_id`. Subsequent `move_to` / `release` calls
/// targeting the same `cursor_id` reuse the same virtual-pointer device, so
/// the compositor treats the sequence as one logical drag rather than as
/// independent clicks. Errors if `cursor_id` already has a held button.
pub fn press(cursor_id: &str, window_id: u64, x: i32, y: i32, button: u8) -> anyhow::Result<()> {
    let (tx_r, rx_r) = bounded(1);
    tx().send(Cmd::Press {
        cursor_id: cursor_id.to_string(),
        window_id,
        x,
        y,
        button,
        reply: tx_r,
    })
    .map_err(|e| anyhow::anyhow!("cua-persistent-vptr thread is dead: {e}"))?;
    rx_r.recv()
        .map_err(|e| anyhow::anyhow!("reply channel closed: {e}"))?
}

/// Emit motion_absolute on the held cursor's virtual-pointer. Errors if there
/// is no held button for `cursor_id`.
pub fn move_to(cursor_id: &str, x: i32, y: i32) -> anyhow::Result<()> {
    let (tx_r, rx_r) = bounded(1);
    tx().send(Cmd::MoveTo {
        cursor_id: cursor_id.to_string(),
        x,
        y,
        reply: tx_r,
    })
    .map_err(|e| anyhow::anyhow!("cua-persistent-vptr thread is dead: {e}"))?;
    rx_r.recv()
        .map_err(|e| anyhow::anyhow!("reply channel closed: {e}"))?
}

/// Release `button` on the held cursor. If no other buttons remain held the
/// virtual-pointer is destroyed and its Wayland connection torn down.
pub fn release(cursor_id: &str, button: u8) -> anyhow::Result<()> {
    let (tx_r, rx_r) = bounded(1);
    tx().send(Cmd::Release {
        cursor_id: cursor_id.to_string(),
        button,
        reply: tx_r,
    })
    .map_err(|e| anyhow::anyhow!("cua-persistent-vptr thread is dead: {e}"))?;
    rx_r.recv()
        .map_err(|e| anyhow::anyhow!("reply channel closed: {e}"))?
}

/// Drop the entry for `cursor_id` without emitting any Wayland events.
/// Useful for recovery — if the agent thinks a button is held but the
/// compositor disagrees, this clears the local state without trying to
/// send a release that would error.
pub fn forget(cursor_id: &str) -> anyhow::Result<()> {
    let (tx_r, rx_r) = bounded(1);
    tx().send(Cmd::Forget {
        cursor_id: cursor_id.to_string(),
        reply: tx_r,
    })
    .map_err(|e| anyhow::anyhow!("cua-persistent-vptr thread is dead: {e}"))?;
    rx_r.recv()
        .map_err(|e| anyhow::anyhow!("reply channel closed: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Debug, PartialEq)]
    enum Event {
        Motion(u32, u32, (u32, u32)),
        Release(u32),
        Destroy,
    }

    #[derive(Default)]
    struct Recorder(RefCell<Vec<Event>>);

    impl HeldDevice for &Recorder {
        fn motion(&self, x: u32, y: u32, extent: (u32, u32)) {
            self.0.borrow_mut().push(Event::Motion(x, y, extent));
        }

        fn button(&self, button: u32, state: ButtonState) {
            assert_eq!(state, ButtonState::Released);
            self.0.borrow_mut().push(Event::Release(button));
        }

        fn destroy(&self) {
            self.0.borrow_mut().push(Event::Destroy);
        }
    }

    fn held(device: &Recorder) -> HashMap<String, ActivePointer<&Recorder>> {
        HashMap::from([(
            "drag".to_owned(),
            ActivePointer {
                vptr: device,
                held: HashSet::from([0x110]),
                out_w: 1280,
                out_h: 720,
                origin_x: 0,
                origin_y: 0,
            },
        )])
    }

    #[test]
    fn a_held_motion_after_the_pointer_space_changed_releases_instead_of_moving() {
        let device = Recorder::default();
        let mut active = held(&device);
        // Unchanged: the motion goes out in the saved extent.
        move_held(
            &mut active,
            "drag",
            (200, 200),
            || Ok(Some((0, 0, 1280, 720))),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(*device.0.borrow(), [Event::Motion(200, 200, (1280, 720))]);

        // The output went from scale 1 to 1.5 between two motions: scaling
        // by the old extent would land far from the aimed point.
        let stopped = move_held(
            &mut active,
            "drag",
            (267, 267),
            || Ok(Some((0, 0, 854, 480))),
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(stopped
            .downcast_ref::<HeldPointerEnded>()
            .is_some_and(|ended| ended.release_confirmed));
        assert_eq!(
            stopped.to_string(),
            "the niri pointer space changed from (0, 0, 1280, 720) to (0, 0, 854, 480); take a new screenshot; no motion was sent, and the held button was released where the pointer was, which the app may take as a drop"
        );
        assert_eq!(
            device.0.borrow()[1..],
            [Event::Release(0x110), Event::Destroy]
        );
        assert!(active.is_empty());
    }

    #[test]
    fn a_held_motion_whose_cleanup_niri_never_answers_still_ends_the_hold() {
        // niri stopped answering: the pointer space cannot be read and the
        // release is not confirmed. The owner discards the cursor anyway, so
        // the tool must learn that the hold ended, not see a plain failure.
        let device = Recorder::default();
        let mut active = held(&device);
        let stopped = move_held(
            &mut active,
            "drag",
            (200, 200),
            || anyhow::bail!("niri did not answer in time"),
            |_| anyhow::bail!("niri did not answer a Wayland round trip in time"),
        )
        .unwrap_err();
        let ended = stopped
            .downcast_ref::<HeldPointerEnded>()
            .expect("the hold ended");
        assert!(!ended.release_confirmed);
        assert_eq!(
            stopped.to_string(),
            "niri did not answer in time; no motion was sent, and the held button's release was sent but not confirmed (niri did not answer a Wayland round trip in time); the cursor no longer holds it"
        );
        assert_eq!(*device.0.borrow(), [Event::Release(0x110), Event::Destroy]);
        assert!(active.is_empty());
    }
}
