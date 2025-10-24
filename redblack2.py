#!/usr/bin/env python3
"""
Red–Black Tree Visual Demo (Top-Down vs Bottom-Up)
--------------------------------------------------
Visual, step-by-step red–black tree insertions with:
- Bottom-up (CLRS) and Top-down (LLRB) modes
- NIL leaves (black squares) to show black-height explicitly
- Role halos (bottom-up): parent=amber, uncle=blue, grandparent=green
- Optional post-op (attached) frames for stable “after rotation/flip”
- Live-editable sequence; step/back/play/stop; keyboard shortcuts
- Invariant checker; emits a frame if a property is violated

Shortcuts: ←/→ (Back/Step), Space (Play/Pause), R (Reset)

Run:
  python red_black_tree_demo.py --mode topdown --sequence "7 3 18 10 22 8 11 26" --speed 900
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Set

import tkinter as tk
from tkinter import ttk, messagebox

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    raise SystemExit("Matplotlib with TkAgg backend is required. Install matplotlib and try again.")

# ============================================================
# Core helpers
# ============================================================

def inorder_positions(root) -> Tuple[dict, dict]:
    """Return node->x index and node->depth using in-order traversal."""
    pos: Dict[Any, int] = {}
    depth: Dict[Any, int] = {}
    i = 0
    def dfs(n, d):
        nonlocal i
        if not n:
            return
        dfs(getattr(n, 'left', None), d+1)
        pos[n] = i
        depth[n] = d
        i += 1
        dfs(getattr(n, 'right', None), d+1)
    dfs(root, 0)
    return pos, depth

def clone_bu(node: Optional['BUNode']) -> Optional['BUNode']:
    if node is None:
        return None
    c = BUNode(node.key, node.red)
    c.left = clone_bu(node.left)
    if c.left: c.left.parent = c
    c.right = clone_bu(node.right)
    if c.right: c.right.parent = c
    return c

def clone_ll(node: Optional['LLNode']) -> Optional['LLNode']:
    if node is None:
        return None
    c = LLNode(node.key, node.red)
    c.left = clone_ll(node.left)
    c.right = clone_ll(node.right)
    return c

# ============================================================
# Bottom-up (CLRS)
# ============================================================

@dataclass(eq=False)
class BUNode:
    key: int
    red: bool = True
    left: Optional['BUNode'] = None
    right: Optional['BUNode'] = None
    parent: Optional['BUNode'] = None

class BottomUpRBTree:
    def __init__(self, frame_cb=None, postop: bool = True):
        self.root: Optional[BUNode] = None
        self.frame_cb = frame_cb
        self.postop = postop

    def _emit(self, msg: str, highlight_nodes: List[BUNode] = None, roles: Dict[int, str] = None):
        if self.frame_cb:
            keys = {n.key for n in (highlight_nodes or []) if n is not None}
            self.frame_cb(msg, clone_bu(self.root), keys, ghost_key=None, ghost_path=None, role_map=roles or {})

    # Rotations
    def _left_rotate(self, x: BUNode):
        y = x.right; assert y is not None
        self._emit(f"rotate-left at {x.key}", [x, y], roles={x.key:'parent', y.key:'grandparent'})
        x.right = y.left
        if y.left: y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        if self.postop:
            self._emit(f"after rotate-left at {x.key} (attached)")

    def _right_rotate(self, y: BUNode):
        x = y.left; assert x is not None
        self._emit(f"rotate-right at {y.key}", [y, x], roles={y.key:'parent', x.key:'grandparent'})
        y.left = x.right
        if x.right: x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x
        if self.postop:
            self._emit(f"after rotate-right at {y.key} (attached)")

    def insert(self, key: int):
        # Plain BST attach first (emphasize bottom-up)
        z = BUNode(key)
        y = None
        x = self.root
        while x is not None:
            y = x
            x = x.left if key < x.key else x.right
        z.parent = y
        if y is None:
            self.root = z
        elif key < y.key:
            y.left = z
        else:
            y.right = z
        self._emit(f"insert {key}", [z])
        self._insert_fixup(z)

        ok, msg, keys = verify_rb(self.root)
        if not ok and self.frame_cb:
            self.frame_cb(f"INVARIANT VIOLATION: {msg}", clone_bu(self.root), keys, ghost_key=None, ghost_path=[], role_map={})

    def _insert_fixup(self, z: BUNode):
        while z.parent is not None and z.parent.red:
            gp = z.parent.parent
            if z.parent == gp.left:
                y = gp.right
                if y is not None and y.red:
                    self._emit(
                        f"recolor parent {z.parent.key}, uncle {y.key}, grandparent {gp.key}",
                        [z.parent, y, gp],
                        roles={z.parent.key:'parent', y.key:'uncle', gp.key:'grandparent'}
                    )
                    z.parent.red = False; y.red = False; gp.red = True
                    if self.postop:
                        self._emit(f"after recolor at grandparent {gp.key}")
                    z = gp
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._emit("prepare inner case (left rotation)", [z], roles={z.key:'parent'})
                        self._left_rotate(z)
                    self._emit(
                        f"recolor & rotate-right at grandparent {gp.key}",
                        [gp, gp.left] if gp.left else [gp],
                        roles={gp.key:'grandparent', (gp.left.key if gp.left else gp.key):'parent'}
                    )
                    z.parent.red = False
                    gp.red = True
                    self._right_rotate(gp)
            else:
                y = gp.left
                if y is not None and y.red:
                    self._emit(
                        f"recolor parent {z.parent.key}, uncle {y.key}, grandparent {gp.key}",
                        [z.parent, y, gp],
                        roles={z.parent.key:'parent', y.key:'uncle', gp.key:'grandparent'}
                    )
                    z.parent.red = False; y.red = False; gp.red = True
                    if self.postop:
                        self._emit(f"after recolor at grandparent {gp.key}")
                    z = gp
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._emit("prepare inner case (right rotation)", [z], roles={z.key:'parent'})
                        self._right_rotate(z)
                    self._emit(
                        f"recolor & rotate-left at grandparent {gp.key}",
                        [gp, gp.right] if gp.right else [gp],
                        roles={gp.key:'grandparent', (gp.right.key if gp.right else gp.key):'parent'}
                    )
                    z.parent.red = False
                    gp.red = True
                    self._left_rotate(gp)
        if self.root and self.root.red:
            self._emit("recolor root black", [self.root], roles={self.root.key:'grandparent'})
            self.root.red = False
            if self.postop:
                self._emit("after recolor root black")

# ============================================================
# RB invariant checker (Properties 1–5)
# ============================================================

def verify_rb(root) -> Tuple[bool, str, Set[int]]:
    """Return (ok, message, highlight_keys)."""
    if root is None:
        return True, "ok", set()

    # 2) Root is black
    if getattr(root, 'red', False):
        return False, "Root must be black", {getattr(root, 'key', -1)}

    bad = set()
    def check(node) -> Tuple[int, bool]:
        # returns (black_height, ok)
        if node is None:
            return 1, True  # NIL leaf counts as one black

        # 4) red node cannot have red child
        if getattr(node, 'red', False):
            for ch in (getattr(node, 'left', None), getattr(node, 'right', None)):
                if ch is not None and getattr(ch, 'red', False):
                    bad.update({getattr(node, 'key', -1), getattr(ch, 'key', -1)})
                    return 0, False

        bl, ok = check(getattr(node, 'left', None))
        if not ok: return 0, False
        br, ok = check(getattr(node, 'right', None))
        if not ok: return 0, False

        # 5) equal black height for all paths
        if bl != br:
            bad.add(getattr(node, 'key', -1))
            return 0, False

        return bl + (0 if getattr(node, 'red', False) else 1), True

    _, ok = check(root)
    if not ok:
        return False, "Black-height mismatch or red violation", bad
    return True, "ok", set()

# ============================================================
# Top-down (LLRB) — **iterative** with sentinel root
# ============================================================

@dataclass(eq=False)
class LLNode:
    key: int
    red: bool = True
    left: Optional['LLNode'] = None
    right: Optional['LLNode'] = None

def is_red(n: Optional[LLNode]) -> bool:
    return bool(n and n.red)

class TopDownLLRB:
    def __init__(self, frame_cb=None, postop: bool = True):
        self.root: Optional[LLNode] = None
        self.frame_cb = frame_cb
        self.postop = postop
        self._ghost_key: Optional[int] = None
        self._ghost_path: List[int] = []

    # snapshot helper
    def _emit(self, msg: str, keys: List[int] = None):
        if self.frame_cb:
            self.frame_cb(msg, clone_ll(self.root), set(keys or []),
                          ghost_key=self._ghost_key, ghost_path=list(self._ghost_path), role_map={})

    # rotations *at* parent's child pointer (keeps whole tree attached)
    def _rotate_left_at(self, parent: LLNode, child_attr: str):
        h: LLNode = getattr(parent, child_attr); assert h and is_red(h.right)
        self._emit(f"fix: right-lean → rotate-left at {h.key}", [h.key])
        x = h.right
        h.right = x.left
        x.left = h
        x.red = h.red
        h.red = True
        setattr(parent, child_attr, x)
        if self.postop:
            self._emit(f"after rotate-left at subtree root {x.key}", [x.key])

    def _rotate_right_at(self, parent: LLNode, child_attr: str):
        h: LLNode = getattr(parent, child_attr); assert h and is_red(h.left)
        self._emit(f"fix: two reds on left → rotate-right at {h.key}", [h.key])
        x = h.left
        h.left = x.right
        x.right = h
        x.red = h.red
        h.red = True
        setattr(parent, child_attr, x)
        if self.postop:
            self._emit(f"after rotate-right at subtree root {x.key}", [x.key])

    def _color_flip_here(self, node: LLNode):
        # Show explicit before/after
        keys = [node.key]
        if node.left:  keys.append(node.left.key)
        if node.right: keys.append(node.right.key)
        self._emit(f"split 4-node at {node.key} → color-flip (before)", keys)
        node.red = not node.red
        if node.left:  node.left.red  = not node.left.red
        if node.right: node.right.red = not node.right.red
        if self.postop:
            self._emit(f"after color-flip at {node.key} (attached)", [node.key])

    def insert(self, key: int):
        self._ghost_key = key
        self._ghost_path = []

        if self.root is None:
            self.root = LLNode(key, red=False)  # root black
            # pre-insertion frame (empty → attach)
            self._emit(f"attach at leaf {key}", [key])
            self._emit(f"attach new node {key}", [])
            self._ghost_key = None; self._ghost_path = []
            return

        # sentinel parent to simplify root rotations
        dummy = LLNode(-1, red=False)
        dummy.left = self.root

        parent = dummy
        child_attr = 'left'  # where 'h' hangs from its parent
        h = getattr(parent, child_attr)

        while h is not None:
            # path / narration
            self._ghost_path.append(h.key)
            self._emit(f"descend through {h.key}", [h.key])

            # Pre-descent: split 4-node if both children are red
            if is_red(h.left) and is_red(h.right):
                self._color_flip_here(h)

            # Fix right-lean
            if is_red(h.right) and not is_red(h.left):
                self._rotate_left_at(parent, child_attr)
                h = getattr(parent, child_attr)  # new subtree root after rotate

            # Fix two reds on left
            if is_red(h.left) and is_red(h.left.left):
                self._rotate_right_at(parent, child_attr)
                h = getattr(parent, child_attr)

            # Descend or attach
            if key < h.key:
                if h.left is None:
                    h.left = LLNode(key, red=True)
                    self._emit(f"attach at leaf {key}", [key])
                    break
                parent = h; child_attr = 'left'; h = h.left
            elif key > h.key:
                if h.right is None:
                    h.right = LLNode(key, red=True)
                    self._emit(f"attach at leaf {key}", [key])
                    break
                parent = h; child_attr = 'right'; h = h.right
            else:
                # duplicate key; nothing changes
                break

        # finalize root & snapshot
        self.root = dummy.left
        if self.root and self.root.red:
            self._emit("recolor root black", [self.root.key])
            self.root.red = False
            if self.postop:
                self._emit("after recolor root black", [self.root.key])

        self._emit(f"attach new node {key}", [])
        self._ghost_key = None
        self._ghost_path = []

        # verify invariants
        ok, msg, keys = verify_rb(self.root)
        if not ok and self.frame_cb:
            self.frame_cb(f"INVARIANT VIOLATION: {msg}", clone_ll(self.root), keys,
                          ghost_key=None, ghost_path=[], role_map={})

# ============================================================
# GUI
# ============================================================

@dataclass
class Frame:
    msg: str
    root_any: Any  # BUNode or LLNode clone
    highlight_keys: Set[int]
    ghost_key: Optional[int] = None
    ghost_path: Optional[List[int]] = None
    role_map: Dict[int, str] = None  # 'parent' | 'uncle' | 'grandparent'

class VisualRBApp(tk.Tk):
    def __init__(self, init_mode: str = "bottomup", init_sequence: str = "7 3 18 10 22 8 11 26", init_speed: int = 900):
        super().__init__()
        self.title("Red–Black Tree Visual Demo")
        self.geometry("1200x900")
        self._playing = False

        # Toolbar
        toolbar = ttk.Frame(self, padding=(8,6))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(toolbar, text="Mode:").grid(row=0, column=0, padx=(0,4))
        self.mode_var = tk.StringVar(value=init_mode)
        ttk.OptionMenu(toolbar, self.mode_var, init_mode, "bottomup", "topdown",
                       command=lambda _=None: self.rebuild_frames()).grid(row=0, column=1, padx=(0,10))

        ttk.Label(toolbar, text="Sequence:").grid(row=0, column=2, padx=(0,4))
        self.seq_var = tk.StringVar(value=init_sequence)
        seq_entry = ttk.Entry(toolbar, textvariable=self.seq_var, width=42)
        seq_entry.grid(row=0, column=3, padx=(0,10))
        seq_entry.bind("<Return>", lambda e: self.rebuild_frames())
        self.seq_var.trace_add('write', lambda *_: setattr(self, 'seq_dirty', True))
        self.seq_dirty = False

        ttk.Label(toolbar, text="Speed (ms):").grid(row=0, column=4, padx=(0,4))
        self.speed_var = tk.IntVar(value=init_speed)
        ttk.Spinbox(toolbar, from_=100, to=4000, increment=100,
                    textvariable=self.speed_var, width=6).grid(row=0, column=5, padx=(0,10))

        self.btn_reset = ttk.Button(toolbar, text="Reset/Load", command=self.rebuild_frames)
        self.btn_prev  = ttk.Button(toolbar, text="← Back",   command=self.step_back)
        self.btn_step  = ttk.Button(toolbar, text="Step →",   command=self.step_forward)
        self.btn_play  = ttk.Button(toolbar, text="Play ▶︎",   command=self.play_sequence)
        self.btn_stop  = ttk.Button(toolbar, text="Stop ⏹",    command=self.stop_play, state=tk.DISABLED)
        self.btn_reset.grid(row=0, column=6, padx=4)
        self.btn_prev.grid( row=0, column=7, padx=4)
        self.btn_step.grid( row=0, column=8, padx=4)
        self.btn_play.grid( row=0, column=9, padx=4)
        self.btn_stop.grid( row=0, column=10, padx=4)

        # Toggles
        self.show_nil_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Show NIL leaves", variable=self.show_nil_var,
                        command=self.redraw).grid(row=0, column=11, padx=(12,6))

        self.postop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Show post-op (attached) frames",
                        variable=self.postop_var, command=self.rebuild_frames)\
                        .grid(row=0, column=12, padx=(0,6))

        # Status
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor='w',
                  padding=(8,4)).pack(side=tk.BOTTOM, fill=tk.X)

        # Figure/canvas
        self.fig = plt.Figure(figsize=(12, 7.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Frame state
        self.frames: List[Frame] = []
        self.fi: int = -1
        self.last_seq_str: str = ""

        # Shortcuts
        self.bind("<Left>",  lambda e: self.step_back())
        self.bind("<Right>", lambda e: self.step_forward())
        self.bind("<space>", lambda e: (self.stop_play() if self._playing else self.play_sequence()))
        self.bind("<Key-r>", lambda e: self.rebuild_frames())
        self.bind("<Key-R>", lambda e: self.rebuild_frames())

        self.rebuild_frames()

    # ---------------- Frame building ----------------
    def parse_sequence(self) -> List[int]:
        raw = self.seq_var.get().replace(',', ' ')
        try:
            return [int(x) for x in raw.split() if x.strip()]
        except ValueError:
            messagebox.showerror("Sequence error", "Please enter integers separated by space or comma.")
            return []

    def rebuild_frames(self):
        seq = self.parse_sequence()
        if not seq:
            return
        self.frames.clear()
        self.fi = -1
        mode = self.mode_var.get()
        self.last_seq_str = self.seq_var.get()
        self.seq_dirty = False

        def push_frame(msg: str, root_clone, highlight_keys: Set[int],
                       ghost_key=None, ghost_path=None, role_map=None):
            self.frames.append(Frame(msg, root_clone, set(highlight_keys or set()),
                                     ghost_key, ghost_path or [], role_map or {}))

        if mode == 'bottomup':
            t = BottomUpRBTree(frame_cb=push_frame, postop=self.postop_var.get())
        else:
            t = TopDownLLRB(frame_cb=push_frame, postop=self.postop_var.get())
        for k in seq:
            t.insert(k)

        if not self.frames:
            self.frames.append(Frame("<empty>", None, set()))
        self.status.set(f"Built {len(self.frames)} frames for {mode} with sequence {seq}.")
        self.fi = 0
        self.redraw()

    def ensure_frames_current(self):
        if self.seq_var.get() != self.last_seq_str or self.seq_dirty:
            self.rebuild_frames()

    # ---------------- Controls ----------------
    def step_forward(self):
        self.ensure_frames_current()
        if self.fi < len(self.frames) - 1:
            self.fi += 1
            self.redraw()
        else:
            self.status.set("End of frames.")

    def step_back(self):
        self.ensure_frames_current()
        if self.fi > 0:
            self.fi -= 1
            self.redraw()
        else:
            self.status.set("Start of frames.")

    def play_sequence(self):
        self.ensure_frames_current()
        if self._playing:
            return
        self._playing = True
        self.btn_play.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._play_tick()

    def _play_tick(self):
        if not self._playing:
            return
        if self.fi < len(self.frames) - 1:
            self.fi += 1
            self.redraw()
            self.after(max(50, int(self.speed_var.get())), self._play_tick)
        else:
            self.stop_play()

    def stop_play(self):
        self._playing = False
        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    # ---------------- Drawing ----------------
    def _draw_nil(self, x, y):
        s = 0.12
        self.ax.add_patch(plt.Rectangle((x - s/2, y - s/2), s, s,
                         facecolor='black', edgecolor='black'))

    def redraw(self):
        self.ax.clear()
        self.ax.axis('off')
        if self.fi < 0 or self.fi >= len(self.frames):
            self.canvas.draw(); return
        fr = self.frames[self.fi]
        root = fr.root_any
        if root is None:
            self.ax.text(0.5, 0.5, "<empty>", ha='center', va='center')
            self.canvas.draw(); return

        pos, depth = inorder_positions(root)
        if not pos:
            self.canvas.draw(); return
        xs = [pos[n] for n in pos]; ys = [depth[n] for n in pos]
        x_min, x_max = min(xs)-1, max(xs)+1
        y_min, y_max = -max(ys)-1, 1

        # Edges + optional NIL leaves
        for n in pos:
            if getattr(n, 'left', None):
                self.ax.plot([pos[n], pos[n.left]], [-(depth[n]), -(depth[n.left])])
            else:
                gx, gy = pos[n] - 0.6, -depth[n] - 1
                self.ax.plot([pos[n], gx], [-(depth[n]), gy]); self._draw_nil(gx, gy)
            if getattr(n, 'right', None):
                self.ax.plot([pos[n], pos[n.right]], [-(depth[n]), -(depth[n.right])])
            else:
                gx, gy = pos[n] + 0.6, -depth[n] - 1
                self.ax.plot([pos[n], gx], [-(depth[n]), gy]); self._draw_nil(gx, gy)

        # Ghost path (top-down)
        if fr.ghost_path:
            path_nodes = [next((nn for nn in pos if getattr(nn, 'key', None)==k), None)
                          for k in fr.ghost_path]
            path_nodes = [pn for pn in path_nodes if pn is not None]
            for a, b in zip(path_nodes, path_nodes[1:]):
                self.ax.plot([pos[a], pos[b]], [-(depth[a]), -(depth[b])], linestyle='--')

        # Nodes + halos
        role_colors = {'parent':'#f2b01e', 'uncle':'#1e90ff', 'grandparent':'#2e8b57'}
        for n in pos:
            face = 'red' if getattr(n,'red',False) else 'black'
            circ = plt.Circle((pos[n], -depth[n]), 0.28,
                              edgecolor='black', facecolor=face, linewidth=2)
            self.ax.add_patch(circ)
            halo_color = None
            if fr.role_map and getattr(n, 'key', None) in fr.role_map:
                halo_color = role_colors.get(fr.role_map[getattr(n, 'key')])
            if getattr(n, 'key', None) in fr.highlight_keys and not halo_color:
                halo_color = 'gold'
            if halo_color:
                self.ax.add_patch(plt.Circle((pos[n], -depth[n]), 0.33,
                                  edgecolor=halo_color, facecolor='none', linewidth=3))
            self.ax.text(pos[n], -depth[n], str(getattr(n,'key','?')),
                         ha='center', va='center', fontsize=11,
                         color='white', fontweight='bold')

        # Ghost node circle
        if fr.ghost_key is not None and fr.ghost_path:
            lastk = fr.ghost_path[-1]
            parent = next((nn for nn in pos if getattr(nn, 'key', None)==lastk), None)
            if parent is not None:
                side = -0.6 if fr.ghost_key < getattr(parent,'key',0) else 0.6
                gx = pos[parent] + side; gy = -depth[parent] - 0.8
                self.ax.add_patch(plt.Circle((gx, gy), 0.25, edgecolor='black',
                                  facecolor='none', linewidth=2, linestyle='--'))
                self.ax.text(gx, gy, str(fr.ghost_key), ha='center', va='center', fontsize=10)

        # Legend & overlay
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='red',
                     markeredgecolor='black', label='Red node')
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='black',
                     markeredgecolor='black', label='Black node')
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='none',
                     markeredgecolor='gold', label='Highlighted')
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='none',
                     markeredgecolor='#f2b01e', label='Parent (halo)')
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='none',
                     markeredgecolor='#1e90ff', label='Uncle (halo)')
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='none',
                     markeredgecolor='#2e8b57', label='Grandparent (halo)')
        self.ax.plot([], [], linestyle='--', label='Ghost path / node')
        self.ax.legend(loc='upper right', frameon=True)

        self.ax.text(0.02, 0.98, fr.msg, transform=self.ax.transAxes,
                     ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white',
                               alpha=0.85, edgecolor='gray'))
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal')
        self.fig.tight_layout()
        self.canvas.draw()

# ============================================================
# Main
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Red–Black Tree Visual Demo")
    ap.add_argument('--mode', choices=['bottomup', 'topdown'], default='bottomup')
    ap.add_argument('--sequence', type=str, default='7 3 18 10 22 8 11 26')
    ap.add_argument('--speed', type=int, default=900)
    args = ap.parse_args()

    app = VisualRBApp(init_mode=args.mode, init_sequence=args.sequence, init_speed=args.speed)
    app.mainloop()

if __name__ == '__main__':
    main()