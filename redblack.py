#!/usr/bin/env python3
"""
Red–Black Tree Visual Demo (Top‑Down vs Bottom‑Up)
--------------------------------------------------
A classroom‑friendly GUI that *visually* shows red–black tree insertions with
step‑by‑step frames you can play, pause, and **go backward** through.

Highlights
* **Edit the sequence live** while the app is running (press Enter or just hit Step/Play; it auto‑reloads).
* **Two strategies**: Bottom‑up (CLRS fix‑after) and Top‑down (LLRB fix‑during‑descent).
* **Granular frames**: each recolor, rotation, and structural change is its own frame.
* **Ghost node** (top‑down): the new key is shown as a dashed, hollow node during the descent; it is only attached in the final frame to emphasize the concept.
* **Backward/Forward** navigation and **slower animation** control.
* **Clear visuals**: solid red/black nodes with white text; gold halo highlights; dashed ghost node and dashed path; legend on canvas.

Run
    python red_black_tree_demo.py [--mode bottomup|topdown] [--sequence "7 3 18 10 22 8 11 26"] [--speed 900]
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
except Exception as e:
    raise SystemExit("Matplotlib with TkAgg backend is required. Install matplotlib and try again.")

# ============================================================
# Core drawing helpers
# ============================================================

def inorder_positions(root) -> Tuple[dict, dict]:
    pos: Dict[Any, int] = {}
    depth: Dict[Any, int] = {}
    i = 0
    def dfs(n, d):
        nonlocal i
        if not n:
            return
        dfs(n.left, d+1)
        pos[n] = i
        depth[n] = d
        i += 1
        dfs(n.right, d+1)
    dfs(root, 0)
    return pos, depth

# Snapshot cloning so frames can be revisited precisely

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
# Bottom‑up (CLRS‑style)
# ============================================================

@dataclass(eq=False)
class BUNode:
    key: int
    red: bool = True
    left: Optional['BUNode'] = None
    right: Optional['BUNode'] = None
    parent: Optional['BUNode'] = None

class BottomUpRBTree:
    def __init__(self, frame_cb=None):
        self.root: Optional[BUNode] = None
        self.steps: List[str] = []
        self.frame_cb = frame_cb  # function(msg:str, highlight_keys:Set[int], ghost_key:Optional[int])

    def _emit(self, msg: str, highlight_nodes: List[BUNode] = None):
        self.steps.append(msg)
        if self.frame_cb:
            keys = {n.key for n in (highlight_nodes or []) if n is not None}
            self.frame_cb(msg, clone_bu(self.root), keys, ghost_key=None, ghost_path=None)

    # Rotations
    def _left_rotate(self, x: BUNode):
        y = x.right; assert y is not None
        self._emit(f"rotate-left at {x.key}", [x, y])
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
        self._emit(f"after rotate-left at {x.key}")

    def _right_rotate(self, y: BUNode):
        x = y.left; assert x is not None
        self._emit(f"rotate-right at {y.key}", [y, x])
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
        self._emit(f"after rotate-right at {y.key}")

    def insert(self, key: int):
        # Plain BST attach first (to emphasize bottom‑up)
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

    def _insert_fixup(self, z: BUNode):
        while z.parent is not None and z.parent.red:
            gp = z.parent.parent
            if z.parent == gp.left:
                y = gp.right
                if y is not None and y.red:
                    self._emit(f"recolor parent {z.parent.key}, uncle {y.key}, grandparent {gp.key}", [z.parent, y, gp])
                    z.parent.red = False; y.red = False; gp.red = True
                    self._emit(f"after recolor at grandparent {gp.key}")
                    z = gp
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._emit(f"prepare inner case (left rotation)", [z])
                        self._left_rotate(z)
                    self._emit(f"recolor & rotate-right at grandparent {gp.key}", [gp, gp.left] if gp.left else [gp])
                    z.parent.red = False
                    gp.red = True
                    self._right_rotate(gp)
            else:
                y = gp.left
                if y is not None and y.red:
                    self._emit(f"recolor parent {z.parent.key}, uncle {y.key}, grandparent {gp.key}", [z.parent, y, gp])
                    z.parent.red = False; y.red = False; gp.red = True
                    self._emit(f"after recolor at grandparent {gp.key}")
                    z = gp
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._emit(f"prepare inner case (right rotation)", [z])
                        self._right_rotate(z)
                    self._emit(f"recolor & rotate-left at grandparent {gp.key}", [gp, gp.right] if gp.right else [gp])
                    z.parent.red = False
                    gp.red = True
                    self._left_rotate(gp)
        if self.root and self.root.red:
            self._emit("recolor root black", [self.root])
            self.root.red = False
            self._emit("after recolor root black")

# ============================================================
# Top‑down (LLRB / Sedgewick)
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
    def __init__(self, frame_cb=None):
        self.root: Optional[LLNode] = None
        self.steps: List[str] = []
        self.frame_cb = frame_cb
        self._ghost_key: Optional[int] = None
        self._ghost_path: List[int] = []  # keys along descent path for dashed hint

    def _emit(self, msg: str, highlight_nodes: List[LLNode] = None):
        self.steps.append(msg)
        if self.frame_cb:
            keys = {n.key for n in (highlight_nodes or []) if n is not None}
            self.frame_cb(msg, clone_ll(self.root), keys, ghost_key=self._ghost_key, ghost_path=list(self._ghost_path))

    def _rotate_left(self, h: LLNode) -> LLNode:
        assert h.right and is_red(h.right)
        x = h.right
        h.right = x.left
        x.left = h
        x.red = h.red
        h.red = True
        return x

    def _rotate_right(self, h: LLNode) -> LLNode:
        assert h.left and is_red(h.left)
        x = h.left
        h.left = x.right
        x.right = h
        x.red = h.red
        h.red = True
        return x

    def _color_flip(self, h: LLNode):
        # pure mutation, no emits here to avoid mid-stack snapshots
        h.red = not h.red
        if h.left:
            h.left.red = not h.left.red
        if h.right:
            h.right.red = not h.right.red

    def insert(self, key: int):
        # For teaching: set up a ghost key visible during the whole descent.
        self._ghost_key = key
        self._ghost_path = []
        self._emit(f"begin insert {key} (ghost descending)")
        self.root = self._insert(self.root, key)
        if self.root and self.root.red:
            self._emit("recolor root black", [self.root])
            self.root.red = False
            self._emit("after recolor root black")
        # Attach completes: clear ghost
        self._emit(f"attach new node {key}")
        self._ghost_key = None
        self._ghost_path = []

    def _insert(self, h: Optional[LLNode], key: int) -> LLNode:
        if h is None:
            # Final step of top‑down: actual attach happens here
            n = LLNode(key=key, red=True)
            # show a frame just before and just after attach
            self._emit(f"attach at leaf {key}")
            return n
        # Record the descent path (for dashed path hints)
        self._ghost_path.append(h.key)
        self._emit(f"descend through {h.key}")
        if key < h.key:
            h.left = self._insert(h.left, key)
        elif key > h.key:
            h.right = self._insert(h.right, key)
        else:
            # duplicate ignored
            pass
        # Fix-ups in LLRB (top‑down feel): emit a frame for each op
        if is_red(h.right) and not is_red(h.left):
            self._emit(f"fix: right-lean → rotate-left at {h.key}", [h, h.right])
            h = self._rotate_left(h)
        if is_red(h.left) and is_red(h.left.left):
            self._emit(f"fix: two reds on left → rotate-right at {h.key}", [h, h.left])
            h = self._rotate_right(h)
        if is_red(h.left) and is_red(h.right):
            self._emit(f"split 4-node at {h.key} → color-flip", [h, h.left, h.right])
            self._color_flip(h)
        return h

# ============================================================
# GUI Application (Tk + Matplotlib)
# ============================================================

@dataclass
class Frame:
    msg: str
    root_any: Any  # BUNode or LLNode clone
    highlight_keys: Set[int]
    ghost_key: Optional[int] = None
    ghost_path: Optional[List[int]] = None
    role_map: Dict[int, str] = None  # key -> role ('parent','uncle','grandparent')

class VisualRBApp(tk.Tk):
    def __init__(self, init_mode: str = "bottomup", init_sequence: str = "7 3 18 10 22 8 11 26", init_speed: int = 900):
        super().__init__()
        self.title("Red–Black Tree Visual Demo")
        self.geometry("1200x820")
        self._building = False

        # Controls frame
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(ctrl, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value=init_mode)
        mode_menu = ttk.OptionMenu(ctrl, self.mode_var, init_mode, "bottomup", "topdown", command=lambda _=None: self.rebuild_frames())
        mode_menu.pack(side=tk.LEFT, padx=6)

        ttk.Label(ctrl, text="Sequence:").pack(side=tk.LEFT, padx=(12, 2))
        self.seq_var = tk.StringVar(value=init_sequence)
        seq_entry = ttk.Entry(ctrl, textvariable=self.seq_var, width=48)
        seq_entry.pack(side=tk.LEFT, padx=6)
        seq_entry.bind("<Return>", lambda e: self.rebuild_frames())
        self.seq_var.trace_add('write', lambda *_: setattr(self, 'seq_dirty', True))
        self.seq_dirty = False

        ttk.Label(ctrl, text="Speed (ms):").pack(side=tk.LEFT, padx=(12, 2))
        self.speed_var = tk.IntVar(value=init_speed)
        ttk.Spinbox(ctrl, from_=200, to=3000, increment=100, textvariable=self.speed_var, width=6).pack(side=tk.LEFT)

        self.btn_reset = ttk.Button(ctrl, text="Reset/Load", command=self.rebuild_frames)
        self.btn_reset.pack(side=tk.LEFT, padx=4)
        self.btn_prev = ttk.Button(ctrl, text="← Back", command=self.step_back)
        self.btn_prev.pack(side=tk.LEFT, padx=4)
        self.btn_step = ttk.Button(ctrl, text="Step →", command=self.step_forward)
        self.btn_step.pack(side=tk.LEFT, padx=4)
        self.btn_play = ttk.Button(ctrl, text="Play ▶︎", command=self.play_sequence)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        self.btn_stop = ttk.Button(ctrl, text="Stop ⏹", command=self.stop_play, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        # Status / narration
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, anchor='w').pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)

        # Matplotlib figure
        self.fig = plt.Figure(figsize=(12, 6.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Frame state
        self.frames: List[Frame] = []
        self.fi: int = -1  # frame index
        self.last_seq_str: str = ""

        self.rebuild_frames()  # initial

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
        # Frame callback collector
        def push_frame(msg: str, root_clone, highlight_keys: Set[int], ghost_key=None, ghost_path=None):
            self.frames.append(Frame(msg, root_clone, set(highlight_keys or set()), ghost_key, ghost_path or []))
        if mode == 'bottomup':
            t = BottomUpRBTree(frame_cb=push_frame)
            for k in seq:
                t.insert(k)
        else:
            t = TopDownLLRB(frame_cb=push_frame)
            for k in seq:
                t.insert(k)
        # Ensure at least one frame present
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
        if getattr(self, '_playing', False):
            return
        self._playing = True
        self.btn_play.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._play_tick()

    def _play_tick(self):
        if not getattr(self, '_playing', False):
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
    def redraw(self):
        self.ax.clear()
        self.ax.axis('off')
        if self.fi < 0 or self.fi >= len(self.frames):
            self.canvas.draw(); return
        fr = self.frames[self.fi]
        root = fr.root_any
        # Draw tree
        if root is None:
            self.ax.text(0.5, 0.5, "<empty>", ha='center', va='center')
            self.canvas.draw(); return
        pos, depth = inorder_positions(root)
        if not pos:
            self.canvas.draw(); return
        xs = [pos[n] for n in pos]
        ys = [depth[n] for n in pos]
        x_min, x_max = min(xs)-1, max(xs)+1
        y_min, y_max = -max(ys)-1, 1

        # Edges (neutral gray); dashed path for ghost descent
        for n in pos:
            if getattr(n, 'left', None):
                self.ax.plot([pos[n], pos[n.left]], [-(depth[n]), -(depth[n.left])])
            if getattr(n, 'right', None):
                self.ax.plot([pos[n], pos[n.right]], [-(depth[n]), -(depth[n.right])])

        # Ghost path (top‑down only)
        if fr.ghost_path:
            # draw dashed connectors along keys in path
            path_nodes = [next((nn for nn in pos if getattr(nn, 'key', None)==k), None) for k in fr.ghost_path]
            path_nodes = [pn for pn in path_nodes if pn is not None]
            for a, b in zip(path_nodes, path_nodes[1:]):
                self.ax.plot([pos[a], pos[b]], [-(depth[a]), -(depth[b])], linestyle='--')

        # Nodes (solid fill; white text) with highlights
        for n in pos:
            is_red = bool(getattr(n, 'red', False))
            face = 'red' if is_red else 'black'
            circ = plt.Circle((pos[n], -depth[n]), 0.28, edgecolor='black', facecolor=face, linewidth=2)
            self.ax.add_patch(circ)
            if getattr(n, 'key', None) in fr.highlight_keys:
                halo = plt.Circle((pos[n], -depth[n]), 0.33, edgecolor='gold', facecolor='none', linewidth=3)
                self.ax.add_patch(halo)
            self.ax.text(pos[n], -depth[n], str(getattr(n, 'key', '?')), ha='center', va='center', fontsize=11, color='white', fontweight='bold')

        # Ghost node (top‑down attach pending)
        if fr.ghost_key is not None:
            # place at the would‑be child of last node in ghost path
            parent = None
            if fr.ghost_path:
                lastk = fr.ghost_path[-1]
                parent = next((nn for nn in pos if getattr(nn, 'key', None)==lastk), None)
            if parent is not None:
                # offset slightly below parent toward the side
                side = -0.6 if fr.ghost_key < getattr(parent,'key',0) else 0.6
                gx = pos[parent] + side
                gy = -depth[parent] - 0.8
                circ = plt.Circle((gx, gy), 0.25, edgecolor='black', facecolor='none', linewidth=2, linestyle='--')
                self.ax.add_patch(circ)
                self.ax.text(gx, gy, str(fr.ghost_key), ha='center', va='center', fontsize=10)

        # Legend & overlay
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='red', markeredgecolor='black', label='Red node')
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='black', markeredgecolor='black', label='Black node')
        self.ax.plot([], [], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor='gold', label='Highlighted')
        self.ax.plot([], [], linestyle='--', label='Ghost path / node')
        self.ax.legend(loc='upper right', frameon=True)

        overlay_text = fr.msg
        self.ax.text(0.02, 0.98, overlay_text, transform=self.ax.transAxes,
                     ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))

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
