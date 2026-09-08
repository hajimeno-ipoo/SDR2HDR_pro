"""Reference-inspired desktop palette and restrained interaction feedback."""
import tkinter as tk
from tkinter import ttk

INK = "#14200e"
PAPER = "#fffdf6"
PINK = "#efb4eb"
GREEN = "#a9d994"


def apply_theme(root: tk.Tk, reduced=None) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=INK)
    root.option_add("*TCombobox*Listbox.background", PAPER)
    root.option_add("*TCombobox*Listbox.foreground", INK)
    root.option_add("*TCombobox*Listbox.selectBackground", GREEN)
    root.option_add("*TCombobox*Listbox.selectForeground", INK)
    style.configure(".", background=PAPER, foreground=INK, font=("Helvetica Neue", 11), bordercolor=INK, lightcolor=INK, darkcolor=INK)
    style.configure("Shell.TFrame", background=INK)
    style.configure("Card.TFrame", background=PAPER, borderwidth=3, relief="solid")
    style.configure("Green.TFrame", background=GREEN)
    style.configure("Green.TLabel", background=GREEN)
    style.configure("Status.TLabel", background=GREEN, font=("Helvetica Neue", 16, "bold"))
    style.configure("Section.TLabel", font=("Impact", 26))
    style.configure("Group.TLabel", background=GREEN, font=("Helvetica Neue", 10, "bold"), padding=(6, 2))
    style.configure("Error.TEntry", bordercolor="#b34232", lightcolor="#b34232", darkcolor="#b34232")
    style.configure("Muted.TLabel", foreground="#505748", font=("Helvetica Neue", 10))
    style.configure("TButton", padding=(10, 8), borderwidth=2, relief="solid", focusthickness=2, focuscolor=INK)
    style.map("TButton", background=[("disabled", "#e7e8df"), ("pressed", GREEN), ("active", "#e8efdf")], foreground=[("disabled", "#74796c")])
    for name, color, foreground in (("Accent.TButton", PINK, INK), ("Primary.TButton", INK, PAPER)):
        style.configure(name, background=color, foreground=foreground, font=("Helvetica Neue", 12, "bold"), padding=(12, 12))
        style.map(name, background=[("disabled", "#e7e8df"), ("pressed", GREEN), ("active", GREEN)], foreground=[("disabled", "#74796c"), ("pressed", INK), ("active", INK)])
    style.configure("TEntry", fieldbackground=PAPER, padding=4, borderwidth=1)
    style.configure("TCombobox", fieldbackground=PAPER, padding=3, arrowsize=14)
    style.map("TCombobox", fieldbackground=[("readonly", PAPER)], foreground=[("disabled", "#74796c"), ("readonly", INK)], selectbackground=[("readonly", PAPER)], selectforeground=[("readonly", INK)])
    style.configure("TNotebook", borderwidth=0, background=PAPER, tabmargins=(0, 0, 0, 8))
    style.configure("TNotebook.Tab", padding=(14, 10), background=PAPER, borderwidth=2)
    style.map("TNotebook.Tab", background=[("selected", PINK), ("active", GREEN)], foreground=[("selected", INK)])
    style.configure("Treeview", background=PAPER, fieldbackground=PAPER, rowheight=32, borderwidth=1)
    style.configure("Treeview.Heading", background=PINK, padding=(8, 9), font=("Helvetica Neue", 10, "bold"), relief="flat")
    style.map("Treeview", background=[("selected", GREEN)], foreground=[("selected", INK)])
    style.configure("Horizontal.TProgressbar", troughcolor=PAPER, background=INK, borderwidth=1, thickness=10)
    style.configure("Horizontal.TScale", troughcolor="#dfe4d8", background=GREEN)



class FeedbackMotion:
    """One short, interruptible cue. Text and final colors remain without motion."""
    def __init__(self, root, reduced, label, output_button):
        self.root, self.reduced = root, reduced
        self.label, self.output_button = label, output_button
        self.pending = None
        self.restore = lambda: None
        self.style = ttk.Style(root)
        reduced.trace_add('write', lambda *_: self.cancel())

    def cancel(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.restore()
        self.restore = lambda: None

    def cue(self, kind, tree=None, row=None, target=None):
        import math
        self.cancel()
        if kind == 'complete' and hasattr(self.output_button, 'motion_slot'):
            self.output_button.motion_slot.bounce()
        rows = tuple(row) if isinstance(row, (list, tuple)) else ((row,) if row is not None else ())
        self.style.configure('Status.TLabel', background=GREEN)
        self.output_button.configure(style='Accent.TButton' if kind == 'complete' else 'TButton')
        if target is not None:
            target.configure(style='Error.TEntry')
            target.bind('<KeyRelease>', lambda _: target.configure(style='TEntry'), add='+')
        if tree is not None:
            tree.tag_configure('recent', background=GREEN)
            for item in rows:
                if tree.exists(item):
                    tree.item(item, tags=('recent',))
            if rows and tree.exists(rows[-1]):
                tree.see(rows[-1])

        def reset():
            if hasattr(self.output_button, "motion_slot"):
                self.output_button.motion_slot.reset()
            if tree is not None:
                for item in rows:
                    if tree.exists(item):
                        tree.item(item, tags=())

        self.restore = reset
        if self.reduced.get():
            return

        def frame(index=0):
            if not self.label.winfo_exists():
                self.pending = None
                return
            t = index / 16
            if kind == 'error' and target is not None:
                # A single border emphasis stays on the field that needs attention.
                red = round(179 + 45 * math.sin(math.pi*t))
                self.style.configure('Error.TEntry', bordercolor=f'#{red:02x}4232')
            if tree is not None:
                blend = (1-t)**3
                rgb = tuple(round(int(PAPER[i:i+2],16)*(1-blend)+int(GREEN[i:i+2],16)*blend) for i in (1,3,5))
                tree.tag_configure('recent', background='#%02x%02x%02x' % rgb)
            if index < 16:
                self.pending = self.root.after(20, frame, index+1)
            else:
                self.pending = None
                reset()
        frame()


class ButtonSlot(ttk.Frame):
    """Reserve a constant rectangle; translate the whole native button inside it."""
    def __init__(self, parent, reduced, **kwargs):
        super().__init__(parent)
        self.reduced = reduced
        self.pending = None
        self.button = ttk.Button(self, **kwargs)
        self.button.motion_slot = self
        self.update_idletasks()
        self.configure(height=self.button.winfo_reqheight() + 12, width=self.button.winfo_reqwidth() + 12)
        self.button.place(x=6, y=6, relwidth=1, width=-12)
        self.button.bind('<ButtonPress-1>', self.press, add='+')
        self.button.bind('<ButtonRelease-1>', lambda _: self.bounce(), add='+')
        self.button.bind('<Leave>', lambda _: self.reset(), add='+')
        reduced.trace_add('write', lambda *_: self.reset())
        self.bind('<Destroy>', lambda _: self.reset(), add='+')

    def reset(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
            self.pending = None
        if self.button.winfo_exists():
            self.button.place_configure(y=6)

    def press(self, _):
        self.reset()
        if not self.reduced.get() and not self.button.instate(['disabled']):
            self.button.place_configure(y=9)

    def bounce(self):
        import math
        import time
        self.reset()
        if self.reduced.get():
            return
        started = time.monotonic()
        def frame():
            t = min(1, (time.monotonic()-started)/0.3)
            # One shallow upward arc, settling within the reserved six-pixel inset.
            self.button.place_configure(y=6-round(5*math.sin(math.pi*t)))
            if t < 1:
                self.pending = self.after(16, frame)
            else:
                self.pending = None
                self.button.place_configure(y=6)
        frame()
