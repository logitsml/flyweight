"""The one-stack model running continuously for the live viewer (scripts/run/serve.py).

Every window is report_ms of fly time. With Qwen driving, Qwen first reads a report of its fly (the
world's description, the visitor's message if any, the senses the fly's neurons print) and its
answer writes the command registers; with the visitor driving, the registers hold the levels they
set with DRIVE tokens. Qwen's instructions are the ones every run uses: nothing tells it to follow
the visitor. A message reaches the fly only through what it does to Qwen's move logits, and stays
in every report until the visitor replaces or clears it. The world writes the sense registers (a visitor can add to them), the fly's
layers run step by step, and the body moves from the motor readout. Each window comes back in the
viewer's formats (flyweight/viewdata.py).

fly_only: the fly's layers alone (no Qwen, about 4 GB): only the visitor drives.
memory: Qwen reads each report after its last `memory` reports and answers (QwenFlyRunner memory).
adapt: the fly runs the adaptation variant (fly_layers.ADAPT).
"""

import base64
import queue
import time

import torch

from . import fly_layers, report, setup, viewdata
from .flyworld import ARENA, BITTER, DROP_R, PERCH, PERCH_R, PERCH_TOP, SUGAR, Body
from .sets import DRIVEN, REG, ROLES, SENSES, WORDS

TOPK = 96  # Qwen MLP neurons recorded per layer at a report's last token
MESSAGE_CHARS = 300

b64 = lambda raw: base64.b64encode(raw).decode()


class LiveFly:
    def __init__(self, fly_only=False, report_ms=report.STEP_MS, dust=0.0, memory=0, adapt=None):
        self.fly_only, self.report_ms, self.memory, self.adapt = fly_only, report_ms, memory, adapt
        con, A, world, groups, roles = setup.world_and_groups()
        self.con, self.A, self.world = con, A, world
        self.controls = queue.Queue()
        self.driver = "you" if fly_only else "qwen"
        self.levels = {c: 0 for c, _ in DRIVEN}  # the visitor's command levels (DRIVE tokens) when they drive
        self.sense_add = {s: 0 for s in SENSES}  # the visitor's sense stimulation, on top of the world's
        self.message, self.paused = None, False
        self.events = []  # what the visitor did since the last window (transcript lines)
        if fly_only:
            self.fly, self.tok, self.r, self.L = fly_layers.FlyModel(con, groups, roles, adapt=adapt), None, None, 0
        else:
            self.tok, qwen = setup.load_qwen()
            S = setup.standard(self.tok, qwen, adapt=adapt)
            self.m = S["m"]
            from .merged import QwenFlyRunner
            self.r = QwenFlyRunner(self.m, memory=memory)
            self.r.text(self.tok(report.PROMPT).input_ids)
            self.r.snapshot_qwen()
            self.r.start_fly()
            self.fly = self.r.flybrain
            self.L = len(self.m.qlayers)
            self.wid = [self.tok(" " + w).input_ids[0] for w in setup.ANSWER_WORDS]
            self.buf = torch.zeros(self.L, qwen.config.text_config.intermediate_size, device="cuda")
            for li, layer in enumerate(self.m.qlayers):  # Qwen's MLP neurons at the report's last token (the viewer's wall)
                layer.mlp.down_proj.register_forward_pre_hook(self._hook(li))
        self.n_windows = self.n_tokens = 0
        self.sense_now = {g: 0 for g in SENSES}
        self.reset(dust)
        ro = {k: v for k, v in world.readout.items()}
        groups_v, roles_v, labels = viewdata.groups_and_roles(con, A, ro)
        self.labels = labels
        self.header = dict(prompt=report.PROMPT, groups=groups_v, roles=roles_v, labels={str(k): v for k, v in labels.items()},
                           world=dict(sugar=SUGAR.tolist(), bitter=BITTER.tolist(), perch=PERCH.tolist(), perch_top=PERCH_TOP,
                                      arena=ARENA, drop_r=DROP_R, perch_r=PERCH_R),
                           flies=["fly"], fed_ms=[0.0], moves=dict(DRIVEN), body_cols=viewdata.BODY_COLS, leg_hz=viewdata.LEG_HZ, n_neurons=int(con.n), mismatch=None, block_mismatch=0,
                           say_bad=0, qwen=dict(layers=self.L, neurons=9216, topk=viewdata.KEEP), vocab=None, engine=viewdata.ENGINE,
                           design="live: " + ("you drive the fly's registers" if fly_only else "Qwen's answer drives, the fly prints its senses"),
                           live=dict(fly_only=fly_only, report_ms=report_ms, words=WORDS, commands=[c for c, _ in DRIVEN], senses=SENSES,
                                     memory=memory, adapt=adapt))

    def _hook(self, li):
        def f(mod, inp):  # returns None: a pre-hook that returns a value would replace the module's input
            self.buf[li].copy_(inp[0][0, -1].float())
        return f

    def reset(self, dust=0.0):
        """The fly back at the start, at rest, its registers cleared."""
        from .flyworld import START
        if self.fly_only:
            self.fly._start()
        else:
            self.r.start_fly()
            self.r.reset_memory()
        self.world.bodies = [Body("fly", START[0], START[1], yaw=START[2], dust=dust)]  # the world's clock runs on: the viewer's time never goes back
        self.sense_now = {g: 0 for g in SENSES}

    def apply(self, c):
        """One control from a visitor: {"type": "message" | "driver" | "levels" | "senses" | "world" | "pause"}."""
        t = c.get("type")
        if t == "message" and not self.fly_only:
            text = " ".join(str(c.get("text", "")).split())[:MESSAGE_CHARS]  # one line: the report's layout stays
            if text != (self.message or ""):
                self.events.append(f"You say: {text}" if text else "You clear your message.")
            self.message = text or None
        elif t == "driver" and not self.fly_only:
            self.driver = "you" if c.get("value") == "you" else "qwen"
            self.events.append("You drive the fly's registers." if self.driver == "you" else "Qwen's answer drives the fly.")
        elif t == "levels":
            for k, v in (c.get("levels") or {}).items():
                if k in self.levels:
                    self.levels[k] = int(max(0, min(15, int(v))))
        elif t == "senses":
            for k, v in (c.get("levels") or {}).items():
                if k in self.sense_add:
                    self.sense_add[k] = int(max(0, min(15, int(v))))
        elif t == "world":
            B = self.world.bodies[0]
            if c.get("action") == "dust":
                B.dust = 1.0
                self.events.append("Dust on the fly's antennae.")
            elif c.get("action") == "reset":
                self.reset()
                self.events.append("The fly is back at the start.")
        elif t == "pause":
            self.paused = bool(c.get("value", True))

    def _report(self):
        """Qwen reads one report; its answer writes the command registers. Returns the viewer's segment pieces."""
        W = self.world
        head = report.head(*W.observe(0))
        if self.message:
            head += f' The user says: "{self.message}"'
        lg, words = setup.read_report(self.r, self.tok, W.t / 10, head)
        line = report.text(W.t / 10, head, words)
        top = self.buf.abs().topk(TOPK, dim=1)
        val = torch.gather(self.buf, 1, top.indices)
        acts = [(top.indices[l].to(torch.int16).cpu(), val[l].half().cpu(), float(self.buf[l].abs().mean())) for l in range(self.L)]
        p = torch.softmax(lg[self.wid].float(), 0).cpu().numpy()
        answer = setup.ANSWER_WORDS[int(p.argmax())]
        self.r.end_report(setup.answer_ids(self.tok, answer))  # with memory, Qwen's log keeps its answer
        levels = self.r.levels(self.r.registers())[:len(DRIVEN)]
        return dict(read=line, words=[w.strip() for w in words], token=" " + answer, acts=acts,
                    probs={w: float(v) for w, v in zip(setup.ANSWER_WORDS, p)}, levels=[round(float(x), 2) for x in levels]), line

    def window(self):
        """report_ms of fly time; returns the message for the viewer (or None while paused)."""
        while True:
            try:
                self.apply(self.controls.get_nowait())
            except queue.Empty:
                break
        if self.paused:
            return None
        W, B = self.world, self.world.bodies[0]
        tq = time.time()
        msg = dict(type="window", segments=[], tokens=[], qwen=None)
        for e in self.events:
            msg["segments"].append(dict(kind="user", text=e))
        self.events = []
        line, levels, probs, words = report.head(*W.observe(0)), None, None, None
        if self.driver == "qwen":
            seg, line = self._report()
            levels, probs, words = seg["levels"], seg["probs"], seg["words"]
            msg["segments"].append(dict(kind="qwen", start=self.n_tokens, count=1, read=seg["read"], probs=probs, words=words, levels=levels))
            msg["tokens"] = [seg["token"]]
            msg["qwen"] = b64(viewdata.qwen_payload([seg["acts"]], self.L, viewdata.KEEP))
            self.n_tokens += 1
        else:  # the visitor's levels, written by DRIVE tokens
            self.fly.set_levels({REG[c]: self.levels[c] for c, _ in DRIVEN})
            levels = [float(self.levels[c]) for c, _ in DRIVEN]
        tq = time.time() - tq
        cmd_now = [c for (c, _), lv in zip(DRIVEN, levels) if lv >= 3] or ["rest"]
        steps, spikes, membr = self.report_ms * 10, [], []
        tw, f0 = time.time(), self.fly.n_forward
        for s in range(steps):
            sl = W.sense_levels(0)
            sl = {g: min(15, sl[g] + self.sense_add[g]) for g in SENSES}
            ch = {REG[g]: sl[g] for g in SENSES if sl[g] != self.sense_now[g]}
            if ch:
                self.fly.set_levels(ch)
                self.sense_now.update(sl)
            out = self.fly.step()[0]
            spikes += [(W.t, i) for i in out]
            if s % 50 == 0:
                idx, mv = self.fly.membrane()[0]
                membr.append((W.t, idx, mv))
            W.t += 1
            if W.t % 10 == 0:
                W.update_body(0, "+".join(cmd_now), dict(zip(ROLES, self.fly.move()[0].tolist())))
        # the command levels the fly's first layer holds at the window's end (inside Qwen: read back, whoever drives)
        held = None if self.fly_only else [round(float(x), 2) for x in self.r.levels(self.r.registers())[:len(DRIVEN)]]
        w = dict(cmds={"fly": cmd_now}, ms=self.report_ms, spikes=[spikes], pois=[[]], membrane=[membr], wall=time.time() - tw,
                 forwards=self.fly.n_forward - f0, obs=[line], t0=W.t - steps, probs=probs, levels=levels, words=words)
        new_labels = {}
        payload = viewdata.window_payload(w, self.A, new_labels)
        new_labels = {str(k): v for k, v in new_labels.items() if k not in self.labels}
        self.labels.update({int(k): v for k, v in new_labels.items()})
        rows = viewdata.body_rows(B.log)
        B.log.clear()  # sent; a live session runs for hours
        self.fly.tokens = [[] for _ in self.fly.tokens]
        del self.fly.bad_log[:-100]
        if self.r is not None:
            self.r.streams = [[] for _ in self.r.streams]
        msg["segments"].append(dict(kind="fly", run=self.n_windows, summary="", args={"fly": cmd_now}))
        msg.update(window=viewdata.window_meta(w, len(payload)), blob=b64(payload), body=b64(rows.tobytes()), labels=new_labels,
                   state=dict(t_ms=W.t / 10, x=B.x, y=B.y, z=B.z, mode=B.mode, fed_ms=B.fed_ms, dust=B.dust, driver=self.driver,
                              levels=self.levels, senses=self.sense_add, held=held, message=self.message,
                              qwen_s=round(tq, 3), fly_s=round(time.time() - tw, 3), block=[self.fly.mismatch, self.fly.swaps]))
        self.n_windows += 1
        return msg
