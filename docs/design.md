# FlyWeight: design and checks

Qwen3.5-4B's 32 decoder layers and the fly's 3 form a single model that shares its residual stream,
vocabulary, cache and token stream. Qwen's computation sets the drive on the fly's descending neurons,
the fly's neurons print what it senses as text that Qwen reads, and greedy decoding alone chooses
each spike. Results are in the [README](../README.md).

Notation: n = 166,700 neurons (Janelia male CNS v1.0); H = 6 x 4^9 = 1,572,864 residual dims.

## The fly's layers (`flyweight/fly_layers.py`)

Tokens: `TICK_t` (t = 0..22, one 0.1 ms step each), `spk_j` (one per neuron), `INIT`, `PAD`,
`DRIVE_g_l` (10 registers x 16 levels), `MOVE_r` (14 readout rows, read and never decoded) and `SAY_k`
(4 printed-word slots).

- **Layer 1: registers.** 23 phase heads hold which TICK came last. 10 register heads hold the drive:
  7 commands (written by Qwen's answer or a `DRIVE` token) and 3 senses (written by the world). Its MLP
  writes relu(register - offset) for each inhibition the answer bridge uses: the steering pair inhibit
  each other, and both inhibit forward.
- **Layer 2: the spike ring.** 23 slots keyed by phase. One read type looks 19 slots back (the 1.8 ms
  synaptic delay) and its output projection is the connectome: 24,939,905 connections, each once, at
  count x 0.275 mV x 0.73, excitatory and inhibitory in separate channels. The other reads the last 22
  slots: the 2.2 ms refractory flag, which also counts each readout role's recent spikes. The MLP adds
  synaptic input unless refractory and, at every TICK, each register's tonic input: per member,
  `STEP x gain x (relu(r - 1) - relu(r - 16))` with r = alpha x register + beta (an inhibited register
  first subtracts its inhibitors' relu terms). `up` reads the TICK flag, so a row is exactly 0 off a
  TICK whatever the register holds; the member's refractory flag closes it at -2^60, for any register
  below 2^50 levels. Gains are fixed per neuron in 13/16..19/16, so a driven population fires at a
  spread of rates.
- **Layer 3: membranes and the motor readout.** Each membrane is two decay modes (membrane and
  synaptic time constants), each split into + and - channels, decayed by the delta rule's beta and
  reset by the neuron's own spike token. Each readout role integrates its refractory count over 30 ms.
- **LM head.** Spike logits are 1024 x (v - v_th), so greedy decoding is the threshold: every neuron
  above the next TICK's logit spikes, in descending order. MOVE rows give each role's rate in Hz per
  member. Inside Qwen, SAY rows give the printed words (below).

**The adaptation variant** (`fly_layers.ADAPT`; `--adapt` on the scripts that build the fly). Shiu's
model has no spike-frequency adaptation. In the variant, one more layer-3 head per neuron integrates the neuron's
refractory flag (its spike in the last 22 steps) over tau = 200 ms into a threshold rise theta, kappa =
16 mV times the flag's leaky mean, written as -theta into the neuron's membrane dim, which only the
spike logit reads (so it reads v - theta against the threshold). A neuron firing at 100 Hz
has its threshold ~3.5 mV up. The executor (`flyweight/lif.py`) and Brian2
(`scripts/checks/check_tonic_brian.py`) run the same rule.

**Exactness.** Fly tokens carry 2^33 in 6 ballast dims, so every residual RMSNorm on a fly token
divides by exactly 2^24, which the weights fold in. The per-head norms of layers 1 and 3 carry a
power-of-two ballast (layer 1: 2^40 with Qwen's answer bridge, else 2^30; layer 3: 2^40) and divide by
exactly half of it. Layer 2's heads hold spike indicators in one channel and return 0 or 1 (1 to
within 6e-8: the delta rule scales queries by 1/sqrt(23)). The convolution's SiLU passes a value
unchanged above 17.3: every synaptic input and any drive above 1/150 of a level (membrane scale
2^10), every relu row above 1/60 of its unit (slopes 2^10). The fly's layers always take HF's
recurrent delta rule (flash-linear-attention's fused recurrent kernel when installed), so a block
of tokens and single tokens give the same numbers. Up to 16 token rows per forward run one CSR
matrix-vector product each, so a block, a batch of two flies and a lone fly agree bit for bit.
cuSPARSE's matrix-vector kernel splits the connectome's long rows across threads, so that one product
can differ from run to run in its last bit (membranes by ~1e-6 mV; spikes only at a threshold tie).
With `SparseLinear.deterministic` every sparse product reads only the columns its input touches (for
the connectome, the neurons that spiked 19 steps ago and the refractory ones) and sums each target in
a fixed segmented reduction: every run is then the same, bit for bit, at some cost in speed.

**Stepping on the GPU.** A step's time goes to kernel launches, so its blocks replay as CUDA graphs,
one per block length (`BlockGraphs`): the same kernels in the same order, without Python per layer. Transformers updates the DeltaNet cache in place, so its addresses stay fixed. Only each
fly's last token in a block (its TICK) reads the spike ring, so the connectome's product runs on those
rows alone with no host sync. Every step checks that the block's other rows stayed zero. The next
step's spikes are chosen on the GPU (a stable order: logit descending, ties by neuron) and come back
with this step's argmax checks, one transfer per step. Graphs and eager stepping give identical
spikes, step for step.

## One stack (`flyweight/merged.py`)

- **Residual.** Qwen's dims 0..2559, then markers: `TXT` (2^-20 on every text token), six `SIT` dims
  (the control bridge) and `CLS` (2^-20 on the report-end token `"):"`), then the fly's dims.
- **Qwen's weights are unchanged** (dense bf16 blocks reading and writing dims 0..2559). Its residual
  norms average over H, so their gains carry sqrt(2560 / H) and their eps 1e-6 x 2560 / H.
- **Isolation by weights.** On a fly token, mute columns on the fly's ballast set Qwen's DeltaNet decay
  to 1, its write to 0, its output to 0 and its attention output gate to 0: Qwen's state and dims stay
  exactly unchanged. On a text token every fly write strength gets -K x TXT, so the fly writes
  nothing, and every fly LM row gets -K x TXT, so it is never decoded there. Qwen's full attention
  ignores fly positions through the attention mask, a standard input.
- **Fast path.** Qwen's layers are skipped on fly tokens (their convolution windows get the zeros fly
  tokens would have put there), and the fly's layers run only on report-end tokens. On text tokens
  Qwen's layers run on the residual's first 2,569 dims, the only ones a text token holds before the
  fly's layers; the norms still divide by H, so this is the same computation without the zeros. On fly
  tokens the LM head computes only the fly's rows and the printed words' rows: Qwen's dims are 0 there,
  so every other text row is exactly -2^13. On fly tokens each residual RMSNorm is the reference's own
  two products, (x 2^-24)(1 + w), without its squares, mean and rsqrt (the ballast makes the divisor
  exactly 2^24), and products with an all-zero matrix are skipped.
- **Full forward.** All 35 layers on every token over the whole residual; `check_merged` compares it
  with the fast path.

## The neurons (`flyweight/sets.py`, `flyweight/flyworld.py`)

Every set is chosen by Janelia's cell types and checked in the connectome itself
(`scripts/checks/check_wiring.py`, signed synapse counts).

| set | male CNS types | neurons (L/R) | why |
|---|---|---|---|
| forward | DNp09 | 1/1 | P9, which initiates forward walking (Bidaye et al. 2020) |
| back | MDN | 2/2 | the moonwalker descending neurons, which drive backward walking (Bidaye et al. 2014); 67% of their output synapses are on nerve-cord interneurons, led by the hind-leg neuromere |
| groom | DNg12_e | 3/3 | the antennal grooming descending neuron (aDN, Hampel et al. 2015; MANC's DNg12); 70% of its output synapses are in the front-leg neuromere, 11% straight onto motor neurons |
| turn left, right | DNa02 | 1/1 | steering descending neurons (Rayshubskiy et al.); 92% of each one's synapses are on its own side; each synapses on its side's turn pool (1,606 and 1,928 synapses, against 34 and 17 on the other pool) and inhibits the other pool two synapses away |
| takeoff | DNp01 | 1/1 | the giant fiber. Each drives its own side's jump motor neuron (TTMn) and PSI through gap junctions that follow it spike for spike (shakB; Tanouye and Wyman 1980). The chemical connectome holds only 20 and 70 synapses onto TTMn, too few against TTMn's inhibition, so each junction is added as 250 excitatory synapses (`flyweight/malecns.py`) |
| fly | DNg02_a..g | 15/14 | the population that sets wing stroke amplitude (Namiki et al. 2022); excites the flight power motor neurons two synapses away |
| sugar | LB3c | 12/11 | FlyWire's "sugar" labellar taste neurons; excite MN9 three synapses away |
| bitter | LB1a..d | 19/19 | FlyWire's "bitter" labellar taste neurons; inhibit MN9 |
| Johnston's organ | JO-C*, JO-E* | 203/132 | static antennal deflection, driving antennal grooming (Hampel et al. 2015); the right side has 130 JO neurons without a type |
| forward, turn, backward and grooming pools | nerve-cord, ascending and descending neurons | 40, 6, 18, 40, 40 | what each command recruits in the fly's own layers and the others do not (`calib_motor.py`, `pick_readout.py`): at least 10 Hz under its command, at most 2 Hz under every other command and 1 Hz at rest, top 40. Forward drive also recruits the left-turn pool, by a share that changes with its level |
| jump | TTMn | 1/1 | tergotrochanteral motor neurons |
| flight power | DLMn, DVMn | 12/12 | indirect flight muscle motor neurons (5 DLM and 7 DVM per side) |
| flight steering | b1 MN, b2 MN | 2/2 | basalar steering muscle motor neurons |
| proboscis | MN9 | 1/1 | proboscis extension (FlyWire CB0701, Shiu et al.'s MN9) |
| legs | every nerve-cord motor neuron of one leg's neuromere (front, middle, hind), by side | 68/67, 58/58, 66/64 | each leg's own motor pool: what `check_legs.py` and the viewer's legs read |

Excluded: LB3b (FlyWire "sugar/low_salt", a mixed type), LB1e ("glutamate") and JO-A/B/D/F (sound
and vibration). The senses are bilateral: the labellum meets a drop with both halves, and a wall or
dust meets both antennae. No readout neuron is one a register drives.

## The answer bridge

At the report's last token (`"):"`, marked `CLS`) the fly's first layer reads Qwen's final hidden
state through its input norm, whose gains and eps are Qwen's final norm's: the vector Qwen's LM head
reads. Register r projects it on 64 x (U_w - U_ref), Qwen's unembedding row of its word minus the row
of a reference word, so it holds s_w, Qwen's logit of the move over the reference. A register written
on a text token has no ballast, and layer 1's per-head norm eps is 2^42, so it is divided by exactly
2^21. The convolution's SiLU passes it unchanged above 0.27 and keeps it within 0.005 of 0 below 0,
so the register holds max(s_w, 0). The same token sets the register heads' decay to 0 before they write
(their decay input reads the CLS marker, exp(-softplus(2e9 CLS - 1e9 TXT - 200)), which is 0 there and 1
on every other token), so the write is exact whatever the register held: a `DRIVE` token's value is
~2^18 times larger inside the head and would otherwise round away its low bits. The tonic rows turn
the register's margin m_w into a drive level, clip(15 x (m_w - theta_w) / delta_w, 0, 15):

| command | descending neurons | reference | margin m | theta | delta |
|---|---|---|---|---|---|
| forward | P9 | stop | s - relu(s_left - 3) - relu(s_right - 3) - relu(s_groom) | 2 | 0.9375 |
| left, right | DNa02 | stop | s - relu(s_other) - relu(s_groom) | 0 | 0.234 |
| jump | giant fiber | forward | s | 1.5 | 0.234 |
| fly | DNg02 | forward | s | 0 | 0.9375 |
| back | MDN | forward | s - relu(s_groom) | 1.5 | 0.234 |
| groom | aDN | stop | s | 0.25 | 0.234 |

The steering pair inhibit each other, so at most one side turns, by Qwen's margin for it over both
stop and the other side. Both inhibit forward, so a strong pull to one side stops the walk and the
fly turns in place; in the connectome, P9 and DNa02 held together at middling levels suppress each
other's pools, and the fly neither walks nor turns. Groom inhibits every walking register: a grooming
fly stands, and Qwen often answers back or right for dusty antennae, which would walk the fly away
from the grooming it needs. Back, jump and fly are read against forward (the fit chooses each
reference). Read against stop, back would fall as soon as the antennae left the wall while it was
still within 3 mm, and the forward pool's rebound would walk the fly back into it. In the air, Qwen's
logit of fly over stop is near zero whenever it turns, so read against stop the wings would stop on
about half its turns; its logit of fly over forward separates flight from the floor and keeps them
beating through every turn, and through a stop answer too: Qwen answers stop for many reports right
after the jump, while the wings are still, and that drive is what starts them. The cost is that Qwen
cannot land the fly on the perch ([related.md](related.md#open-directions)).

Every parameter is fitted by least squares to the pilot's levels (15 x the targets in
`flyweight/report.py`), over the synthetic reports where Qwen's own move is the pilot's (292 of 400,
`scripts/calibrate/check_answers.py`). Fitting only where they agree keeps the map from reversing
Qwen's other choices. The fit sets, for jump and groom, each one's reference (stop or forward), theta
and delta; then groom's offset c_g together with the walking registers: one delta for the steering
pair, forward's offset c, theta and delta, and back's reference, theta and delta. Fly is set by rule:
read against forward from no offset, at the slope that takes a 0.6 nat margin (Qwen's margin just
after a jump) to level 9.6, where DNg02 already holds the fly up (`check_motor.py`). Every report on
the floor stays below -2 nats, at level 0. Least squares, which weighs every report alike, put those
first moments in the air below the power that holds the fly up.

Each slope 15 / delta is a power of two and each theta a multiple of 1/4, so `DRIVE` tokens (the
oracle control, a visitor's drive) write the same registers exactly. A `DRIVE` token's level 0 writes
0, the bottom of the register's dead zone, so an idle register inhibits nothing. Qwen reads each report
from a saved post-instructions state of its layers; the fly's state is left as it is. The instructions
state the rules and hold 28 worked reports: on the floor, back when the arena wall is within 3 mm
ahead; groom when the antennae are touching with the wall farther away (dust); otherwise turn toward
the sugar or walk to it, stop on it until fed, then jump, fly toward the perch and stop over it.

**Memory** (`QwenFlyRunner(memory=N)`, `--memory N`). Without it, every report is read from the
post-instructions state, so Qwen sees one moment. With it, Qwen reads each report after its own log:
each earlier report and its answer (the move word and a newline) stay in its cache, and the answer
just given is read before the next report. When the log reaches 2N entries, Qwen's layers go back to
the post-instructions state and re-read the last N, so the context stays between N and 2N reports.

The control bridge reads six groups of 8 of Qwen's MLP neurons (L#N#,
`scripts/calibrate/discover_drive.py`) into the `SIT` dims through extra down-projection rows, and a hand-written map turns them into the
same registers.

## Printed senses (`flyweight/report.py`)

Inside each report the host writes " Its neurons: taste", ", proboscis", ", antennae" and ", wings".
After each, a `SAY_k` token runs through the fly's layers (it writes nothing) and the greedy next
token over the whole vocabulary is the fly's word, a real text token. LM-head rows add 2^16 at their
own slot plus 1/8 per Hz of a role's rate above a threshold: " sweet" (sugar neurons, 20 Hz) or
" bitter" (bitter neurons, 20 Hz), else " nothing"; " extended" (MN9, 15 Hz) or " folded"; " touching"
(Johnston's organ, 10 Hz) or " free"; " beating" (flight power, 100 Hz: flight drives it past 130 Hz,
walking below 50) or " still". On text tokens the rows read only zero dims; on TICKs they sit ~2^13
below the fly's rows.

## Body and world (`flyweight/flyworld.py`)

Outside the neurons, as in NeuroMechFly or flybody. The world writes the sense registers: taste
neurons while the labellum is on a drop (sugar level 13, bitter 4), Johnston's organ while the
antennae meet the arena wall ahead (within 1.5 mm) or carry dust (level 3). Every report on the floor
states the distance to the wall ahead.

On the floor the body moves from the MOVE rows. Forward comes from the forward pool, minus what
backward drive alone recruits in it, and backward from the backward pool (14 and 8 mm/s at full
drive). Turning comes from the turn pools, each minus what forward drive alone recruits in it at the
forward pool's present rate (`scripts/calibrate/calib_forward.py`, P9 at every level, fitted
monotone, and `scripts/calibrate/calib_body.py` for the top levels and MDN). The grooming pool makes
the fly stand, and 0.3 s of full grooming cleans the dust off. The fly takes off when both jump motor
neurons pass 55 Hz (under the giant fiber the left one reaches 70-83 Hz and the right one ~120; a
stray spike reaches ~35), and feeds while MN9 fires on the sugar.

In flight the power motor neurons set the ground speed (120 mm/s at full power) and the height the
fly holds (20 mm at full power, lower with less, climbing or sinking at up to 150 mm/s), both through
the body's inertia. Below the power that holds the fly up (0.42 of full) the wings stop and it falls;
after DNg02 lets go, the flight circuit leaves the power motor neurons at 35-55 Hz. Under DNg02 the
steering motor neurons b1 and b2 are silent (each takes ~6,000 inhibitory synapses that flight drive
recruits; driven alone they fire at 130-150 Hz), and one side's DNg02 shifts the two sides' power
motor neurons by only 3-5%. Yaw therefore comes from the turn pools: each pool's excess over what
steady flight leaves in it (1.2 and 0.004 Hz), against what a full DNa02 command adds with the wings
powered (14.4 and 1.5 Hz, against 34 and 32 on the floor; `calib_body.py`), through the body's
rotational inertia (0.1 s). The jump recruits both pools for its first ~250 ms of flight, the right
one past a full right command, so the fly yaws ~30 degrees right as it takes off. Within 12 mm of a
wall a flying fly turns away in a saccade, faster the nearer the wall and the more directly it heads
into it (up to 1,500 deg/s, from 7 mm), until it heads away at 30 degrees or more. The held height and
the saccade stand in for the visual and haltere control that the model has no input for.

The six leg pools are logged beside the body and scale each leg's stride in the viewer. The viewer
draws stepping and the wingbeat; the LIF connectome has no rhythm generator
(`scripts/experiments/probe_rhythm.py`).

**The body in integers** (`flyweight/bodyint.py`). The body and the senses in integer arithmetic
(adds, multiplies, shifts, compares and one 65,536-entry sine table): positions in 2^-20 mm, heading
in 2^-24 turns, fractions in Q16, rates in 1/1024 Hz, divisions rounded to nearest. A circuit of stock
layers can compute this form exactly, so it is the first step toward moving the body into the
weights; the running body is the float one. `scripts/checks/check_bodyint.py` runs both on the same
random motor streams.

## Checks and pass criteria

| check | passes when |
|---|---|
| `check_wiring.py` (CPU, the built weights) | every set is the types above and bilateral where stated; the connectome checks above hold; every register reaches exactly its members with their gains, gating and inhibition; DRIVE tokens read back exactly; every connection, MOVE row and printed word reads exactly its source; the answer bridge's coefficients are Qwen's unembedding rows |
| `check_fly_layers.py` | random 400-neuron connectome, 400 ms of random register schedules: no step differs from the float64 executor; whole CNS: the first difference is a threshold tie (both sides within 1e-3 mV) |
| `check_tonic_brian.py` | the executor's tonic drive equals Brian2 running Shiu's equations, spike for spike |
| `check_merged.py` | fly and printed-word logits of the fast path equal the full forward's; text fidelity against base Qwen; the fly inside Qwen equals the fly alone |
| `check_bridge.py` | registers written by Qwen's answer equal float64 from the residual; the printed words equal the host's rule on the fly's rates; the fly's layers on report-end tokens only equal them on every text token |
| `check_checkpoint.py` | the model loaded from its safetensors checkpoint alone replays a test stream bit for bit |
| `check_graphs.py` | the fly's steps replayed as CUDA graphs equal eager stepping, spike for spike and in the motor readout, for one fly and a batch of four |
| `check_body.py` | on the fly's own layers, read by the body's code: forward walks straight, each turn turns its way alone and while walking, back walks backward, groom grooms, the giant fiber takes off, DNg02 powers the wings, rest does nothing |
| `check_motor.py` | each command in the body's units: forward faster with level and straight (heading drift within 12 deg/s at every level), each turn its own way (45 deg/s in place), back at 4 mm/s, groom grooms, no walking command takes off; the giant fiber hops and lands, with DNg02 at 9 the fly leaves the floor and at 12 or 15 it stays aloft and covers ground, flying on it holds its heading within 30 deg/s, a turn command in flight turns it its own way at 90 deg/s (yaw rates in flight leave out the wall saccades), released it lands, and a long flight spends under a fifth of its time at a wall |
| `check_persist.py` | each walking and turning command's readout falls below 20% of its peak within 150 ms of release (the grooming latch is reported, not required) |
| `check_bodyint.py` (CPU) | the integer body against the float body on the same random motor streams: positions within 0.05 mm, headings within 0.01 rad, height within 0.05 mm, modes and senses equal on all but 0.5% of ms, dust within 0.01 |
| `check_live.py` | the live loop through every control: Qwen drives, a message goes in and is cleared, each command the visitor takes moves the body its way, reset and pause work, the registers hold the levels set; memory stays flat over the run |
| `tests/` (pytest, CPU) | sparse products give each row's own numbers; the fly matches the float64 executor spike for spike; the tonic rows stay shut off a TICK and for a refractory member whatever the register holds; inhibited registers drive by their margin over their inhibitors; MOVE tokens are inert; the fly-token norm equals the reference; the body's walking, backing, grooming and senses; each report situation's words and its pilot move |

Reported without a pass criterion: `check_answers.py` (Qwen's move against the pilot's, and the fitted map's error
against the pilot's levels, in sample and held out); `check_messages.py` (what a visitor's message does
to Qwen's move and the register levels); `check_legs.py` (each command's leg motor pools against four
predictions from the literature, each also on five degree-preserving wiring shuffles); `probe_rhythm.py`
(a walking rhythm: a 2-30 Hz peak, promotors and remotors in antiphase, absent on shuffles);
`sweep_adapt.py` (the grooming latch and each command's pool against the adaptation strength).

## Trials and controls (`scripts/experiments/trials.py`)

One Qwen and one fly; episodes run one after another, the fly reset between them. Conditions: answer
(Qwen's answer drives), groups (the control bridge), constant (the residual the fly reads replaced
by its mean over 40 walking reports: the same drive every report), oracle (the host pilot writes
`DRIVE` tokens from what the report states, its bearing rounded; in the air it stops once the perch
is within 10 mm, the glide a fly needs from full power down to the perch's top, where Qwen's
instructions say to stop over it) and none. Oracle and none read no text, so they run the fly's
layers alone, the same computation step for step. Orienting: 500 ms on
the floor with the sugar 10 mm away at +-30..150 degrees; scored on the signed heading change (more
than 10 degrees toward or away from the drop's side). Walking: up to 3 s with the drop 7 mm away at a
random bearing; did the labellum reach it? Grooming: up to 600 ms with dust on the antennae; were they
cleaned (the none condition measures the connectome's own Johnston's organ -> aDN reflex)? Backing:
up to 1 s facing a wall with the antennae against it; did the head get 3 mm clear, and by walking
backward? Perching: up to 2 s, fed, on the floor beside the drop; did the fly take off, fly to the perch
(a 12 mm post ~18 mm away) and land on its top? Success needs the giant fiber's jump, wing power from
DNg02, steering in the air and the wings stopped in time for the fly to glide down onto the top.
Requested grooming: up to 300 ms with clean antennae and a visitor's "Groom your antennae." in every
report (`--say`); did the grooming pool reach half strength? Johnston's organ is silent, so the
reflex is off and only Qwen's answer can make the fly groom. `--memory N` and `--adapt` run the
variants. Rates carry Wilson 95% intervals.

Shiu et al.'s figures (`scripts/experiments/shiu_battery.py`): sugar GRNs to MN9, sugar with bitter,
water, Johnston's organ to the DNg12 family, and silencing MN9's strongest excitatory input types, in three
columns: Shiu's model on FlyWire, the male CNS with the same Poisson input, and the fly inside Qwen.

Text benchmarks (`scripts/experiments/text_eval.py`): MMLU, ARC-Challenge, HellaSwag, GSM8K, IFEval,
wikitext perplexity and a needle at 8k tokens, for base Qwen (sdpa), base Qwen (eager, the noise floor) and
the merged model, paired.

## Live (`scripts/run/serve.py`, `flyweight/live.py`)

The one-stack model runs one 20 ms window at a time while anyone watches, and streams each window to
the browser in the viewer's formats (`flyweight/viewdata.py`). A visitor who joins starts from the
newest window. The fly starts with clean antennae. With Qwen driving, Qwen reads the report
and its answer writes the command registers, as in a session. A visitor can type anything. Their words
go into every report, after the world's part, as `The user says: "..."`, until they replace or clear
them. Qwen's instructions are the ones every run uses and never mention a visitor, so the words reach
the fly only through what they do to Qwen's state at the report's end, where the fly's first layer
reads its move logits. Qwen writes no reply: its only output is the move. A visitor can also take the
command registers (`DRIVE` tokens, as the oracle condition writes them), add to the sense registers,
dust the antennae, reset the fly or pause the model. `scripts/experiments/check_messages.py` measures
what messages do to Qwen's move and to the fly's register levels over random world states.

## Outside the weights

The world, the body and the report's scaffold (as in NeuroMechFly); Qwen's full attention ignoring
fly positions (the attention mask); and the reset of Qwen's cache to its post-instructions state
before each report.
