# Related work and open directions

## Related work

- **Embodied whole-brain LIF flies.** Shiu et al.'s model on FlyWire drives NeuroMechFly in several
  projects (Eon Systems; Minnick et al., on the BANC connectome). In each, the walking rhythm comes from
  a hand-built controller and the neurons scale it. Minnick et al. read six per-leg motor-neuron pools
  as per-leg amplitudes, as FlyWeight's leg pools are read.
- **A connectome-derived walking rhythm.** Pugliese et al. found a delayed-inhibition loop in the nerve
  cord (DNg100 -> IN17A001 -> INXXX466 -> IN16B036) that oscillates in a rate model with size-scaled
  excitability, antiphase on the coxa promotor and remotor motor neurons. `scripts/experiments/probe_rhythm.py`
  looks for it in Shiu's uniform LIF model on the male CNS.
- **Learned decoders.** Decoders trained on top of a connectome (Fly-Saber, FlyGM, the Digital Sphinx)
  can make shuffled or foreign wiring perform as well as the real one, so credit to the connectome
  needs wiring-shuffle controls. FlyWeight's body has no learned parameters, and `check_legs.py` and
  `probe_rhythm.py` run shuffles.
- **Programs compiled into stock LLM layers.** QwenPU compiles a RISC-V CPU and a vector unit into
  stock Qwen3.5 layers in front of Qwen3.5-4B, and FlyWeight's method comes from it (the README's
  [Relation to QwenPU](../README.md#relation-to-qwenpu) compares the two technique by technique). Both
  work under the same constraint: within one forward pass information flows only up the layers, so
  the fly reaches Qwen through tokens, the printed words.

## Open directions

1. **The body inside the weights.** `flyweight/bodyint.py` is the body and senses as integer arithmetic.
   Compiled into stock layers after the fly's layer 3, body tokens would commit the state every ms and
   sense slots would emit the `DRIVE` tokens that write the sense registers, checked in lockstep against
   the integer reference. The host would then inject only TICKs, slots and the report scaffold.
2. **Steering in flight through the wings.** Flies steer in flight by left-right differences in wing
   stroke, which the DNg02 population and the steering motor neurons set (Namiki et al. 2022). In Shiu's
   model on the male CNS the steering motor neurons b1 and b2 never fire under DNg02 or DNa02 drive (each
   takes ~6,000 inhibitory synapses that flight drive recruits), and DNg02 on one side moves the two sides'
   power motor neurons apart by only 3-5%. The body therefore reads a turn in flight from the turn pools,
   which DNa02 still reaches with the wings powered, against what steady flight leaves in them. A model of
   the wing motor system's own dynamics would let the wings steer the fly.
3. **A stop that stops the wings.** The answer bridge reads fly against forward, so the wings keep beating
   while Qwen turns in the air, and also when it answers stop: Qwen answers stop as the fly nears the
   perch, and the fly flies on over it. A register for Qwen's logit of stop over fly, inhibiting DNg02
   unless a turn also beats stop, separates Qwen's stop answers in the air from its turns, but on its own
   it grounds the fly: Qwen answers stop for many reports right after the jump, while the wings are still,
   and the drive the current bridge gives those reports is what starts the wings. It needs Qwen to answer
   fly after a jump and to stop about 10 mm before the perch, the glide a fly needs from full power; asking
   Qwen3.5-4B for the early stop in its instructions made it turn instead and lean left in flight.
4. **A richer printed channel.** More SAY slots for more neuron groups, or graded rates as digits, still
   chosen by the fly's neurons.
5. **More commands.** Graded steering (DNa01, DNg13), song (pIP10 and the wing motor neurons), landing
   (DNp10, DNp07) and the feeding motor neurons. Each needs a selective readout, a move word with worked
   examples and its checks.
6. **Speed.** A report read in one text forward instead of five, since the fly's words do not depend on
   Qwen's text, and a fly path on a compact residual. Token-level stepping on one GPU runs about 70 times
   slower than real time.
7. **A physics renderer.** MuJoCo in the browser posed from the model's joint angles, with no feedback
   into the model.

## Sources

- Shiu et al. 2024, doi 10.1038/s41586-024-07763-9; code at github.com/philshiu/Drosophila_brain_model.
- Minnick et al., arXiv 2610.33881.
- Pugliese et al., doi 10.1038/s41586-026-11195-y.
- NeuroMechFly v2, doi 10.1038/s41592-024-02497-y; flybody, doi 10.1038/s41586-025-09029-4.
- Cheong et al. (MANC), doi 10.7554/eLife.96084; Braun et al. 2024, doi 10.1038/s41586-024-07523-9.
- Fly-Saber, github.com/jackie-0110/Fly-Saber; the Digital Sphinx, elifesciences.org/reviewed-preprints/111516v1.
- QwenPU, github.com/divagr18/QwenPU.
