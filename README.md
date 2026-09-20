# Operator Time

Working notebook and executable toy for a thought that became visible while looking across the recent neuron / operator / self / timing repos:

> **The important object may not be a fixed operator acting on changing data. The current state and its history may determine which operator exists *now*.**

The repository slug `emaNYrotisopeR` is left untouched. Inside it, the working name is **Operator Time**.

This is not a theory of consciousness, not a claim that transformers and brains are the same machine, and not a claim that biological frequency is mysterious. The goal is narrower: write down the recurring computational motif cleanly enough that we can attack it.

## The prime

Keep the substrate parameters fixed:

```
theta_(t+1) = theta_t
```

but allow resident state to change the effective operator:

```
r_(t+1) = lambda r_t + write(event_t, perspective_t, object_t)

O_t = O_theta(
    self_anchor,
    resident_state=r_t,
    current_perspective,
    current_event
)

x_(t+1) = O_t x_t
```

Then it is perfectly possible to have:

```
theta_(t+1) = theta_t
O_(t+1) != O_t
```

**Operator time** is the trajectory through those effective operators.

Wall-clock time says *when* a computation happened.

Operator time asks:

> **What computation was available at that moment, given everything still resident from the path that led there?**

That distinction is the center of this repo.

## The transformer analogy

A transformer is a useful mathematical example because its learned parameters may stay fixed while its effective routing changes with context.

Very schematically:

```
A(X) = softmax(Q(X) K(X)^T)
Y    = A(X) V(X)
```

The learned projection matrices can be unchanged while `A(X)` changes as the resident context changes.

In autoregressive inference, past token keys and values remain addressable in the KV cache. That cache is **not biological short-term memory**, but it gives a clean digital example of the shape we care about:

```
new residue enters resident context
        ↓
future routing changes
        ↓
future effective operator changes
```

Likewise a residual stream has the generic form:

```
resident state + computed residue
```

and the next layer acts on the changed resident state.

The analogy is structural, not an equivalence.

## The biological question

For the biological line, the recurring question is no longer:

> where is the weight?

It is:

> **what determines the operator available at this instant?**

A point-neuron abstraction mostly hides that question in weights.

A spatial dynamical neuron/circuit could make the answer depend on some mixture of:

```
geometry
membrane / conductance state
synaptic and axonal connectivity
recent input
resident biochemical/electrical state
frequency / mode
phase / coherence
event-relative time
perspective / route
neuromodulatory context
developmental history
```

Frequency does not have to be magic. Structured matter has modes. The scientific question is how living matter creates, modifies, couples and uses them.

The old cable / resonator / whorl / frequency-addressed repos can therefore be read as different attacks on the same problem:

> **How does physical or simulated structure make only some operations accessible now?**

## SELF and transient objects

The morning thought that seeded this repo was also simple:

> **SELF may be the unusually persistent object/reference process; most other objects come and go.**

That does *not* require a magical SELF vector.

There can be a representation **of** self, like there are representations of mouse, person, place or tool.

Separately, there can be a persistent reference process relative to which relations such as these are evaluated:

```
here / there
mine / yours
I acted / another acted
danger to me
what would the other see?
what happened before this?
```

A transient OTHER perspective can be instantiated, do work, and leave residue without replacing the persistent control origin:

```
SELF
  -> object
  -> OTHER perspective
  -> computation
  -> SELF'
```

with:

```
SELF' is continuous with SELF
SELF' is not unchanged
```

That connects directly to **SelfAndOtherObjectsInTime**.

## Frozen residues: words, books and inherited operators

There is a more speculative but useful lens here.

A written sentence is not an operator in the strict mathematical sense. But it is a **frozen residue of a trajectory through another cognitive system**.

When read later, it can alter the reader's resident state and therefore alter the operations available next.

So the chain can be pictured as:

```
earlier minds / culture / language
        ↓
filtered expression
        ↓
frozen text
        ↓
current resident state
        ↓
new effective operators
        ↓
new expression
```

Plato is not literally running in the reader. But some structure produced by his process can still perturb another process thousands of years later.

The same applies recursively: the writer was already shaped by earlier people, language and culture.

That suggests a **pyramid of historical operator constraints**:

```
evolution
development
circuit construction
learning
long-term memory
recent event history
current resident state
current cue
        ↓
operator available now
```

This is a metaphor until a level is formalized. The repo will keep that distinction explicit.

## Relation to the older line

This is not a new start. It is an attempt to name the machine that kept reappearing.

- **GAx / ThirdWay** — algorithmic populations and routes can diverge under repeated transforms; context changes which mode/path is amplified.
- **FrequencyAddressedState-dependentOperatorComposition** — made state-dependent operator composition explicit.
- **SimpleNeuron / FusionMachine** — resident state changes what later input can do.
- **Sihti / SighImageFactorization** — repeated dynamics separate/preserve components and residue rather than treating processing as erase-and-replace.
- **WhatToLookAt / AnotherOddThing / ReadWrite** — the current state can actively determine what should be observed or probed next.
- **SelfAndOtherObjectsInTime** — gives resident computation event ownership, reference frames, nested local time and consequence-dependent temporal bandwidth.
- **ArtificialCortex / the_whorl** — points toward phase/field coordinates generated by substrate rather than handed in as software variables.

None of those repositories proves the larger synthesis. They provide executable pieces that motivated it.

## Gate 1 — operator time without weight change

The first executable test asks for the minimum possible claim.

A fixed synthetic substrate contains:

- one permanent **self anchor**;
- transient object vectors;
- a decaying resident state;
- fixed basis operators;
- a fixed map from resident state to operator mixture coefficients;
- one identical current probe.

There are **no parameter updates** during an episode.

Two trajectories can arrive at the same current probe with different recent histories. If operator time is real in this toy, the same present should be processed by different effective operators.

Across 64 deterministic worlds:

| mechanism / diagnostic | mean |
|---|---:|
| dynamic resident-state readout accuracy | **0.9924** |
| fixed-operator baseline | 0.4956 |
| erase-residue baseline | 0.4972 |
| relative operator distance after reversing identical history multiset | **0.0438** |
| current-output distance after reversing history | **0.2196** |
| self-anchor fidelity | **1.0000** |
| parameter drift | **0.0000** |

The external linear readout is only a probe: it asks whether the recent transient object class is still accessible in the **current operator response**. The operator factory itself is fixed and untrained.

The important matched attack is history reversal:

```
A -> B -> same current probe
B -> A -> same current probe
```

The object multiset is identical. The current input is identical. Learned/substrate parameters are identical.

Only the trajectory differs.

Yet the effective operator differs.

That is the smallest executable statement of **operator time**.

## What Gate 1 does not establish

It does not establish that:

- transformers implement this toy mechanism;
- brains use this exact resident-state equation;
- the self is one vector;
- biological rhythms are operator addresses;
- eigenmodes explain thought;
- text literally stores operators.

Those remain hypotheses, analogies or future tests.

## Next gates

The next useful attacks are already visible:

1. **Perspective residue** — hold current object and SELF fixed; visit another perspective; ask whether return changes the operator without moving the anchor.
2. **Frozen external history** — feed a fixed stored sequence into two identical systems and test whether it reproducibly seeds a later operator trajectory.
3. **Route divergence** — start from the same operator family, give branches different histories, then test whether their accessible operator families separate.
4. **Oscillatory address** — replace a software route tag with locally generated phase/frequency state.
5. **Event ownership** — combine this operator-time formulation with nested clocks from SelfAndOtherObjectsInTime.
6. **Slow operator time** — let the operator factory itself change through plasticity/development, creating nested timescales.

## Run

```bash
python -m pip install -r requirements.txt
python experiment.py --assert-gate --seeds 64
pytest -q
```

The deterministic receipt is committed at `results/gate1.json`.

---

The working question is now:

> **What determines the operator available at this instant — and what parts of the past are still physically or computationally present enough to determine it?**
