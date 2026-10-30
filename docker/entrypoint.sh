#!/bin/sh
# Apply this container's baseline network profile, then hand off to the component.
#
# The profile is applied here, inside the container at startup, rather than from
# the host after the fact. Applying it from outside leaves a window in which the
# first requests traverse an unshaped link, which would contaminate the early
# samples of every run.
#
# Layout (verified on the WSL2 kernel: sch_tbf and sch_netem are both present):
#
#   root  1:   tbf    -- the rate cap / token bucket
#     └── 10:  netem  -- delay, jitter, loss
#
# netem shapes EGRESS only, so a one-way hop A->B costs A's delay and the HTTP
# response B->A costs B's delay. Per-tier delay is therefore set to that tier's
# one-way link latency l, and a request/response round trip over one hop costs
# l_A + l_B. The analysis measures one-way hop time directly from the shared
# host clock, so it reports l, not the round trip.
set -e

DEV="${TC_DEV:-eth0}"

if [ -n "${TC_ENABLE}" ]; then
    RATE="${TC_RATE_MBIT:-1000}"
    DELAY="${TC_DELAY_MS:-0}"
    JITTER="${TC_JITTER_MS:-0}"
    LOSS="${TC_LOSS_PCT:-0}"
    BURST="${TC_BURST_KBIT:-32}"

    tc qdisc del dev "$DEV" root 2>/dev/null || true
    tc qdisc add dev "$DEV" root handle 1: tbf \
        rate "${RATE}mbit" burst "${BURST}kbit" latency 50ms

    NETEM="delay ${DELAY}ms"
    if [ "$JITTER" != "0" ]; then NETEM="${NETEM} ${JITTER}ms distribution normal"; fi
    if [ "$LOSS" != "0" ]; then NETEM="${NETEM} loss ${LOSS}%"; fi

    # shellcheck disable=SC2086
    tc qdisc add dev "$DEV" parent 1:1 handle 10: netem $NETEM

    echo "[tc] $(hostname) dev=$DEV rate=${RATE}mbit delay=${DELAY}ms jitter=${JITTER}ms loss=${LOSS}%" >&2
    tc qdisc show dev "$DEV" >&2
fi

exec python -u -m "$COMPONENT_MODULE"
