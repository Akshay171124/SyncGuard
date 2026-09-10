# Consecutive-failure guard for SLURM auto-resubmit loops.
#
# On 2026-04-07 a resubmit trap produced ~40 identical failing jobs because
# the in-process guard flag did not survive across submissions. This keeps
# the count in a file so it persists between jobs.
#
# Reset the counter whenever real progress is made (an epoch completes), so
# a long multi-job run is never throttled — only a genuine crash-loop is.

_guard_file() {
    mkdir -p outputs/logs
    echo "outputs/logs/.resubmit_count_${1}"
}

guard_count() {
    local f
    f="$(_guard_file "$1")"
    if [ -f "$f" ]; then cat "$f"; else echo 0; fi
}

guard_record_failure() {
    local f n
    f="$(_guard_file "$1")"
    n=$(guard_count "$1")
    echo $((n + 1)) > "$f"
}

guard_reset() {
    local f
    f="$(_guard_file "$1")"
    echo 0 > "$f"
}

# guard_may_resubmit <name> <max>  -> exit 0 if allowed, 1 if limit reached
guard_may_resubmit() {
    local n max
    n=$(guard_count "$1")
    max="${2:-3}"
    if [ "$n" -ge "$max" ]; then
        echo "GUARD: ${n} consecutive failures for '$1' (limit ${max}). Not resubmitting." >&2
        return 1
    fi
    return 0
}

# guard_note_progress <name> <marker>
# Reset the failure counter when the progress marker has changed since the
# last job. A crash-loop produces no new checkpoint, so its marker is
# unchanged and the counter keeps climbing; a healthy resume produces a new
# checkpoint, so the counter clears and a long multi-job run is never throttled.
guard_note_progress() {
    local f marker stored
    f="$(_guard_file "$1")_marker"
    marker="$2"
    stored=""
    [ -f "$f" ] && stored="$(cat "$f")"
    if [ "$marker" != "$stored" ]; then
        echo "$marker" > "$f"
        guard_reset "$1"
    fi
}
