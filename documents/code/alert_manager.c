/*
 * REQTRACE AI SYNTHETIC SOURCE CODE
 * Fictional demonstration data only.
 *
 * Candidate traceability:
 *   Requirements: REQ-041, REQ-042, REQ-050
 *   Observations: OBS-052, OBS-061
 */

#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    ALERT_LOW = 0,
    ALERT_HIGH = 1
} alert_severity_t;

typedef struct
{
    bool active;
    bool priority_indicator;
    uint32_t alert_id;
} alert_state_t;

void process_alert(
    alert_state_t *state,
    uint32_t alert_id,
    alert_severity_t severity)
{
    if (state == NULL)
    {
        return;
    }

    state->active = true;
    state->alert_id = alert_id;

    if (severity == ALERT_HIGH)
    {
        state->priority_indicator = true;
    }
}

void reset_alert(
    alert_state_t *state,
    uint32_t measured_elapsed_ms)
{
    if (state == NULL)
    {
        return;
    }

    /*
     * The supplied function records the measured time but does not establish
     * which FRD revision applies to the software baseline.
     */
    if (measured_elapsed_ms >= 760U)
    {
        state->active = false;
        state->priority_indicator = false;
    }
}
