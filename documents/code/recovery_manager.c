/*
 * REQTRACE AI SYNTHETIC ORIGINAL-BASELINE SOURCE CODE
 * Fictional demonstration data only.
 *
 * Traceability:
 *   Requirement: REQ-060
 *   Observation: OBS-070
 *   Test: TC-401
 *
 * Closure limitation:
 *   No corrected implementation is supplied in the Phase 3 evidence set.
 */

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    uint32_t recovery_counter;
    bool recovery_active;
} recovery_state_t;

void start_recovery(
    recovery_state_t *state,
    bool request_valid)
{
    if ((state == NULL) || (!request_valid))
    {
        return;
    }

    /*
     * Intentional synthetic defect:
     * recovery_counter is not initialized to zero.
     */
    state->recovery_active = true;
}
