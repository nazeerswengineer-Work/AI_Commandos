/*
 * REQTRACE AI SYNTHETIC PARTIAL SOURCE CODE
 * Fictional demonstration data only.
 *
 * Traceability:
 *   Requirement: REQ-033
 *   Observation: OBS-044
 *   Test: TC-301
 *
 * Limitation:
 *   This file intentionally omits the final mode-assignment implementation.
 */

#include <stdbool.h>

typedef enum
{
    MODE_REQUEST_REJECTED = 0,
    MODE_REQUEST_ACCEPTED = 1
} mode_request_status_t;

mode_request_status_t request_active_mode(
    bool enable_request_valid,
    bool primary_ready)
{
    if ((!enable_request_valid) || (!primary_ready))
    {
        return MODE_REQUEST_REJECTED;
    }

    /*
     * The secondary readiness check and final mode assignment are not
     * included in the supplied evidence.
     */
    return MODE_REQUEST_ACCEPTED;
}
