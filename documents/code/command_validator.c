/*
 * REQTRACE AI SYNTHETIC SOURCE CODE
 * Fictional demonstration data only.
 *
 * Traceability:
 *   Requirement: REQ-021
 *   Observation: OBS-031
 *   Test: TC-201
 */

#include <stdint.h>

#define COMMAND_ID_MINIMUM 1U
#define COMMAND_ID_MAXIMUM 16U

typedef enum
{
    COMMAND_VALID = 0,
    COMMAND_INVALID_ID = 1
} command_status_t;

command_status_t validate_command(uint8_t command_id)
{
    if ((command_id < COMMAND_ID_MINIMUM) ||
        (command_id > COMMAND_ID_MAXIMUM))
    {
        return COMMAND_INVALID_ID;
    }

    return COMMAND_VALID;
}
