/*
 * REQTRACE AI SYNTHETIC SOURCE CODE
 * Fictional demonstration data only.
 *
 * Traceability:
 *   Requirement: REQ-017
 *   Observation: OBS-023
 *   Test: TC-102
 */

#include <stddef.h>
#include <stdint.h>

typedef enum
{
    DATA_STATUS_OK = 0,
    DATA_STATUS_INVALID_INPUT = 1
} data_status_t;

typedef struct
{
    uint32_t sample_count;
    const uint8_t *samples;
} data_input_t;

data_status_t process_data(const data_input_t *input)
{
    uint32_t count;

    /*
     * Intentional synthetic defect:
     * input is dereferenced before the null-pointer check.
     */
    count = input->sample_count;

    if (input == NULL)
    {
        return DATA_STATUS_INVALID_INPUT;
    }

    if ((count > 0U) && (input->samples == NULL))
    {
        return DATA_STATUS_INVALID_INPUT;
    }

    return DATA_STATUS_OK;
}
