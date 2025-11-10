/* PufferLib binding for F16 Waypoint Environment
 * This file provides the Python interface following PufferLib conventions
 * 
 * Action Space (4D continuous):
 *   [0] Nz     - Normal acceleration command (G's)
 *   [1] ps     - Roll rate command (rad/s)
 *   [2] Ny_r   - Lateral acceleration command (G's)
 *   [3] throttle - Throttle command (0-1)
 * 
 * Observation Space (28D):
 *   See OBSERVATION_SPACE.md for detailed description
 *   [0-18]  F16 state (angles as sin/cos, velocities, rates, altitudes)
 *   [19-21] Previous action (Nz, ps, Ny_r)
 *   [22-27] Waypoint in spherical coordinates (azimuth, elevation, range)
 */

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "f16_waypoint.h"

#define Env F16Waypoint
#include "../env_binding.h"

/* Helper function to unpack float from kwargs with default value */
static float unpack_float(PyObject* kwargs, const char* key, float default_value) {
    if (kwargs == NULL) {
        return default_value;
    }
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        return default_value;
    }
    if (PyFloat_Check(val)) {
        return (float)PyFloat_AsDouble(val);
    }
    if (PyLong_Check(val)) {
        return (float)PyLong_AsLong(val);
    }
    return default_value;
}

/* Helper function to unpack int from kwargs with default value */
static int unpack_int(PyObject* kwargs, const char* key, int default_value) {
    if (kwargs == NULL) {
        return default_value;
    }
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        return default_value;
    }
    if (PyLong_Check(val)) {
        return (int)PyLong_AsLong(val);
    }
    if (PyFloat_Check(val)) {
        return (int)PyFloat_AsDouble(val);
    }
    return default_value;
}

/* Initialize environment with Python kwargs
 * Called once per environment instance
 */
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    /* Extract parameters from Python kwargs */
    env->step_size = unpack_float(kwargs, "step_size", 1.0f / 30.0f);
    env->time_limit = unpack_float(kwargs, "time_limit", 100.0f);
    env->seed = unpack_int(kwargs, "seed", 0);
    
    /* Seed C random number generator */
    if (env->seed == 0) {
        env->seed = (unsigned int)time(NULL);
    }
    srand(env->seed);
    
    /* Initialize log to zero */
    memset(&env->log, 0, sizeof(Log));
    
    return 0;
}

/* Export log data to Python dictionary
 * Called periodically to report episode statistics
 */
static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
