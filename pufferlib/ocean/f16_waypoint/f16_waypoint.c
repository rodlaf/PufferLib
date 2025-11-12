/* F-16 Waypoint Neural Network Inference Demo - Standalone C version
 * Build it with:
 * bash scripts/build_ocean.sh f16_waypoint local (debug)
 * bash scripts/build_ocean.sh f16_waypoint fast
 * 
 * This replaces the autopilot with a trained neural network policy.
 * Weights are exported by running: puffer export puffer_f16_waypoint
 */

#include "f16_waypoint.h"
#include "puffernet.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

/* Playback configuration */
#define PLAYBACK_SPEED 1.0   /* 1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed */
#define RENDER_FPS 60        /* Target frames per second for rendering */

/* Box-Muller transform for sampling from Normal distribution */
double randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

/* LinearContLSTM structure for continuous action policies
 * This matches the F16Waypoint Python model structure exactly:
 * - encoder: Linear(input_dim -> 1024)
 * - GELU activation
 * - LSTM(1024 -> 1024)
 * - actor: Linear(1024 -> num_actions)
 * - value_fn: Linear(1024 -> 1)
 * - log_std: trainable parameter (num_actions floats)
 */
typedef struct LinearContLSTM LinearContLSTM;
struct LinearContLSTM {
    int num_agents;
    float *obs;
    float *log_std;
    Linear *encoder;
    GELU *gelu1;
    LSTM *lstm;
    Linear *actor;
    Linear *value_fn;
    int num_actions;
};

LinearContLSTM *make_linearcontlstm(Weights *weights, int num_agents, int input_dim,
                                    int logit_sizes[], int num_actions) {
    LinearContLSTM *net = calloc(1, sizeof(LinearContLSTM));
    net->num_agents = num_agents;
    net->obs = calloc(num_agents * input_dim, sizeof(float));
    net->num_actions = logit_sizes[0];
    
    // Read log_std parameters
    net->log_std = get_weights(weights, net->num_actions);
    
    // Build network: encoder -> GELU -> LSTM -> actor/value
    net->encoder = make_linear(weights, num_agents, input_dim, 1024);
    net->gelu1 = make_gelu(num_agents, 1024);
    
    int atn_sum = 0;
    for (int i = 0; i < num_actions; i++) {
        atn_sum += logit_sizes[i];
    }
    
    net->actor = make_linear(weights, num_agents, 1024, atn_sum);
    net->value_fn = make_linear(weights, num_agents, 1024, 1);
    net->lstm = make_lstm(weights, num_agents, 1024, 1024);
    
    return net;
}

void free_linearcontlstm(LinearContLSTM *net) {
    free(net->obs);
    free(net->encoder);
    free(net->gelu1);
    free(net->actor);
    free(net->value_fn);
    free(net->lstm);
    free(net);
}

void forward_linearcontlstm(LinearContLSTM *net, float *observations, float *actions) {
    // Forward pass: encoder -> GELU -> LSTM -> actor
    linear(net->encoder, observations);
    gelu(net->gelu1, net->encoder->output);
    lstm(net->lstm, net->gelu1->output);
    linear(net->actor, net->lstm->state_h);
    linear(net->value_fn, net->lstm->state_h);
    
    // Sample actions from Normal(mean, std)
    for (int i = 0; i < net->num_actions; i++) {
        float std = expf(net->log_std[i]);
        float mean = net->actor->output[i];
        actions[i] = randn(mean, std);
    }
}

/* Get wall-clock time in seconds */
static double get_wall_time(void) {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

/* Main simulation loop */
int main(int argc, char** argv) {
    srand(time(NULL));
    
    printf("F-16 Waypoint Neural Network Demo\n");
    printf("===================================\n\n");
    
    /* Create environment on stack (not heap) */
    F16Waypoint env;
    env.step_size = 1.0 / 30.0;
    env.time_limit = 100.0;
    env.seed = (unsigned int)time(NULL);
    
    /* Allocate buffers for vectorized API compatibility */
    size_t obs_size = OBS_DIM_TOTAL;  // 28
    size_t act_size = 4;
    float *observations = (float *)calloc(obs_size, sizeof(float));
    float *action_buffer = (float *)calloc(act_size, sizeof(float));
    
    if (!observations || !action_buffer) {
        fprintf(stderr, "ERROR: Failed to allocate memory for buffers.\n");
        free(observations);
        free(action_buffer);
        return 1;
    }
    
    /* Load neural network weights */
    const char* weights_path = "pufferlib/resources/f16_waypoint/f16_waypoint_weights.bin";
    int num_weights = 8431625;  // Exact count from exported weights
    
    printf("Loading weights from: %s\n", weights_path);
    Weights* weights = load_weights(weights_path, num_weights);
    if (!weights) {
        printf("WARNING: Failed to load weights from %s\n", weights_path);
        printf("Using random actions instead.\n");
        printf("To train and export weights:\n");
        printf("  1. Train: puffer train puffer_f16_waypoint\n");
        printf("  2. Export: puffer export puffer_f16_waypoint\n\n");
    } else {
        printf("Weights loaded successfully!\n\n");
    }
    
    /* Create neural network (if weights loaded) */
    LinearContLSTM *net = NULL;
    if (weights) {
        int logit_sizes[1] = {4};  // 4 continuous actions
        net = make_linearcontlstm(weights, 1, obs_size, logit_sizes, 1);
    }
    
    /* Reset environment */
    c_reset(&env);
    
    printf("Starting simulation...\n");
    printf("Initial waypoint: E=%.1f N=%.1f Alt=%.1f\n\n",
           env.waypoint[0], env.waypoint[1], env.waypoint[2]);
    printf("Press ESC to exit\n\n");
    
    /* Timing variables */
    double target_render_dt = 1.0 / RENDER_FPS;
    double last_render_time = get_wall_time();
    double sim_steps_per_frame = (PLAYBACK_SPEED * target_render_dt) / env.step_size;
    
    int waypoint_count = 1;
    int episode_count = 0;
    float total_return = 0.0f;
    
    /* Main loop */
    c_render(&env);
    
    while (!c_should_close()) {
        double current_time = get_wall_time();
        double time_since_render = current_time - last_render_time;
        
        if (time_since_render >= target_render_dt) {
            int steps_this_frame = (int)(sim_steps_per_frame + 0.5);
            
            for (int step = 0; step < steps_this_frame; step++) {
                /* Get observation */
                float obs[OBS_DIM_TOTAL];
                get_observation(&env, obs);
                memcpy(observations, obs, obs_size * sizeof(float));
                
                /* Generate actions */
                if (net) {
                    // Use neural network
                    forward_linearcontlstm(net, observations, action_buffer);
                    for (int i = 0; i < act_size; i++) {
                        env.u_ref[i] = (double)action_buffer[i];
                    }
                } else {
                    // Random actions as fallback
                    for (int i = 0; i < act_size; i++) {
                        env.u_ref[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                    }
                }
                
                /* Step environment */
                c_step(&env);
                total_return += env.reward;
                
                if (env.terminal) {
                    episode_count++;
                    int is_success = (env.log.perf > 0);
                    
                    printf("Episode %d complete | Return: %.2f | Success: %s\n",
                           episode_count, total_return, is_success ? "YES" : "NO");
                    
                    c_clear_trail();
                    
                    if (is_success) {
                        waypoint_count++;
                        printf("  Waypoint %d reached!\n", waypoint_count - 1);
                        printf("  New waypoint %d: E=%.1f N=%.1f Alt=%.1f\n\n",
                               waypoint_count, env.waypoint[0], env.waypoint[1], env.waypoint[2]);
                    } else {
                        printf("  Physics violation or timeout\n\n");
                        waypoint_count = 1;
                    }
                    
                    total_return = 0.0f;
                    break;
                }
            }
            
            c_render(&env);
            last_render_time = current_time;
        }
        
        /* Small sleep to avoid busy waiting */
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 1000000;  /* 1ms */
        nanosleep(&ts, NULL);
    }
    
    /* Cleanup */
    if (net) {
        free_linearcontlstm(net);
    }
    free(observations);
    free(action_buffer);
    c_close(&env);
    
    printf("\nSimulation closed\n");
    printf("Total episodes: %d\n", episode_count);
    printf("Waypoints reached: %d\n", waypoint_count - 1);
    
    return 0;
}
