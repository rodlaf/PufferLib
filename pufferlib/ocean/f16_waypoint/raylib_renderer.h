/*
 * Raylib Renderer - Header-only C implementation
 * Single-file renderer for F-16 simulation visualization
 */

#ifndef RAYLIB_RENDERER_H
#define RAYLIB_RENDERER_H

#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#ifndef RAD2DEG
#define RAD2DEG (180.0f / 3.14159265358979323846f)
#endif

/* Rendering constants */
#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define TARGET_FPS 60
#define CAMERA_DEFAULT_DISTANCE 10000.0f  // Default camera distance from target
#define CAMERA_ZOOM_MIN_DISTANCE 500.0f   // Closest zoom (camera distance)
#define CAMERA_ZOOM_MAX_DISTANCE 50000.0f  // Farthest zoom (camera distance)
#define CAMERA_DEFAULT_ELEVATION 0.3f     // Default camera elevation angle in radians (~17 degrees up)
#define CHASE_ELEVATION 150.0f
#define F16_SCALE 150.0f
#define GRID_SPACING 1000.0f
#define ALTITUDE_LINE_SPACING 300.0f
#define RIBBON_WIDTH 150.0f
#define MAX_TRAIL_POINTS 10000

/* Colors */
static const Color SKY_COLOR = {0, 0, 0, 255};
static const Color GRID_COLOR = {40, 120, 40, 180};
static const Color TRAIL_COLOR = {253, 249, 0, 255};  /* YELLOW */
static const Color ALTITUDE_MARKER_COLOR = {255, 255, 0, 128};
static const Color BOUNDS_COLOR = {255, 0, 0, 80};  /* RED transparent */

/* Render state structure */
typedef struct {
    float time_sec;
    float speed_fps;
    float alpha_rad;
    float beta_rad;
    float phi_rad;
    float theta_rad;
    float psi_rad;
    float pos_e;
    float pos_n;
    float altitude;
    float nz_g;
    float ps_rad_s;
    float waypoint_e;
    float waypoint_n;
    float waypoint_alt;
    float waypoint_radius;
    float world_bounds_e_min;
    float world_bounds_e_max;
    float world_bounds_n_min;
    float world_bounds_n_max;
    float world_bounds_alt_max;
    float reward;
} RenderState;

/* Trail marker structure */
typedef struct {
    Vector3 pos;
    float roll;
    float pitch;
    float yaw;
} TrailMarker;

/* Module-level state */
static Camera3D _camera = {0};
static float _near_plane = 0.1f;
static float _far_plane = 200000.0f;  /* 200km far plane for extreme zoom out */
static TrailMarker _altitude_markers[MAX_TRAIL_POINTS];
static int _num_markers = 0;
static Vector3 _last_marker_pos = {0};
static bool _has_last_marker = false;
static Font _font = {0};
static Vector2 _manual_camera_offset = {0.0f, CAMERA_DEFAULT_ELEVATION};  /* azimuth, elevation */
static float _manual_zoom = 0.0f;
static Vector2 _last_mouse_pos = {0};
static bool _mouse_was_down = false;
static bool _initialized = false;

/* Forward declarations */
static void _initialize(void);
static void _handle_camera_input(void);
static void _update_camera(Vector3 position);
static void _draw_simple_plane(Vector3 position, float roll, float pitch, float yaw, float size);
static void _draw_hud(RenderState* state);

/* Initialize renderer */
static void _initialize(void) {
    if (_initialized) return;
    
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Aerobench 3D");
    SetTargetFPS(TARGET_FPS);
    
    _camera.position = (Vector3){0.0f, CHASE_ELEVATION, -CAMERA_DEFAULT_DISTANCE};
    _camera.target = (Vector3){0.0f, 0.0f, 0.0f};
    _camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    _camera.fovy = 60.0f;
    _camera.projection = CAMERA_PERSPECTIVE;
    
    _font = GetFontDefault();
    _initialized = true;
}

/* Close renderer */
void raylib_renderer_close(void) {
    if (_initialized) {
        CloseWindow();
        _initialized = false;
    }
}

/* Clear trail only (called when episode ends but continuing) */
void raylib_renderer_clear_trail(void) {
    _num_markers = 0;
    _has_last_marker = false;
}

/* Handle camera input */
static void _handle_camera_input(void) {
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        _manual_zoom += wheel * 200.0f;
        float min_zoom = -(CAMERA_ZOOM_MAX_DISTANCE - CAMERA_DEFAULT_DISTANCE);
        float max_zoom = CAMERA_DEFAULT_DISTANCE - CAMERA_ZOOM_MIN_DISTANCE;
        if (_manual_zoom < min_zoom) _manual_zoom = min_zoom;
        if (_manual_zoom > max_zoom) _manual_zoom = max_zoom;
    }
    
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_pos = GetMousePosition();
        if (_mouse_was_down) {
            float delta_x = mouse_pos.x - _last_mouse_pos.x;
            float delta_y = mouse_pos.y - _last_mouse_pos.y;
            _manual_camera_offset.x -= delta_x * 0.01f;
            _manual_camera_offset.y += delta_y * 0.01f;
            
            float max_el = 3.14159265358979323846f / 2.0f - 0.1f;
            if (_manual_camera_offset.y < -max_el) _manual_camera_offset.y = -max_el;
            if (_manual_camera_offset.y > max_el) _manual_camera_offset.y = max_el;
        }
        _last_mouse_pos = mouse_pos;
        _mouse_was_down = true;
    } else {
        _mouse_was_down = false;
    }
}

/* Update camera position */
static void _update_camera(Vector3 position) {
    _camera.target = position;
    _camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    
    float d = CAMERA_DEFAULT_DISTANCE - _manual_zoom;
    
    float az = _manual_camera_offset.x;
    float el = _manual_camera_offset.y;
    float ce = cosf(el);
    float se = sinf(el);
    float ca = cosf(az);
    float sa = sinf(az);
    
    _camera.position = (Vector3){
        position.x + d * ce * sa,
        position.y + d * se,
        position.z + d * ce * ca
    };
}

/* Draw F-16 using navigation equations */
static void _draw_simple_plane(Vector3 position, float roll, float pitch, float yaw, float size) {
    float sp = sinf(roll), cp = cosf(roll);
    float st = sinf(pitch), ct = cosf(pitch);
    float sy = sinf(yaw), cy = cosf(yaw);
    
    /* F16 navigation: body -> world (north, east, up) -> viz (East, Up, North) */
    Vector3 n = {ct * sy, st, ct * cy};
    Vector3 r = {sp * sy * st + cp * cy, -sp * ct, sp * cy * st - cp * sy};
    
    /* Cross product for up vector */
    Vector3 u = {
        n.y * r.z - n.z * r.y,
        n.z * r.x - n.x * r.z,
        n.x * r.y - n.y * r.x
    };
    float ul = sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
    u.x /= ul; u.y /= ul; u.z /= ul;
    
    /* Scaled components */
    Vector3 nose = {position.x + n.x*size, position.y + n.y*size, position.z + n.z*size};
    Vector3 tail = {position.x - n.x*size*0.4f, position.y - n.y*size*0.4f, position.z - n.z*size*0.4f};
    Vector3 lwing = {position.x - r.x*size*0.6f, position.y - r.y*size*0.6f, position.z - r.z*size*0.6f};
    Vector3 rwing = {position.x + r.x*size*0.6f, position.y + r.y*size*0.6f, position.z + r.z*size*0.6f};
    Vector3 vtail = {tail.x + u.x*size*0.3f, tail.y + u.y*size*0.3f, tail.z + u.z*size*0.3f};
    
    DrawLine3D(position, nose, RED);
    DrawSphere(nose, size * 0.08f, RED);
    DrawLine3D(lwing, rwing, GREEN);
    DrawSphere(lwing, size * 0.08f, GREEN);
    DrawSphere(rwing, size * 0.08f, GREEN);
    DrawLine3D(position, tail, BLUE);
    DrawSphere(tail, size * 0.08f, BLUE);
    DrawLine3D(tail, vtail, SKYBLUE);
    DrawSphere(vtail, size * 0.08f, SKYBLUE);
    DrawSphere(position, size * 0.12f, LIGHTGRAY);
}

/* Draw HUD */
static void _draw_hud(RenderState* state) {
    int p = 12, lh = 22;
    char text[256];
    
    sprintf(text, "t = %.2f s", state->time_sec);
    DrawTextEx(_font, text, (Vector2){p, p + 0*lh}, 20, 1, RAYWHITE);
    
    sprintf(text, "h = %.1f ft", state->altitude);
    DrawTextEx(_font, text, (Vector2){p, p + 1*lh}, 20, 1, RAYWHITE);
    
    sprintf(text, "V = %.1f ft/s", state->speed_fps);
    DrawTextEx(_font, text, (Vector2){p, p + 2*lh}, 20, 1, RAYWHITE);
    
    sprintf(text, "alpha = %.1f deg", state->alpha_rad * RAD2DEG);
    DrawTextEx(_font, text, (Vector2){p, p + 3*lh}, 20, 1, RAYWHITE);
    
    sprintf(text, "beta = %.1f deg", state->beta_rad * RAD2DEG);
    DrawTextEx(_font, text, (Vector2){p, p + 4*lh}, 20, 1, RAYWHITE);
    
    sprintf(text, "Nz = %.2f g", state->nz_g);
    DrawTextEx(_font, text, (Vector2){p, p + 5*lh}, 20, 1, RAYWHITE);
    
    sprintf(text, "ps = %.1f deg/s", state->ps_rad_s * RAD2DEG);
    DrawTextEx(_font, text, (Vector2){p, p + 6*lh}, 20, 1, RAYWHITE);
    
    sprintf(text, "phi/theta/psi = %.1f/%.1f/%.1f deg", 
            state->phi_rad * RAD2DEG, state->theta_rad * RAD2DEG, state->psi_rad * RAD2DEG);
    DrawTextEx(_font, text, (Vector2){p, p + 7*lh}, 20, 1, RAYWHITE);
    
    sprintf(text, "reward = %.4f", state->reward);
    DrawTextEx(_font, text, (Vector2){p, p + 8*lh}, 20, 1, YELLOW);
    
    DrawTextEx(_font, "Left click + drag: rotate camera", (Vector2){p, p + 9*lh}, 20, 1, RAYWHITE);
    DrawTextEx(_font, "Mouse wheel: zoom in/out", (Vector2){p, p + 10*lh}, 20, 1, RAYWHITE);
}

/* Main render function */
void raylib_renderer_render(RenderState* state) {
    if (!_initialized) {
        _initialize();
    }
    
    _handle_camera_input();
    
    Vector3 position = {state->pos_e, state->altitude, state->pos_n};
    
    /* Add altitude marker and trail point at same spacing */
    if (!_has_last_marker) {
        if (_num_markers < MAX_TRAIL_POINTS) {
            _altitude_markers[_num_markers].pos = position;
            _altitude_markers[_num_markers].roll = state->phi_rad;
            _altitude_markers[_num_markers].pitch = state->theta_rad;
            _altitude_markers[_num_markers].yaw = state->psi_rad;
            _num_markers++;
        }
        _last_marker_pos = position;
        _has_last_marker = true;
    } else {
        Vector3 diff = {position.x - _last_marker_pos.x, 
                       position.y - _last_marker_pos.y, 
                       position.z - _last_marker_pos.z};
        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
        
        if (dist >= ALTITUDE_LINE_SPACING) {
            if (_num_markers < MAX_TRAIL_POINTS) {
                _altitude_markers[_num_markers].pos = position;
                _altitude_markers[_num_markers].roll = state->phi_rad;
                _altitude_markers[_num_markers].pitch = state->theta_rad;
                _altitude_markers[_num_markers].yaw = state->psi_rad;
                _num_markers++;
            }
            _last_marker_pos = position;
        }
    }
    
    _update_camera(position);
    
    BeginDrawing();
    ClearBackground(SKY_COLOR);
    BeginMode3D(_camera);
    
    /* Override projection for extended far clipping plane */
    float aspect = (float)GetScreenWidth() / (float)GetScreenHeight();
    float top = _near_plane * tanf(_camera.fovy * 0.5f * 3.14159265358979323846f / 180.0f);
    rlMatrixMode(RL_PROJECTION);
    rlLoadIdentity();
    rlFrustum(-top * aspect, top * aspect, -top, top, _near_plane, _far_plane);
    rlMatrixMode(RL_MODELVIEW);
    
    /* Draw ground grid (constrained to world bounds) */
    float e_min = state->world_bounds_e_min;
    float e_max = state->world_bounds_e_max;
    float n_min = state->world_bounds_n_min;
    float n_max = state->world_bounds_n_max;
    
    int grid_step = (int)GRID_SPACING;
    
    int x_start = ((int)(e_min / GRID_SPACING)) * grid_step;
    int x_end = ((int)(e_max / GRID_SPACING)) * grid_step;
    for (int x = x_start; x <= x_end; x += grid_step) {
        DrawLine3D((Vector3){(float)x, 0.0f, n_min},
                  (Vector3){(float)x, 0.0f, n_max}, GRID_COLOR);
    }
    
    int z_start = ((int)(n_min / GRID_SPACING)) * grid_step;
    int z_end = ((int)(n_max / GRID_SPACING)) * grid_step;
    for (int z = z_start; z <= z_end; z += grid_step) {
        DrawLine3D((Vector3){e_min, 0.0f, (float)z},
                  (Vector3){e_max, 0.0f, (float)z}, GRID_COLOR);
    }
    
    /* Draw altitude markers with T-shaped markers and ribbon */
    float hw = RIBBON_WIDTH * 0.5f;
    for (int i = 0; i < _num_markers; i++) {
        TrailMarker* m = &_altitude_markers[i];
        float px = m->pos.x, py = m->pos.y, pz = m->pos.z;
        
        /* Vertical line (ground to position) */
        DrawLine3D((Vector3){px, 0.0f, pz}, (Vector3){px, py, pz}, ALTITUDE_MARKER_COLOR);
        
        /* Horizontal line (showing roll orientation) */
        float sp = sinf(m->roll), cp = cosf(m->roll);
        float st = sinf(m->pitch), ct = cosf(m->pitch);
        float sy = sinf(m->yaw), cy = cosf(m->yaw);
        
        float rx = sp * sy * st + cp * cy;
        float ry = -sp * ct;
        float rz = sp * cy * st - cp * sy;
        float rnorm = sqrtf(rx*rx + ry*ry + rz*rz) + 1e-10f;
        rx = rx/rnorm * hw;
        ry = ry/rnorm * hw;
        rz = rz/rnorm * hw;
        
        Vector3 left_pt = {px - rx, py - ry, pz - rz};
        Vector3 right_pt = {px + rx, py + ry, pz + rz};
        DrawLine3D(left_pt, right_pt, TRAIL_COLOR);
        
        /* Center line and edge lines to next marker */
        if (i < _num_markers - 1) {
            TrailMarker* next_m = &_altitude_markers[i + 1];
            float nx = next_m->pos.x, ny = next_m->pos.y, nz = next_m->pos.z;
            
            float nsp = sinf(next_m->roll), ncp = cosf(next_m->roll);
            float nst = sinf(next_m->pitch), nct = cosf(next_m->pitch);
            float nsy = sinf(next_m->yaw), ncy = cosf(next_m->yaw);
            
            float nrx = nsp * nsy * nst + ncp * ncy;
            float nry = -nsp * nct;
            float nrz = nsp * ncy * nst - ncp * nsy;
            float nrnorm = sqrtf(nrx*nrx + nry*nry + nrz*nrz) + 1e-10f;
            nrx = nrx/nrnorm * hw;
            nry = nry/nrnorm * hw;
            nrz = nrz/nrnorm * hw;
            
            Vector3 next_left = {nx - nrx, ny - nry, nz - nrz};
            Vector3 next_right = {nx + nrx, ny + nry, nz + nrz};
            
            DrawLine3D((Vector3){px, py, pz}, (Vector3){nx, ny, nz}, TRAIL_COLOR);  /* center */
            DrawLine3D(left_pt, next_left, TRAIL_COLOR);  /* left edge */
            DrawLine3D(right_pt, next_right, TRAIL_COLOR);  /* right edge */
        }
    }
    
    /* Current position marker */
    DrawLine3D((Vector3){position.x, 0.0f, position.z}, position, ALTITUDE_MARKER_COLOR);
    
    /* Draw waypoint */
    Vector3 wp_pos = {state->waypoint_e, state->waypoint_alt, state->waypoint_n};
    DrawSphere(wp_pos, state->waypoint_radius, PURPLE);
    DrawLine3D((Vector3){wp_pos.x, 0.0f, wp_pos.z}, wp_pos, PURPLE);
    
    /* Draw world bounds cube with thick lines */
    float alt_max = state->world_bounds_alt_max;
    
    /* Bottom square */
    DrawLine3D((Vector3){e_min, 0, n_min}, (Vector3){e_max, 0, n_min}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_max, 0, n_min}, (Vector3){e_max, 0, n_max}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_max, 0, n_max}, (Vector3){e_min, 0, n_max}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_min, 0, n_max}, (Vector3){e_min, 0, n_min}, BOUNDS_COLOR);
    
    /* Top square */
    DrawLine3D((Vector3){e_min, alt_max, n_min}, (Vector3){e_max, alt_max, n_min}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_max, alt_max, n_min}, (Vector3){e_max, alt_max, n_max}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_max, alt_max, n_max}, (Vector3){e_min, alt_max, n_max}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_min, alt_max, n_max}, (Vector3){e_min, alt_max, n_min}, BOUNDS_COLOR);
    
    /* Vertical edges */
    DrawLine3D((Vector3){e_min, 0, n_min}, (Vector3){e_min, alt_max, n_min}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_max, 0, n_min}, (Vector3){e_max, alt_max, n_min}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_max, 0, n_max}, (Vector3){e_max, alt_max, n_max}, BOUNDS_COLOR);
    DrawLine3D((Vector3){e_min, 0, n_max}, (Vector3){e_min, alt_max, n_max}, BOUNDS_COLOR);
    
    /* Draw plane */
    _draw_simple_plane(position, state->phi_rad, state->theta_rad, state->psi_rad, F16_SCALE);
    
    EndMode3D();
    
    _draw_hud(state);
    EndDrawing();
}

/* Check if window should close */
bool raylib_renderer_should_close(void) {
    if (!_initialized) return false;
    return WindowShouldClose();
}

#endif /* RAYLIB_RENDERER_H */
