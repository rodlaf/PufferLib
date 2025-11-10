#include <stddef.h>
#include <math.h>

// round towards zero
static inline int fix(double ele) {
    return (ele > 0) ? (int)floor(ele) : (int)ceil(ele);
}

// sign of a number
static inline int sign(double ele) {
    return (ele < 0) ? -1 : (ele == 0) ? 0 : 1;
}

// converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure
// (qbar) See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd
// edition
static inline void adc(double vt, double alt, double *amach, double *qbar) {
    double ro = 2.377e-3;
    double tfac = 1 - .703e-5 * alt;
    double t = (alt >= 35000) ? 390 : 519 * tfac;
    double rho = ro * pow(tfac, 4.14);
    double a = sqrt(1.4 * 1716.3 * t);
    *amach = vt / a;
    *qbar = .5 * rho * vt * vt;
}

// dampp function
static inline void dampp(double alpha, double d[9]) {
    static const double a[12][9] = {
        {-.267, .882, -.108, -8.80, -.126, -.360, -7.21, -.380, .061},
        {-.110, .852, -.108, -25.8, -.026, -.359, -.540, -.363, .052},
        {.308, .876, -.188, -28.9, .063, -.443, -5.23, -.378, .052},
        {1.34, .958, .110, -31.4, .113, -.420, -5.26, -.386, -.012},
        {2.08, .962, .258, -31.2, .208, -.383, -6.11, -.370, -.013},
        {2.91, .974, .226, -30.7, .230, -.375, -6.64, -.453, -.024},
        {2.76, .819, .344, -27.7, .319, -.329, -5.69, -.550, .050},
        {2.05, .483, .362, -28.2, .437, -.294, -6.00, -.582, .150},
        {1.50, .590, .611, -29.0, .680, -.230, -6.20, -.595, .130},
        {1.49, 1.21, .529, -29.8, .100, -.210, -6.40, -.637, .158},
        {1.83, -.493, .298, -38.3, .447, -.120, -6.60, -1.02, .240},
        {1.21, -1.04, -2.27, -35.3, -.330, -.100, -6.00, -.840, .150}
    };
    
    double s = .2 * alpha;
    int k = fix(s);
    
    if (k <= -2) k = -1;
    if (k >= 9) k = 8;
    
    double da = s - k;
    int l = k + fix(1.1 * sign(da));
    k = k + 3;
    l = l + 3;
    
    for (int i = 0; i < 9; i++) {
        d[i] = a[k-1][i] + fabs(da) * (a[l-1][i] - a[k-1][i]);
    }
}

// rtau function
static inline double rtau(double dp) {
    double rt;
    if (dp <= 25) {
        rt = 1.0;
    } else if (dp >= 50) {
        rt = .1;
    } else {
        rt = 1.9 - .036 * dp;
    }
    return rt;
}

// pdot function
static inline double pdot(double p3, double p1) {
    double p2, t;
    if (p1 >= 50) {
        if (p3 >= 50) {
            t = 5;
            p2 = p1;
        } else {
            p2 = 60;
            t = rtau(p2 - p3);
        }
    } else {
        if (p3 >= 50) {
            t = 5;
            p2 = 40;
        } else {
            p2 = p1;
            t = rtau(p2 - p3);
        }
    }
    double pd = t * (p2 - p3);
    return pd;
}

// tgear function
static inline double tgear(double thtl) {
    double tg;
    if (thtl <= .77) {
        tg = 64.94 * thtl;
    } else {
        tg = 217.38 * thtl - 117.38;
    }
    return tg;
}

// thrust lookup-table version
static inline double thrust(double power, double alt, double rmach) {
    static const double a[6][6] = {
        {1060, 670, 880, 1140, 1500, 1860},
        {635, 425, 690, 1010, 1330, 1700},
        {60, 25, 345, 755, 1130, 1525},
        {-1020, -170, -300, 350, 910, 1360},
        {-2700, -1900, -1300, -247, 600, 1100},
        {-3600, -1400, -595, -342, -200, 700}
    };
    
    static const double b[6][6] = {
        {12680, 9150, 6200, 3950, 2450, 1400},
        {12680, 9150, 6313, 4040, 2470, 1400},
        {12610, 9312, 6610, 4290, 2600, 1560},
        {12640, 9839, 7090, 4660, 2840, 1660},
        {12390, 10176, 7750, 5320, 3250, 1930},
        {11680, 9848, 8050, 6100, 3800, 2310}
    };
    
    static const double c[6][6] = {
        {20000, 15000, 10800, 7000, 4000, 2500},
        {21420, 15700, 11225, 7323, 4435, 2600},
        {22700, 16860, 12250, 8154, 5000, 2835},
        {24240, 18910, 13760, 9285, 5700, 3215},
        {26070, 21075, 15975, 11115, 6860, 3950},
        {28886, 23319, 18300, 13484, 8642, 5057}
    };
    
    if (alt < 0) alt = 0.01; // uh, why not 0?
    
    double h = .0001 * alt;
    int i = fix(h);
    
    if (i >= 5) i = 4;
    
    double dh = h - i;
    double rm = 5 * rmach;
    int m = fix(rm);
    
    if (m >= 5) m = 4;
    else if (m <= 0) m = 0;
    
    double dm = rm - m;
    double cdh = 1 - dh;
    
    double s = b[i][m] * cdh + b[i + 1][m] * dh;
    double t = b[i][m + 1] * cdh + b[i + 1][m + 1] * dh;
    double tmil = s + (t - s) * dm;
    
    double thrst;
    if (power < 50) {
        s = a[i][m] * cdh + a[i + 1][m] * dh;
        t = a[i][m + 1] * cdh + a[i + 1][m + 1] * dh;
        double tidl = s + (t - s) * dm;
        thrst = tidl + (tmil - tidl) * power * .02;
    } else {
        s = c[i][m] * cdh + c[i + 1][m] * dh;
        t = c[i][m + 1] * cdh + c[i + 1][m + 1] * dh;
        double tmax = s + (t - s) * dm;
        thrst = tmil + (tmax - tmil) * (power - 50) * .02;
    }
    
    return thrst;
}

// Morelli dynamics (Polynomial interpolation)
static inline void Morellif16(double alpha, double beta, double de, double da, double dr, 
                               double p, double q, double r, double cbar, double b, double V, 
                               double xcg, double xcgref, 
                               double *Cx, double *Cy, double *Cz, double *Cl, double *Cm, double *Cn) {
    double phat = p * b / (2 * V);
    double qhat = q * cbar / (2 * V);
    double rhat = r * b / (2 * V);
    
    double a0 = -1.943367e-2, a1 = 2.136104e-1, a2 = -2.903457e-1, a3 = -3.348641e-3;
    double a4 = -2.060504e-1, a5 = 6.988016e-1, a6 = -9.035381e-1;
    double b0 = 4.833383e-1, b1 = 8.644627, b2 = 1.131098e1, b3 = -7.422961e1, b4 = 6.075776e1;
    double c0 = -1.145916, c1 = 6.016057e-2, c2 = 1.642479e-1;
    double d0 = -1.006733e-1, d1 = 8.679799e-1, d2 = 4.260586, d3 = -6.923267;
    double e0 = 8.071648e-1, e1 = 1.189633e-1, e2 = 4.177702, e3 = -9.162236;
    double f0 = -1.378278e-1, f1 = -4.211369, f2 = 4.775187, f3 = -1.026225e1, f4 = 8.399763, f5 = -4.354000e-1;
    double g0 = -3.054956e1, g1 = -4.132305e1, g2 = 3.292788e2, g3 = -6.848038e2, g4 = 4.080244e2;
    double h0 = -1.05853e-1, h1 = -5.776677e-1, h2 = -1.672435e-2, h3 = 1.357256e-1;
    double h4 = 2.172952e-1, h5 = 3.464156, h6 = -2.835451, h7 = -1.098104;
    double i0 = -4.126806e-1, i1 = -1.189974e-1, i2 = 1.247721, i3 = -7.391132e-1;
    double j0 = 6.250437e-2, j1 = 6.067723e-1, j2 = -1.101964, j3 = 9.100087, j4 = -1.192672e1;
    double k0 = -1.463144e-1, k1 = -4.07391e-2, k2 = 3.253159e-2, k3 = 4.851209e-1;
    double k4 = 2.978850e-1, k5 = -3.746393e-1, k6 = -3.213068e-1;
    double l0 = 2.635729e-2, l1 = -2.192910e-2, l2 = -3.152901e-3, l3 = -5.817803e-2;
    double l4 = 4.516159e-1, l5 = -4.928702e-1, l6 = -1.579864e-2;
    double m0 = -2.029370e-2, m1 = 4.660702e-2, m2 = -6.012308e-1, m3 = -8.062977e-2;
    double m4 = 8.320429e-2, m5 = 5.018538e-1, m6 = 6.378864e-1, m7 = 4.226356e-1;
    double n0 = -5.19153, n1 = -3.554716, n2 = -3.598636e1, n3 = 2.247355e2, n4 = -4.120991e2, n5 = 2.411750e2;
    double o0 = 2.993363e-1, o1 = 6.594004e-2, o2 = -2.003125e-1, o3 = -6.233977e-2;
    double o4 = -2.107885, o5 = 2.141420, o6 = 8.476901e-1;
    double p0 = 2.677652e-2, p1 = -3.298246e-1, p2 = 1.926178e-1, p3 = 4.013325, p4 = -4.404302;
    double q0 = -3.698756e-1, q1 = -1.167551e-1, q2 = -7.641297e-1;
    double r0 = -3.348717e-2, r1 = 4.276655e-2, r2 = 6.573646e-3, r3 = 3.535831e-1, r4 = -1.373308;
    double r5 = 1.237582, r6 = 2.302543e-1, r7 = -2.512876e-1, r8 = 1.588105e-1, r9 = -5.199526e-1;
    double s0 = -8.115894e-2, s1 = -1.156580e-2, s2 = 2.514167e-2, s3 = 2.038748e-1, s4 = -3.337476e-1, s5 = 1.004297e-1;
    
    double Cx0 = a0 + a1*alpha + a2*de*de + a3*de + a4*alpha*de + a5*alpha*alpha + a6*alpha*alpha*alpha;
    double Cxq = b0 + b1*alpha + b2*alpha*alpha + b3*alpha*alpha*alpha + b4*alpha*alpha*alpha*alpha;
    double Cy0 = c0*beta + c1*da + c2*dr;
    double Cyp = d0 + d1*alpha + d2*alpha*alpha + d3*alpha*alpha*alpha;
    double Cyr = e0 + e1*alpha + e2*alpha*alpha + e3*alpha*alpha*alpha;
    double Cz0 = (f0 + f1*alpha + f2*alpha*alpha + f3*alpha*alpha*alpha + f4*alpha*alpha*alpha*alpha)*(1 - beta*beta) + f5*de;
    double Czq = g0 + g1*alpha + g2*alpha*alpha + g3*alpha*alpha*alpha + g4*alpha*alpha*alpha*alpha;
    double Cl0 = h0*beta + h1*alpha*beta + h2*alpha*alpha*beta + h3*beta*beta + h4*alpha*beta*beta + h5*alpha*alpha*alpha*beta + h6*alpha*alpha*alpha*alpha*beta + h7*alpha*alpha*beta*beta;
    double Clp = i0 + i1*alpha + i2*alpha*alpha + i3*alpha*alpha*alpha;
    double Clr = j0 + j1*alpha + j2*alpha*alpha + j3*alpha*alpha*alpha + j4*alpha*alpha*alpha*alpha;
    double Clda = k0 + k1*alpha + k2*beta + k3*alpha*alpha + k4*alpha*beta + k5*alpha*alpha*beta + k6*alpha*alpha*alpha;
    double Cldr = l0 + l1*alpha + l2*beta + l3*alpha*beta + l4*alpha*alpha*beta + l5*alpha*alpha*alpha*beta + l6*beta*beta;
    double Cm0 = m0 + m1*alpha + m2*de + m3*alpha*de + m4*de*de + m5*alpha*alpha*de + m6*de*de*de + m7*alpha*de*de;
    double Cmq = n0 + n1*alpha + n2*alpha*alpha + n3*alpha*alpha*alpha + n4*alpha*alpha*alpha*alpha + n5*alpha*alpha*alpha*alpha*alpha;
    double Cn0 = o0*beta + o1*alpha*beta + o2*beta*beta + o3*alpha*beta*beta + o4*alpha*alpha*beta + o5*alpha*alpha*beta*beta + o6*alpha*alpha*alpha*beta;
    double Cnp = p0 + p1*alpha + p2*alpha*alpha + p3*alpha*alpha*alpha + p4*alpha*alpha*alpha*alpha;
    double Cnr = q0 + q1*alpha + q2*alpha*alpha;
    double Cnda = r0 + r1*alpha + r2*beta + r3*alpha*beta + r4*alpha*alpha*beta + r5*alpha*alpha*alpha*beta + r6*alpha*alpha + r7*alpha*alpha*alpha + r8*beta*beta*beta + r9*alpha*beta*beta*beta;
    double Cndr = s0 + s1*alpha + s2*beta + s3*alpha*beta + s4*alpha*alpha*beta + s5*alpha*alpha;
    
    *Cx = Cx0 + Cxq*qhat;
    *Cy = Cy0 + Cyp*phat + Cyr*rhat;
    *Cz = Cz0 + Czq*qhat;
    *Cl = Cl0 + Clp*phat + Clr*rhat + Clda*da + Cldr*dr;
    *Cm = Cm0 + Cmq*qhat + (*Cz)*(xcgref - xcg);
    *Cn = Cn0 + Cnp*phat + Cnr*rhat + Cnda*da + Cndr*dr - (*Cy)*(xcgref - xcg)*(cbar/b);
}

// output aircraft state vector derivative for a given input
// The reference for the model is Appendix A of Stevens & Lewis
//
//         x[0] = air speed, VT    (ft/sec)
//         x[1] = angle of attack, alpha  (rad)
//         x[2] = angle of sideslip, beta (rad)
//         x[3] = roll angle, phi  (rad)
//         x[4] = pitch angle, theta  (rad)
//         x[5] = yaw angle, psi  (rad)
//         x[6] = roll rate, P  (rad/sec)
//         x[7] = pitch rate, Q  (rad/sec)
//         x[8] = yaw rate, R  (rad/sec)
//         x[9] = northward horizontal displacement, pn  (feet)
//         x[10] = eastward horizontal displacement, pe  (feet)
//         x[11] = altitude, h  (feet)
//         x[12] = engine thrust dynamics lag state, pow
//
//         u[0] = throttle command  0.0 < u(1) < 1.0
//         u[1] = elevator command in degrees
//         u[2] = aileron command in degrees
//         u[3] = rudder command in degrees
static inline void subf16_model(double x[13], double u[4], double xd[13],
                                 double *Nz, double *Ny, double *az_out, double *ay_out) {
    double xcg = 0.35;
    double thtlc = u[0], el = u[1], ail = u[2], rdr = u[3];

    double s = 300, b = 30, cbar = 11.32, rm = 1.57e-3, xcgr = .35, he = 160.0;
    double c1 = -.770, c2 = .02755, c3 = 1.055e-4, c4 = 1.642e-6, c5 = .9604;
    double c6 = 1.759e-2, c7 = 1.792e-5, c8 = -.7336, c9 = 1.587e-5;
    double rtod = 57.29578, g = 32.17;
    const double deg_to_rad = 0.01745329251994329577;

    double vt = x[0], alpha = x[1] * rtod, beta = x[2] * rtod;
    double phi = x[3], theta = x[4], psi = x[5];
    double p = x[6], q = x[7], r = x[8], alt = x[11], power = x[12];

    // air data computer and engine model
    double amach, qbar;
    adc(vt, alt, &amach, &qbar);
    double cpow = tgear(thtlc);
    xd[12] = pdot(power, cpow);
    double t = thrust(power, alt, amach);

    // component build up
    double cxt = 0.0, cyt = 0.0, czt = 0.0, clt = 0.0, cmt = 0.0, cnt = 0.0;

    // morelli model (polynomial version)
    Morellif16(alpha * deg_to_rad, beta * deg_to_rad, el * deg_to_rad, ail * deg_to_rad, rdr * deg_to_rad,
                p, q, r, cbar, b, vt, xcg, xcgr, &cxt, &cyt, &czt, &clt, &cmt, &cnt);

    // add damping derivatives
    double tvt = .5 / vt, b2v = b * tvt, cq = cbar * q * tvt;
    double d[9];
    dampp(alpha, d);
    cxt = cxt + cq * d[0];
    cyt = cyt + b2v * (d[1] * r + d[2] * p);
    czt = czt + cq * d[3];
    clt = clt + b2v * (d[4] * r + d[5] * p);
    cmt = cmt + cq * d[6] + czt * (xcgr - xcg);
    cnt = cnt + b2v * (d[7] * r + d[8] * p) - cyt * (xcgr - xcg) * cbar / b;

    double cbta = cos(x[2]);
    double u_vel = vt * cos(x[1]) * cbta;
    double v = vt * sin(x[2]);
    double w = vt * sin(x[1]) * cbta;
    double sth = sin(theta), cth = cos(theta), sph = sin(phi), cph = cos(phi);
    double spsi = sin(psi), cpsi = cos(psi);
    double qs = qbar * s, qsb = qs * b, rmqs = rm * qs, gcth = g * cth, qsph = q * sph;
    double ay = rmqs * cyt, az = rmqs * czt;

    // force equations
    double udot = r * v - q * w - g * sth + rm * (qs * cxt + t);
    double vdot = p * w - r * u_vel + gcth * sph + ay;
    double wdot = q * u_vel - p * v + gcth * cph + az;
    double dum = (u_vel * u_vel + w * w);

    xd[0] = (u_vel * udot + v * vdot + w * wdot) / vt;
    xd[1] = (u_vel * wdot - w * udot) / dum;
    xd[2] = (vt * vdot - v * xd[0]) * cbta / dum;

    // kinematics
    xd[3] = p + (sth / cth) * (qsph + r * cph);
    xd[4] = q * cph - r * sph;
    xd[5] = (qsph + r * cph) / cth;

    // moments
    xd[6] = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt);
    xd[7] = (c5 * p - c7 * he) * r + c6 * (r * r - p * p) + qs * cbar * c7 * cmt;
    xd[8] = (c8 * p - c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt);

    // navigation
    double t1 = sph * cpsi, t2 = cph * sth, t3 = sph * spsi;
    double s1 = cth * cpsi, s2 = cth * spsi, s3 = t1 * sth - cph * spsi;
    double s4 = t3 * sth + cph * cpsi, s5 = sph * cth;
    double s6 = t2 * cpsi + t3, s7 = t2 * spsi - t1, s8 = cph * cth;
    xd[9]  = u_vel * s1 + v * s3 + w * s6;  // north speed
    xd[10] = u_vel * s2 + v * s4 + w * s7;  // east speed
    xd[11] = u_vel * sth - v * s5 - w * s8; // vertical speed

    // outputs
    double xa = 15.0; // sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
    az = az - xa * xd[7]; // moves normal accel in front of c.g.
    ay = ay + xa * xd[8]; // moves side accel in front of c.g.

    // For extraction of Nz
    *Nz = (-az / g) - 1; // zeroed at 1 g, positive g = pulling up
    *Ny = ay / g;
    *az_out = az;
    *ay_out = ay;
}

// Stanley Bak
// Python Version of F-16 GCAS
//
// Low-level flight controller
//
// and
//
// ODE derivative code (controlled F16)

static const double THROTTLE_MAX = 1.0;
static const double THROTTLE_MIN = 0.0;
static const double ELEVATOR_MAX_DEG = 25.0;
static const double ELEVATOR_MIN_DEG = -25.0;
static const double AILERON_MAX_DEG = 21.5;
static const double AILERON_MIN_DEG = -21.5;
static const double RUDDER_MAX_DEG = 30.0;
static const double RUDDER_MIN_DEG = -30.0;
static const double RAD_PER_DEG = 0.01745329251994329577;

// Control Limits
static const double K_long[3] = {-156.8801506723475, -31.037008068526642, -38.72983346216317};
static const double K_lat[2][5] = {
    {37.84483, -25.40956, -6.82876, -332.88343, -17.15997},
    {-23.91233, 5.69968, -21.63431, 64.49490, -88.36203}
};

// Longitudinal Gains
static const double K_lqr[3][8] = {
    {-156.8801506723475, -31.037008068526642, -38.72983346216317, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 37.84483, -25.40956, -6.82876, -332.88343, -17.15997},
    {0.0, 0.0, 0.0, -23.91233, 5.69968, -21.63431, 64.49490, -88.36203}
};

static const double xequil[13] = {502.0, 0.0389, 0.0, 0.0, 0.0389, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 1000.0, 9.0567};
static const double uequil[4] = {0.1395, -0.7496, 0.0, 0.0};

static inline double clamp(double value, double min_val, double max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// get the reference commands for the control surfaces
static inline void get_u_deg(const double u_ref[4], const double f16_state[16],
                             double x_ctrl[8], double u_deg[4]) {
    // Calculate perturbation from trim state
    double x_delta[16];
    for (int i = 0; i < 16; ++i) x_delta[i] = f16_state[i];
    for (int i = 0; i < 13; ++i) x_delta[i] -= xequil[i];

    // ## Implement LQR Feedback Control
    // Reorder states to match controller:
    // [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
    static const int idx_map[8] = {1, 7, 13, 2, 6, 8, 14, 15};
    for (int i = 0; i < 8; ++i) x_ctrl[i] = x_delta[idx_map[i]];

    // Initialize control vectors
    for (int i = 0; i < 4; ++i) u_deg[i] = 0.0;

    // Calculate control using LQR gains
    for (int row = 0; row < 3; ++row) {
        double sum = 0.0;
        for (int col = 0; col < 8; ++col) sum += K_lqr[row][col] * x_ctrl[col];
        u_deg[row + 1] = -sum; // Full Control
    }

    // Set throttle as directed from output of getOuterLoopCtrl(...)
    u_deg[0] = u_ref[3];

    // Add in equilibrium control
    for (int i = 0; i < 4; ++i) u_deg[i] += uequil[i];

    // ## Limit controls to saturation limits
    // Limit throttle from 0 to 1
    u_deg[0] = clamp(u_deg[0], THROTTLE_MIN, THROTTLE_MAX);
    // Limit elevator from -25 to 25 deg
    u_deg[1] = clamp(u_deg[1], ELEVATOR_MIN_DEG, ELEVATOR_MAX_DEG);
    // Limit aileron from -21.5 to 21.5 deg
    u_deg[2] = clamp(u_deg[2], AILERON_MIN_DEG, AILERON_MAX_DEG);
    // Limit rudder from -30 to 30 deg
    u_deg[3] = clamp(u_deg[3], RUDDER_MIN_DEG, RUDDER_MAX_DEG);
}

// get the derivatives of the integrators in the low-level controller
static inline void get_integrator_derivatives(const double u_ref[4], double Nz, double ps, double Ny_r,
                                              double derivatives[3]) {
    derivatives[0] = Nz - u_ref[0];
    derivatives[1] = ps - u_ref[1];
    derivatives[2] = Ny_r - u_ref[2];
}

// returns the LQR-controlled F-16 state derivatives and more
static inline void controlled_f16(const double x_f16[16], const double u_ref[4],
                                  double xd[16], double u_rad[7],
                                  double *Nz_out, double *ps_out, double *Ny_r_out) {
    double x_ctrl[8], u_deg[4];
    get_u_deg(u_ref, x_f16, x_ctrl, u_deg);

    // Note: Control vector (u) for subF16 is in units of degrees
    double x_state[13];
    for (int i = 0; i < 13; ++i) x_state[i] = x_f16[i];
    double xd_model[13], Nz, Ny, az, ay;
    subf16_model(x_state, u_deg, xd_model, &Nz, &Ny, &az, &ay);

    // Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
    double ps = x_ctrl[4] * cos(x_ctrl[0]) + x_ctrl[5] * sin(x_ctrl[0]);

    // Calculate (side force + yaw rate) term
    double Ny_r = Ny + x_ctrl[5];

    for (int i = 0; i < 16; ++i) xd[i] = 0.0;
    for (int i = 0; i < 13; ++i) xd[i] = xd_model[i];

    // integrators from low-level controller
    double int_der[3];
    get_integrator_derivatives(u_ref, Nz, ps, Ny_r, int_der);
    for (int i = 0; i < 3; ++i) xd[13 + i] = int_der[i];

    // Convert all degree values to radians for output
    // throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    for (int i = 0; i < 7; ++i) u_rad[i] = 0.0;
    u_rad[0] = u_deg[0]; // throttle
    for (int i = 1; i < 4; ++i) u_rad[i] = u_deg[i] * RAD_PER_DEG;
    u_rad[4] = u_ref[0];
    u_rad[5] = u_ref[1];
    u_rad[6] = u_ref[2]; // inner-loop commands are 4-7

    if (Nz_out) *Nz_out = Nz;
    if (ps_out) *ps_out = ps;
    if (Ny_r_out) *Ny_r_out = Ny_r;
}