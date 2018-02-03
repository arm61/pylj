void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                           double *distances, double box_length);

void scale_velocities(int len_particles, double *xvel, double *yvel, double average_temp, double tempature);