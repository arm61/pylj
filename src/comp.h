void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                           double *distances_arr, double *xforces, double *yforce, double box_l, double *force_arr);

double compute_pressure(int number_of_particles, const double *xpos, const double *ypos, double box_length,
                        double temperature);

void scale_velocities(int len_particles, double *xvel, double *yvel, double average_temp, double tempature);
