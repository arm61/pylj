double compute_pressure(int number_of_particles, const double *xpos, const double *ypos, double box_length,
                        double temperature);

void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                           double *distances, double *xforces, double *yforce, double box_length, double *force_arr);

void compute_energy_and_force(int len_particles, const double *xpos, const double *ypos, double *energy, double *xforce,
                              double *yforce, double *xforcedash, double *yforcedash, double box_length);

void compute_force(int len_particles, const double *xpos, const double *ypos, double *xforce, double *yforce,
                   double box_length);

void scale_velocities(int len_particles, double *xvel, double *yvel, double average_temp, double tempature);
