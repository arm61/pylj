void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                           double *distances_arr, double box_l, double *force_arr, double *energy_arr, double cut);

void compute_energies(int len_particles, const double *xpos, const double *ypos, double *distances_arr, double box_l,
                    double *energy_arr, double cut);

double compute_pressure(int number_of_particles, const double *xpos, const double *ypos, double box_length,
                        double temperature, double cut);

void scale_velocities(int len_particles, double *xvel, double *yvel, double average_temp, double tempature);
