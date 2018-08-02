#include "comp.h"
#include <math.h>
#include <stdio.h>
#include <iostream>

void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc,
		double *yacc, double *distances_arr, double box_l, double *force_arr, double *energy_arr, double cut)
{
    int ii = 0;
    double dx, dy, dr, f, e;
    for (ii = 0; ii < len_particles; ii++)
    {
        xacc[ii] = 0.;
        yacc[ii] = 0.;
        force_arr[ii] = 0;
    }
    int k = 0;
    int i = 0;
    double inv_dr_1, inv_dr_2, inv_dr_6;
    // values of A and B where determined from from A. Rahman "Correlations in the Motion of Atoms in Liquid Argon",
    // Physical Review 136 pp. A405–A411 (1964)
    double A = 1.363e-134; // joules / metre ^{12}
    double B = 9.273e-78; // joules / meter ^{6}
    double atomic_mass_unit = 1.660539e-27; // kilograms
    double mass_of_argon_amu = 39.948;  // amu
    double mass_of_argon = mass_of_argon_amu * atomic_mass_unit; // kilograms
    double inv_mass_of_argon = 1 / mass_of_argon;
    for (i = 0; i < len_particles - 1; i++)
    {
        int j = 0;
        for (j = i + 1; j < len_particles; j++)
        {
            dx = xpos[i] - xpos[j];
            dy = ypos[i] - ypos[j];
            if (fabs(dx) > 0.5 * box_l)
            {
                dx *= 1 - box_l / fabs(dx);
            }
            if (fabs(dy) > 0.5 * box_l)
            {
                dy *= 1 - box_l / fabs(dy);
            }
            dr = sqrt(dx * dx + dy * dy);
            distances_arr[k] = dr;
            if (dr <= cut)
            {
                inv_dr_1 = 1.0 / dr;
                inv_dr_2 = inv_dr_1 * inv_dr_1;
                inv_dr_6 = inv_dr_2 * inv_dr_2 * inv_dr_2;
                f = (12 * A * (inv_dr_1 * inv_dr_6 * inv_dr_6) - 6 * B * (inv_dr_1 * inv_dr_6));
                force_arr[k] = f;
                e = (A * (inv_dr_6 * inv_dr_6) - B * inv_dr_6);
                energy_arr[k] = e;
                xacc[i] += (f * dx * inv_dr_1) * inv_mass_of_argon;
                yacc[i] += (f * dy * inv_dr_1) * inv_mass_of_argon;
                xacc[j] -= (f * dx * inv_dr_1) * inv_mass_of_argon;
                yacc[j] -= (f * dy * inv_dr_1) * inv_mass_of_argon;
            }
            else
            {
                force_arr[k] = 0.;
                energy_arr[k] = 0.;
            }
            k++;
        }
    }
}

void compute_energies(int len_particles, const double *xpos, const double *ypos, double *distances_arr, double box_l,
                      double *energy_arr, double cut)
{
    double dx, dy, dr, e;
    int k = 0;
    int i = 0;
    // values of A and B where determined from from A. Rahman "Correlations in the Motion of Atoms in Liquid Argon",
    // Physical Review 136 pp. A405–A411 (1964)
    double A = 1.363e-134; // joules / metre ^{12}
    double B = 9.273e-78; // joules / meter ^{6}
    for (i = 0; i < len_particles - 1; i++)
    {
        int j = 0;
        for (j = i + 1; j < len_particles; j++)
        {
            dx = xpos[i] - xpos[j];
            dy = ypos[i] - ypos[j];
            if (fabs(dx) > 0.5 * box_l)
            {
                dx *= 1 - box_l / fabs(dx);
            }
            if (fabs(dy) > 0.5 * box_l)
            {
                dy *= 1 - box_l / fabs(dy);
            }
            dr = sqrt(dx * dx + dy * dy);
            distances_arr[k] = dr;
            if (dr <= cut)
            {
                e = (A * pow(dr, -12.) - B * pow(dr, -6.));
	            energy_arr[k] = e;
            }
            else
            {
                energy_arr[k] = 0.;
            }
	        k++;
	    }
    }
}

double compute_pressure(int number_of_particles, const double *xpos, const double *ypos, double box_length,
                        double temperature, double cut)
{
	double pres = 0.;
	int i, j;
	double dx, dy, dr, f;
	double A = 1.363e-134; // joules / metre ^{12}
    double B = 9.273e-78; // joules / meter ^{6}
	for (i = 0; i < number_of_particles - 1; i++)
	{
		for (j = i + 1; j < number_of_particles; j++)
		{
			dx = xpos[i] - xpos[j];
            dy = ypos[i] - ypos[j];
            if (fabs(dx) > 0.5 * box_length)
            {
                dx *= 1 - box_length / fabs(dx);
            }
            if (fabs(dy) > 0.5 * box_length)
            {
                dy *= 1 - box_length / fabs(dy);
            }
            dr = sqrt(dx * dx + dy * dy);
            if (dr <= cut)
            {
                f = (12 * A * pow(dr, -13.) - 6 * B * pow(dr, -7.));
                pres += f * dr;
            }
		}
	}
	double boltzmann_constant = 1.3806e-23; // joules/kelvin
	pres = 1. / (2 * box_length * box_length) * pres +
	       ((double)number_of_particles / (box_length * box_length) * boltzmann_constant * temperature);
	return pres;
}

void scale_velocities(int len_particles, double *xvel, double *yvel, double average_temp, double temperature)
{
    int i = 0;
    for (i = 0; i < len_particles; i++)
    {
        xvel[i] = xvel[i] * sqrt(temperature / average_temp);
        yvel[i] = yvel[i] * sqrt(temperature / average_temp);
    }
}