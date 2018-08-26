#include "comp.h"
#include <math.h>
#include <stdio.h>
#include <iostream>

void compute_accelerations(int len_particles, const double *xpos,
                           const double *ypos, double *xacc, double *yacc,
                           double *distances_arr, double box_l,
                           double *force_arr, double *energy_arr, double cut,
                           double ac, double bc, double massc)
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
    double inv_dr_1;
    // values of A and B where determined from from A. Rahman "Correlations
    // in the Motion of Atoms in Liquid Argon", Physical Review 136 pp.
    // A405–A411 (1964)
    double A = ac; // joules / metre ^{12}
    double B = bc; // joules / meter ^{6}
    double atomic_mass_unit = 1.660539e-27; // kilograms
    double mass_amu = massc;  // amu
    double mass_kg = mass_amu * atomic_mass_unit; // kilograms
    double inv_mass_kg = 1 / mass_kg;
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
            inv_dr_1 = 1 / dr;
            distances_arr[k] = dr;
            if (dr <= cut)
            {
                f = lennard_jones_force(A, B, inv_dr_1);
                force_arr[k] = f;
                e = lennard_jones_energy(A, B, inv_dr_1);
                energy_arr[k] = e;
                xacc[i] += (f * dx * inv_dr_1) * inv_mass_kg;
                yacc[i] += (f * dy * inv_dr_1) * inv_mass_kg;
                xacc[j] -= (f * dx * inv_dr_1) * inv_mass_kg;
                yacc[j] -= (f * dy * inv_dr_1) * inv_mass_kg;
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

void compute_energies(int len_particles, const double *xpos,
                      const double *ypos, double *distances_arr, double box_l,
                      double *energy_arr, double cut, double ac, double bc)
{
    double dx, dy, dr, e, inv_dr_1;
    int k = 0;
    int i = 0;
    // values of A and B where determined from from A. Rahman "Correlations
    // in the Motion of Atoms in Liquid Argon", Physical Review 136 pp.
    // A405–A411 (1964)
    double A = ac; // joules / metre ^{12}
    double B = bc; // joules / meter ^{6}
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
            inv_dr_1 = 1. / dr;
            distances_arr[k] = dr;
            if (dr <= cut)
            {
                e = lennard_jones_energy(A, B, inv_dr_1);
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

double compute_pressure(int number_of_particles, const double *xpos,
                        const double *ypos, double box_length,
                        double temperature, double cut, double ac, double bc)
{
	double pres = 0.;
	int i, j;
	double dx, dy, dr, f, inv_dr_1;
	double A = ac; // joules / metre ^{12}
    double B = bc; // joules / meter ^{6}
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
            inv_dr_1 = 1. / dr;
            if (dr <= cut)
            {
                f = lennard_jones_force(A, B, inv_dr_1);
                pres += f * dr;
            }
		}
	}
	double boltzmann_constant = 1.3806e-23; // joules/kelvin
	pres = 1. / (2 * box_length * box_length) * pres +
	       ((double)number_of_particles / (box_length * box_length) *
           boltzmann_constant * temperature);
	return pres;
}

void scale_velocities(int len_particles, double *xvel, double *yvel,
                      double average_temp, double temperature)
{
    int i = 0;
    for (i = 0; i < len_particles; i++)
    {
        xvel[i] = xvel[i] * sqrt(temperature / average_temp);
        yvel[i] = yvel[i] * sqrt(temperature / average_temp);
    }
}

double lennard_jones_force(double A, double B, double inv_dr_1)
{
    double inv_dr_2 = inv_dr_1 * inv_dr_1;
    double inv_dr_6 = inv_dr_2 * inv_dr_2 * inv_dr_2;
    double f = (12 * A * (inv_dr_1 * inv_dr_6 * inv_dr_6) - 6 * B *
               (inv_dr_1 * inv_dr_6));
    return f;
}

double lennard_jones_energy(double A, double B, double inv_dr_1)
{
    double inv_dr_2 = inv_dr_1 * inv_dr_1;
    double inv_dr_6 = inv_dr_2 * inv_dr_2 * inv_dr_2;
    double e = (A * (inv_dr_6 * inv_dr_6) - B * inv_dr_6);
    return e;
}

void get_distances(int len_particles, double *xpositions, double *ypositions,
                   double box_l, double *distances, double *xdistances,
                   double *ydistances)
{
    double dx, dy, dr;
    int k = 0;
    int i = 0;
    for (i = 0; i < len_particles - 1; i++)
    {
        int j = 0;
        for (j = i + 1; j < len_particles; j++)
        {
            dx = xpositions[i] - xpositions[j];
            dy = ypositions[i] - ypositions[j];
            if (fabs(dx) > 0.5 * box_l)
            {
                dx *= 1 - box_l / fabs(dx);
            }
            if (fabs(dy) > 0.5 * box_l)
            {
                dy *= 1 - box_l / fabs(dy);
            }
            dr = sqrt(dx * dx + dy * dy);
            distances[k] = dr;
            xdistances[k] = dx;
            ydistances[k] = dy;
            k += 1;
        }
    }
}
