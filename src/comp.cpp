#include "comp.h"
#include <math.h>
#include <stdio.h>

void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc,
		double *yacc, double *distances_arr, double *xforce, double *yforce, double box_l,
		double *force_arr)
{
    int ii = 0;
    double dx, dy, dr, f;
    for (ii = 0; ii < len_particles; ii++)
    {
        xacc[ii] = 0.;
        yacc[ii] = 0.;
    }
    int k = 0;
    int i = 0;
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
            f = (1.89774e-13 * pow(dr, -13.) - 5.1186e-19 * pow(dr, -7.));
	        force_arr[k] = f;
            xforce[i] += f * dx / dr;
            yforce[i] += f * dy / dr;
            xforce[j] -= f * dx / dr;
            yforce[j] -= f * dy / dr;
	        k++;
            xacc[i] += (f * dx / dr) / 66.234e-27;
            yacc[i] += (f * dy / dr) / 66.234e-27;
            xacc[j] -= (f * dx / dr) / 66.234e-27;
            yacc[j] -= (f * dy / dr) / 66.234e-27;
        }
    }
}

double compute_pressure(int number_of_particles, const double *xpos, const double *ypos, double box_length,
                        double temperature)
{
	int k = 0;
	double pres = 0.;
	int i, j;
	double dx, dy, dr, f;
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
            f = (1.89774e-13 * pow(dr, -13.) - 5.1186e-19 * pow(dr, -7.));
			pres += f * dr;
		}
	}
	pres = 1. / (3 * box_length * box_length) * pres +
	       ((double)number_of_particles / (box_length * box_length) * 1.3806e-3 * temperature);
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