#include "force.h"
#include <math.h>
#include <stdio.h>

void compute_accelerations(int len_particles, const double *xpos, const double *ypos, double *xacc, double *yacc,
                           double *distances, double *forces, double box_length)
{
    int i = 0;
    double dx, dy, dr, f;
    for (i = 0; i < len_particles; i++)
    {
        xacc[i] = 0.;
        yacc[i] = 0.;
    }
    int k = 0;
    for (i = 0; i < len_particles - 1; i++)
    {
        int j = 0;
        for (j = i + 1; j < len_particles; j++)
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
            distances[k] = dr;
            k++;
            f = 48. * pow(dr, -13.) - 24. * pow(dr, -7.);
            forces[k] = f;
            xacc[i] += f * dx / dr;
            yacc[i] += f * dy / dr;
            xacc[j] -= f * dx / dr;
            yacc[j] -= f * dy / dr;
        }
    }
}


void compute_force(int len_particles, const double *xpos, const double *ypos, double *xforce, double *yforce,
                   double box_length)
{
    int i = 0;
    double dx, dy, dr;
    for (i = 0; i < len_particles - 1; i++)
    {
        int j = 0;
        for (j = i + 1; j < len_particles; j++)
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
            xforce[i] += (48. * pow(dr, -13.) - 24. * pow(dr, -7.)) * dx / dr;
            yforce[i] += (48. * pow(dr, -13.) - 24. * pow(dr, -7.)) * dy / dr;
            xforce[j] -= (48. * pow(dr, -13.) - 24. * pow(dr, -7.)) * dx / dr;
            yforce[j] -= (48. * pow(dr, -13.) - 24. * pow(dr, -7.)) * dy / dr;
        }
    }
}


void compute_sd(int len_particles, const double *xpos, const double *ypos, double *energy, double *xforce,
                double *yforce, double box_length)
{
    int i = 0;
    double dx, dy, dr;
    for (i = 0; i < len_particles - 1; i++)
    {
        int j = 0;
        for (j = i + 1; j < len_particles; j++)
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
            energy[i] += (4. * pow(dr, -12.) - 4. * pow(dr, -6.)) * dx / dr;
            energy[j] += (4. * pow(dr, -12.) - 4. * pow(dr, -6.)) * dx / dr;
            xforce[i] += (-48. * pow(dr, -13.) + 24. * pow(dr, -7.)) * dx / dr;
            yforce[i] += (-48. * pow(dr, -13.) + 24. * pow(dr, -7.)) * dy / dr;
            xforce[j] -= (-48. * pow(dr, -13.) + 24. * pow(dr, -7.)) * dx / dr;
            yforce[j] -= (-48. * pow(dr, -13.) + 24. * pow(dr, -7.)) * dy / dr;
        }
    }
}

void compute_energy_and_force(int len_particles, const double *xpos, const double *ypos, double *energy, double *xforce,
                              double *yforce, double *xforcedash, double *yforcedash, double box_length)
{
    int i = 0;
    double dx, dy, dr;
    for (i = 0; i < len_particles - 1; i++)
    {
        int j = 0;
        for (j = i + 1; j < len_particles; j++)
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
            energy[i] += 4. * pow(dr, -12.) - 4. * pow(dr, -6.);
            energy[j] += 4. * pow(dr, -12.) - 4. * pow(dr, -6.);
            xforce[i] += (-48. * pow(dr, -13.) + 24. * pow(dr, -7.)) * dx / dr;
            yforce[i] += (-48. * pow(dr, -13.) + 24. * pow(dr, -7.)) * dy / dr;
            xforcedash[i] += (624 * pow(dr, -14.) - 168 * pow(dr, -8.)) * dx / dr;
            yforcedash[i] += (624 * pow(dr, -14.) - 168 * pow(dr, -8.)) * dy / dr;
            xforce[j] += (-48. * pow(dr, -13.) + 24. * pow(dr, -7.)) * dx / dr;
            yforce[j] += (-48. * pow(dr, -13.) + 24. * pow(dr, -7.)) * dy / dr;
            xforcedash[j] += (624 * pow(dr, -14.) - 168 * pow(dr, -8.)) * dx / dr;
            yforcedash[j] += (624 * pow(dr, -14.) - 168 * pow(dr, -8.)) * dy / dr;
        }
    }
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