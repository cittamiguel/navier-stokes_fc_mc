// solver.h
#ifndef SOLVER_H_INCLUDED
#define SOLVER_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif


typedef enum { NONE = 0,
               VERTICAL = 1,
               HORIZONTAL = 2 } boundary;

typedef enum { RED, BLACK } grid_color;

void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt);
void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt);

#ifdef __cplusplus
}
#endif

#endif /* SOLVER_H_INCLUDED */

