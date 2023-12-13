/************************************************
** This is a local search solver for Minimum Vertex Cover.                                                       
************************************************/


/************************************************
** Date:	2011.7.1  
** TSEWF (Two Stage Exchange and Weighting with Forgetting) 
** Author: Shaowei Cai, shaoweicai.cs@gmail.com    
**		   School of EECS, Peking University   
**		   Beijing, China                      
** 
** Date:    	2011.10.28
** Modify: Shaowei Cai
** use dynamic memory for v_adj[][] and v_edge[][], tidy codes.                                                        
************************************************/

/************************************************
** NuMVC Version 2011.11.7                                                    
************************************************/


/************************************************
** NuMVC Version 2015.8.20     
** This version uses dynamic memory allocation for arrays, and
** implements the initialization function via heap (an advanced data
** structure). These two improvements make NuMVC can handle massive
** graphs well and much more efficient.                                             
************************************************/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <string.h>


#include <sys/times.h>
#include <cmath>

using namespace std;


#define pop(stack) stack[--stack ## _fill_pointer]
#define push(item, stack) stack[stack ## _fill_pointer++] = item

tms start, finish;
int start_time;

struct Edge{
	int v1;
	int v2;
};

/*parameters of algorithm*/
long long	max_steps;			//step limit
int			cutoff_time;			//time limit
long long	step;
int			optimal_size;			//terminate the algorithm before step limit if it finds a vertex cover of optimal_size

/*parameters of the instance*/
int		v_num;//|V|: 1...v
int		e_num;//|E|: 0...e-1

/*structures about edge*/
Edge*	edge;
int*		edge_weight;

/*structures about vertex*/
int*	dscore;						//dscore of v
long long*	time_stamp;
int		best_cov_v;		//the vertex of the highest dscore in C

//from vertex to it's edges and neighbors
int**	v_edges;			//edges related to v, v_edges[i][k] means vertex v_i's k_th edge
int**	v_adj;				//v_adj[v_i][k] = v_j(actually, that is v_i's k_th neighbor)
int*	v_edge_count;		//amount of edges (neighbors) related to v


/* structures about solution */
//current candidate solution
int		c_size;						//cardinality of C
int*	v_in_c;						//a flag indicates whether a vertex is in C
int*	remove_cand;				//remove candidates, an array consists of only vertices in C, not including tabu_remove
int*	index_in_remove_cand;
int		remove_cand_size;

//best solution found
int		best_c_size;
int*	best_v_in_c;				//a flag indicates whether a vertex is in best solution
double  best_comp_time;
long    best_step;


//uncovered edge stack
int*	uncov_stack;				//store the uncov edge number
int		uncov_stack_fill_pointer;
int*	index_in_uncov_stack;		//which position is an edge in the uncov_stack


//CC and taboo
int*	conf_change;
int		tabu_remove=0;

//smooth 
int		ave_weight=1;
int		delta_total_weight=0;
int		threshold;
float	p_scale=0.3;//w=w*p

