/************************************************
** This is a local search solver for Minimum Vertex Cover.                                                       
************************************************/


/************************************************
** Date:	2011.7.1  
** TSEWF (Two Stage Exchange and Weighting with Forgetting) 
** Author: Shaowei Cai, shaowei_cai@126.com    
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

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <sys/times.h>
#include <cmath>

using namespace std;


#define pop(stack) stack[--stack ## _fill_pointer]
#define push(item, stack) stack[stack ## _fill_pointer++] = item

/*max vertex count and max edge count*/
#define	MAXV	5000000
#define MAXE	80000000


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
Edge	edge[MAXE];  
int		edge_weight[MAXE];

/*structures about vertex*/
int		dscore[MAXV];			//dscore of v
long long	time_stamp[MAXV];
int		best_cov_v;		//the vertex of the highest dscore in C

//from vertex to it's edges and neighbors
int*	v_edges[MAXV];	//edges related to v, v_edges[i][k] means vertex v_i's k_th edge
int*	v_adj[MAXV];		//v_adj[v_i][k] = v_j(actually, that is v_i's k_th neighbor)
int		v_edge_count[MAXV];	//amount of edges (neighbors) related to v


/* structures about solution */
//current candidate solution
int		c_size;						//cardinality of C
int		v_in_c[MAXV];				//a flag indicates whether a vertex is in C
int		remove_cand[MAXV];			//remove candidates, an array consists of only vertices in C, not including tabu_remove
int		index_in_remove_cand[MAXV];
int		remove_cand_size;

//best solution found
int		best_c_size;
int		best_v_in_c[MAXV];			//a flag indicates whether a vertex is in best solution
double  best_comp_time;
long    best_step;


//uncovered edge stack
int		uncov_stack[MAXE];		//store the uncov edge number
int		uncov_stack_fill_pointer;
int		index_in_uncov_stack[MAXE];//which position is an edge in the uncov_stack


//CC and taboo
int 	conf_change[MAXV];
int		tabu_remove=0;

//smooth 
int		ave_weight=1;
int		delta_total_weight=0;
int		threshold;
float	p_scale=0.3;//w=w*p


/* functions declaration */
int build_instance(char *filename);
void init_sol();
void cover_LS();
void add(int v);
void remove(int v);
void update_edge_weight();
void cover_rest_edges();
int check_solution();



void update_best_sol()
{
	int i;

	for (i=1;i<=v_num;i++)
	{
		best_v_in_c[i] = v_in_c[i];
	}
	
	best_c_size = c_size;
	times(&finish);
	best_comp_time = double(finish.tms_utime - start.tms_utime + finish.tms_stime - start.tms_stime)/sysconf(_SC_CLK_TCK);
	best_comp_time = round(best_comp_time * 100)/100.0;
	best_step = step;
	
}



int v_edge_count_tmp[MAXV];

int build_instance(char *filename)
{
	char line[1024];
	char tempstr1[10];
	char tempstr2[10];
	int  v,e;
	
	char	tmp;
	int		v1,v2;
	
	ifstream infile(filename);
    if(infile==NULL) return 0;

	/*** build problem data structures of the instance ***/
	infile.getline(line,1024);
	while (line[0] != 'p')
		infile.getline(line,1024);
	sscanf(line, "%s %s %d %d", tempstr1, tempstr2, &v_num, &e_num);

	/* read edges and compute v_edge_count */
	for (v=1; v<=v_num; v++)
		v_edge_count[v] = 0;
	
	for (e=0; e<e_num; e++)
	{
		infile>>tmp>>v1>>v2;
		v_edge_count[v1]++;
		v_edge_count[v2]++;
		
		edge[e].v1 = v1;
		edge[e].v2 = v2;
	}
	infile.close();
	
	/* build v_adj and v_edges arrays */
	for (v=1; v<=v_num; v++)
	{
		v_adj[v] = new int[v_edge_count[v]];
		v_edges[v] = new int[v_edge_count[v]];
	}

		
	for(v=1; v<=v_num; v++)
		v_edge_count_tmp[v]=0;
	for (e=0; e<e_num; e++)
	{
		
		v1=edge[e].v1;
		v2=edge[e].v2;

		v_edges[v1][v_edge_count_tmp[v1]] = e;
		v_edges[v2][v_edge_count_tmp[v2]] = e;

		v_adj[v1][v_edge_count_tmp[v1]] = v2;
		v_adj[v2][v_edge_count_tmp[v2]] = v1;

		v_edge_count_tmp[v1]++;
		v_edge_count_tmp[v2]++;
	}
	

	return 1;

}


void free_memory()
{
	for (int v=1; v<=v_num; v++)
	{
		delete[] v_adj[v];
		delete[] v_edges[v];
	}
}

void reset_remove_cand()
{
	int v,j;
	j=0;
	for (v=1;v<=v_num;v++)
	{
		if(v_in_c[v]==1)// && v!=tabu_remove)
		{
			remove_cand[j] = v;
			index_in_remove_cand[v]=j;
			j++;
		}
		else index_in_remove_cand[v]=0;
	}
	
	remove_cand_size = j;
	
}




void update_target_size()
{
	c_size--;
	
	int v;
	int max_improvement;
	int max_vertex;//vertex with the highest improvement in C

	max_improvement=-100000000;
	for (int v=1; v<=v_num; ++v)
	{
		if(v_in_c[v]==0)continue;
		if (dscore[v]>max_improvement)
		{
			max_improvement = dscore[v];
			max_vertex = v;
		}
	}
	remove(max_vertex);
	
	reset_remove_cand();
}




//update the best vertex in C 

void update_best_cov_v()
{
	int i,v;
	best_cov_v = remove_cand[0];
	for (i=1; i<remove_cand_size; ++i)
	{
		v = remove_cand[i];
		if(v==tabu_remove) continue;
		
		if( dscore[v] < dscore[best_cov_v])
			continue;
		else if( dscore[v]> dscore[best_cov_v] )
			best_cov_v = v;
		else if (time_stamp[v]<time_stamp[best_cov_v])
			best_cov_v = v;
	}
}



inline
void uncover(int e) 
{
	index_in_uncov_stack[e] = uncov_stack_fill_pointer;
	push(e,uncov_stack);
}


inline
void cover(int e)
{
	int index,last_uncov_edge;

	//since the edge is satisfied, its position can be reused to store the last_uncov_edge
	last_uncov_edge = pop(uncov_stack);
	index = index_in_uncov_stack[e];
	uncov_stack[index] = last_uncov_edge;
	index_in_uncov_stack[last_uncov_edge] = index;
}



int best_vertex_improvement;
int best_count;
int best_array[MAXV];

void init_sol()
{
	int i,v,e;

	/*** build solution data structures of the instance ***/
	//init vertex cover
	for (v=1; v<=v_num; v++)
	{
		v_in_c[v] = 0;
		dscore[v] = 0;
		
		conf_change[v] = 1;
		time_stamp[v]= 0; // to break ties
	}

	for (e=0; e<e_num; e++)
	{
		edge_weight[e] = 1;
		dscore[edge[e].v1]+=edge_weight[e];
		dscore[edge[e].v2]+=edge_weight[e];
	}

	//init uncovered edge stack and cover_vertrex_count_of_edge array
	uncov_stack_fill_pointer = 0;
	for (e=0; e<e_num; e++)
		uncover(e);

	

	for (i=0; uncov_stack_fill_pointer>0; )
	{
		best_vertex_improvement = 0;
		best_count = 0;
		for (v=1; v<=v_num; ++v)
		{
			if(v_in_c[v]==1)continue;

			if (dscore[v]>best_vertex_improvement)
			{
				best_vertex_improvement = dscore[v];
				best_array[0] = v;
				best_count = 1;
			}
			else if (dscore[v]==best_vertex_improvement)
			{
				best_array[best_count] = v;
				best_count++;
			}
		}

		if(best_count>0)
		{
			add(best_array[rand()%best_count]);
			++i;
		}
	}

	c_size = i;
	
	update_best_sol();
	
	reset_remove_cand();
	
	update_best_cov_v();
	
}


void add(int v)
{
	v_in_c[v] = 1;
	dscore[v] = -dscore[v];
	
	int i,e,n;

	int edge_count = v_edge_count[v];
	
	for (i=0; i<edge_count; ++i)
	{
		e = v_edges[v][i];// v's i'th edge
		n = v_adj[v][i];//v's i'th neighbor

		if (v_in_c[n]==0)//this adj isn't in cover set
		{
			dscore[n] -= edge_weight[e];
			conf_change[n] = 1;

			cover(e);
		}
		else
		{
			dscore[n] += edge_weight[e];
		}
	}
	
}

void remove(int v)
{
	v_in_c[v] = 0;
	dscore[v] = -dscore[v];
	conf_change[v] = 0;

	int i,e,n;

	int edge_count = v_edge_count[v];
	for (i=0; i<edge_count; ++i)
	{
		e = v_edges[v][i];
		n = v_adj[v][i];

		if (v_in_c[n]==0)//this adj isn't in cover set
		{
			dscore[n] += edge_weight[e];
			conf_change[n] = 1;

			uncover(e);
		}
		else
		{
			dscore[n] -= edge_weight[e];
		}
	}

}

/*On solution*/

void print_solution()
{
	int mis_vertex_count=0;
	
	for (int i=1; i<=v_num; i++)
	{
		if (best_v_in_c[i]!=1)
			mis_vertex_count++;
	}
	
	if(mis_vertex_count+best_c_size!=v_num)
		cout<<"The size of independent set + the size of vertex cover is not equal to |V(G)|!"<<endl;
	
	cout<<"c Best found independent set size = "<<mis_vertex_count<<endl;
	cout<<"c The following output is the found independent set."<<endl;


	for (int i=1; i<=v_num; i++)
	{
		if (best_v_in_c[i]!=1)//output max independent set
			cout<<i<<'\t';
	}
	cout<<endl;

}

//check whether the solution found is a proper solution
int check_solution()
{
	int e;
	
	for(e=0; e<e_num; ++e)
	{
		if(best_v_in_c[edge[e].v1]!=1 && best_v_in_c[edge[e].v2]!=1)
		{
			cout<<"uncovered edge "<<e<<endl;
			return 0;
		}
	}
	
	return 1;
}

