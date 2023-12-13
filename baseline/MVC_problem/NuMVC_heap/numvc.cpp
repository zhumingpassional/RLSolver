#include "numvc.h"

int try_step=100;

void forget_edge_weights()
{
	int v,e;
	int new_total_weight=0;
	
	for(v=1; v<=v_num; v++)
		dscore[v]=0;

	//scale_ave=ave_weight*q_scale;
	for (e = 0; e<e_num; e++)
	{
		edge_weight[e] = edge_weight[e]*p_scale;
			
		new_total_weight+=edge_weight[e];
		
		//update dscore
		if (v_in_c[edge[e].v1]+v_in_c[edge[e].v2]==0){
			dscore[edge[e].v1]+=edge_weight[e];
			dscore[edge[e].v2]+=edge_weight[e];
			}
		else if(v_in_c[edge[e].v1]+v_in_c[edge[e].v2]==1){
			if(v_in_c[edge[e].v1]==1)dscore[edge[e].v1]-=edge_weight[e];
			else  dscore[edge[e].v2]-=edge_weight[e];
		}
	}
	ave_weight=new_total_weight/e_num;

}


void update_edge_weight()
{
	int i,e;
	for(i=0; i<uncov_stack_fill_pointer; ++i)
	{
		e = uncov_stack[i];
		
		edge_weight[e]+= 1;
		dscore[edge[e].v1] += 1;
		dscore[edge[e].v2] += 1;
	}
	
	
	delta_total_weight+=uncov_stack_fill_pointer;
	if(delta_total_weight>=e_num)
	{
		ave_weight+=1;
		delta_total_weight -= e_num;
	}
	
	//smooth weights
	if(ave_weight>=threshold)
	{
		forget_edge_weights();
	}
	
}


void cover_LS()
{
	int		best_add_v;
	int		e,v1,v2;

	step = 1;

	while(1)// wihin cutoff_time
	//while(step<=max_steps)
	{
		if (uncov_stack_fill_pointer == 0)//update best solution if needed
		{
			update_best_sol();
			
			if (c_size==optimal_size)
				return;
				
			update_target_size();
			
			continue;
		}
		
		//if(step>=try_step)// wihin cutoff_time
		if(step%try_step==0)
		{
			times(&finish);
			double elap_time = (finish.tms_utime + finish.tms_stime - start_time)/sysconf(_SC_CLK_TCK);
			if(elap_time >= cutoff_time)return;
		}
		
	
		update_best_cov_v();
		
		remove(best_cov_v);
		
		e = uncov_stack[rand()%uncov_stack_fill_pointer];
		v1 = edge[e].v1;
		v2 = edge[e].v2;
		
		if(conf_change[v1]==0 ) best_add_v=v2;
		else if(conf_change[v2]==0) best_add_v=v1;
		
		else{
			if(dscore[v1]>dscore[v2] || (dscore[v1]==dscore[v2] && time_stamp[v1]<time_stamp[v2]) )
				best_add_v=v1;
			else best_add_v=v2;
		}
		
		add(best_add_v);
		
		int index = index_in_remove_cand[best_cov_v];
		index_in_remove_cand[best_cov_v] = 0;
		
		remove_cand[index] = best_add_v;
		index_in_remove_cand[best_add_v] = index;
		
		time_stamp[best_add_v]=time_stamp[best_cov_v]=step;

		tabu_remove = best_add_v;
		
		update_edge_weight();
		
		step++;

	}

}

int main(int argc, char* argv[])
{
	if (argc != 5) {
		cout << "usage: " << endl;
		return 1;
	}
	int seed;
	//float scale_threshold;
	
	if(build_instance(argv[1])!=1){
		cout<<"can't open instance file"<<endl;
		return -1;
	}
	
		sscanf(argv[2],"%d",&optimal_size);//if you want to stop the algorithm only cutoff time is reached, set optimal_size to 0.
		sscanf(argv[3],"%d",&seed);
		sscanf(argv[4],"%d",&cutoff_time);
	
		threshold = (int)(0.5*v_num); 
	
		srand(seed);

		//cout<<seed<<' ';
//		cout<<"c This is NuMVC, a local search solver for the Minimum Vertex Cover (and also Maximum Independent Set) problem."<<endl;

		times(&start);
		start_time = start.tms_utime + start.tms_stime;

    	init_sol();
    	
	if(c_size + uncov_stack_fill_pointer > optimal_size ) 
	{
		//cout<<"c Start local search..."<<endl;
		cover_LS();
	}
		
		//check solution
		if(check_solution()==1)
		{
			cout<<"c Best found vertex cover size = "<<best_c_size<<endl;
//			print_solution();
			cout<<"c searchSteps = "<<best_step<<endl;
			cout<<"c solveTime = "<<best_comp_time<<endl;
			
			//cout<<best_c_size<<' '<<best_comp_time<<' '<<best_step<<endl;
		}
		else
		{
			cout<<"the solution is wrong."<<endl;
			//print_solution();
		}
	
		free_memory();

	return 0;
}
