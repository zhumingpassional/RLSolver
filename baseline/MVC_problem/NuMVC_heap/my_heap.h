#include "basic.h"

int *pos_in_my_heap;

int *my_heap;
int my_heap_count;

void my_heap_swap(int a, int b)
{
	int t = my_heap[a];
	
	my_heap[a] = my_heap[b];
	pos_in_my_heap[my_heap[a]] = a;
	
	my_heap[b] = t;
	pos_in_my_heap[my_heap[b]] = b;
}

bool my_heap_is_leaf(int pos)
{
	if( (pos>=my_heap_count/2) && (pos<my_heap_count) ) return true;
	else return false;
}

int my_heap_left_child(int pos)
{
	return (2*pos+1);
}

int my_heap_right_child(int pos)
{
	return (2*pos+2);
}

int my_heap_parent(int pos)
{
	return (pos-1)/2;
}

void my_heap_shiftdown(int pos)
{
	while(!my_heap_is_leaf(pos))
	{
		int j = my_heap_left_child(pos);
		int rc = my_heap_right_child(pos);
		if( (rc<my_heap_count) && (dscore[my_heap[rc]]>dscore[my_heap[j]]) )
			j = rc;
		if(dscore[my_heap[pos]]>dscore[my_heap[j]]) return;
		my_heap_swap(pos, j);
		pos = j;
	}
}

void my_heap_insert(int v)
{
	int curr = my_heap_count++;
	my_heap[curr] = v;
	pos_in_my_heap[v] = curr;
	
	while(curr!=0 && dscore[my_heap[curr]]>dscore[my_heap[my_heap_parent(curr)]])
	{
		my_heap_swap(curr, my_heap_parent(curr));
		curr = my_heap_parent(curr);
	}
}

int my_heap_remove_first()
{
	my_heap_swap(0, --my_heap_count);
	if(my_heap_count!=0) my_heap_shiftdown(0);
	return my_heap[my_heap_count];
}

int my_heap_remove(int pos)
{
	if(pos==(my_heap_count-1)) my_heap_count--;
	else
	{
		my_heap_swap(pos, --my_heap_count);
		while( (pos!=0) && (dscore[my_heap[pos]]>dscore[my_heap[my_heap_parent(pos)]]) )
		{
			my_heap_swap(pos, my_heap_parent(pos));
			pos = my_heap_parent(pos);
		}
		if(my_heap_count!=0) my_heap_shiftdown(pos);
	}
	return my_heap[my_heap_count];
}
