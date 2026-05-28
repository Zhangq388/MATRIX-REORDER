#include "rcm.h"
void degreeCalculation(int n, int nz, node* nodes, int* rows, int* cols)
{
    // As degree of each node is considered the #neighbors
    for(int i=0; i<nz; i++)
    {
        if(rows[i] != cols[i])
        {
            nodes[rows[i]].degree += 1;
            nodes[cols[i]].degree += 1;
        }
    }

    // Allocate space for every node's neighbors which is in this case the degree of the node
    #pragma omp parallel for 
    for(int i=0; i<n; i++)
    {
        nodes[i].neighbors = (int *)malloc(nodes[i].degree*(sizeof(int)));
        if(nodes[i].neighbors == NULL) printf("Could not allocate neighbors[%d]!", i);
    }
}

void minimumNode(int n, node* nodes, queue* Q)
{
    int min = 0;
    int first_time = 1;

    // Find the node with the minimum degree
    for(int i=0; i<n; i++)
    {
        if(first_time == 1 && nodes[i].inside_perm == false)
        {
            first_time = 0;
            min = i;
        }
        if(nodes[i].degree < nodes[min].degree && nodes[i].inside_perm == false && nodes[min].inside_perm == false)
        {
            min = i;
        }
    }
    nodes[min].inside_q = true;

    queueAdd(Q, nodes[min]);
}

void findAllNeighbors(int n, int nz, int* rows,int* cols, node* nodes)
{
    for(int i=0; i<nz; i++)
    {
        if(rows[i] != cols[i])
        {
            // Because we only have the lower part of the matrix whenever we add a neighbor we add its symmetric too
            nodes[rows[i]].neighbors[nodes[rows[i]].index++] = cols[i];
            nodes[cols[i]].neighbors[nodes[cols[i]].index++] = rows[i];
        }
    }
}

void findNeighbors(int nz, int numberOfnode, int* rows, int* cols, node* nodes, queue* Q, queue* temp_neighbors)
{
    // Counter that holds how many neighbors are found 
    int  counter = 0;
    node del;
    // Find the neighbors of the corresponding node
    for(int i=0; i<nodes[numberOfnode].degree; i++)
    {
        // If the nodes are neighbors matrix(numberOfnodes,i) == 1 and the node is not in the Queue or the permutation
        if(nodes[nodes[numberOfnode].neighbors[i]].inside_perm == false && nodes[nodes[numberOfnode].neighbors[i]].inside_q == false )
        {
            nodes[nodes[numberOfnode].neighbors[i]].inside_q = true;

            // Add the node as a neighbor of numberOfnode
            queueAdd(temp_neighbors,nodes[nodes[numberOfnode].neighbors[i]]);  
            counter++;
        }
    }
    // Sort the neighbors acording to their degree and add them in the Q queue in increasing order of degree
    queueSort(temp_neighbors);
    for(int i=0; i<counter; i++)
    {
        queueDel(temp_neighbors,&del);
        queueAdd(Q,del);
    }
}

void Cuthill_Mckee(int n, int nz, int* rows, int* cols, unsigned int* permutation)
{
    // Initialize an array of n nodes
    node* nodes = nodeInit(n);
    int   nodes_added = 0;
    int   index = 0;

    // Queue where the nodes wait to be added in the permutation
    queue* Q = queueInit(n);

    // Queue that holds the neighbors of the node added in the permutation
    queue* temp_neighbors = queueInit(n);

    // Node extracted from Q and added in the permutation
    node extracted_node;

    // Calcutation of all the degrees
    degreeCalculation(n, nz, nodes, rows, cols);

    // Find all the neighbors in advance
    findAllNeighbors(n, nz, rows, cols, nodes);
    
    // Find the node with the minimum degree and its neighbors 
    minimumNode(n, nodes, Q);

    // While the Q is not empty 
    while(!Q->empty)
    {
        // Extract the first node in the Q
        queueDel(Q, &extracted_node);
        nodes[extracted_node.num].inside_perm = true;

        // Add the node to the permutation
        permutation[index] = extracted_node.num;
        nodes_added++;
        ++index;
        
        // Find the neighbors of the extracted node and add them in increasing order of degree in the Q
        findNeighbors(nz, extracted_node.num, rows, cols, nodes, Q, temp_neighbors);
        if(Q->empty == 1 && nodes_added != n)
        {
            minimumNode(n, nodes, Q);
        }
    }

    // Deallocate the space used and no longer needed 
    queueDelete(Q);
    queueDelete(temp_neighbors);
    nodeDelete(nodes, n);
}
// Reverse the indices in the permutation array

void R_Cuthill_Mckee(int n, int nz, int* rows, int* cols, unsigned int* permutation)
{
    Cuthill_Mckee(n, nz, rows, cols, permutation);
    //Value to hold how many swaps are going to happen, if n is even then the number of swaps needed is floor((n-1)/2)+1 else is floor(n/2)+1
    int new_n = n;
    new_n = new_n/2;
    
    int temp;
    // Apply the swap
    #pragma omp parallel for
    for(int i=0; i<new_n; i++)
    {
        temp = permutation[n-i-1];
        permutation[n-i-1] = permutation[i];
        permutation[i] = temp;
    }
}
