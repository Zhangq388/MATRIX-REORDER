/*
 *	File	: 
 *
 *	Title	: Demo Producer/Consumer. Only the Queue part is obtained and modified.
 *
 *	Short	: A solution to the producer consumer problem using
 *		pthreads.	
 *
 *	Long 	:
 *
 *	Author	: Andrae Muys
 *
 *	Date	: 18 September 1997
 *
 *	Revised	: Doinakis Michalis
 *      e-mail: doinakis@ece.auth.gr
 */
#include "queue.h"

queue *queueInit(int n)
{
  queue *q;

  q = (queue *)malloc(sizeof (queue));
  if(q == NULL)
    printf("Couldn't allocate memory for q .File: queue.c Function: queueInit");
  q->buf = (node*)malloc(n*sizeof(node));
  if(q->buf == NULL)
    printf("Couldn't allocate memory for q->buf .File: queue.c Function: queueInit");

  q->empty = 1;
  q->full = 0;
  q->head = 0;
  q->tail = 0;
  q->buf_size = n;
  return (q);
}

void queueSort(queue *q)
{
  // Sort the queue in increasing order of degree
    node temp;
    for(int i=q->head;i<q->tail; i++)
    {
        for(int j=i+1; j<q->tail;j++)
        {
            if(q->buf[i].degree >= q->buf[j].degree)
            {
                temp = q->buf[i];
                q->buf[i] = q->buf[j];
                q->buf[j] = temp;
            }
        }
    }
}

void queueDelete(queue *q)
{
  free(q->buf);
  free(q);
}


//changed the in variable to be type node
void queueAdd(queue *q, node in)
{
  q->buf[q->tail] = in;
  q->tail++;
  if (q->tail == q->buf_size)
    q->tail = 0;
  if (q->tail == q->head)
    q->full = 1;
  q->empty = 0;

  return;
}

// Changed the out variable to be type node
void queueDel(queue *q, node *out)
{
  *out = q->buf[q->head];
  q->head++;
  if (q->head == q->buf_size)
    q->head = 0;
  if (q->head == q->tail)
    q->empty = 1;
  q->full = 0;

  return;
}

node *nodeInit(int n)
{
  node *nodes = (node *)malloc(n*sizeof(node));
  if(nodes == NULL)
    printf("Unable to allocate nodes.(Function nodeInit queue.c file");

  for(int i=0;i<n;i++)
  {
    nodes[i].num = i;
    nodes[i].degree = 0;
    nodes[i].inside_perm = false;
    nodes[i].inside_q = false;
    nodes[i].neighbors = NULL;
    nodes[i].index = 0;
  }

  return nodes;
}

void nodeDelete(node *node,int n)
{
  for(int i=0; i<n; i++)
  {
    free(node[i].neighbors);
  }
  free(node);
}





