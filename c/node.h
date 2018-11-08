#ifndef NODE_H
#define NODE_H

typedef struct sNode
{
    int value;
    struct sNode * ptrNext;
} node;

typedef node * ptrNode;
void printNode(node * pNode);

#endif
