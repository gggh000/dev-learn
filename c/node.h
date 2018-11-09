#ifndef NODE_H
#define NODE_H

typedef struct sNode
{
    int value;

    // following member used only for linkedlist.

    struct sNode * ptrNext;

    // following members used only for tree.

    struct sNode * ptrLeft;
    struct sNode * ptrRight;
} node;

typedef node * ptrNode;
void printNode(node * pNode);

#endif
