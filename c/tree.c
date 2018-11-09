#include <stdio.h>
#include "node.h"

void printTree(node * pNode) {
    int DEBUG = 0;
    node * lNode = pNode;

    if (pNode == NULL) {
        if (DEBUG == 1) {
            printf("leaf is reached!\n");
        }
        return;
    }
    
    printTree(pNode->ptrLeft);
    printf("%08x: %04d.\n", pNode, pNode->value);
    printTree(pNode->ptrRight);
}

int insertNode(ptrNode * pNode, int pValue) {
    int DEBUG = 0;

    // take care of case where pNode is NULL.

    if (*pNode == NULL) {
            printf("node is NULL now, creating node at the tip, value: %04d...\n", pValue);    

        *pNode = malloc(sizeof(node));
        (*pNode)->ptrLeft = NULL;
        (*pNode)->ptrRight = NULL;
        (*pNode)->value = pValue;
        return 0;
    }

    if (pValue < (*pNode)->value) {
        if (DEBUG == 1) {
            printf("recursing with left pointer...\n");
        }
        insertNode(&(*pNode)->ptrLeft, pValue);
    } else {
        if (DEBUG == 1) {
            printf("recursing with right pointer...\n");
        }
        insertNode(&(*pNode)->ptrRight, pValue);
    }
    return 0;
}

int main()
{
    node * n1;

    n1 = NULL;

    insertNode(&n1, 101);
    insertNode(&n1, 222);
    insertNode(&n1, 455);
    insertNode(&n1, 15);
    insertNode(&n1, 1);
    insertNode(&n1, 10);
    insertNode(&n1, 1220);

    printf("\nAfter insert back...\n");
    printTree(n1);

    printf("done.\n");
    return 0;
}

