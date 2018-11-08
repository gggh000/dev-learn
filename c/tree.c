#include <stdio.h>
#include "node.h"

void printTree(node * pNode) {
    node * lNode = pNode;

    while (pNode != NULL) {
        printNode(lNode);

        if (lNode->ptrNext != NULL) {
            lNode = lNode->ptrNext;                
        } else {
            return; 
        }
    }
}

int insertNodeTail(node * pNode, int pValue) {
    node * lNode; 
    lNode = malloc(sizeof(node));
    lNode->value = pValue;
    lNode->ptrNext = NULL;

    node * lNodeTmp; 
    lNodeTmp = pNode;

    if (lNodeTmp == NULL) {
        printf("insertNodeTail: list is empty! (head ptr is NULL).\n");
        return 1;
    }    
    while (lNodeTmp->ptrNext != NULL) {
        lNodeTmp = lNodeTmp->ptrNext;
    }

    lNodeTmp->ptrNext = lNode;
    return 0;
}

//  Inserts node at the front of linked list.
//  input:
//  - node * ptrNode - pointer to the node pointer. Note that since head pointer of the list is changin
//  when we are inserting at the front, therefore pointer to head pointer must be passed.
//  - int value - node value to be inserted as new node.

int insertNodeFront(ptrNode * pNode, int pValue) {
    node * lNode;
    lNode = malloc(sizeof(node));
    lNode->value = pValue;
    lNode->ptrNext = NULL;

    printf("insertNodeFront: created new node: %08x: %04d.\n", lNode, lNode->value);

    if (lNode == NULL) {
        printf("malloc error:");
        return 1;
    }

    lNode->ptrNext = * pNode;
    * pNode = lNode;
    return 0; 
}

int main()
{
    node * n1;
    n1 = malloc(sizeof(node));

    n1->value = 100;
    n1->ptrNext = NULL;

    printf("\nBefore insert...\n");
    printTree(n1);
    insertNodeFront(&n1, 101);
    insertNodeFront(&n1, 150);
    insertNodeFront(&n1, 222);
    insertNodeFront(&n1, 455);

    printf("\nAfter insert front...\n");
    printTree(n1);
    insertNodeTail(n1, 1000);
    insertNodeTail(n1, 1330);
    insertNodeTail(n1, 5000);

    printf("\nAfter insert back...\n");
    printTree(n1);

    printf("done.\n");
    return 0;
}

