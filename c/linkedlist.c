#include <stdio.h>

typedef struct sNode 
{
    int value;
    struct sNode * ptrNext;
} node; 

void printNode(node * pNode) {
    if (pNode == NULL) {
        printf("Error. Pointer is null.");
        return ;
    }

    printf("%x: %d.\n", pNode->ptrNext, pNode->value);
    return;
}

void printLinkedList(node * pNode) {
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

int main()
{
    node * n1;
    n1 = malloc(sizeof(node));

    n1->value = 100;
    n1->ptrNext = NULL;

    //printNode(n1);        
    printLinkedList(n1);
    printf("done.\n");
    return 0;
}

