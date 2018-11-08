#include <stdio.h>
#include "node.h"

void printNode(node * pNode) {
    if (pNode == NULL) {
        printf("Error. Pointer is null.");
        return ;
    }

    printf("%08x: %04d.\n", pNode, pNode->value);
    return;
}

