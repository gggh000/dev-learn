#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#define THREAD_NO 1000
/* single global variable */
/* shared, accessible, modifiable by all threads */

int accum = 0;

//  Function to compute square on x. 

void* square(void* x) {
    int xi = (int)x;
    accum += xi * xi;
    return NULL;                            // nothing to return, prevent warning.
}

int main(int argc, char** argv) 
{
    int i;
    pthread_t ths[THREAD_NO];

    for (i = 0; i < THREAD_NO; i++) {
        pthread_create(&ths[i], NULL, square, (void*)(i + 1));
    }

    for (i = 0; i < THREAD_NO; i++) {
        void* res;
        pthread_join(ths[i], &res);
    }

    printf("accum = %d\n", accum);
    return 0;
}
