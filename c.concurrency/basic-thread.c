#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

void* func(void* x) {
    int xi = (int)x;

    printf("Inside thread; x = %d\n", xi);

    return (void*)(xi + 123);
}

int main(int argc, char** argv) {
    pthread_t th;
    pthread_create(&th, NULL, func, (void*)100);

    void* ret_from_thread;
    int ri;
    pthread_join(th, &ret_from_thread);
    ri = (int)ret_from_thread;

    printf("Outside thread, which returned %d\n", ri);
    return 0;
}
