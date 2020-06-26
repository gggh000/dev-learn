
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

struct DataBlock {
	unsigned char * dev_bitmap;

	// need book library.

	CPUAnimBitmap * bitmap;
};
void cleanup(DataBlock * d);

void generate_frame(DataBlock * d, int ticks);

__global__ void kernel(unsigned char * ptr, int ticks) 
{
	// map from threadidx/blockidxs to pixel position.

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdy.y + blockIdy.y * blockDim.y;
	int offset = x + y * blockDim.x + gridDim.x;

	// calculate the value at that position.

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

int main()
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	cudaMalloc(viud **)&data.dev_bitmap, bitmap.image_size());
	bitmap.anim_and_exit((void(*)(void *, int) )generate_frame, (void(*) (void*)) cleanup);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void generate_frame(DataBlock * d, int ticks)
{
	dim3 blocks[DIM / 16, DIM / 16];
	dim3 threads[16, 16];
	kernel << <blocks, threads >> > (d->dev_bitmap, ticks);
	cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost);

}

void cleanup(DataBlock * d) {
	cudaFree(d->dev_bitmap);
}

