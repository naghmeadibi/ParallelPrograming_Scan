//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "scan.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuerrors.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"


#define tx threadIdx.x
#define bdx blockDim.x
#define bx blockIdx.x
#define gdx gridDim.x



__global__ void gpuKernelMan(uint8_t* matrix, uint8_t* c, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of);
__global__ void KernelFunc(uint8_t* matrix, uint8_t* c, const int m, const int n, const int f, uint8_t* alpha_to, uint8_t* index_of);
__global__ void KernelMultiply(uint8_t* matrix, uint8_t* matrixd, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of);
__global__ void KernelMultiplyFunc(uint8_t* matrixd, const int m, const int n, const int f,uint8_t* alpha_to, uint8_t* index_of);


__device__ uint8_t multiplyd(uint8_t a, uint8_t b, uint8_t* alpha_to, uint8_t* index_of)
{
	if (a == 0 || b == 0)
	{
		return 0;
	}
	else
	{
		return alpha_to[(uint32_t(index_of[a]) + uint32_t(index_of[b])) % 255];	
	}
	
}
__device__ void matrix_to_vector_multiplyd(uint8_t* a, uint8_t* v, uint8_t* res, int dim, uint8_t* alpha_to, uint8_t* index_of)
{
	for (int i=0; i<dim; i++)
	{
		res[i] = 0;
		for (int j=0; j<dim; j++)
		{
			res[i] ^= multiplyd(a[i*dim+j], v[j], alpha_to, index_of);
		}
	}
	
}
__device__ void matrix_to_matrix_multiplyd(uint8_t* a, uint8_t* b, uint8_t* c, int dim, uint8_t* alpha_to, uint8_t* index_of)
{
	for (int i=0; i<dim; i++)
	{
		for (int j=0; j<dim; j++)
		{
			c[i*dim+j] = 0;
			for (int k=0; k<dim; k++)
			{
				c[i*dim+j] ^= multiplyd(a[i*dim+k], b[k*dim+j], alpha_to, index_of);
			}
		}
	}
}
uint8_t multiply(uint8_t a, uint8_t b, uint8_t* alpha_to, uint8_t* index_of)
{
	if (a == 0 || b == 0)
	{
		return 0;
	}
	else
	{
		return alpha_to[(uint32_t(index_of[a]) + uint32_t(index_of[b])) % 255];	
	}
	
}
void matrix_to_matrix_multiply(uint8_t* a, uint8_t* b, uint8_t* c, int dim, uint8_t* alpha_to, uint8_t* index_of)
{
	for (int i=0; i<dim; i++)
	{
		for (int j=0; j<dim; j++)
		{
			c[i*dim+j] = 0;
			for (int k=0; k<dim; k++)
			{
				c[i*dim+j] ^= multiply(a[i*dim+k], b[k*dim+j], alpha_to, index_of);
			}
		}
	}
}








__global__ void gpuKernelMan(uint8_t* matrix, uint8_t* c, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of)
{
	int chunkSize = bdx*gdx;
	for (int i = bx * bdx + tx; i < bx * bdx + tx + int((n/((1)<<(25))*3/2)) * chunkSize + 1; i+=chunkSize)
	{
	__shared__ int d;
	__shared__ uint8_t save[4 * 512];
	__shared__ uint8_t cs[512*4];

	if(tx==0) {
		d = (m<=9 ? m : 9);
	}
	__syncthreads();
	if (i<((n)))
	{
		for (int counter = 0; counter < 4; counter++)
		{
			cs[tx*4+counter] = c[i*4+counter];
		}
	}
	__syncthreads();

	
	for (int j = 0; j < d; j++)
	{
		if ((tx >= ((1)<<(j))) && (tx<n))
		{
			 
			matrix_to_vector_multiplyd(&matrix[j*16], &(cs[(tx*4-4*((1)<<(j)))]), &save[tx*4], 4, alpha_to, index_of);
		}
		__syncthreads();	
		if ((tx >= ((1)<<(j))) && (tx<n))
		{	
			for (int counter = 0; counter < 4; counter++)
			{	
				cs[tx*4 + counter] ^= save[tx*4+counter];
			} 


		}
		__syncthreads();

	}

	if (i<((n)))
	{
		for (int counter = 0; counter < 4; counter++)
		{
			c[i*4+counter] = cs[tx*4+counter];
		}
	}
	__syncthreads();

	}
}





__global__ void KernelFunc(uint8_t* matrix, uint8_t* c, const int m, const int n, const int f, uint8_t* alpha_to, uint8_t* index_of)
{
	int chunkSize = gdx;
	for (int i = 0; i < int(n/((1)<<(26)))+1; i++)
	{
	__shared__ uint8_t save[4 * 512];
	int s = ((1)<<(f));

	matrix_to_vector_multiplyd(&matrix[(tx + (((bx+i*chunkSize) % s) * bdx)) * 16], &(c[(2048*s+4096*s*(int((bx+i*chunkSize) / s))-4)]), &save[4*tx], 4, alpha_to, index_of);
	__syncthreads();
	for (int counter = 0; counter < 4; counter++)
	{
		c[(2048*s)+tx*4+counter+2048*((bx+i*chunkSize) % s)+4096*s*(int((bx+i*chunkSize) / s))] ^= save[4*tx+counter];
	}
	__syncthreads();
	}
}




__global__ void KernelMultiply(uint8_t* matrix, uint8_t* matrixd, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of)
{
	int i = tx;
	__shared__ uint8_t save[16 * 512];
	__shared__ uint8_t matrixds[16 * 512];
	for (int counter = 0; counter < 16; counter++)
	{
		matrixds[i*16 + counter] = matrix[counter];
	}
	__syncthreads();
	
	for (int j = 0; j < 9; j++)
	{
		if ((i+((1)<<(j))) < 512)
		{
			matrix_to_matrix_multiplyd(&matrixds[i*16], &matrixds[i*16+16*((1)<<(j))], &save[tx*16], 4, alpha_to, index_of);
		}
		__syncthreads();
		if ((i+((1)<<(j))) < 512)
		{
			for (int counter = 0; counter < 16; counter++)
			{	
				matrixds[i*16+16*((1)<<(j)) + counter] = save[tx*16 + counter];
			}	
		}
		__syncthreads();
	}
	
	for (int co = 0; co < n/1024; co++)
	{
		for (int counter = 0; counter < 16; counter++)
		{
			matrixd[i*16 + counter + 512*16*co] = matrixds[i*16 + counter];
		}
	}
	__syncthreads();
	
}





__global__ void KernelMultiplyFunc(uint8_t* matrixd, const int m, const int n, const int f,uint8_t* alpha_to, uint8_t* index_of)
{

	__shared__ uint8_t save[16 * 512];
	int s = ((1)<<(f));


	matrix_to_matrix_multiplyd(&matrixd[((int(bx/s))*1024*s+(512*s)-1)*16], &matrixd[((int(bx/s))*1024*s+(bx%s)*512+tx+(512*s))*16], &save[16*tx], 4, alpha_to, index_of);
	__syncthreads();
	for (int counter = 0; counter < 16; counter++)
	{
		matrixd[(((int(bx/s))*1024*s+(bx%s)*512+tx+(512*s))*16)+counter] = save[16*tx+counter];
	}
	__syncthreads();
	
}






//-----------------------------------------------------------------------------
void gpuKernel(const uint8_t* const a, const uint8_t* const matrix, uint8_t* c, const int m, const int n, uint8_t* alpha_to, uint8_t* index_of)
{
	uint8_t* matrixdd;
	uint8_t* cd;
	uint8_t* alpha_tod;
	uint8_t* index_ofd;
	//uint8_t* mymatrix;
	uint8_t* amatrix;
	uint8_t* matrixd;
	uint8_t* matrixtoM;
	//mymatrix = (uint8_t*)malloc(((n/2)+1) * 4 * 4 * sizeof(uint8_t));
	matrixd = (uint8_t*)malloc(((m>9 ? 9 : m)) * 4 * 4 * sizeof(uint8_t));

    HANDLE_ERROR(cudaMalloc((void**)&matrixdd, ((m>9 ? 9 : m)) * 4 * 4 * sizeof(uint8_t)));
	HANDLE_ERROR(cudaMalloc((void**)&matrixtoM, 4 * 4 * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&cd, n*4 * sizeof(uint8_t)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_tod, 256 * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&index_ofd, 256 * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc((void**)&amatrix, ((n/2)) * 16 * sizeof(uint8_t)));
	HANDLE_ERROR(cudaMemcpy(cd, a, n*4 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_tod, alpha_to, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(index_ofd, index_of, 256 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(matrixtoM, matrix, 16 * sizeof(uint8_t), cudaMemcpyHostToDevice));

	if (m>9)
	{ 
		/*
		for (int co = 0; co < 16; co++)
		{
			mymatrix[co] = matrix[co];
		}
		for (int co = 0; co < n/2; co++)
		{
			matrix_to_matrix_multiply(&mymatrix[0], &mymatrix[co*16], &mymatrix[(co+1)*16], 4, alpha_to, index_of);
		}
		
		 
		for (int co = 0; co < 16; co++)
		{
			mymatrix[co] = matrix[co];
		}
		for (int co = 0; co < 511; co++)
		{
			matrix_to_matrix_multiply(&mymatrix[0], &mymatrix[co*16], &mymatrix[(co+1)*16], 4, alpha_to, index_of);
		}
		
		for (int counter = 512*16; counter < (n/2)*16; counter++)
		{
			mymatrix[counter] = mymatrix[counter%(512*16)];
		} 
		
		HANDLE_ERROR(cudaMemcpy(amatrix, mymatrix, ((n/2)+1) * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice));
		*/
		dim3 block1(512,1,1);
    	dim3 grid1(1,1,1); 
		KernelMultiply<<< grid1, block1 >>>(matrixtoM, amatrix, m, n, alpha_tod, index_ofd); 
		
		for (int f=0; f<(m-10); f++) {
			dim3 block2(512,1,1);
    		dim3 grid2(n/2048,1,1); 
			KernelMultiplyFunc<<< grid2, block2 >>>(amatrix, m, n, f, alpha_tod, index_ofd); 
		}
		
	} 
	for (int co = 0; co < 16; co++)
	{
		matrixd[co] = matrix[co];
	}
	for (int co = 0; co < (m>9 ? 8 : m-1); co++)
	{
		matrix_to_matrix_multiply(&matrixd[co*16], &matrixd[co*16], &matrixd[(co+1)*16], 4, alpha_to, index_of);
	}
	

    HANDLE_ERROR(cudaMemcpy(matrixdd, matrixd, ((m>9 ? 9 : m)) * 4 * 4 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(amatrix, mymatrix, ((n/2)) * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice));

	int num = (n > 512 ? 512 : n);
	int num2 = (n > 512 ? (m>=25 ? (1)<<(15) : n/512) : 1);
	dim3 blockSize(num,1,1);
    dim3 gridSize(num2,1,1); 
	gpuKernelMan<<< gridSize, blockSize >>>(matrixdd, cd, m, n, alpha_tod, index_ofd); 

	dim3 blockSize2(512,1,1);
    dim3 gridSize2(m>=26 ? (1)<<(15) : n/1024,1,1); 
	for (int f=0; f<(m-9); f++) {
	KernelFunc<<< gridSize2, blockSize2 >>>(amatrix, cd, m, n, f, alpha_tod, index_ofd); 
	}
    
	HANDLE_ERROR(cudaMemcpy(c, cd, n*4 * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	//free(mymatrix);
	free(matrixd);
    HANDLE_ERROR(cudaFree(matrixdd));
    HANDLE_ERROR(cudaFree(cd));
	HANDLE_ERROR(cudaFree(alpha_tod));
    HANDLE_ERROR(cudaFree(index_ofd));
	HANDLE_ERROR(cudaFree(amatrix));

	
}
