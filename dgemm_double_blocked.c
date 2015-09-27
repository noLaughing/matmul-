const char* dgemm_desc = "Simple blocked dgemm.";
#include <stdlib.h>
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 256)
#endif

#ifndef BLOCK_SIZE_2
#define BLOCK_SIZE_2 ((int) 32)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                // swapped indices due to AT
                cij += A[i*lda+k] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}

void transpose(const int dim, const double *A, double *AT) {
    int i, j;
    for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; j++) {
            AT[i + j*dim] = A[j + i*dim];
        }
    }
}

// read (block_size x block_size) matrix from A(i,j) into block_A
void read_to_contiguous(const int M, const double *A, double *block_A,
                        const int i, const int j, const int block_size)
{
    // guard against matrix edge case
    const int mBound = (i+block_size > M? M-i : block_size);
    const int nBound = (j+block_size > M? M-j : block_size);
    
    // offset is index of upper left corner of desired block within A
    const int offset = i + M*j;
    int m, n;
    for (n = 0; n < nBound; ++n) {
        for (m = 0; m < mBound; ++m) {
            block_A[m + block_size*n] = A[offset + m + M*n];
        }
    }
}

// write block_C into C(i,j)
void write_block_to_original(const int M, const double *block_C, double *C,
                             const int i, const int j, const int block_size)
{
    // guard against matrix edge case
    const int mBound = (i+block_size > M? M-i : block_size);
    const int nBound = (j+block_size > M? M-j : block_size);
    
    int m, n;
    const int offset = i + M*j;
    for (n = 0; n < nBound; ++n) {
        for (m = 0; m < mBound; ++m) {
            C[offset + m + M*n] = block_C[m + block_size*n];
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    // compute loop lengths to stay inside matrix bounds
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void second_block(const int M, const double *A, const double *B, double *C)
{
	double* AT = (double*) malloc(M * M * sizeof(double));
    transpose(M, A, AT);
	
	const int n_blocks = M / BLOCK_SIZE_2 + (M%BLOCK_SIZE_2? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE_2;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE_2;
            double* block_C = (double*) malloc(BLOCK_SIZE_2 * BLOCK_SIZE_2
                                       * sizeof(double));
            read_to_contiguous(M, C, block_C, i, j, BLOCK_SIZE_2);
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE_2;
                double* block_A = (double*) malloc(BLOCK_SIZE_2 * BLOCK_SIZE_2
                                       * sizeof(double));
                double* block_B = (double*) malloc(BLOCK_SIZE_2 * BLOCK_SIZE_2
                                       * sizeof(double));
                
                // swapped indices due to AT
                read_to_contiguous(M, A, block_A, k, i, BLOCK_SIZE_2);
                read_to_contiguous(M, B, block_B, k, j, BLOCK_SIZE_2);
                
                basic_dgemm(BLOCK_SIZE_2, BLOCK_SIZE_2, BLOCK_SIZE_2, BLOCK_SIZE_2,
                            block_A, block_B, block_C);
            }
            write_block_to_original(M, block_C, C, i, j, BLOCK_SIZE_2);
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    double* AT = (double*) malloc(M * M * sizeof(double));
    transpose(M, A, AT);
    
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    printf("Dimension %d has %d blocks\n", M, n_blocks);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            double* block_C = (double*) malloc(BLOCK_SIZE * BLOCK_SIZE
                                       * sizeof(double));
            read_to_contiguous(M, C, block_C, i, j, BLOCK_SIZE);
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                double* block_A = (double*) malloc(BLOCK_SIZE * BLOCK_SIZE
                                       * sizeof(double));
                double* block_B = (double*) malloc(BLOCK_SIZE * BLOCK_SIZE
                                       * sizeof(double));
                
                // swapped indices due to AT
                read_to_contiguous(M, AT, block_A, k, i, BLOCK_SIZE);
                read_to_contiguous(M, B, block_B, k, j, BLOCK_SIZE);
                
                //basic_dgemm(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE,
                //            block_A, block_B, block_C);
				second_block(BLOCK_SIZE, block_A, block_B, block_C);
            }
            write_block_to_original(M, block_C, C, i, j, BLOCK_SIZE);
        }
    }
}

