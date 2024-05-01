#include <iostream>
#include <complex>
#include"\include\cblas.h"
//#include"mkl.h"

using namespace std;


float** new_matrix(const int N) {
    float** a = new float* [N];
    float* memp = new float[N * N];
    for (int i = 0; i < N; i++) {
        a[i] = &memp[i * N];
    }
    return a;
}

void matrix_multiply(float** A, float** B, float** result, int N) {
#pragma omp parallel for num_threads(128)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = 0.0f;
            for (int k = 0; k < N; ++k) {
                result[i][j] += A[i][k] * B[k][j]; 
            }
        }
    }
}


bool compare_matrices(float** A, float** B, int rows, int cols, float epsilon = 1e-2f) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fabs(A[i][j] - B[i][j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

void optimized_matrix_multiply(float** A, float** B, float** C, int N, int block_size = 64) {
#pragma omp parallel for
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                for (int ii = i; ii < min(i + block_size, N); ++ii) {
                    for (int jj = j; jj < min(j + block_size, N); ++jj) {
                        for (int kk = k; kk < min(k + block_size, N); ++kk) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
}


int main()

{

    setlocale(LC_ALL, "Russian");

    const int N = 2048;

    float** a = new_matrix(N);
    float** b = new_matrix(N);
    float** c1 = new_matrix(N);
    float** c2 = new_matrix(N);


    cout << "Перебейнос Константин Евгеньевич \t" << "090301-ПОВа-023" << endl<<endl;


    // Заполнение матриц случайными значениями от 0 до 1

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            a[i][j] = static_cast<float>(rand()) / RAND_MAX * 5.0f;
            b[i][j] = static_cast<float>(rand()) / RAND_MAX * 5.0f;
        }


    cout << "a[10][20] = " << a[10][20] << "\n";

    cout << "b[20][10] = " << b[20][10] << "\n";



    float alpha = 1.0f, beta = 0.0f;



    clock_t start = clock();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, alpha, a[0], N, b[0], N, beta, c1[0], N);

    clock_t end = clock();

    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    cout << "Время выполнения умножения матриц: " << elapsed_secs << " секунд.\n";

    cout << "Достигнутая производительность " << 2.0 * (double)N * (double)N * N / elapsed_secs * 1.0e-6 << " MFlops.\n";



    cout << "c1[10][20] = " << c1[10][20] << "\n";



    start = clock();

    matrix_multiply(a, b, c2, N);

    end = clock();

    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    cout << "Время выполнения умножения матриц: " << elapsed_secs << " секунд.\n";

    cout << "Достигнутая производительность " << 2.0 * (double)N * (double)N * N / elapsed_secs * 1.0e-6 << " MFlops.\n";

    cout << "c2[10][20] = " << c2[10][20] << "\n";


    start = clock();

    optimized_matrix_multiply(a, b, c2, N);

    end = clock();

    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    cout << "Время выполнения умножения матриц: " << elapsed_secs << " секунд.\n";

    cout << "Достигнутая производительность " << 2.0 * (double)N * (double)N * N / elapsed_secs * 1.0e-6 << " MFlops.\n";

    cout << "c2[10][20] = " << c2[10][20] << "\n";


    if (compare_matrices(c1, c2, N, N))

        cout << "Матрицы c1 и c2 равны.\n";

    else

        cout << "Матрицы c1 и c2 не равны!\n";


    system("pause");

    return 0;

}

